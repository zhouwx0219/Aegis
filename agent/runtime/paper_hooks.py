"""Operation hooks that execute the paper ATCC action inside a transaction."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from .context import LockAction, LockClass, TransactionPhase
from .operation_interceptor import NoopTransactionHooks


class PaperATCCHooks(NoopTransactionHooks):
    def __init__(self, manager: Any):
        self.manager = manager

    @staticmethod
    def enabled(txn: Any) -> bool:
        return bool(txn.metadata.get("paper_atcc", False))

    @staticmethod
    def coordinated_backend(txn: Any) -> bool:
        return bool(txn.metadata.get("paper_atcc_backend", False))

    def on_begin(self, txn: Any) -> None:
        if not self.enabled(txn):
            return
        # Category write protection is a commit-admission decision.  Holding
        # every planned write through multi-round Agent reasoning creates a
        # background convoy; only an exact object learned from a failed retry
        # is promoted early.
        txn.metadata["_defer_policy_write_locks"] = bool(
            txn.metadata.get("commit_admission_write_protection", False)
        )
        planned = tuple(
            str(value)
            for value in txn.metadata.get("_planned_snapshot_object_ids", ())
        )
        profiled_hot = self.manager.hotness_tracker.hot_targets(planned)
        profiled_hot_ratio = len(profiled_hot) / len(set(planned)) if planned else 0.0
        txn.metadata["_profiled_hot_ratio"] = profiled_hot_ratio
        context = dict(txn.metadata.get("context", {}) or {})
        ycsb_high = bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and str(context.get("level", "")).strip().lower() == "high"
        )
        txn.metadata["_version_risk_exact_mode"] = ycsb_high
        metrics = self.manager.paper_runtime_metrics()
        planned_reads = set(planned) - set(txn.context.planned_write_targets)
        version_risk_reads = (
            self.manager.version_manager.top_background_changed(
                planned_reads,
                limit=2,
                min_changes=2,
                min_share=0.005,
                min_total_changes=32,
            )
            if ycsb_high and txn.context.retry_count <= 1
            else ()
        )
        txn.metadata["_version_risk_read_targets"] = version_risk_reads
        if not self.manager.low_conflict_occ_guard:
            txn.metadata["_cold_occ_fast_task"] = False
            return
        cold_profile = bool(
            planned
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and float(metrics.get("conflict_abort_rate", 0.0) or 0.0) < 0.30
            and int(metrics.get("waiter_count", 0) or 0) == 0
            and not version_risk_reads
        )
        txn.metadata["_cold_occ_fast_task"] = cold_profile

    def before_read(self, txn: Any, object_id: str) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        hot = self.manager.is_hot(object_id)
        if hot:
            txn.context.hot_read_targets.add(str(object_id))
        key = str(object_id)
        exact_mode = bool(txn.metadata.get("_version_risk_exact_mode", False))
        exact_version_risk = key in set(
            txn.metadata.get("_version_risk_read_targets", ())
        )
        exact_write = key in txn.context.retry_conflict_write_targets
        exact_read = key in txn.context.retry_conflict_read_targets
        planned_write = key in txn.context.planned_write_targets
        profiled_hot = self.manager.hotness_tracker.is_profiled_hot(key)
        profiled_shared = self.manager.hotness_tracker.is_profiled_shared(key)
        if (
            (
                profiled_hot
                or (
                    profiled_shared
                    and (
                        not self.manager.low_conflict_occ_guard
                        or hot
                        or txn.context.retry_count > 0
                        or txn.context.retry_conflict_mask
                    )
                )
            )
            and txn.context.phase == TransactionPhase.EXPLORE
            and not txn.metadata.get("_version_risk_exact_mode", False)
        ):
            guard_bit = (
                LockClass.HOT_WRITE if planned_write and hot
                else LockClass.COLD_WRITE if planned_write
                else LockClass.HOT_READ if hot
                else LockClass.COLD_READ
            )
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                )
        write_class_protected = bool(
            not exact_mode
            and txn.context.action.protects(hot=hot, write=True)
        )
        if exact_write or (
            planned_write
            and write_class_protected
            and not txn.metadata.get("_defer_policy_write_locks", False)
        ):
            self.manager.refresh_atcc_priority(txn)
            if key in txn.read_set:
                snapshot = txn.snapshot[key]
                self.manager.atcc_locks.validate_and_wlock(
                    key,
                    txn.context,
                    snapshot.version,
                    lambda: int(
                        self.manager.version_manager.read_committed(key).version
                    ),
                )
            else:
                # _ensure_snapshot has prepared a value, but it has not yet
                # been returned to the Agent. Establish the WLock first and
                # read the current committed base inside that protection.
                self.manager.atcc_locks.wlock(key, txn.context)
                txn.refresh_unobserved_locked_snapshot(key)
            txn.context.policy_write_lock_targets.add(key)
        elif exact_version_risk:
            self.manager.refresh_atcc_priority(txn)
            # Keep the old committed version through the long reasoning
            # interval without holding an RLock. Commit installs a short read
            # guard, so Agent writers remain serialized while background
            # publishers can keep creating private versions.
            self.manager.ensure_snapshot_epoch(txn)
            snapshot = txn.snapshot[key]
            if self.manager.version_manager.can_lock_pinned_version(
                txn.context.snapshot_epoch,
                key,
                int(snapshot.version),
                tid=txn.context.tid,
            ):
                txn.metadata.setdefault(
                    "_version_risk_pinned_read_targets", set()
                ).add(key)
                self.manager.version_manager.note_version_risk_read_lock()
                return
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: (
                    int(snapshot.version)
                    if self.manager.version_manager.can_lock_pinned_version(
                        txn.context.snapshot_epoch,
                        key,
                        int(snapshot.version),
                        tid=txn.context.tid,
                    )
                    else int(
                        self.manager.version_manager.read_committed(
                            key
                        ).version
                    )
                ),
            )
            txn.context.policy_read_lock_targets.add(key)
        elif (
            exact_read
            or (
                not exact_mode
                and txn.context.action.protects(hot=hot, write=False)
            )
        ):
            self.manager.refresh_atcc_priority(txn)
            snapshot = txn.snapshot[key]
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: (
                    int(snapshot.version)
                    if self.manager.version_manager.can_lock_pinned_version(
                        txn.context.snapshot_epoch,
                        key,
                        int(snapshot.version),
                        tid=txn.context.tid,
                    )
                    else int(
                        self.manager.version_manager.read_committed(
                            key
                        ).version
                    )
                ),
            )
            txn.context.policy_read_lock_targets.add(key)

    def before_write(self, txn: Any, object_id: str) -> None:
        if self.coordinated_backend(txn):
            # Short-lived backend writes remain optimistic and buffered; the
            # unified commit protocol acquires their WLocks as one write set.
            return
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        hot = self.manager.is_hot(object_id)
        if hot:
            txn.context.hot_write_targets.add(str(object_id))
        key = str(object_id)
        exact_write = key in txn.context.retry_conflict_write_targets
        profiled_shared = self.manager.hotness_tracker.is_profiled_shared(key)
        if (
            profiled_shared
            and (
                not self.manager.low_conflict_occ_guard
                or hot
                or txn.context.retry_count > 0
                or txn.context.retry_conflict_mask
            )
            and txn.context.phase in {
                TransactionPhase.REFINE,
                TransactionPhase.COMMIT,
            }
            and not txn.metadata.get("_version_risk_exact_mode", False)
        ):
            guard_bit = LockClass.HOT_WRITE if hot else LockClass.COLD_WRITE
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                )
        exact_mode = bool(txn.metadata.get("_version_risk_exact_mode", False))
        category_write_protected = bool(
            not exact_mode
            and txn.context.action.protects(hot=hot, write=True)
        )
        if (
            category_write_protected
            and txn.metadata.get("_defer_policy_write_locks", False)
            and not exact_write
        ):
            txn.context.policy_write_lock_targets.add(key)
            return
        if exact_write or category_write_protected:
            self.manager.refresh_atcc_priority(txn)
            write_class = (
                LockClass.HOT_WRITE if hot else LockClass.COLD_WRITE
            )
            late_action_write = bool(
                txn.context.phase
                in {TransactionPhase.REFINE, TransactionPhase.COMMIT}
            )
            if (
                exact_write
                or late_action_write
                or LockClass(int(txn.context.retry_conflict_mask)) & write_class
            ):
                # A known failed object is protected regardless of hot/cold
                # reclassification. In a late phase the selected action is
                # executed faithfully; unselected classes remain private.
                if key in txn.read_set:
                    snapshot = txn.snapshot[key]
                    self.manager.atcc_locks.validate_and_wlock(
                        key,
                        txn.context,
                        snapshot.version,
                        lambda: int(
                            self.manager.version_manager.read_committed(key).version
                        ),
                    )
                else:
                    # _ensure_snapshot materializes the transaction snapshot
                    # before this hook, but a blind write has not observed it.
                    # Lock first, then rebase the private write on the latest
                    # committed version while publication is excluded.
                    self.manager.atcc_locks.wlock(key, txn.context)
                    txn.refresh_unobserved_locked_snapshot(key)
            # Otherwise the write remains private until unified commit.
            txn.context.policy_write_lock_targets.add(key)

    def before_commit(self, txn: Any) -> None:
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if self.enabled(txn):
            self.manager.interceptor.account_agent_interval(txn)
            if not txn.metadata.get("_version_risk_exact_mode", False):
                self._select_and_apply(txn, self._state(txn))
        if self.enabled(txn) or (
            self.coordinated_backend(txn) and bool(txn.write_set)
        ):
            self.manager.refresh_atcc_priority(txn)

    def on_phase_change(self, txn: Any, phase: TransactionPhase) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if txn.metadata.get("_version_risk_exact_mode", False):
            return
        initial_explore = (
            phase == TransactionPhase.EXPLORE
            and not txn.context.read_versions
            and not txn.context.write_targets
        )
        if initial_explore:
            if self.manager.collect_trajectories:
                state = self._state(txn)
                self.manager.trajectory_collector.decision(
                    txn.context.tid,
                    state,
                    int(txn.context.action.protected),
                    1.0,
                    blocked_time_ms=txn.context.blocked_time_ms,
                    lock_count=self._lock_count(txn),
                    **self._externality_metrics(txn),
                )
            return
        state = self._state(txn)
        self._select_and_apply(txn, state)

    def _select_and_apply(self, txn: Any, state: Any) -> None:
        policy_started = time.perf_counter()
        policy = self.manager.paper_policy.snapshot()
        distribution_sampler = getattr(policy, "select_with_distribution", None)
        behavior_action_probabilities = ()
        if distribution_sampler is not None:
            selected, behavior_action_probabilities = distribution_sampler(state)
            behavior_probability = behavior_action_probabilities[
                int(selected.protected)
            ]
        else:
            sampler = getattr(policy, "select_with_probability", None)
            if sampler is not None:
                selected, behavior_probability = sampler(state)
            else:
                selected = policy.select(state)
                behavior_probability = 1.0
        if self._low_conflict_occ_guard(txn, state):
            selected = LockAction(LockClass.NONE)
        selected = LockAction(selected.protected | txn.context.action.protected)
        selected = self._apply_protection_guard(txn, state, selected)
        self.manager.add_commit_timing(
            txn,
            "policy",
            (time.perf_counter() - policy_started) * 1000.0,
        )
        if self.manager.collect_trajectories:
            self.manager.trajectory_collector.decision(
                txn.context.tid,
                state,
                int(selected.protected),
                behavior_probability,
                blocked_time_ms=txn.context.blocked_time_ms,
                lock_count=self._lock_count(txn),
                behavior_action_probabilities=behavior_action_probabilities,
                **self._externality_metrics(txn),
            )
        if selected.protected != txn.context.action.protected:
            self.manager.transition_atcc_action(txn, selected)
            # Action-transition lock wait is represented by Blocked(T), not
            # the separate inter-round agent interval in the priority formula.
            self.manager.interceptor.reset_agent_interval(txn)

    @staticmethod
    def _apply_protection_guard(txn: Any, state: Any, selected: LockAction) -> LockAction:
        """Protect observed hot reads before late phases under conflict pressure."""
        if txn.context.phase not in {TransactionPhase.REFINE, TransactionPhase.COMMIT}:
            return selected
        read_targets = set(txn.context.read_versions)
        hot_reads = read_targets & set(txn.context.hot_read_targets)
        hot_read_ratio = len(hot_reads) / len(read_targets) if read_targets else 0.0
        validation_pressure = max(
            0.0,
            float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0),
        )
        guarded = selected.protected
        if hot_read_ratio >= 0.25:
            guarded |= LockClass.HOT_READ
        if read_targets and validation_pressure >= 0.30:
            guarded |= LockClass.HOT_READ if hot_reads else LockClass.NONE
            if read_targets - hot_reads:
                guarded |= LockClass.COLD_READ
        if guarded != selected.protected:
            return LockAction(guarded)
        return selected

    def _low_conflict_occ_guard(self, txn: Any, state: Any) -> bool:
        """Keep the zero-protection action while every observed risk signal is cold."""
        return bool(
            self.manager.low_conflict_occ_guard
            and txn.context.action.protected == LockClass.NONE
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and not txn.context.hot_read_targets
            and not txn.context.hot_write_targets
            and float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0)
            <= 0.001
            and int(getattr(state, "global_waiter_count", 0) or 0) == 0
        )

    def on_finish(self, txn: Any) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if not self.manager.collect_trajectories:
            return
        state = self._state(txn)
        committed = bool(getattr(getattr(txn, "result", None), "committed", False))
        self.manager.trajectory_collector.finish(
            txn.context.tid,
            state,
            committed=committed,
            operation_cost_ms=txn.context.operation_cost_ms,
            agent_cost_ms=txn.context.agent_cost_ms,
            retry_cost_ms=txn.context.prior_retry_cost_ms,
            blocked_time_ms=txn.context.blocked_time_ms,
            lock_count=self._lock_count(txn),
            **self._externality_metrics(txn),
        )

    def _state(self, txn: Any) -> Any:
        hotness_started = time.perf_counter()
        self.manager.refresh_hot_targets(txn)
        self.manager.add_commit_timing(
            txn,
            "hotness",
            (time.perf_counter() - hotness_started) * 1000.0,
        )
        state = self.manager.state_collector.snapshot(txn.context)
        metrics = self.manager.paper_runtime_metrics()
        return dataclasses.replace(
            state,
            global_active_transactions=int(metrics["active_transactions"]),
            global_waiter_count=int(metrics["waiter_count"]),
            global_abort_rate=float(metrics["abort_rate"]),
            global_throughput=float(metrics["throughput"]),
            global_avg_latency_ms=float(metrics["average_latency_ms"]),
            global_tail_latency_ms=float(metrics["tail_latency_ms"]),
            global_agent_task_throughput=float(metrics["agent_task_throughput"]),
            global_agent_task_avg_latency_ms=float(metrics["agent_task_average_latency_ms"]),
            global_agent_task_tail_latency_ms=float(metrics["agent_task_tail_latency_ms"]),
            global_conflict_abort_rate=float(metrics["conflict_abort_rate"]),
            global_background_throughput=float(metrics["background_throughput"]),
            global_background_abort_rate=float(metrics["background_abort_rate"]),
        )

    @staticmethod
    def _externality_metrics(txn: Any) -> dict[str, float | int]:
        context = txn.context
        return {
            "background_blocked_ms_caused": context.background_blocked_ms_caused,
            "background_aborts_caused": context.background_aborts_caused,
            "agent_blocked_ms_caused": context.agent_blocked_ms_caused,
            "agent_aborts_caused": context.agent_aborts_caused,
        }

    @staticmethod
    def _lock_count(txn: Any) -> int:
        return len(
            txn.context.policy_read_lock_targets
            | txn.context.policy_write_lock_targets
        )
