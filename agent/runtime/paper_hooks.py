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

    @staticmethod
    def online_observed(txn: Any) -> bool:
        return (
            str(txn.metadata.get("access_set_visibility", "")).strip().lower()
            == "online_observed"
        )

    @classmethod
    def online_ycsb_high_mixed(cls, txn: Any) -> bool:
        context = dict(txn.metadata.get("context", {}) or {})
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        return bool(
            cls.online_observed(txn)
            and str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and str(context.get("level", "")).strip().lower() == "high"
            and int(agentic.get("background_workers", 0) or 0) > 0
        )

    @classmethod
    def mark_online_bypass_read(cls, txn: Any, object_id: str) -> None:
        if cls.online_ycsb_high_mixed(txn) and txn.metadata.get(
            "paper_atcc_optimized", False
        ):
            txn.metadata.setdefault("_online_bypass_read_targets", set()).add(
                str(object_id)
            )

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
        level = str(context.get("level", "")).strip().lower()
        workload = str(txn.metadata.get("workload", "")).strip().lower()
        ycsb_high = bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and level == "high"
        )
        txn.metadata["_version_risk_exact_mode"] = bool(
            ycsb_high and txn.metadata.get("paper_atcc_optimized", False)
        )
        metrics = self.manager.paper_runtime_metrics()
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        background_workers = int(agentic.get("background_workers", 0) or 0)
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
            and (
                (
                    level == "low"
                    and not (workload == "tpcc" and background_workers > 0)
                )
                or (level == "medium" and background_workers <= 0)
            )
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
        self._start_online_observed_prefix(txn)
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
        if self.online_ycsb_high_mixed(txn) and exact_write:
            # A previous conflict may identify this exact future write, but it
            # must not turn into a task-wide or reasoning-long WLock. Treat its
            # read as a bypassable observed dependency and upgrade only when
            # the write operation is actually issued.
            exact_write = False
            exact_read = True
        planned_write = key in txn.context.planned_write_targets
        deferred_tpcc_root_write = bool(
            txn.metadata.get("_deferred_reasoning_replay", False)
            and str(txn.metadata.get("workload", "")).strip().lower() == "tpcc"
            and str(
                dict(txn.metadata.get("context", {}) or {}).get("level", "")
            ).strip().lower()
            == "high"
            and planned_write
            and key.startswith(
                (
                    "tpcc:warehouse:",
                    "tpcc:district:",
                    "tpcc:customer:",
                )
            )
        )
        if deferred_tpcc_root_write:
            # The reasoning interval has already elapsed, so an exact WLock
            # protects only the short replay/commit suffix.  This serializes
            # one-warehouse roots before they are observed without recreating
            # a long Agent-side 2PL convoy.
            exact_write = True
        exact_retry_only = self._tpcc_all_agent_exact_retry(txn)
        if exact_retry_only:
            exact_write = False
            exact_read = False
        if key in txn.context.held_write_locks:
            txn.context.policy_write_lock_targets.add(key)
            return
        if key in txn.context.held_read_locks and not (exact_write or planned_write):
            txn.context.policy_read_lock_targets.add(key)
            return
        if exact_version_risk:
            guard_bit = LockClass.HOT_READ if hot else LockClass.COLD_READ
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                    timeout_s=self._action_transition_timeout_s(txn),
                )
        write_class_protected = bool(
            txn.context.action.protects(hot=hot, write=True)
            and not exact_retry_only
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
        elif (
            exact_version_risk
            and exact_mode
            and self.online_observed(txn)
        ):
            self.manager.refresh_atcc_priority(txn)
            snapshot = txn.snapshot[key]
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_read_lock_targets.add(key)
            self.mark_online_bypass_read(txn, key)
            self.manager.version_manager.note_version_risk_read_lock()
        elif (
            exact_version_risk
            and exact_mode
            and not txn.context.planned_write_targets
            and not txn.write_set
        ):
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
            self.mark_online_bypass_read(txn, key)
            if exact_version_risk:
                self.manager.version_manager.note_version_risk_read_lock()

        elif (
            exact_read
            or (
                txn.context.action.protects(hot=hot, write=False)
                and not exact_retry_only
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
            self.mark_online_bypass_read(txn, key)

            if exact_version_risk:
                self.manager.version_manager.note_version_risk_read_lock()

    def before_write(self, txn: Any, object_id: str) -> None:
        if self.coordinated_backend(txn):
            # Short-lived backend writes remain optimistic and buffered; the
            # unified commit protocol acquires their WLocks as one write set.
            return
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        self._start_online_observed_prefix(txn)
        hot = self.manager.is_hot(object_id)
        if hot:
            txn.context.hot_write_targets.add(str(object_id))
        key = str(object_id)
        exact_write = key in txn.context.retry_conflict_write_targets
        exact_retry_only = self._tpcc_all_agent_exact_retry(txn)
        if exact_retry_only:
            exact_write = False
        if key in txn.context.held_write_locks:
            txn.context.policy_write_lock_targets.add(key)
            return
        profiled_shared = self.manager.hotness_tracker.is_profiled_shared(key)
        if (
            profiled_shared
            and not exact_retry_only
            and (
                txn.context.retry_count > 0
                or txn.context.retry_conflict_mask
            )
            and txn.context.phase in {
                TransactionPhase.REFINE,
                TransactionPhase.COMMIT,
            }
        ):
            guard_bit = LockClass.HOT_WRITE if hot else LockClass.COLD_WRITE
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                    timeout_s=self._action_transition_timeout_s(txn),
                )
        category_write_protected = bool(
            txn.context.action.protects(hot=hot, write=True)
            and not exact_retry_only
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
            # In strict paper mode, selecting a write class means acquiring
            # its WLock when the object is accessed, independent of phase.
            # Only paper-atcc-opt may defer that acquisition to commit.
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
            txn.context.policy_write_lock_targets.add(key)

    def before_commit(self, txn: Any) -> None:
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if self.enabled(txn):
            self.manager.interceptor.account_agent_interval(txn)
            self._select_and_apply(txn, self._state(txn))
            self._protect_tpcc_read_before_write(txn, require_evidence=False)
        if self.enabled(txn) or (
            self.coordinated_backend(txn) and bool(txn.write_set)
        ):
            self.manager.refresh_atcc_priority(txn)

    def _protect_tpcc_read_before_write(
        self,
        txn: Any,
        *,
        require_evidence: bool,
    ) -> None:
        """Late-protect exact TPC-C warehouse/district read-before-writes."""
        context = dict(txn.metadata.get("context", {}) or {})
        if (
            str(txn.metadata.get("workload", "")).strip().lower() != "tpcc"
            or str(context.get("level", "")).strip().lower() != "high"
        ):
            return
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        if (
            int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and not txn.metadata.get("_deferred_reasoning_replay", False)
        ):
            # Start one-warehouse all-Agent TPC-C optimistically. A failed
            # attempt feeds its exact root object into retry protection.
            txn.metadata["_tpcc_first_attempt_exact_only"] = True
            return
        candidates = set(txn.read_set) & (
            set(txn.write_set) | set(txn.context.planned_write_targets)
        )
        if self._tpcc_all_agent_exact_retry(txn):
            candidates &= set(txn.context.retry_conflict_write_targets)
        targets = tuple(
            sorted(
                object_id
                for object_id in candidates
                if object_id.startswith("tpcc:warehouse:")
                or object_id.startswith("tpcc:district:")
            )
        )
        newly_protected = []
        for object_id in targets:
            if object_id in txn.context.held_write_locks:
                continue
            parts = object_id.split(":")
            object_type = parts[1] if len(parts) > 1 else ""
            field = parts[-1] if parts else ""
            exact_changes, family_changes, total_changes = (
                self.manager.version_manager.background_change_evidence(
                    object_id,
                    prefix=f"tpcc:{object_type}:",
                    suffix=f":{field}",
                )
            )
            sufficient = bool(
                total_changes >= 32
                and (exact_changes >= 2 or family_changes >= 2)
            )
            self.manager.atcc_locks.note_tpcc_exact_guard_evidence(
                exact_changes=exact_changes,
                family_changes=family_changes,
                total_changes=total_changes,
                sufficient=sufficient,
            )
            if require_evidence and not sufficient:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_wlock(
                object_id,
                txn.context,
                int(txn.read_set[object_id].version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_write_lock_targets.add(object_id)
            newly_protected.append(object_id)
            txn.metadata.setdefault("_tpcc_exact_risk_targets", set()).add(
                object_id
            )
            self.manager.atcc_locks.note_tpcc_exact_risk_wlock(
                family_fallback=exact_changes < 2 <= family_changes,
            )
        if newly_protected:
            txn.metadata.setdefault("_tpcc_late_protected_targets", set()).update(
                newly_protected
            )

    def on_phase_change(self, txn: Any, phase: TransactionPhase) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        initial_explore = (
            phase == TransactionPhase.EXPLORE
            and not txn.context.read_versions
            and not txn.context.write_targets
        )
        if initial_explore:
            # A retry already knows the exact objects that invalidated the
            # previous attempt. Protect those objects before another long
            # interaction round starts; waiting until the object is visited
            # again leaves it exposed between the atomic snapshot and access.
            self._protect_retry_targets(txn)
            # Paper section 4.2 starts every transaction in OCC. The first
            # policy lookup occurs only after the initial operation batch,
            # when its observed access and cost state is available.
            return
        state = self._state(txn)
        self._select_and_apply(txn, state)
        if phase == TransactionPhase.COMMIT:
            # Keep explore/refine optimistic, then protect only the already
            # observed high-frequency root dependency before the write suffix.
            self._protect_tpcc_read_before_write(txn, require_evidence=True)

    def _protect_retry_targets(self, txn: Any) -> None:
        if txn.context.retry_count <= 0:
            return
        if self._tpcc_all_agent_exact_retry(txn):
            # TPC-C high reasoning is deferred before begin().  A cached retry
            # therefore has no long Agent interval inside the transaction.  Lock
            # only the roots that actually conflicted on the preceding attempt
            # at the replay boundary; waiting until commit lets the same
            # warehouse/district version change again and creates retry storms.
            # No future/unobserved target is introduced here.
            txn.metadata["_tpcc_retry_root_prelock"] = True
        read_targets = set(txn.context.retry_conflict_read_targets)
        write_targets = set(txn.context.retry_conflict_write_targets)
        defer_exact_writes = self.online_ycsb_high_mixed(txn)
        for object_id in sorted(() if defer_exact_writes else write_targets):
            if object_id in txn.context.held_write_locks:
                continue
            snapshot = txn.snapshot.get(object_id)
            if snapshot is None:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_wlock(
                object_id,
                txn.context,
                int(snapshot.version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_write_lock_targets.add(object_id)
        for object_id in sorted(read_targets - write_targets):
            if (
                object_id in txn.context.held_read_locks
                or object_id in txn.context.held_write_locks
            ):
                continue
            snapshot = txn.snapshot.get(object_id)
            if snapshot is None:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_rlock(
                object_id,
                txn.context,
                int(snapshot.version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_read_lock_targets.add(object_id)
            self.mark_online_bypass_read(txn, object_id)

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
            self.manager.transition_atcc_action(
                txn,
                selected,
                timeout_s=self._action_transition_timeout_s(txn),
            )
            # Action-transition lock wait is represented by Blocked(T), not
            # the separate inter-round agent interval in the priority formula.
            self.manager.interceptor.reset_agent_interval(txn)
        self._finish_online_observed_prefix(txn)

    def _apply_protection_guard(self, txn: Any, state: Any, selected: LockAction) -> LockAction:
        """Protect observed hot reads before late phases under conflict pressure."""
        context = dict(txn.metadata.get("context", {}) or {})
        level = str(context.get("level", "")).strip().lower()
        workload = str(txn.metadata.get("workload", "")).strip().lower()
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        tpcc_high_all_agent = bool(
            workload == "tpcc"
            and level == "high"
            and int(agentic.get("background_workers", 0) or 0) == 0
        )
        tpcc_high_deferred_replay = bool(
            workload == "tpcc"
            and level == "high"
            and txn.metadata.get("_deferred_reasoning_replay", False)
        )
        tpcc_high_all_agent_first_attempt = bool(
            tpcc_high_all_agent
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        )
        tpcc_high_exact_first_attempt = bool(
            (tpcc_high_all_agent or tpcc_high_deferred_replay)
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        )
        if (
            self.manager.low_conflict_occ_guard
            and level in {"low", "medium"}
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and txn.context.action.protected == LockClass.NONE
        ):
            # Low/medium contention is sensitive to unnecessary first-attempt
            # locks. Let an observed retry conflict activate the learned
            # monotonic protection path on the same logical transaction.
            return LockAction(LockClass.NONE)
        online_mixed_pressure = bool(
            self.online_ycsb_high_mixed(txn)
            and (
                int(agentic.get("background_workers", 0) or 0) >= 4
                or float(
                    getattr(state, "global_background_abort_rate", 0.0) or 0.0
                )
                >= 0.05
                or int(getattr(state, "global_waiter_count", 0) or 0) > 0
                or float(
                    getattr(state, "global_conflict_abort_rate", 0.0) or 0.0
                )
                >= 0.20
            )
        )
        if txn.context.phase not in {TransactionPhase.REFINE, TransactionPhase.COMMIT}:
            if online_mixed_pressure:
                observed_reads = set(txn.context.read_versions)
                if int(agentic.get("background_workers", 0) or 0) == 6:
                    # At the intermediate pressure bucket, two Agent slots can
                    # otherwise race past the first decision on a twice-seen
                    # key. Promote only those already-observed candidates; the
                    # 8-background-worker bucket deliberately keeps the normal
                    # HotnessTracker threshold to avoid over-coverage.
                    for object_id in observed_reads:
                        heat = self.manager.hotness_tracker.object_snapshot(
                            object_id
                        )
                        if int(heat.get("accesses", 0) or 0) >= 2:
                            txn.context.hot_read_targets.add(object_id)
                observed_hot_reads = bool(
                    observed_reads & set(txn.context.hot_read_targets)
                )
                return LockAction(
                    txn.context.action.protected
                    | (selected.protected & LockClass.HOT_READ)
                    | (LockClass.HOT_READ if observed_hot_reads else LockClass.NONE)
                )
            return selected
        read_targets = set(txn.context.read_versions)
        hot_reads = read_targets & set(txn.context.hot_read_targets)
        hot_read_ratio = len(hot_reads) / len(read_targets) if read_targets else 0.0
        validation_pressure = max(
            0.0,
            float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0),
        )
        guarded = selected.protected
        if (
            self.manager.low_conflict_occ_guard
            and level in {"low", "medium", "high"}
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        ):
            # A system-wide pressure sample is insufficient evidence that a
            # particular transaction needs category-wide cold protection.
            # Preserve already-held bits, but require a concrete retry or very
            # high pressure before adding a new cold class.
            cold_classes = LockClass.COLD_READ | LockClass.COLD_WRITE
            guarded &= ~cold_classes
            guarded |= txn.context.action.protected & cold_classes
        if level in {"medium", "high"} and txn.context.retry_count <= 0:
            # The paper protocol permits deferred writes. A sparse compiled
            # policy can otherwise turn a first attempt into category-wide
            # 2PL at refine, which blocks unrelated background rows. Medium
            # and high contention both require concrete retry evidence before
            # adding a write class. TPC-C root read-before-writes are handled
            # by the exact guard below.
            write_classes = LockClass.HOT_WRITE | LockClass.COLD_WRITE
            guarded &= ~write_classes
            guarded |= txn.context.action.protected & write_classes
        if online_mixed_pressure:
            # Under real mixed pressure, broad action-15 protection creates a
            # write convoy. Keep only observed hot reads; actual writes use the
            # unified short commit admission, while exact retry writes upgrade
            # only when their operation is reached.
            guarded = txn.context.action.protected | (guarded & LockClass.HOT_READ)
        if tpcc_high_all_agent and txn.context.retry_count > 0:
            # Retry state already carries the exact failed objects. Keep the
            # monotonic action at those classes instead of letting a sparse
            # nearest-neighbor row expand one conflict into action-15.
            exact_retry = LockClass(int(txn.context.retry_conflict_mask) & 0xF)
            allowed = txn.context.action.protected | exact_retry
            guarded &= allowed
        if tpcc_high_exact_first_attempt:
            # The exact warehouse/district read-before-write guard below is
            # sufficient for one-warehouse TPC-C, including deferred mixed
            # replay. Broad action-1 read locks create an Agent convoy without
            # protecting any additional write dependency.
            guarded &= ~LockClass.HOT_READ
            guarded |= txn.context.action.protected & LockClass.HOT_READ
            txn.metadata["_tpcc_first_attempt_exact_only"] = True
        if hot_read_ratio >= 0.25 and not tpcc_high_exact_first_attempt:
            guarded |= LockClass.HOT_READ
        if (
            hot_reads
            and validation_pressure >= 0.30
            and not tpcc_high_exact_first_attempt
        ):
            # Global pressure justifies protecting observed hotspots, but it
            # is not transaction-specific evidence for a category-wide cold
            # read lock. Cold bits are introduced only by retry evidence.
            guarded |= LockClass.HOT_READ
        if guarded != selected.protected:
            return LockAction(guarded)
        return selected

    @staticmethod
    def _action_transition_timeout_s(txn: Any) -> float:
        context = dict(txn.metadata.get("context", {}) or {})
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        saturated_ycsb_high = bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and str(context.get("level", "")).strip().lower() == "high"
            and int(agentic.get("background_workers", 0) or 0) >= 8
        )
        return 0.025 if saturated_ycsb_high else 5.0

    @staticmethod
    def _tpcc_all_agent_exact_retry(txn: Any) -> bool:
        context = dict(txn.metadata.get("context", {}) or {})
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        return bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "tpcc"
            and str(context.get("level", "")).strip().lower() == "high"
            and int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.retry_count > 0
        )

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
        self._finish_online_observed_prefix(txn)
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

    def _start_online_observed_prefix(self, txn: Any) -> None:
        if not self.online_ycsb_high_mixed(txn) or txn.metadata.get(
            "_online_prefix_active", False
        ):
            return
        txn.metadata["_online_prefix_active"] = True
        self.manager.enter_online_observed_prefix()

    def _finish_online_observed_prefix(self, txn: Any) -> None:
        if not txn.metadata.pop("_online_prefix_active", False):
            return
        self.manager.leave_online_observed_prefix()

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
