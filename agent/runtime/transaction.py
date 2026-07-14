"""Single-plan agent transaction runtime over a versioned KV store."""

from __future__ import annotations

import dataclasses
import collections
import threading
import time
from typing import Any, Dict, Iterable, Optional

from agent.cc import ConcurrencyControlRegistry, ExclusiveLockTable, LockConflict, ReservationTable, TwoPhaseLockTable
from agent.cc.locks import normalize_conflict_kind
from agent.cc.traditional import lock_conflict_result
from agent.native import load_cast_core
from agent.runtime.types import (
    ConflictDetail,
    ReadRecord,
    SnapshotValue,
    TransactionEvent,
    TransactionResult,
    TransactionState,
    WriteRecord,
)
from agent.runtime.context import (
    LockAction,
    LockClass,
    TransactionContext,
    TransactionPhase,
    TransactionStatus,
)
from agent.runtime.operation_interceptor import OperationInterceptor, TransactionHooks
from agent.runtime.atcc_lock_manager import PaperATCCLockManager
from agent.runtime.priority import PriorityManager
from agent.runtime.state_collector import StateCollector
from agent.runtime.undo_log import UndoLog
from agent.runtime.version_manager import VersionManager
from agent.runtime.paper_policy import AtomicPolicyManager, CompiledPhasePolicy
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.runtime.trajectory import TrajectoryCollector
from agent.runtime.hotness import HotnessTracker

cc = load_cast_core()


@dataclasses.dataclass
class _RetryProtectionState:
    mask: LockClass = LockClass.NONE
    validation_conflicts: int = 0
    last_conflicts: tuple[ConflictDetail, ...] = ()
    protected_read_targets: frozenset[str] = frozenset()
    protected_write_targets: frozenset[str] = frozenset()


class AgentTransaction:
    """One logical agent task transaction with one concrete plan."""

    def __init__(
        self,
        manager: "AgentTransactionManager",
        task_id: str,
        snapshot: Dict[str, SnapshotValue],
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TransactionContext] = None,
    ):
        self.manager = manager
        self.task_id = str(task_id)
        self.snapshot = snapshot
        self.metadata = dict(metadata or {})
        self.context = context or TransactionContext(
            task_id=self.task_id,
            attempt_id=int(self.metadata.get("retry_count", 0) or 0),
            generation=int(self.metadata.get("generation", 0) or 0),
            snapshot_epoch=int(self.metadata.get("snapshot_epoch", -1)),
            retry_count=int(self.metadata.get("retry_count", 0) or 0),
            retry_validation_conflicts=int(
                self.metadata.get("retry_validation_conflicts", 0) or 0
            ),
            retry_conflict_mask=int(
                self.metadata.get("retry_conflict_mask", 0) or 0
            ),
            retry_conflict_read_targets={
                str(value)
                for value in self.metadata.get("retry_conflict_read_targets", ())
            },
            retry_conflict_write_targets={
                str(value)
                for value in self.metadata.get("retry_conflict_write_targets", ())
            },
            action=LockAction(
                LockClass(int(self.metadata.get("retry_protection_mask", 0) or 0) & 0xF)
            ),
            prior_retry_cost_ms=float(self.metadata.get("prior_retry_cost_ms", 0.0) or 0.0),
            recent_conflict_kind=normalize_conflict_kind(
                self.metadata.get("previous_failure_reason", "none")
            ),
            is_background=bool(self.metadata.get("paper_atcc_backend", False)),
            planned_write_targets={
                str(value)
                for value in self.metadata.get("planned_write_targets", ())
            },
        )
        self.state = TransactionState.ACTIVE
        self.started_at = time.perf_counter()
        self.events: list[TransactionEvent] = []
        self.read_set: Dict[str, ReadRecord] = {}
        self.write_set: Dict[str, WriteRecord] = {}
        self.result: Optional[TransactionResult] = None
        self._lifecycle_lock = threading.RLock()
        self._event("begin", {"snapshot_objects": len(snapshot)})

    def _ensure_active(self) -> None:
        if self.state != TransactionState.ACTIVE or self.context.status != TransactionStatus.ACTIVE:
            if self.context.status in {TransactionStatus.ABORTING, TransactionStatus.ABORTED}:
                raise LockConflict(
                    "transaction was wounded",
                    tuple(self.context.held_read_locks | self.context.held_write_locks),
                    kind="lock-preempted",
                )
            raise RuntimeError(f"transaction is no longer active: {self.state.value}")

    def _event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TransactionEvent(time.perf_counter() - self.started_at, kind, dict(detail or {}))
        )

    def read(self, object_id: str) -> SnapshotValue:
        with self._lifecycle_lock:
            operation_started = time.perf_counter()
            blocked_before_ms = self.context.blocked_time_ms
            self._ensure_active()
            key = str(object_id)
            if key not in self.manager._catalog:
                raise KeyError(f"object is not registered: {key}")
            self._ensure_snapshot(key)
            self.manager.interceptor.before_read(self, key)
            # A first-access protection hook may lock and refresh an
            # unobserved planned-write object to the current committed base.
            snapshot = self.snapshot[key]
            self.read_set.setdefault(key, ReadRecord(key, snapshot.version))
            self.manager.interceptor.after_read(self, key, snapshot.version)
            self._event("read", {"object_id": key, "version": snapshot.version})
            self.manager.interceptor.operation_finished(
                self,
                elapsed_ms=(time.perf_counter() - operation_started) * 1000.0,
                blocked_before_ms=blocked_before_ms,
            )
            return snapshot

    def write(
        self,
        object_id: str,
        value: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentTransaction":
        with self._lifecycle_lock:
            operation_started = time.perf_counter()
            blocked_before_ms = self.context.blocked_time_ms
            self._ensure_active()
            key = str(object_id)
            if key not in self.manager._catalog:
                raise KeyError(f"object is not registered: {key}")
            if key in self.write_set:
                raise ValueError(f"transaction already writes object: {key}")
            self._ensure_snapshot(key)
            self.manager.interceptor.before_write(self, key)
            snapshot = self.snapshot[key]
            self.write_set[key] = WriteRecord(
                object_id=key,
                base_value=snapshot.value,
                base_version=snapshot.version,
                value=str(value),
                metadata=dict(metadata or {}),
            )
            self._event(
                "write",
                {
                    "object_id": key,
                    "base_version": snapshot.version,
                },
            )
            self.manager.interceptor.after_write(self, key)
            self.manager.interceptor.operation_finished(
                self,
                elapsed_ms=(time.perf_counter() - operation_started) * 1000.0,
                blocked_before_ms=blocked_before_ms,
            )
            return self

    def _ensure_snapshot(self, object_id: str) -> None:
        key = str(object_id)
        if key in self.snapshot:
            return
        if self.context.is_background:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )
        elif self.manager.paper_versioning_enabled:
            self.manager.ensure_snapshot_epoch(self)
            self.snapshot[key] = self.manager.version_manager.read_at(
                self.context.snapshot_epoch,
                key,
            )
        else:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )

    def refresh_unobserved_locked_snapshot(self, object_id: str) -> None:
        key = str(object_id)
        if key in self.read_set or key in self.write_set:
            return
        if self.manager.paper_versioning_enabled:
            self.snapshot[key] = self.manager.version_manager.read_committed(key)
        else:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )

    def commit(self, strategy: str = "occ") -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            self.manager.interceptor.before_commit(self)
            return self.manager.commit(self, strategy=strategy)

    def enter_phase(self, phase: str | TransactionPhase) -> None:
        if isinstance(phase, TransactionPhase):
            normalized = phase
        else:
            phase_name = str(phase).strip().lower()
            normalized = TransactionPhase.EXPLORE if phase_name == "plan" else TransactionPhase(phase_name)
        initial_explore = (
            normalized == TransactionPhase.EXPLORE
            and not self.context.read_versions
            and not self.context.write_targets
        )
        if self.metadata.get("paper_atcc", False) and not initial_explore:
            self.manager.interceptor.account_agent_interval(self)
            self.manager.refresh_atcc_priority(self)
        self.manager.interceptor.phase_change(self, normalized)
        self._event("phase", {"name": normalized.value, "source_name": str(phase)})

    def prepare_phase(self, operations: Iterable[Any]) -> None:
        # The paper policy observes completed execution only; it never inspects
        # the upcoming operation batch.
        _ = operations

    def abort(
        self,
        reason: str,
        *,
        strategy: str = "",
        conflict_object_ids: Iterable[str] = (),
    ) -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            conflict_ids = tuple(str(value) for value in conflict_object_ids)
            conflict_details, retry_mask = self.manager._prepare_retry_feedback(
                self,
                reason=reason,
                conflict_object_ids=conflict_ids,
            )
            self.context.transition(TransactionStatus.ABORTING)
            self.context.note_conflict(normalize_conflict_kind(reason))
            self.manager.interceptor.abort(self, str(reason))
            self.state = TransactionState.ABORTED
            self.context.transition(TransactionStatus.ABORTED)
            self._event("abort", {"reason": str(reason), "strategy": str(strategy)})
            self.result = self._result(
                strategy=strategy,
                committed=False,
                action="abort",
                reason=str(reason),
                conflict_object_ids=conflict_ids,
                conflict_details=conflict_details,
                retry_protection_mask=retry_mask,
            )
            self.manager._note_runtime_outcome(self, False)
            with self.manager._lock:
                self.manager._paper_aborts += 1
            self.manager._record(self)
            self.manager.interceptor.finish(self)
            self.manager.atcc_locks.release_all(self.context)
            if self.manager.paper_versioning_enabled:
                self.manager.version_manager.finish(self.context.tid, committed=False)
            self.manager._live_transactions.pop(self.context.tid, None)
            return self.result

    def _result(
        self,
        *,
        strategy: str,
        committed: bool,
        action: str,
        reason: str = "",
        conflict_object_ids: Iterable[str] = (),
        conflict_details: Iterable[ConflictDetail] = (),
        retry_protection_mask: int = 0,
        lock_wait_s: float = 0.0,
    ) -> TransactionResult:
        return TransactionResult(
            task_id=self.task_id,
            state=self.state,
            strategy=str(strategy),
            committed=bool(committed),
            action=str(action),
            reason=str(reason),
            elapsed_s=time.perf_counter() - self.started_at,
            read_count=len(self.read_set),
            write_count=len(self.write_set),
            conflict_object_ids=tuple(str(value) for value in conflict_object_ids),
            conflict_details=tuple(conflict_details),
            retry_protection_mask=int(retry_protection_mask),
            lock_wait_s=float(lock_wait_s),
        )

    def to_trace(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
            "state": self.state.value,
            "snapshot_versions": {key: value.version for key, value in self.snapshot.items()},
            "read_set": {key: dataclasses.asdict(value) for key, value in self.read_set.items()},
            "write_set": {key: dataclasses.asdict(value) for key, value in self.write_set.items()},
            "events": [dataclasses.asdict(event) for event in self.events],
            "result": self.result.to_dict() if self.result else None,
        }


class AgentTransactionManager:
    """Thread-safe transaction manager backed by versioned KV primitives."""

    _KIND_MAP = {
        "generic": cc.ObjectType.kGeneric,
        "row": cc.ObjectType.kRow,
        "text": cc.ObjectType.kText,
        "counter": cc.ObjectType.kCounter,
    }

    def __init__(
        self,
        *,
        store: Optional[Any] = None,
        cc_registry: Optional[ConcurrencyControlRegistry] = None,
        atcc_policy: Optional[Any] = None,
        record_traces: bool = True,
        transaction_hooks: Optional[TransactionHooks] = None,
        undo_log_path: Optional[str] = None,
        paper_policy: Optional[CompiledPhasePolicy] = None,
        collect_trajectories: bool = True,
    ):
        self.store = store if store is not None else cc.Dbx1000VersionedKVStore()
        self.version_manager = VersionManager(self.store)
        self.cc_registry = cc_registry or ConcurrencyControlRegistry(atcc_policy=atcc_policy)
        self.exclusive_locks = ExclusiveLockTable()
        self.two_phase_locks = TwoPhaseLockTable()
        self.reservations = ReservationTable()
        self._lock = threading.RLock()
        self._catalog: Dict[str, Any] = {}
        self._traces: list[Dict[str, Any]] = []
        self._record_traces = bool(record_traces)
        self.state_collector = StateCollector()
        self.priority_manager = PriorityManager()
        self.hotness_tracker = HotnessTracker()
        self.paper_policy = AtomicPolicyManager(paper_policy)
        self.trajectory_collector = TrajectoryCollector()
        self.collect_trajectories = bool(collect_trajectories)
        hooks = transaction_hooks or PaperATCCHooks(self)
        self.interceptor = OperationInterceptor(hooks, state_collector=self.state_collector)
        self._live_transactions: Dict[str, AgentTransaction] = {}
        self.atcc_locks = PaperATCCLockManager(
            wound_callback=self._wound_context,
            priority_callback=self.priority_manager.compute,
            contention_callback=self.hotness_tracker.observe_contention,
        )
        self.undo_log = UndoLog(undo_log_path)
        self._commit_fault_injector: Optional[Any] = None
        self._paper_attempts = 0
        self._paper_aborts = 0
        self._paper_background_aborts = 0
        self._runtime_started_at = time.monotonic()
        self._recent_outcomes = collections.deque()
        self._recent_agent_tasks = collections.deque()
        self._paper_metrics_cache_at = 0.0
        self._paper_metrics_cache: Dict[str, float | int] = {}
        self._retry_protection: Dict[str, _RetryProtectionState] = {}
        self._retry_protection_diagnostics: Dict[str, int] = collections.defaultdict(int)
        self._retry_conflict_objects: Dict[str, int] = collections.defaultdict(int)
        self._paper_versioning_enabled = False
        self._native_dirty_objects: set[str] = set()

    @property
    def paper_versioning_enabled(self) -> bool:
        with self._lock:
            return bool(self._paper_versioning_enabled)

    def _enable_paper_versioning(self) -> None:
        with self._lock:
            if self._paper_versioning_enabled:
                return
            for object_id in sorted(self._native_dirty_objects):
                self.version_manager.synchronize_object(object_id)
            self._native_dirty_objects.clear()
            self._paper_versioning_enabled = True

    def _inject_commit_fault(self, stage: str, txn: AgentTransaction) -> None:
        injector = self._commit_fault_injector
        if injector is not None:
            injector(str(stage), txn)

    def is_hot(self, object_id: str) -> bool:
        return self.hotness_tracker.observe_access(str(object_id))

    def peek_hot(self, object_id: str, *, total_accesses: int | None = None) -> bool:
        _ = total_accesses
        return self.hotness_tracker.is_hot(str(object_id))

    def refresh_hot_targets(self, txn: AgentTransaction) -> None:
        read_targets = set(txn.read_set) | set(txn.context.read_versions)
        write_targets = set(txn.write_set) | set(txn.context.write_targets)
        txn.context.hot_read_targets = self.hotness_tracker.hot_targets(read_targets)
        txn.context.hot_write_targets = self.hotness_tracker.hot_targets(write_targets)

    def transition_atcc_action(self, txn: AgentTransaction, action: Any, *, timeout_s: float = 5.0) -> None:
        """Retroactively validate and lock objects newly protected by an action."""
        if not self.paper_versioning_enabled:
            self._enable_paper_versioning()
        self.ensure_snapshot_epoch(txn)
        previous = txn.context.action
        added = action.added_since(previous)
        self.refresh_hot_targets(txn)
        read_targets = []
        write_targets: dict[str, int] = {}
        for object_id, read in txn.read_set.items():
            hot = object_id in txn.context.hot_read_targets
            planned_write = object_id in txn.context.planned_write_targets
            if (
                planned_write
                and action.protects(hot=hot, write=True)
                and not previous.protects(hot=hot, write=True)
            ):
                write_targets[object_id] = int(read.version)
            elif action.protects(hot=hot, write=False) and not previous.protects(hot=hot, write=False):
                read_targets.append((object_id, int(read.version)))
        for object_id, write in txn.write_set.items():
            hot = object_id in txn.context.hot_write_targets
            if action.protects(hot=hot, write=True) and not previous.protects(hot=hot, write=True):
                write_targets[object_id] = int(write.base_version)
        try:
            for object_id, version in sorted(read_targets):
                self.atcc_locks.validate_and_rlock(
                    object_id,
                    txn.context,
                    version,
                    lambda key=object_id, observed=version: (
                        int(observed)
                        if self.version_manager.can_lock_pinned_version(
                            txn.context.snapshot_epoch,
                            key,
                            int(observed),
                            tid=txn.context.tid,
                        )
                        else int(self.version_manager.read_committed(key).version)
                    ),
                    timeout_s=timeout_s,
                )
                txn.context.policy_read_lock_targets.add(str(object_id))
            for object_id, version in sorted(write_targets.items()):
                self.atcc_locks.validate_and_wlock(
                    object_id,
                    txn.context,
                    version,
                    lambda key=object_id: int(
                        self.version_manager.read_committed(key).version
                    ),
                    timeout_s=timeout_s,
                )
                txn.context.policy_write_lock_targets.add(str(object_id))
            self.interceptor.action_change(txn, action)
            txn._event("action-change", {"added": int(added), "action": int(action.protected)})
        except LockConflict as exc:
            self.atcc_locks.release_all(txn.context)
            txn.abort(
                exc.reason,
                strategy="paper-atcc",
                conflict_object_ids=exc.targets,
            )
            raise

    def refresh_atcc_priority(self, txn: AgentTransaction) -> int:
        return self.priority_manager.refresh(txn.context, self.atcc_locks)

    def paper_runtime_metrics(self) -> Dict[str, float | int]:
        now = time.monotonic()
        with self._lock:
            if now - self._paper_metrics_cache_at < 0.010 and self._paper_metrics_cache:
                cached = dict(self._paper_metrics_cache)
                cached["active_transactions"] = len(self._live_transactions)
                cached["waiter_count"] = self.atcc_locks.global_waiter_count()
                cached["background_aborts"] = self._paper_background_aborts
                return cached
            while self._recent_outcomes and now - self._recent_outcomes[0][0] > 1.0:
                self._recent_outcomes.popleft()
            while self._recent_agent_tasks and now - self._recent_agent_tasks[0][0] > 1.0:
                self._recent_agent_tasks.popleft()
            outcomes = tuple(self._recent_outcomes)
            agent_tasks = tuple(self._recent_agent_tasks)
            active_transactions = len(self._live_transactions)
            background_aborts = self._paper_background_aborts
        commits = [row for row in outcomes if row[1]]
        agent_attempts = [row for row in outcomes if not row[3]]
        background_attempts = [row for row in outcomes if row[3]]
        committed_background = [row for row in background_attempts if row[1]]
        committed_agent_tasks = [row for row in agent_tasks if row[1]]
        elapsed = max(0.1, min(1.0, now - self._runtime_started_at))
        latencies = sorted(row[2] for row in commits)
        tail_index = max(0, min(len(latencies) - 1, int(0.99 * len(latencies))))
        agent_task_latencies = sorted(row[2] for row in committed_agent_tasks)
        agent_tail_index = max(
            0,
            min(len(agent_task_latencies) - 1, int(0.99 * len(agent_task_latencies))),
        )
        conflict_kinds = {"version-conflict", "lock-preempted", "lock-timeout", "lock-conflict"}
        metrics = {
            "active_transactions": active_transactions,
            "waiter_count": self.atcc_locks.global_waiter_count(),
            "abort_rate": (
                sum(1 for row in outcomes if not row[1]) / len(outcomes)
                if outcomes else 0.0
            ),
            "throughput": len(commits) / elapsed,
            "average_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "tail_latency_ms": latencies[tail_index] if latencies else 0.0,
            "agent_task_throughput": len(committed_agent_tasks) / elapsed,
            "agent_task_average_latency_ms": (
                sum(agent_task_latencies) / len(agent_task_latencies)
                if agent_task_latencies else 0.0
            ),
            "agent_task_tail_latency_ms": (
                agent_task_latencies[agent_tail_index] if agent_task_latencies else 0.0
            ),
            "conflict_abort_rate": (
                sum(1 for row in agent_attempts if not row[1] and row[4] in conflict_kinds)
                / len(agent_attempts)
                if agent_attempts else 0.0
            ),
            "background_throughput": len(committed_background) / elapsed,
            "background_abort_rate": (
                sum(1 for row in background_attempts if not row[1])
                / len(background_attempts)
                if background_attempts else 0.0
            ),
            "background_aborts": background_aborts,
        }
        with self._lock:
            self._paper_metrics_cache_at = now
            self._paper_metrics_cache = dict(metrics)
        return metrics

    def _note_runtime_outcome(self, txn: AgentTransaction, committed: bool) -> None:
        with self._lock:
            self._recent_outcomes.append(
                (
                    time.monotonic(),
                    bool(committed),
                    (time.perf_counter() - txn.started_at) * 1000.0,
                    bool(txn.context.is_background),
                    normalize_conflict_kind(txn.context.recent_conflict_kind) if not committed else "none",
                )
            )

    def note_agent_task_outcome(self, *, committed: bool, latency_ms: float) -> None:
        """Publish one logical task outcome for the paper Delta-Psys window."""
        with self._lock:
            self._recent_agent_tasks.append(
                (time.monotonic(), bool(committed), max(0.0, float(latency_ms)))
            )
            self._paper_metrics_cache_at = 0.0

    def note_background_abort(self) -> None:
        with self._lock:
            self._paper_background_aborts += 1

    @property
    def backend_name(self) -> str:
        return str(self.store.backend_name)

    def register_object(self, object_id: str, initial_value: Any, *, kind: str = "generic") -> None:
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        key = str(object_id)
        with self._lock:
            if key in self._catalog:
                raise ValueError(f"object already registered: {key}")
            self.store.put(key, str(initial_value))
            self._catalog[key] = self._KIND_MAP[kind]
            self.version_manager.register_object(key)

    def begin(
        self,
        task_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        snapshot_object_ids: Optional[Iterable[str]] = None,
        strategy: str | None = None,
    ) -> AgentTransaction:
        metadata = dict(metadata or {})
        selected_strategy = str(
            strategy or metadata.get("strategy") or metadata.get("cc") or ""
        ).strip()
        if selected_strategy:
            strategy_impl = self.cc_registry.resolve(selected_strategy)
            metadata.setdefault("strategy", strategy_impl.name)
            if strategy_impl.family == "paper-atcc":
                metadata["paper_atcc"] = True
        task_key = str(task_id)
        retry_count = int(metadata.get("retry_count", 0) or 0)
        if metadata.get("paper_atcc", False) and not metadata.get("paper_atcc_backend", False):
            explicit_mask = LockClass(
                int(metadata.get("retry_protection_mask", 0) or 0) & 0xF
            )
            with self._lock:
                if retry_count <= 0:
                    self._retry_protection.pop(task_key, None)
                inherited = self._retry_protection.get(task_key)
                if inherited is not None:
                    explicit_mask |= inherited.mask
                    metadata["retry_validation_conflicts"] = int(
                        inherited.validation_conflicts
                    )
                    retry_conflict_mask = LockClass.NONE
                    for detail in inherited.last_conflicts:
                        retry_conflict_mask |= LockClass(int(detail.protection_bit))
                    metadata["retry_conflict_mask"] = int(retry_conflict_mask)
                    metadata["retry_conflict_read_targets"] = sorted(
                        inherited.protected_read_targets
                    )
                    metadata["retry_conflict_write_targets"] = sorted(
                        inherited.protected_write_targets
                    )
                    self._retry_protection_diagnostics["inherited_attempts"] += 1
            metadata["retry_protection_mask"] = int(explicit_mask)
        if metadata.get("paper_atcc", False) or metadata.get("paper_atcc_backend", False):
            self._enable_paper_versioning()
        snapshot_ids = (
            tuple(dict.fromkeys(str(value) for value in snapshot_object_ids))
            if snapshot_object_ids is not None
            else ()
        )
        if snapshot_ids and self.paper_versioning_enabled:
            snapshot_epoch, snapshot = self.version_manager.snapshot_current(
                snapshot_ids
            )
        else:
            snapshot_epoch = -1
            snapshot = {}
        metadata["snapshot_epoch"] = snapshot_epoch
        txn = AgentTransaction(self, str(task_id), snapshot, metadata)
        if snapshot_epoch >= 0 and self.paper_versioning_enabled:
            self.version_manager.pin(
                txn.context.tid,
                snapshot_epoch,
                materialized=bool(snapshot_ids),
            )
        with self._lock:
            self._live_transactions[txn.context.tid] = txn
            self._paper_attempts += 1
            txn.metadata.setdefault(
                "paper_background_aborts_at_begin", self._paper_background_aborts
            )
        self.interceptor.begin(txn)
        return txn

    def ensure_snapshot_epoch(self, txn: AgentTransaction) -> int:
        if txn.context.snapshot_epoch >= 0:
            return int(txn.context.snapshot_epoch)
        epoch = self.version_manager.pin_lazy(
            txn.context.tid,
            self._catalog,
        )
        txn.context.snapshot_epoch = epoch
        return epoch

    def _conflict_details(
        self,
        txn: AgentTransaction,
        conflict_object_ids: Iterable[str],
    ) -> tuple[ConflictDetail, ...]:
        details = []
        for object_id in sorted(set(str(value) for value in conflict_object_ids)):
            write = (
                object_id in txn.write_set
                or object_id in txn.context.write_targets
                or object_id in txn.context.planned_write_targets
            )
            hot = self.peek_hot(object_id)
            bit = (
                LockClass.HOT_WRITE if hot and write
                else LockClass.COLD_WRITE if write
                else LockClass.HOT_READ if hot
                else LockClass.COLD_READ
            )
            details.append(
                ConflictDetail(
                    object_id=object_id,
                    access_kind="write" if write else "read",
                    hot=hot,
                    protection_class=bit.name.lower().replace("_", "-"),
                    protection_bit=int(bit),
                )
            )
        return tuple(details)

    def _observed_protection_mask(self, txn: AgentTransaction) -> LockClass:
        mask = LockClass.NONE
        read_targets = set(txn.read_set) | set(txn.context.read_versions)
        write_targets = set(txn.write_set) | set(txn.context.write_targets)
        planned_observed_writes = read_targets & set(txn.context.planned_write_targets)
        write_targets.update(planned_observed_writes)
        for object_id in read_targets - planned_observed_writes:
            mask |= (
                LockClass.HOT_READ
                if self.peek_hot(object_id)
                else LockClass.COLD_READ
            )
        for object_id in write_targets:
            mask |= (
                LockClass.HOT_WRITE
                if self.peek_hot(object_id)
                else LockClass.COLD_WRITE
            )
        return mask

    def _prepare_retry_feedback(
        self,
        txn: AgentTransaction,
        *,
        reason: str,
        conflict_object_ids: Iterable[str],
    ) -> tuple[tuple[ConflictDetail, ...], int]:
        if (
            normalize_conflict_kind(reason) != "version-conflict"
            or not txn.metadata.get("paper_atcc", False)
            or txn.context.is_background
        ):
            return (), int(txn.context.action.protected)
        details = self._conflict_details(txn, conflict_object_ids)
        conflict_mask = LockClass.NONE
        for detail in details:
            conflict_mask |= LockClass(int(detail.protection_bit))
        with self._lock:
            previous = self._retry_protection.get(txn.task_id, _RetryProtectionState())
            conflict_count = previous.validation_conflicts + 1
            mask = previous.mask | txn.context.action.protected | conflict_mask
            protected_reads = set(previous.protected_read_targets)
            protected_writes = set(previous.protected_write_targets)
            for detail in details:
                target_set = (
                    protected_writes
                    if detail.access_kind == "write"
                    else protected_reads
                )
                target_set.add(detail.object_id)
            if conflict_count >= 2:
                mask |= self._observed_protection_mask(txn)
                observed_reads = set(txn.read_set) | set(txn.context.read_versions)
                planned_observed_writes = (
                    observed_reads & set(txn.context.planned_write_targets)
                )
                protected_reads.update(observed_reads - planned_observed_writes)
                protected_writes.update(planned_observed_writes)
                protected_writes.update(txn.write_set)
                protected_writes.update(txn.context.write_targets)
                self._retry_protection_diagnostics["full_observed_escalations"] += 1
            if mask != previous.mask:
                self._retry_protection_diagnostics["mask_escalations"] += 1
            self._retry_protection[txn.task_id] = _RetryProtectionState(
                mask=mask,
                validation_conflicts=conflict_count,
                last_conflicts=details,
                protected_read_targets=frozenset(protected_reads),
                protected_write_targets=frozenset(protected_writes),
            )
            self._retry_protection_diagnostics["validation_conflicts"] += 1
            attempt_bucket = (
                "first_attempt"
                if int(txn.context.retry_count) <= 0
                else "retry_attempt"
            )
            self._retry_protection_diagnostics[
                f"validation_conflicts_{attempt_bucket}"
            ] += 1
            for detail in details:
                self._retry_conflict_objects[detail.object_id] += 1
                conflict_class = detail.protection_class.replace("-", "_")
                self._retry_protection_diagnostics[
                    f"conflict_{conflict_class}"
                ] += 1
                if detail.access_kind == "write":
                    dependency = (
                        "read_before_write"
                        if detail.object_id in txn.read_set
                        else "blind_write"
                    )
                    self._retry_protection_diagnostics[
                        f"conflict_{dependency}"
                    ] += 1
                parts = detail.object_id.split(":")
                object_type = parts[1] if len(parts) > 1 else "other"
                if object_type not in {
                    "warehouse",
                    "district",
                    "stock",
                    "customer",
                }:
                    object_type = "other"
                self._retry_protection_diagnostics[
                    f"conflict_object_{object_type}"
                ] += 1
        txn.context.retry_validation_conflicts = conflict_count
        txn._event(
            "retry-protection",
            {
                "validation_conflicts": conflict_count,
                "mask": int(mask),
                "conflicts": [dataclasses.asdict(detail) for detail in details],
                "protected_read_targets": sorted(protected_reads),
                "protected_write_targets": sorted(protected_writes),
            },
        )
        return details, int(mask)

    def retry_protection_diagnostics(self) -> Dict[str, object]:
        with self._lock:
            return {
                **dict(self._retry_protection_diagnostics),
                "tracked_tasks": len(self._retry_protection),
                "conflict_objects": dict(sorted(self._retry_conflict_objects.items())),
            }

    def _clear_retry_protection(self, txn: AgentTransaction) -> None:
        if txn.context.is_background:
            return
        with self._lock:
            self._retry_protection.pop(txn.task_id, None)

    def _wound_context(self, context: TransactionContext, reason: str) -> None:
        txn = self._live_transactions.get(context.tid)
        if txn is None or txn.state != TransactionState.ACTIVE:
            return
        txn.state = TransactionState.ABORTED
        context.note_conflict("lock-preempted")
        if context.status == TransactionStatus.ABORTING:
            context.transition(TransactionStatus.ABORTED)
        txn._event("wound", {"reason": str(reason)})
        txn.result = txn._result(
            strategy="paper-atcc",
            committed=False,
            action="abort",
            reason=str(reason),
        )
        self._note_runtime_outcome(txn, False)
        with self._lock:
            self._paper_aborts += 1
        self._record(txn)
        self.interceptor.abort(txn, str(reason))
        self.interceptor.finish(txn)
        if self.paper_versioning_enabled:
            self.version_manager.finish(context.tid, committed=False)
        self._live_transactions.pop(context.tid, None)

    def commit(self, txn: AgentTransaction, *, strategy: str = "occ") -> TransactionResult:
        strategy_impl = self.cc_registry.resolve(strategy)
        plan = strategy_impl.plan(txn)
        txn._event(
            "validate",
            {
                "strategy": plan.strategy,
                "family": plan.family,
                "lock_targets": list(plan.lock_targets),
                "validate_reads": plan.validate_reads,
                "validate_writes": plan.validate_writes,
                "metadata": dict(plan.metadata),
            },
        )
        started_wait = time.perf_counter()
        try:
            lock_table = str(plan.metadata.get("lock_table", ""))
            if lock_table == "2pl":
                mode = "x" if txn.write_set else "s"
                if self._has_prelock(txn, plan, "2pl"):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
                with self.two_phase_locks.acquire(
                    plan.lock_targets,
                    owner=txn,
                    mode=mode,
                    policy=str(plan.metadata.get("policy", "nowait")),
                ):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
            if lock_table == "exclusive" and plan.lock_targets:
                if self._has_prelock(txn, plan, "exclusive"):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
                with self.exclusive_locks.acquire(
                    plan.lock_targets,
                    owner=txn,
                    wait=bool(plan.metadata.get("wait", True)),
                    priority=int(plan.metadata.get("priority", 0) or 0),
                ):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
            return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
        except LockConflict as exc:
            validation = lock_conflict_result(exc)
            result = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason=validation.reason,
                conflict_object_ids=validation.conflict_object_ids,
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, result, txn)
            return result

    def _has_prelock(self, txn: AgentTransaction, plan: Any, lock_table: str) -> bool:
        metadata = dict(getattr(txn, "metadata", {}) or {})
        if str(metadata.get("prelocked_lock_table", "")) != str(lock_table):
            return False
        targets = {str(target) for target in metadata.get("prelocked_targets", ())}
        return set(str(target) for target in plan.lock_targets).issubset(targets)

    def _commit_under_lock(
        self,
        txn: AgentTransaction,
        strategy_impl: Any,
        plan: Any,
        started_wait: float,
    ) -> TransactionResult:
        coordinated_backend = bool(txn.metadata.get("paper_atcc_backend", False))
        paper_atcc = bool(
            txn.context.action.protected
            or txn.context.held_read_locks
            or txn.context.held_write_locks
            or txn.metadata.get("paper_atcc", False)
            or coordinated_backend
            or getattr(plan, "family", "") == "paper-atcc"
        )

        if coordinated_backend:
            used_fast_path, result = self.atcc_locks.try_uncontended_background_publish(
                txn.write_set,
                txn.context,
                lambda: self._commit_after_admission(
                    txn,
                    strategy_impl,
                    plan,
                    started_wait,
                    paper_atcc=True,
                    release_atcc_locks=False,
                ),
            )
            if used_fast_path:
                return result

        if paper_atcc:
            try:
                # A low-priority backend must not occupy a worker for the
                # lifetime of a costly Agent transaction. If its private
                # publish collides with an Agent writer, briefly yield and
                # retry another trace transaction instead of joining a
                # multi-second lock queue.
                self.atcc_locks.acquire_write_set(
                    txn.write_set,
                    txn.context,
                    timeout_s=0.005 if coordinated_backend else 5.0,
                )
            except LockConflict as exc:
                txn.context.note_conflict(exc.kind)
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason=exc.kind,
                    conflict_object_ids=exc.targets,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        if paper_atcc:
            if not self.atcc_locks.begin_committing(txn.context):
                if txn.result is not None:
                    return txn.result
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="transaction was wounded before commit became non-preemptible",
                    conflict_object_ids=(),
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        else:
            txn.context.transition(TransactionStatus.COMMITTING)

        return self._commit_after_admission(
            txn,
            strategy_impl,
            plan,
            started_wait,
            paper_atcc=paper_atcc,
            release_atcc_locks=paper_atcc,
        )

    def _commit_after_admission(
        self,
        txn: AgentTransaction,
        strategy_impl: Any,
        plan: Any,
        started_wait: float,
        *,
        paper_atcc: bool,
        release_atcc_locks: bool,
    ) -> TransactionResult:
        if not paper_atcc:
            validation = strategy_impl.validate(txn, self.store)
            if not validation.ok:
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason=validation.reason,
                    conflict_object_ids=validation.conflict_object_ids,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        read_checks = []
        for read in txn.read_set.values():
            protected = (
                read.object_id in txn.context.held_read_locks
                or read.object_id in txn.context.held_write_locks
            )
            if (
                paper_atcc
                and not txn.context.is_background
                and not protected
                and self.atcc_locks.has_foreign_committing_writer(
                    read.object_id, txn.context
                )
            ):
                self.hotness_tracker.observe_contention(
                    read.object_id, "validation-failure"
                )
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="optimistic read conflicts with committing writer",
                    conflict_object_ids=(read.object_id,),
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
            if plan.validate_reads and not (paper_atcc and protected):
                read_checks.append((read.object_id, int(read.version)))
        checks = list(read_checks)
        checks.extend(
            (write.object_id, int(write.base_version))
            for write in txn.write_set.values()
            if not txn.context.is_background or write.object_id in txn.read_set
        )
        writes = [(write.object_id, write.value) for write in txn.write_set.values()]
        durable_undo = self.undo_log.path is not None
        if durable_undo:
            self.undo_log.begin(txn.context.tid)
        if durable_undo and not txn.context.is_background:
            self.undo_log.update_batch(
                txn.context.tid,
                (
                    (write.object_id, write.base_value, write.base_version)
                    for write in txn.write_set.values()
                ),
            )
        if not txn.context.is_background:
            self._inject_commit_fault("after_undo_flush_before_install", txn)

        def install_private_versions() -> bool:
            nonlocal checks
            background_rebase = bool(
                txn.context.is_background
                and (durable_undo or self.version_manager.has_lazy_pins())
            )
            if background_rebase:
                for object_id, write in tuple(txn.write_set.items()):
                    if object_id in txn.read_set:
                        continue
                    current = self.store.get(object_id)
                    txn.write_set[object_id] = dataclasses.replace(
                        write,
                        base_value=str(current.value),
                        base_version=int(current.version),
                    )
                if durable_undo:
                    self.undo_log.update_batch(
                        txn.context.tid,
                        (
                            (write.object_id, write.base_value, write.base_version)
                            for write in txn.write_set.values()
                        ),
                    )
            if txn.context.is_background:
                self._inject_commit_fault("after_undo_flush_before_install", txn)
            installed = bool(self.store.batch_put_if_version(checks, writes))
            if not installed:
                return False
            self._inject_commit_fault("after_install_before_publish", txn)
            if durable_undo:
                self.undo_log.commit(txn.context.tid)
            return True

        if self.paper_versioning_enabled:
            ok = self.version_manager.atomic_publish(
                txn.context.tid,
                writes,
                install_private_versions,
                background=txn.context.is_background,
                published_version=lambda object_id: int(
                    txn.write_set[str(object_id)].base_version
                )
                + 1,
            )
        else:
            ok = install_private_versions()
        if not ok:
            if durable_undo:
                self.undo_log.abort(txn.context.tid)
            if txn.context.is_background:
                self.atcc_locks.note_background_publish_fallback(
                    "version_mismatch",
                    count_total=False,
                )
            result = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason="atomic version check failed",
                conflict_object_ids=self._conflict_targets(checks),
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, result, txn)
            return result
        self._inject_commit_fault("after_publish", txn)
        txn.state = TransactionState.COMMITTED
        txn.context.transition(TransactionStatus.COMMITTED)
        txn._event("finish", {"state": txn.state.value, "action": "commit"})
        txn.result = txn._result(
            strategy=plan.strategy,
            committed=True,
            action="commit",
            lock_wait_s=time.perf_counter() - started_wait,
        )
        if not self.paper_versioning_enabled and txn.write_set:
            with self._lock:
                self._native_dirty_objects.update(txn.write_set)
        self._note_runtime_outcome(txn, True)
        self._clear_retry_protection(txn)
        self._record(txn)
        self.interceptor.finish(txn)
        if release_atcc_locks:
            self.atcc_locks.release_all(txn.context)
        if self.paper_versioning_enabled:
            self.version_manager.finish(txn.context.tid, committed=True)
        self._live_transactions.pop(txn.context.tid, None)
        self._observe_strategy(strategy_impl, plan, txn.result, txn)
        return txn.result

    def recover(self) -> list[str]:
        return self.undo_log.recover(self.store)

    def _observe_strategy(
        self,
        strategy_impl: Any,
        plan: Any,
        result: TransactionResult,
        txn: AgentTransaction,
    ) -> None:
        observer = getattr(strategy_impl, "observe", None)
        if observer is None:
            return
        try:
            observer(plan, result, txn)
        except TypeError:
            observer(plan, result)

    def _finish_abort(
        self,
        txn: AgentTransaction,
        *,
        strategy: str,
        reason: str,
        conflict_object_ids: Iterable[str],
        lock_wait_s: float,
    ) -> TransactionResult:
        conflict_ids = tuple(str(value) for value in conflict_object_ids)
        conflict_details, retry_mask = self._prepare_retry_feedback(
            txn,
            reason=reason,
            conflict_object_ids=conflict_ids,
        )
        if txn.context.status not in {TransactionStatus.ABORTING, TransactionStatus.ABORTED}:
            txn.context.transition(TransactionStatus.ABORTING)
            self.interceptor.abort(txn, str(reason))
        txn.context.note_conflict(normalize_conflict_kind(reason))
        txn.state = TransactionState.ABORTED
        if txn.context.status != TransactionStatus.ABORTED:
            txn.context.transition(TransactionStatus.ABORTED)
        txn._event(
            "finish",
            {
                "state": txn.state.value,
                "action": "abort",
                "reason": str(reason),
                "conflict_object_ids": list(conflict_ids),
                "conflict_details": [
                    dataclasses.asdict(detail) for detail in conflict_details
                ],
                "retry_protection_mask": retry_mask,
            },
        )
        txn.result = txn._result(
            strategy=strategy,
            committed=False,
            action="abort",
            reason=str(reason),
            conflict_object_ids=conflict_ids,
            conflict_details=conflict_details,
            retry_protection_mask=retry_mask,
            lock_wait_s=lock_wait_s,
        )
        self._note_runtime_outcome(txn, False)
        with self._lock:
            self._paper_aborts += 1
        self._record(txn)
        self.interceptor.finish(txn)
        self.atcc_locks.release_all(txn.context)
        if self.paper_versioning_enabled:
            self.version_manager.finish(txn.context.tid, committed=False)
        self._live_transactions.pop(txn.context.tid, None)
        return txn.result

    def _snapshot_locked(
        self,
        object_ids: Optional[Iterable[str]] = None,
        *,
        snapshot_epoch: Optional[int] = None,
    ) -> Dict[str, SnapshotValue]:
        if object_ids is None:
            keys = ()
        else:
            keys = tuple(dict.fromkeys(str(object_id) for object_id in object_ids))
            missing = [key for key in keys if key not in self._catalog]
            if missing:
                raise KeyError(f"object is not registered: {missing[0]}")
        epoch = (
            self.version_manager.current_epoch()
            if snapshot_epoch is None
            else int(snapshot_epoch)
        )
        if not self.paper_versioning_enabled:
            return {
                object_id: SnapshotValue(
                    value=(value := self.store.get(object_id)).value,
                    version=int(value.version),
                    exists=bool(value.exists),
                )
                for object_id in keys
            }
        return self.version_manager.read_many_at(epoch, keys)

    def _conflict_targets(self, checks: Iterable[tuple[str, int]]) -> tuple[str, ...]:
        conflicts = []
        for object_id, expected_version in checks:
            if int(self.store.get_version(str(object_id))) != int(expected_version):
                key = str(object_id)
                conflicts.append(key)
                self.hotness_tracker.observe_contention(key, "validation-failure")
        return tuple(sorted(set(conflicts)))

    def value_of(self, object_id: str) -> str:
        with self._lock:
            return self.store.get(str(object_id)).value

    def values(self) -> Dict[str, str]:
        with self._lock:
            return {object_id: self.store.get(object_id).value for object_id in self._catalog}

    def _record(self, txn: AgentTransaction) -> None:
        if not self._record_traces:
            return
        with self._lock:
            self._traces.append(txn.to_trace())

    def traces(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._traces)
