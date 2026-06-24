"""Cost-aware commit protocols for agent transactions."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from agent.native import load_cast_core
from agent.runtime.branching import BranchSemantics, QualityRankedBranchSemantics
from agent.runtime.cc_registry import ConcurrencyControlRegistry
from agent.runtime.types import TransactionResult, TransactionState

cc = load_cast_core()


class ObjectLockTimeout(TimeoutError):
    """Raised when an object lock wait budget is exhausted."""


class ObjectLockLease:
    """Locks held from before snapshot construction through transaction finish."""

    def __init__(
        self,
        targets: Iterable[str],
        locks: List[Any],
        wait_s: float = 0.0,
        *,
        priority: int = 0,
        reason: str = "object-lock",
        target_wait_s: Optional[Dict[str, float]] = None,
        target_queue_depth: Optional[Dict[str, int]] = None,
        target_owner_priority: Optional[Dict[str, int]] = None,
    ):
        self.targets = tuple(targets)
        self._locks = list(locks)
        self.wait_s = float(wait_s)
        self.target_wait_s = dict(target_wait_s or {})
        self.target_queue_depth = {
            str(object_id): int(depth)
            for object_id, depth in dict(target_queue_depth or {}).items()
        }
        self.target_owner_priority = {
            str(object_id): int(priority)
            for object_id, priority in dict(target_owner_priority or {}).items()
        }
        self.target_handoff_count: Dict[str, int] = {}
        self.priority = int(priority)
        self.reason = str(reason)
        self.owner: Optional[Any] = None
        self.wounded = False
        self.wound_reason = ""
        self.committing = False
        self._released = False
        self._release_lock = threading.RLock()

    def bind_owner(self, owner: Any) -> None:
        self.owner = owner

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self.committing = False
            for lock in reversed(self._locks):
                lock.release(self)

    def wound(self, reason: str) -> None:
        with self._release_lock:
            if self._released or self.wounded or self.committing:
                return False
            self.wounded = True
            self.wound_reason = str(reason)
            owner = self.owner
        if owner is not None and hasattr(owner, "_wound_prelock"):
            owner._wound_prelock(str(reason))
        return True

    def enter_committing(self) -> None:
        with self._release_lock:
            if not self._released:
                self.committing = True

    def exit_committing(self) -> None:
        with self._release_lock:
            self.committing = False

    def record_handoff(self, object_id: str) -> None:
        with self._release_lock:
            key = str(object_id)
            self.target_handoff_count[key] = self.target_handoff_count.get(key, 0) + 1

    def __enter__(self) -> "ObjectLockLease":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


class ObjectLockTable:
    """Deterministic object-level lock table for pessimistic CC modules."""

    def __init__(self, *, queue_policy: str = "race", priority_burst: int = 2):
        queue = str(queue_policy or "race").strip().lower()
        if queue not in {"race", "priority", "bounded-priority"}:
            raise ValueError(f"unsupported object lock queue policy: {queue_policy}")
        self.queue_policy = queue
        self.priority_burst = max(1, int(priority_burst))
        self._lock = threading.RLock()
        self._locks: Dict[str, _PriorityObjectLock] = {}

    def ensure(self, object_id: str) -> None:
        with self._lock:
            self._locks.setdefault(
                object_id,
                _PriorityObjectLock(
                    str(object_id),
                    queue_policy=self.queue_policy,
                    priority_burst=self.priority_burst,
                ),
            )

    def acquire_lease(
        self,
        targets: Iterable[str],
        *,
        priority: int = 0,
        reason: str = "object-lock",
        wait_timeout_s: Optional[float] = None,
    ) -> ObjectLockLease:
        target_tuple = tuple(sorted(set(targets)))
        locks: List[Any] = []
        with self._lock:
            for target in target_tuple:
                locks.append(
                    self._locks.setdefault(
                        target,
                        _PriorityObjectLock(
                            target,
                            queue_policy=self.queue_policy,
                            priority_burst=self.priority_burst,
                        ),
                    )
                )
        lease = ObjectLockLease(
            target_tuple,
            locks,
            priority=int(priority),
            reason=str(reason),
        )
        acquired: List[Any] = []
        target_wait_s: Dict[str, float] = {}
        target_queue_depth: Dict[str, int] = {}
        target_owner_priority: Dict[str, int] = {}
        started_at = time.perf_counter()
        deadline = (
            started_at + max(0.0, float(wait_timeout_s))
            if wait_timeout_s is not None
            else None
        )
        try:
            for lock in locks:
                snapshot = lock.snapshot()
                target_queue_depth[lock.object_id] = int(snapshot["queue_depth"])
                target_owner_priority[lock.object_id] = int(snapshot["owner_priority"])
                target_started_at = time.perf_counter()
                lock.acquire(lease, deadline=deadline)
                target_wait_s[lock.object_id] = (
                    time.perf_counter() - target_started_at
                )
                acquired.append(lock)
        except BaseException:
            for lock in reversed(acquired):
                lock.release(lease)
            raise
        lease.wait_s = time.perf_counter() - started_at
        lease.target_wait_s = target_wait_s
        lease.target_queue_depth = target_queue_depth
        lease.target_owner_priority = target_owner_priority
        return lease

    def acquire_budgeted_lease(
        self,
        targets: Iterable[str],
        *,
        priority: int = 0,
        reason: str = "object-lock",
        wait_timeout_s: float,
    ) -> tuple[ObjectLockLease, Tuple[str, ...]]:
        target_tuple = tuple(sorted(set(targets)))
        locks_by_target: List[tuple[str, Any]] = []
        with self._lock:
            for target in target_tuple:
                locks_by_target.append(
                    (
                        target,
                        self._locks.setdefault(
                            target,
                            _PriorityObjectLock(
                                target,
                                queue_policy=self.queue_policy,
                                priority_burst=self.priority_burst,
                            ),
                        ),
                    )
                )
        lease = ObjectLockLease((), [], priority=int(priority), reason=str(reason))
        acquired_locks: List[Any] = []
        acquired_targets: List[str] = []
        skipped_targets: List[str] = []
        target_wait_s: Dict[str, float] = {}
        target_queue_depth: Dict[str, int] = {}
        target_owner_priority: Dict[str, int] = {}
        started_at = time.perf_counter()
        try:
            for target, lock in locks_by_target:
                snapshot = lock.snapshot()
                target_queue_depth[target] = int(snapshot["queue_depth"])
                target_owner_priority[target] = int(snapshot["owner_priority"])
                target_started_at = time.perf_counter()
                try:
                    lock.acquire(
                        lease,
                        deadline=target_started_at + max(0.0, float(wait_timeout_s)),
                    )
                except ObjectLockTimeout:
                    skipped_targets.append(target)
                    target_wait_s[target] = time.perf_counter() - target_started_at
                    continue
                target_wait_s[target] = time.perf_counter() - target_started_at
                acquired_locks.append(lock)
                acquired_targets.append(target)
        except BaseException:
            for lock in reversed(acquired_locks):
                lock.release(lease)
            raise
        lease.targets = tuple(acquired_targets)
        lease._locks = list(acquired_locks)
        lease.wait_s = time.perf_counter() - started_at
        lease.target_wait_s = target_wait_s
        lease.target_queue_depth = target_queue_depth
        lease.target_owner_priority = target_owner_priority
        return lease, tuple(skipped_targets)

    @contextlib.contextmanager
    def acquire(
        self,
        targets: Iterable[str],
        *,
        priority: int = 0,
        wait_timeout_s: Optional[float] = None,
    ):
        lease = self.acquire_lease(
            targets,
            priority=priority,
            wait_timeout_s=wait_timeout_s,
        )
        try:
            yield
        finally:
            lease.release()

    @contextlib.contextmanager
    def scope_for_branches(
        self,
        branches: List[Any],
        cc_module: Any,
        targets: Optional[List[str]] = None,
    ):
        if targets is None and not bool(getattr(cc_module, "requires_object_locks", False)):
            yield
            return
        if targets is None:
            targets = sorted(
                {
                    read.object_id
                    for branch in branches
                    for read in branch.read_set
                }
                | {
                    write.object_id
                    for branch in branches
                    for write in branch.writes
                }
            )
        with self.acquire(targets):
            yield


class _PriorityObjectLock:
    def __init__(
        self,
        object_id: str,
        *,
        queue_policy: str = "race",
        priority_burst: int = 2,
    ):
        self.object_id = str(object_id)
        self.queue_policy = str(queue_policy or "race")
        self.priority_burst = max(1, int(priority_burst))
        self._condition = threading.Condition(threading.RLock())
        self._owner: Optional[ObjectLockLease] = None
        self._waiters: List[Tuple[int, int, ObjectLockLease]] = []
        self._next_sequence = 0
        self._consecutive_priority_grants = 0

    def acquire(
        self,
        requester: ObjectLockLease,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        if self.queue_policy == "race":
            self._acquire_race(requester, deadline=deadline)
            return
        self._acquire_priority(requester, deadline=deadline)

    def _acquire_race(
        self,
        requester: ObjectLockLease,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        while True:
            wounded_owner = None
            with self._condition:
                if self._owner is None:
                    self._owner = requester
                    return
                owner = self._owner
                if (
                    requester.priority > owner.priority
                    and not owner.wounded
                    and not owner.committing
                ):
                    wounded_owner = owner
                else:
                    self._condition.wait(
                        timeout=self._wait_interval(deadline=deadline)
                    )
                    continue
            if wounded_owner is not None:
                wounded = wounded_owner.wound(
                    f"wounded by priority {requester.priority} on {self.object_id}"
                )
                if wounded:
                    with self._condition:
                        self._condition.notify_all()

    def _acquire_priority(
        self,
        requester: ObjectLockLease,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        waiter: Optional[Tuple[int, int, ObjectLockLease]] = None
        try:
            while True:
                wounded_owner = None
                with self._condition:
                    if waiter is None:
                        waiter = (-int(requester.priority), self._next_sequence, requester)
                        self._next_sequence += 1
                        self._waiters.append(waiter)
                    if self._owner is requester:
                        if waiter in self._waiters:
                            self._waiters.remove(waiter)
                        self._condition.notify_all()
                        return
                    if self._owner is None and self._next_waiter() is waiter:
                        self._waiters.remove(waiter)
                        self._owner = requester
                        self._record_grant(requester)
                        self._condition.notify_all()
                        return
                    owner = self._owner
                    if (
                        owner is not None
                        and requester.priority > owner.priority
                        and not owner.wounded
                        and not owner.committing
                    ):
                        wounded_owner = owner
                    else:
                        self._condition.wait(
                            timeout=self._wait_interval(deadline=deadline)
                        )
                        continue
                if wounded_owner is not None:
                    wounded = wounded_owner.wound(
                        f"wounded by priority {requester.priority} on {self.object_id}"
                    )
                    if wounded:
                        with self._condition:
                            self._condition.notify_all()
        except BaseException:
            if waiter is not None:
                with self._condition:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                        self._condition.notify_all()
            raise

    def release(self, requester: ObjectLockLease) -> None:
        with self._condition:
            if self._owner is requester:
                next_waiter = (
                    self._next_waiter()
                    if self.queue_policy in {"priority", "bounded-priority"}
                    else None
                )
                if next_waiter is not None:
                    self._waiters.remove(next_waiter)
                    self._owner = next_waiter[2]
                    next_waiter[2].record_handoff(self.object_id)
                    self._record_grant(next_waiter[2])
                else:
                    self._owner = None
                self._condition.notify_all()

    def snapshot(self) -> Dict[str, int]:
        with self._condition:
            return {
                "queue_depth": len(self._waiters),
                "owner_priority": int(self._owner.priority)
                if self._owner is not None
                else 0,
            }

    def _next_waiter(self) -> Optional[Tuple[int, int, ObjectLockLease]]:
        if not self._waiters:
            return None
        if self.queue_policy == "bounded-priority":
            low_priority_waiters = [
                row for row in self._waiters if int(row[2].priority) <= 0
            ]
            if self._consecutive_priority_grants >= self.priority_burst and low_priority_waiters:
                return min(low_priority_waiters, key=lambda row: row[1])
        return min(self._waiters, key=lambda row: (row[0], row[1]))

    def _record_grant(self, requester: ObjectLockLease) -> None:
        if int(requester.priority) > 0:
            self._consecutive_priority_grants += 1
        else:
            self._consecutive_priority_grants = 0

    def _wait_interval(self, *, deadline: Optional[float]) -> float:
        if deadline is None:
            return 0.001
        remaining = float(deadline) - time.perf_counter()
        if remaining <= 0:
            raise ObjectLockTimeout(
                f"object lock wait budget exhausted on {self.object_id}"
            )
        return min(0.001, remaining)


class CostAwareCommitProtocol:
    """Default cost-aware protocol: merge, reselect, or request regeneration."""

    name = "cost-asymmetric"
    family = "commit-protocol"
    description = (
        "Validate ranked branches through a CC plugin, prefer semantic merge or "
        "candidate reselect, and regenerate only at the explicit boundary."
    )

    def __init__(
        self,
        store: Any,
        model: Any,
        *,
        registry: ConcurrencyControlRegistry,
        branch_semantics: Optional[BranchSemantics] = None,
        lock_table: Optional[ObjectLockTable] = None,
        kernel: Optional[Any] = None,
    ):
        self.store = store
        self.model = model
        self.registry = registry
        self.branch_semantics = branch_semantics or QualityRankedBranchSemantics()
        self.lock_table = lock_table or ObjectLockTable()
        self.kernel = kernel or cc.CostAsymmetricCommit(store, model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "branch_semantics": self.branch_semantics.to_dict(),
        }

    def commit(
        self,
        txn: Any,
        *,
        strategy: str = "cast",
        regenerator: Optional[Callable[[Any], None]] = None,
        max_regenerations: int = 1,
        refresh_snapshot: Callable[[], Dict[str, Any]],
        record: Callable[[Any], None],
    ) -> TransactionResult:
        txn._ensure_active()
        if not txn.candidates:
            return txn.abort("no candidates")
        if max_regenerations < 0:
            raise ValueError("max_regenerations must be non-negative")

        resolution = self.registry.resolve(strategy, txn)
        stats = cc.CostStats()
        regeneration_count = 0
        while True:
            branches = self.branch_semantics.to_core_branches(txn)
            operation_lock_targets = None
            operation_policy_decisions = None
            if self.registry.is_operation_adaptive(resolution.requested_strategy):
                operation_lock_targets, operation_policy_decisions = (
                    self.registry.pessimistic_operation_targets(txn)
                )
            prelocked_targets = set(getattr(txn, "prelocked_targets", ()))
            commit_lock_targets = (
                sorted(set(operation_lock_targets or ()) - prelocked_targets)
                if operation_lock_targets is not None
                else None
            )
            txn._event(
                "validate",
                {
                    "cc": resolution.module.name,
                    "strategy": resolution.requested_strategy,
                    "selected_cc": resolution.selected_strategy,
                    "branch_semantics": self.branch_semantics.name,
                    "commit_protocol": self.name,
                    "candidates": len(branches),
                    "attempt": regeneration_count,
                    "operation_lock_targets": operation_lock_targets or [],
                    "prelocked_targets": sorted(prelocked_targets),
                    "commit_lock_targets": commit_lock_targets or [],
                    "operation_policy_decisions": operation_policy_decisions or [],
                },
            )
            with self.lock_table.scope_for_branches(
                branches, resolution.module, commit_lock_targets
            ):
                if hasattr(txn, "_enter_prelock_committing"):
                    txn._enter_prelock_committing()
                try:
                    outcome = self.kernel.commit_task(branches, resolution.module, stats)
                finally:
                    if hasattr(txn, "_exit_prelock_committing"):
                        txn._exit_prelock_committing()

            if not outcome.needs_regeneration:
                break
            if regenerator is None or regeneration_count >= max_regenerations:
                break

            regeneration_count += 1
            stats.n_regen += 1
            txn.snapshot = refresh_snapshot()
            txn.read_set.clear()
            txn.candidates.clear()
            txn._event(
                "regenerate",
                {
                    "attempt": regeneration_count,
                    "snapshot_versions": {
                        key: value.version for key, value in txn.snapshot.items()
                    },
                },
            )
            regenerator(txn)
            if not txn.candidates:
                raise RuntimeError("regenerator did not add any candidates")
            resolution = self.registry.resolve(strategy, txn)

        if outcome.committed:
            txn.state = TransactionState.COMMITTED
        elif getattr(outcome, "rejected", False):
            txn.state = TransactionState.REJECTED
        else:
            txn.state = TransactionState.ABORTED

        action = "regenerate" if outcome.committed and regeneration_count else outcome.action
        kernel_conflict_object_ids = tuple(
            str(object_id)
            for object_id in (getattr(outcome, "conflict_object_ids", ()) or ())
        )
        conflict_object_ids = (
            kernel_conflict_object_ids or self._stale_operation_targets(txn)
            if not bool(outcome.committed) and not bool(getattr(outcome, "rejected", False))
            else ()
        )
        txn._event(
            "finish",
            {
                "state": txn.state.value,
                "action": action,
                "winner_branch_id": outcome.winner_branch_id,
                "reason": outcome.reason,
                "conflict_object_ids": list(conflict_object_ids),
            },
        )
        txn.result = TransactionResult(
            task_id=txn.task_id,
            state=txn.state,
            committed=bool(outcome.committed),
            rejected=bool(getattr(outcome, "rejected", False)),
            action=action,
            winner_branch_id=outcome.winner_branch_id,
            reason=outcome.reason,
            elapsed_s=time.perf_counter() - txn.started_at,
            model_latency_s=txn.model_latency_s,
            total_tokens=txn.total_tokens,
            candidates=len(txn.candidates),
            n_merge=int(stats.n_merge),
            n_reselect=int(stats.n_reselect),
            n_regen=int(stats.n_regen),
        )
        self.registry.observe_operation_feedback(
            resolution.requested_strategy,
            txn,
            txn.result,
            conflict_object_ids=conflict_object_ids,
        )
        record(txn)
        return txn.result

    def _stale_operation_targets(self, txn: Any) -> tuple[str, ...]:
        targets = set()
        for candidate in getattr(txn, "candidates", ()):
            for write in getattr(candidate, "_writes", ()):
                try:
                    current = self.store.get(write.object_id)
                except Exception:
                    continue
                if int(getattr(current, "version", -1)) != int(write.base_version):
                    targets.add(str(write.object_id))
        for object_id, snapshot in getattr(txn, "read_set", {}).items():
            try:
                current = self.store.get(object_id)
            except Exception:
                continue
            if int(getattr(current, "version", -1)) != int(snapshot.version):
                targets.add(str(object_id))
        return tuple(sorted(targets))
