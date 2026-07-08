"""Single-plan agent transaction runtime over a versioned KV store."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Dict, Iterable, Optional

from agent.cc import ConcurrencyControlRegistry, ExclusiveLockTable, LockConflict, ReservationTable, TwoPhaseLockTable
from agent.cc.traditional import lock_conflict_result
from agent.native import load_cast_core
from agent.runtime.types import (
    ReadRecord,
    SnapshotValue,
    TransactionEvent,
    TransactionResult,
    TransactionState,
    WriteRecord,
)

cc = load_cast_core()


class AgentTransaction:
    """One logical agent task transaction with one concrete plan."""

    def __init__(
        self,
        manager: "AgentTransactionManager",
        task_id: str,
        snapshot: Dict[str, SnapshotValue],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.manager = manager
        self.task_id = str(task_id)
        self.snapshot = snapshot
        self.metadata = dict(metadata or {})
        self.state = TransactionState.ACTIVE
        self.started_at = time.perf_counter()
        self.events: list[TransactionEvent] = []
        self.read_set: Dict[str, ReadRecord] = {}
        self.write_set: Dict[str, WriteRecord] = {}
        self.result: Optional[TransactionResult] = None
        self._lifecycle_lock = threading.RLock()
        self._event("begin", {"snapshot_objects": len(snapshot)})

    def _ensure_active(self) -> None:
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError(f"transaction is no longer active: {self.state.value}")

    def _event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TransactionEvent(time.perf_counter() - self.started_at, kind, dict(detail or {}))
        )

    def read(self, object_id: str) -> SnapshotValue:
        with self._lifecycle_lock:
            self._ensure_active()
            key = str(object_id)
            if key not in self.snapshot:
                raise KeyError(f"object is not registered: {key}")
            snapshot = self.snapshot[key]
            self.read_set.setdefault(key, ReadRecord(key, snapshot.version))
            self._event("read", {"object_id": key, "version": snapshot.version})
            return snapshot

    def write(
        self,
        object_id: str,
        value: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentTransaction":
        with self._lifecycle_lock:
            self._ensure_active()
            key = str(object_id)
            if key not in self.snapshot:
                raise KeyError(f"object is not registered: {key}")
            if key in self.write_set:
                raise ValueError(f"transaction already writes object: {key}")
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
            return self

    def commit(self, strategy: str = "occ") -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            return self.manager.commit(self, strategy=strategy)

    def abort(self, reason: str, *, strategy: str = "") -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            self.state = TransactionState.ABORTED
            self._event("abort", {"reason": str(reason), "strategy": str(strategy)})
            self.result = self._result(
                strategy=strategy,
                committed=False,
                action="abort",
                reason=str(reason),
            )
            self.manager._record(self)
            return self.result

    def _result(
        self,
        *,
        strategy: str,
        committed: bool,
        action: str,
        reason: str = "",
        conflict_object_ids: Iterable[str] = (),
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
    ):
        self.store = store if store is not None else cc.Dbx1000VersionedKVStore()
        self.cc_registry = cc_registry or ConcurrencyControlRegistry(atcc_policy=atcc_policy)
        self.exclusive_locks = ExclusiveLockTable()
        self.two_phase_locks = TwoPhaseLockTable()
        self.reservations = ReservationTable()
        self._lock = threading.RLock()
        self._catalog: Dict[str, Any] = {}
        self._traces: list[Dict[str, Any]] = []

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

    def begin(self, task_id: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentTransaction:
        with self._lock:
            snapshot = self._snapshot_locked()
        return AgentTransaction(self, str(task_id), snapshot, metadata)

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
        checks = []
        for read in txn.read_set.values():
            if plan.validate_reads:
                checks.append((read.object_id, int(read.version)))
        for write in txn.write_set.values():
            checks.append((write.object_id, int(write.base_version)))
        writes = [(write.object_id, write.value) for write in txn.write_set.values()]
        with self._lock:
            ok = self.store.batch_put_if_version(checks, writes)
        if not ok:
            result = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason="atomic version check failed",
                conflict_object_ids=self._conflict_targets(checks),
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, result, txn)
            return result
        txn.state = TransactionState.COMMITTED
        txn._event("finish", {"state": txn.state.value, "action": "commit"})
        txn.result = txn._result(
            strategy=plan.strategy,
            committed=True,
            action="commit",
            lock_wait_s=time.perf_counter() - started_wait,
        )
        self._record(txn)
        self._observe_strategy(strategy_impl, plan, txn.result, txn)
        return txn.result

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
        txn.state = TransactionState.ABORTED
        txn._event(
            "finish",
            {
                "state": txn.state.value,
                "action": "abort",
                "reason": str(reason),
                "conflict_object_ids": [str(value) for value in conflict_object_ids],
            },
        )
        txn.result = txn._result(
            strategy=strategy,
            committed=False,
            action="abort",
            reason=str(reason),
            conflict_object_ids=conflict_object_ids,
            lock_wait_s=lock_wait_s,
        )
        self._record(txn)
        return txn.result

    def _snapshot_locked(self) -> Dict[str, SnapshotValue]:
        return {
            object_id: SnapshotValue(
                value=(value := self.store.get(object_id)).value,
                version=int(value.version),
                exists=bool(value.exists),
            )
            for object_id in self._catalog
        }

    def _conflict_targets(self, checks: Iterable[tuple[str, int]]) -> tuple[str, ...]:
        conflicts = []
        for object_id, expected_version in checks:
            if int(self.store.get_version(str(object_id))) != int(expected_version):
                conflicts.append(str(object_id))
        return tuple(sorted(set(conflicts)))

    def value_of(self, object_id: str) -> str:
        with self._lock:
            return self.store.get(str(object_id)).value

    def values(self) -> Dict[str, str]:
        with self._lock:
            return {object_id: self.store.get(object_id).value for object_id in self._catalog}

    def _record(self, txn: AgentTransaction) -> None:
        with self._lock:
            self._traces.append(txn.to_trace())

    def traces(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._traces)
