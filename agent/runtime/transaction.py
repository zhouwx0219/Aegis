"""Complete agent-side transaction lifecycle over the CAST C++ commit kernel.

The runtime owns the task-level boundary:
begin -> snapshot/read -> model/tool calls -> candidate branches -> winner selection
-> intent-aware commit/reselect/reject -> trace.

It intentionally keeps policy selection explicit. Automatic policy learning and real
backend adapters are outside the current scope.
"""
from __future__ import annotations

import dataclasses
import enum
import threading
import time
from typing import Any, Dict, List, Optional

import cast_core as cc


class TransactionState(str, enum.Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    REJECTED = "rejected"
    ABORTED = "aborted"


@dataclasses.dataclass(frozen=True)
class SnapshotValue:
    value: str
    version: int
    exists: bool


@dataclasses.dataclass
class TransactionEvent:
    at_s: float
    kind: str
    detail: Dict[str, Any]


@dataclasses.dataclass
class TransactionResult:
    task_id: str
    state: TransactionState
    committed: bool
    rejected: bool
    action: str
    winner_branch_id: str
    reason: str
    elapsed_s: float
    model_latency_s: float
    total_tokens: int
    candidates: int
    n_merge: int
    n_reselect: int
    n_regen: int

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["state"] = self.state.value
        return row


class CandidateDraft:
    """One generated candidate and its buffered writes."""

    def __init__(
        self,
        txn: "AgentTransaction",
        branch_id: str,
        quality: float,
        gen_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.txn = txn
        self.branch_id = branch_id
        self.quality = float(quality)
        self.gen_cost = float(gen_cost)
        self.metadata = dict(metadata or {})
        self._writes: List[Any] = []
        self._intent_names: List[str] = []

    def _snapshot(self, object_id: str) -> SnapshotValue:
        if object_id not in self.txn.snapshot:
            raise KeyError(f"object was not in transaction snapshot: {object_id}")
        return self.txn.snapshot[object_id]

    def _write(self, object_id: str, branch_value: str, intent: Any, name: str) -> "CandidateDraft":
        snap = self._snapshot(object_id)
        w = cc.BranchWrite()
        w.object_id = object_id
        w.kind = self.txn.manager.kind_of(object_id)
        w.base_value = snap.value
        w.base_version = snap.version
        w.branch_value = str(branch_value)
        w.intent = intent
        self._writes.append(w)
        self._intent_names.append(name)
        return self

    def overwrite(self, object_id: str, new_value: str) -> "CandidateDraft":
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kOverwrite
        return self._write(object_id, str(new_value), intent, "OVERWRITE")

    def append(self, object_id: str, payload: str, *, commutative: bool = False) -> "CandidateDraft":
        snap = self._snapshot(object_id)
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kAppend
        intent.payload = str(payload)
        intent.commutative = bool(commutative)
        name = "APPEND_COMMUTATIVE" if commutative else "APPEND_ORDERED"
        return self._write(object_id, snap.value + str(payload), intent, name)

    def delta(
        self,
        object_id: str,
        amount: int,
        *,
        constrained: bool = False,
        lower_bound: int = 0,
    ) -> "CandidateDraft":
        snap = self._snapshot(object_id)
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kDelta
        intent.payload = str(int(amount))
        intent.constrained = bool(constrained)
        intent.lower_bound = int(lower_bound)
        name = "DELTA_CONSTRAINED" if constrained else "DELTA"
        return self._write(object_id, str(int(snap.value) + int(amount)), intent, name)

    def cas(self, object_id: str, expected: str, new_value: str) -> "CandidateDraft":
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kCas
        cond = cc.Condition()
        cond.type = cc.ConditionType.kValueEquals
        cond.expected_value = str(expected)
        intent.condition = cond
        return self._write(object_id, str(new_value), intent, "CAS")

    def to_core(self) -> Any:
        branch = cc.SpeculativeBranch()
        branch.branch_id = self.branch_id
        branch.writes = self._writes
        branch.gen_cost = self.gen_cost
        branch.quality = self.quality
        return branch

    def to_trace(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "quality": self.quality,
            "gen_cost": self.gen_cost,
            "metadata": self.metadata,
            "intents": list(self._intent_names),
        }


class AgentTransaction:
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
        self.events: List[TransactionEvent] = []
        self.candidates: List[CandidateDraft] = []
        self.model_latency_s = 0.0
        self.total_tokens = 0
        self.result: Optional[TransactionResult] = None
        self._event("begin", {"snapshot_objects": len(snapshot)})

    def _ensure_active(self) -> None:
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError(f"transaction is no longer active: {self.state.value}")

    def _event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TransactionEvent(time.perf_counter() - self.started_at, kind, dict(detail or {}))
        )

    def read(self, object_id: str) -> SnapshotValue:
        self._ensure_active()
        value = self.snapshot[object_id]
        self._event("read", {"object_id": object_id, "version": value.version})
        return value

    def record_model_call(
        self,
        *,
        model: str,
        latency_s: float,
        usage: Optional[Dict[str, Any]] = None,
        candidates: int = 0,
    ) -> None:
        self._ensure_active()
        usage = dict(usage or {})
        self.model_latency_s += float(latency_s)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self._event(
            "model_call",
            {
                "model": model,
                "latency_s": float(latency_s),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "candidates": int(candidates),
            },
        )

    def record_tool_call(
        self,
        name: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        outcome: str = "ok",
    ) -> None:
        self._ensure_active()
        self._event(
            "tool_call",
            {"name": name, "args": dict(args or {}), "outcome": outcome},
        )

    def add_candidate(
        self,
        branch_id: str,
        *,
        quality: float,
        gen_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CandidateDraft:
        self._ensure_active()
        candidate = CandidateDraft(self, branch_id, quality, gen_cost, metadata)
        self.candidates.append(candidate)
        self._event("candidate", {"branch_id": branch_id, "quality": float(quality)})
        return candidate

    def commit(self, strategy: str = "cast") -> TransactionResult:
        self._ensure_active()
        return self.manager.commit(self, strategy=strategy)

    def abort(self, reason: str) -> TransactionResult:
        self._ensure_active()
        self.state = TransactionState.ABORTED
        self._event("abort", {"reason": reason})
        self.result = TransactionResult(
            task_id=self.task_id,
            state=self.state,
            committed=False,
            rejected=False,
            action="abort",
            winner_branch_id="",
            reason=reason,
            elapsed_s=time.perf_counter() - self.started_at,
            model_latency_s=self.model_latency_s,
            total_tokens=self.total_tokens,
            candidates=len(self.candidates),
            n_merge=0,
            n_reselect=0,
            n_regen=0,
        )
        self.manager._record(self)
        return self.result

    def to_trace(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "metadata": self.metadata,
            "state": self.state.value,
            "snapshot_versions": {k: v.version for k, v in self.snapshot.items()},
            "events": [dataclasses.asdict(e) for e in self.events],
            "candidates": [c.to_trace() for c in self.candidates],
            "result": self.result.to_dict() if self.result else None,
        }


class AgentTransactionManager:
    """Thread-safe upper transaction manager backed by the CAST C++ kernel."""

    _KIND_MAP = {
        "generic": cc.ObjectType.kGeneric,
        "row": cc.ObjectType.kRow,
        "text": cc.ObjectType.kText,
        "counter": cc.ObjectType.kCounter,
        "candidate": cc.ObjectType.kCandidateResult,
    }

    def __init__(self, c_gen: float = 1.0, c_merge: float = 0.01):
        self.store = cc.VersionedObjectStore()
        self.model = cc.CostModel(float(c_gen), float(c_merge))
        self.kernel = cc.CostAsymmetricCommit(self.store, self.model)
        self._lock = threading.RLock()
        self._catalog: Dict[str, Any] = {}
        self._traces: List[Dict[str, Any]] = []

    def register_object(self, object_id: str, initial_value: Any, *, kind: str = "generic") -> None:
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        with self._lock:
            if object_id in self._catalog:
                raise ValueError(f"object already registered: {object_id}")
            self.store.put(object_id, str(initial_value))
            self._catalog[object_id] = self._KIND_MAP[kind]

    def kind_of(self, object_id: str) -> Any:
        return self._catalog.get(object_id, cc.ObjectType.kGeneric)

    def begin(self, task_id: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentTransaction:
        with self._lock:
            snapshot = {
                oid: SnapshotValue(
                    value=(v := self.store.get(oid)).value,
                    version=int(v.version),
                    exists=bool(v.exists),
                )
                for oid in self._catalog
            }
        return AgentTransaction(self, str(task_id), snapshot, metadata)

    def commit(self, txn: AgentTransaction, *, strategy: str = "cast") -> TransactionResult:
        if not txn.candidates:
            return txn.abort("no candidates")
        strategy_enum = {
            "cast": cc.CommitStrategy.kCAST,
            "occ": cc.CommitStrategy.kStrictOCC,
        }.get(strategy.lower())
        if strategy_enum is None:
            raise ValueError(f"unknown strategy: {strategy}")

        stats = cc.CostStats()
        branches = [candidate.to_core() for candidate in txn.candidates]
        txn._event("validate", {"strategy": strategy, "candidates": len(branches)})
        with self._lock:
            outcome = self.kernel.commit_task(branches, strategy_enum, stats)

        if outcome.committed:
            txn.state = TransactionState.COMMITTED
        elif getattr(outcome, "rejected", False):
            txn.state = TransactionState.REJECTED
        else:
            txn.state = TransactionState.ABORTED

        txn._event(
            "finish",
            {
                "state": txn.state.value,
                "action": outcome.action,
                "winner_branch_id": outcome.winner_branch_id,
                "reason": outcome.reason,
            },
        )
        txn.result = TransactionResult(
            task_id=txn.task_id,
            state=txn.state,
            committed=bool(outcome.committed),
            rejected=bool(getattr(outcome, "rejected", False)),
            action=outcome.action,
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
        self._record(txn)
        return txn.result

    def value_of(self, object_id: str) -> str:
        with self._lock:
            return self.store.get(object_id).value

    def values(self) -> Dict[str, str]:
        with self._lock:
            return {oid: self.store.get(oid).value for oid in self._catalog}

    def _record(self, txn: AgentTransaction) -> None:
        with self._lock:
            self._traces.append(txn.to_trace())

    def traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._traces)
