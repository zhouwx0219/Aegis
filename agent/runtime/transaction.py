"""Agent-side transaction lifecycle over pluggable runtime modules.

The manager keeps the task boundary and catalog. Multi-branch semantics,
concurrency-control selection, adaptive policy tables, and commit protocols are
separate modules so the agent layer can evolve without changing the versioned KV
backend.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from agent.native import load_cast_core
from agent.runtime.adaptive import AdaptivePolicyTable, OperationPolicyTable
from agent.runtime.branching import (
    BranchSemantics,
    CandidateDraft,
    QualityRankedBranchSemantics,
)
from agent.runtime.cc_registry import ConcurrencyControlRegistry
from agent.runtime.commit_protocol import (
    CostAwareCommitProtocol,
    ObjectLockTable,
    ObjectLockTimeout,
)
from agent.runtime.types import (
    SnapshotValue,
    TransactionEvent,
    TransactionResult,
    TransactionState,
)

cc = load_cast_core()


class AgentTransaction:
    def __init__(
        self,
        manager: "AgentTransactionManager",
        task_id: str,
        snapshot: Dict[str, SnapshotValue],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        prelock_lease: Optional[Any] = None,
        precomputed_operation_policy_decisions: Iterable[Any] = (),
    ):
        self.manager = manager
        self.task_id = str(task_id)
        self.snapshot = snapshot
        self.metadata = dict(metadata or {})
        self.state = TransactionState.ACTIVE
        self._lifecycle_lock = threading.RLock()
        self.started_at = time.perf_counter()
        self.events: List[TransactionEvent] = []
        self.read_set: Dict[str, SnapshotValue] = {}
        self.candidates: List[CandidateDraft] = []
        self.model_latency_s = 0.0
        self.total_tokens = 0
        self.result: Optional[TransactionResult] = None
        self._prelock_lease = prelock_lease
        self._yielded_prelock_targets: tuple[str, ...] = ()
        self._yielded_prelock_priority = 0
        self._yielded_prelock_reason = "pre-snapshot-atcc"
        self.prelocked_targets = tuple(
            getattr(prelock_lease, "targets", ()) if prelock_lease is not None else ()
        )
        self.prelock_wait_s = float(
            getattr(prelock_lease, "wait_s", 0.0) if prelock_lease is not None else 0.0
        )
        self.prelock_target_wait_s = {
            str(object_id): float(wait_s)
            for object_id, wait_s in dict(
                getattr(prelock_lease, "target_wait_s", {})
                if prelock_lease is not None
                else {}
            ).items()
        }
        self.prelock_target_queue_depth = {
            str(object_id): int(depth)
            for object_id, depth in dict(
                getattr(prelock_lease, "target_queue_depth", {})
                if prelock_lease is not None
                else {}
            ).items()
        }
        self.prelock_target_owner_priority = {
            str(object_id): int(priority)
            for object_id, priority in dict(
                getattr(prelock_lease, "target_owner_priority", {})
                if prelock_lease is not None
                else {}
            ).items()
        }
        self.prelock_target_handoff_count = {
            str(object_id): int(count)
            for object_id, count in dict(
                getattr(prelock_lease, "target_handoff_count", {})
                if prelock_lease is not None
                else {}
            ).items()
        }
        self.prelock_committing_enters = 0
        self.prelock_committing_exits = 0
        self.prelock_committing_target_count = 0
        self.precomputed_operation_policy_decisions = tuple(
            precomputed_operation_policy_decisions
        )
        self._event(
            "begin",
            {
                "snapshot_objects": len(snapshot),
                "prelocked_targets": list(self.prelocked_targets),
                "prelock_wait_s": self.prelock_wait_s,
                "prelock_target_wait_s": dict(self.prelock_target_wait_s),
                "prelock_target_queue_depth": dict(self.prelock_target_queue_depth),
                "prelock_target_owner_priority": dict(
                    self.prelock_target_owner_priority
                ),
                "prelock_target_handoff_count": dict(
                    self.prelock_target_handoff_count
                ),
                "prelock_priority": int(
                    getattr(prelock_lease, "priority", 0)
                    if prelock_lease is not None
                    else 0
                ),
            },
        )

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
            value = self.snapshot[object_id]
            self.read_set.setdefault(object_id, value)
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
        with self._lifecycle_lock:
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
        with self._lifecycle_lock:
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
        with self._lifecycle_lock:
            self._ensure_active()
            candidate = CandidateDraft(self, branch_id, quality, gen_cost, metadata)
            self.candidates.append(candidate)
            self._event("candidate", {"branch_id": branch_id, "quality": float(quality)})
            return candidate

    def commit(
        self,
        strategy: str = "cast",
        *,
        regenerator: Optional[Callable[["AgentTransaction"], None]] = None,
        max_regenerations: int = 1,
    ) -> TransactionResult:
        with self._lifecycle_lock:
            if self.state != TransactionState.ACTIVE:
                if (
                    self.result is not None
                    and self.state == TransactionState.ABORTED
                    and self.result.reason == "priority_wound"
                ):
                    return self.result
                self._ensure_active()
            return self.manager.commit(
                self,
                strategy=strategy,
                regenerator=regenerator,
                max_regenerations=max_regenerations,
            )

    def abort(self, reason: str) -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            try:
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
            finally:
                self._release_prelocks()

    def _release_prelocks(self) -> None:
        lease = self._prelock_lease
        if lease is None:
            return
        self._prelock_lease = None
        self.prelocked_targets = ()
        lease.release()

    def _enter_prelock_committing(self) -> None:
        lease = self._prelock_lease
        if lease is None:
            return
        targets = tuple(getattr(lease, "targets", ()) or ())
        self.prelock_committing_enters += 1
        self.prelock_committing_target_count += len(targets)
        lease.enter_committing()
        self._event(
            "prelock_committing_enter",
            {"targets": list(targets)},
        )

    def _exit_prelock_committing(self) -> None:
        lease = self._prelock_lease
        if lease is None:
            return
        self.prelock_committing_exits += 1
        lease.exit_committing()
        self._event(
            "prelock_committing_exit",
            {"targets": list(getattr(lease, "targets", ()) or ())},
        )

    def yield_prelocks_for_planning(self, reason: str = "planning-yield") -> None:
        with self._lifecycle_lock:
            self._ensure_active()
            lease = self._prelock_lease
            if lease is None:
                return
            self._yielded_prelock_targets = tuple(getattr(lease, "targets", ()) or ())
            self._yielded_prelock_priority = int(getattr(lease, "priority", 0) or 0)
            self._yielded_prelock_reason = str(getattr(lease, "reason", "pre-snapshot-atcc"))
            self._release_prelocks()
            self._event(
                "prelock_yield",
                {
                    "reason": str(reason),
                    "targets": list(self._yielded_prelock_targets),
                    "priority": self._yielded_prelock_priority,
                },
            )

    def reacquire_yielded_prelocks(self) -> None:
        with self._lifecycle_lock:
            self._ensure_active()
            if self._prelock_lease is not None or not self._yielded_prelock_targets:
                return
            lease = self.manager.object_locks.acquire_lease(
                self._yielded_prelock_targets,
                priority=self._yielded_prelock_priority,
                reason=self._yielded_prelock_reason + "-reacquire",
            )
            lease.bind_owner(self)
            self._prelock_lease = lease
            self.prelocked_targets = tuple(getattr(lease, "targets", ()) or ())
            self.prelock_wait_s += float(getattr(lease, "wait_s", 0.0) or 0.0)
            for object_id, wait_s in dict(getattr(lease, "target_wait_s", {}) or {}).items():
                key = str(object_id)
                self.prelock_target_wait_s[key] = (
                    self.prelock_target_wait_s.get(key, 0.0) + float(wait_s)
                )
            for object_id, depth in dict(
                getattr(lease, "target_queue_depth", {}) or {}
            ).items():
                self.prelock_target_queue_depth[str(object_id)] = int(depth)
            for object_id, priority in dict(
                getattr(lease, "target_owner_priority", {}) or {}
            ).items():
                self.prelock_target_owner_priority[str(object_id)] = int(priority)
            for object_id, count in dict(
                getattr(lease, "target_handoff_count", {}) or {}
            ).items():
                key = str(object_id)
                self.prelock_target_handoff_count[key] = (
                    self.prelock_target_handoff_count.get(key, 0) + int(count)
                )
            self._event(
                "prelock_reacquire",
                {
                    "targets": list(self.prelocked_targets),
                    "wait_s": float(getattr(lease, "wait_s", 0.0) or 0.0),
                    "target_wait_s": dict(getattr(lease, "target_wait_s", {}) or {}),
                    "target_queue_depth": dict(
                        getattr(lease, "target_queue_depth", {}) or {}
                    ),
                    "target_owner_priority": dict(
                        getattr(lease, "target_owner_priority", {}) or {}
                    ),
                    "target_handoff_count": dict(
                        getattr(lease, "target_handoff_count", {}) or {}
                    ),
                    "priority": self._yielded_prelock_priority,
                },
            )
            self._yielded_prelock_targets = ()

    def refresh_snapshot_for_regeneration(
        self,
        reason: str = "lease-refresh-regenerate",
    ) -> None:
        with self._lifecycle_lock:
            self._ensure_active()
            self.snapshot = self.manager._snapshot_threadsafe()
            self.read_set.clear()
            self.candidates.clear()
            self._event(
                "refresh_regenerate",
                {
                    "reason": str(reason),
                    "snapshot_versions": {
                        key: value.version for key, value in self.snapshot.items()
                    },
                    "prelocked_targets": list(self.prelocked_targets),
                },
            )

    def _wound_prelock(self, reason: str) -> None:
        with self._lifecycle_lock:
            if self.state != TransactionState.ACTIVE:
                self._release_prelocks()
                return
            self.state = TransactionState.ABORTED
            self._event("wound", {"reason": str(reason)})
            self.result = TransactionResult(
                task_id=self.task_id,
                state=self.state,
                committed=False,
                rejected=False,
                action="abort",
                winner_branch_id="",
                reason="priority_wound",
                elapsed_s=time.perf_counter() - self.started_at,
                model_latency_s=self.model_latency_s,
                total_tokens=self.total_tokens,
                candidates=len(self.candidates),
                n_merge=0,
                n_reselect=0,
                n_regen=0,
            )
            self._release_prelocks()
            self.manager._record(self)

    def _read_set_for_core(self) -> List[Any]:
        reads = []
        for object_id, snapshot in sorted(self.read_set.items()):
            read = cc.BranchRead()
            read.object_id = object_id
            read.version = snapshot.version
            reads.append(read)
        return reads

    def to_trace(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "metadata": self.metadata,
            "state": self.state.value,
            "snapshot_versions": {k: v.version for k, v in self.snapshot.items()},
            "read_set_versions": {k: v.version for k, v in self.read_set.items()},
            "prelocked_targets": list(self.prelocked_targets),
            "prelock_wait_s": self.prelock_wait_s,
            "prelock_target_wait_s": dict(self.prelock_target_wait_s),
            "prelock_target_queue_depth": dict(self.prelock_target_queue_depth),
            "prelock_target_owner_priority": dict(
                self.prelock_target_owner_priority
            ),
            "prelock_target_handoff_count": dict(
                self.prelock_target_handoff_count
            ),
            "prelock_committing_enters": int(self.prelock_committing_enters),
            "prelock_committing_exits": int(self.prelock_committing_exits),
            "prelock_committing_target_count": int(
                self.prelock_committing_target_count
            ),
            "events": [dataclasses.asdict(event) for event in self.events],
            "candidates": [candidate.to_trace() for candidate in self.candidates],
            "result": self.result.to_dict() if self.result else None,
        }


class AgentTransactionManager:
    """Thread-safe upper transaction manager backed by a versioned KV store."""

    _KIND_MAP = {
        "generic": cc.ObjectType.kGeneric,
        "row": cc.ObjectType.kRow,
        "text": cc.ObjectType.kText,
        "counter": cc.ObjectType.kCounter,
        "candidate": cc.ObjectType.kCandidateResult,
    }

    def __init__(
        self,
        c_gen: float = 1.0,
        c_merge: float = 0.01,
        *,
        store: Optional[Any] = None,
        adaptive_policy: Optional[AdaptivePolicyTable] = None,
        operation_policy: Optional[OperationPolicyTable] = None,
        branch_semantics: Optional[BranchSemantics] = None,
        cc_registry: Optional[ConcurrencyControlRegistry] = None,
        commit_protocol: Optional[Any] = None,
        object_lock_queue_policy: str = "race",
        object_lock_priority_burst: int = 2,
        prelock_wait_budget_s: float = 0.0,
        prelock_wait_budget_mode: str = "transaction",
    ):
        self.store = store if store is not None else cc.Dbx1000VersionedKVStore()
        self.model = cc.CostModel(float(c_gen), float(c_merge))
        self._lock = threading.RLock()
        self._catalog: Dict[str, Any] = {}
        self._traces: List[Dict[str, Any]] = []

        self.cc_registry = cc_registry or ConcurrencyControlRegistry(
            adaptive_policy=adaptive_policy,
            operation_policy=operation_policy,
        )
        if cc_registry is not None:
            if adaptive_policy is not None:
                self.cc_registry.set_adaptive_policy(adaptive_policy)
            if operation_policy is not None:
                self.cc_registry.set_operation_policy(operation_policy)

        self.branch_semantics = branch_semantics or QualityRankedBranchSemantics()
        self.object_locks = ObjectLockTable(
            queue_policy=object_lock_queue_policy,
            priority_burst=object_lock_priority_burst,
        )
        self.prelock_wait_budget_s = max(0.0, float(prelock_wait_budget_s))
        mode = str(prelock_wait_budget_mode or "transaction").strip().lower()
        if mode not in {"transaction", "object"}:
            raise ValueError(f"unsupported prelock wait budget mode: {mode}")
        self.prelock_wait_budget_mode = mode
        self.commit_protocol = commit_protocol or CostAwareCommitProtocol(
            self.store,
            self.model,
            registry=self.cc_registry,
            branch_semantics=self.branch_semantics,
            lock_table=self.object_locks,
        )
        self.kernel = getattr(self.commit_protocol, "kernel", None)
        if self.kernel is None:
            self.kernel = cc.CostAsymmetricCommit(self.store, self.model)

    @property
    def backend_name(self) -> str:
        return str(self.store.backend_name)

    @staticmethod
    def _normalize_cc_name(name: str) -> str:
        return ConcurrencyControlRegistry.normalize_name(name)

    def register_cc(
        self,
        name: str,
        module: Any,
        *,
        aliases: tuple[str, ...] = (),
        source: str = "custom",
    ) -> None:
        self.cc_registry.register_cc(name, module, aliases=aliases, source=source)

    def cc_strategies(self) -> Dict[str, Dict[str, Any]]:
        return self.cc_registry.strategies()

    def cc_strategy(self, name: str) -> Dict[str, Any]:
        return self.cc_registry.strategy(name)

    def adaptive_policy(self) -> Dict[str, Any]:
        return self.cc_registry.adaptive_policy()

    def operation_policy(self) -> Dict[str, Any]:
        return self.cc_registry.operation_policy()

    def set_adaptive_policy(self, policy: AdaptivePolicyTable) -> None:
        self.cc_registry.set_adaptive_policy(policy)

    def set_operation_policy(self, policy: OperationPolicyTable) -> None:
        self.cc_registry.set_operation_policy(policy)

    def module_catalog(self) -> Dict[str, Any]:
        return {
            "branch_semantics": self.branch_semantics.to_dict(),
            "commit_protocol": self.commit_protocol.to_dict()
            if hasattr(self.commit_protocol, "to_dict")
            else {"name": type(self.commit_protocol).__name__},
            "cc_strategies": self.cc_strategies(),
        }

    def register_object(self, object_id: str, initial_value: Any, *, kind: str = "generic") -> None:
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        with self._lock:
            if object_id in self._catalog:
                raise ValueError(f"object already registered: {object_id}")
            self.store.put(object_id, str(initial_value))
            self._catalog[object_id] = self._KIND_MAP[kind]
            self.object_locks.ensure(object_id)

    def kind_of(self, object_id: str) -> Any:
        return self._catalog.get(object_id, cc.ObjectType.kGeneric)

    def begin(
        self,
        task_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        prelock_targets: Iterable[str] = (),
        operation_policy_decisions: Iterable[Any] = (),
    ) -> AgentTransaction:
        lease = None
        try:
            targets = tuple(sorted(set(prelock_targets)))
            effective_metadata = dict(metadata or {})
            effective_decisions = tuple(operation_policy_decisions)
            if targets:
                budget_s = (
                    self.prelock_wait_budget_s
                    if self._should_budget_prelock(effective_decisions)
                    else 0.0
                )
                if budget_s > 0.0 and self.prelock_wait_budget_mode == "object":
                    lease, skipped_targets = self.object_locks.acquire_budgeted_lease(
                        targets,
                        priority=self._prelock_priority(metadata, effective_decisions),
                        reason="pre-snapshot-atcc",
                        wait_timeout_s=budget_s,
                    )
                    if skipped_targets:
                        effective_decisions = self._fallback_prelock_decisions(
                            effective_decisions,
                            object_ids=skipped_targets,
                        )
                        self._record_prelock_fallback(
                            effective_metadata,
                            reason="prelock_object_wait_budget_exceeded",
                            targets=skipped_targets,
                            budget_s=budget_s,
                            detail="per-object prelock wait budget exhausted",
                        )
                else:
                    try:
                        lease = self.object_locks.acquire_lease(
                            targets,
                            priority=self._prelock_priority(
                                metadata,
                                effective_decisions,
                            ),
                            reason="pre-snapshot-atcc",
                            wait_timeout_s=(
                                budget_s
                                if budget_s > 0.0
                                else None
                            ),
                        )
                    except ObjectLockTimeout as exc:
                        effective_decisions = self._fallback_prelock_decisions(
                            effective_decisions
                        )
                        self._record_prelock_fallback(
                            effective_metadata,
                            reason="prelock_wait_budget_exceeded",
                            targets=targets,
                            budget_s=budget_s,
                            detail=str(exc),
                        )
            with self._lock:
                snapshot = self._snapshot_locked()
            txn = AgentTransaction(
                self,
                str(task_id),
                snapshot,
                effective_metadata,
                prelock_lease=lease,
                precomputed_operation_policy_decisions=effective_decisions,
            )
            if lease is not None:
                lease.bind_owner(txn)
            return txn
        except BaseException:
            if lease is not None:
                lease.release()
            raise

    def _should_budget_prelock(self, decisions: Iterable[Any]) -> bool:
        if self.prelock_wait_budget_s <= 0.0:
            return False
        rows = tuple(decisions)
        if not rows:
            return False
        return any(
            str(getattr(decision, "rule", ""))
            != "pre-snapshot-2pl-all-operations"
            for decision in rows
        )

    def _fallback_prelock_decisions(
        self,
        decisions: Iterable[Any],
        *,
        object_ids: Iterable[str] = (),
    ) -> tuple[Any, ...]:
        target_ids = {str(object_id) for object_id in object_ids}
        rows = []
        for decision in decisions:
            if getattr(decision, "policy", "") != "pessimistic":
                rows.append(decision)
                continue
            if target_ids and str(getattr(decision, "object_id", "")) not in target_ids:
                rows.append(decision)
                continue
            try:
                rows.append(
                    dataclasses.replace(
                        decision,
                        policy="optimistic",
                        rule="prelock-budget-fallback-optimistic",
                    )
                )
            except TypeError:
                rows.append(decision)
        return tuple(rows)

    def _record_prelock_fallback(
        self,
        metadata: Dict[str, Any],
        *,
        reason: str,
        targets: Iterable[str],
        budget_s: float,
        detail: str,
    ) -> None:
        fallback = {
            "reason": str(reason),
            "targets": [str(target) for target in targets],
            "budget_s": float(budget_s),
            "detail": str(detail),
        }
        context = dict(metadata.get("context", {}) or {})
        context["prelock_fallback"] = fallback
        metadata["context"] = context
        metadata["prelock_fallback"] = fallback

    def _prelock_priority(
        self,
        metadata: Optional[Dict[str, Any]],
        operation_policy_decisions: Iterable[Any],
    ) -> int:
        metadata_dict = dict(metadata or {})
        priorities = [
            int(getattr(decision, "atcc_priority", 0) or 0)
            for decision in operation_policy_decisions
        ]
        context = dict(metadata_dict.get("context", {}) or {})
        for key in ("atcc_priority", "lock_priority", "priority"):
            if key in metadata_dict:
                priorities.append(int(metadata_dict.get(key) or 0))
            if key in context:
                priorities.append(int(context.get(key) or 0))
        return max(priorities or [0])

    def _snapshot_locked(self) -> Dict[str, SnapshotValue]:
        return {
            oid: SnapshotValue(
                value=(value := self.store.get(oid)).value,
                version=int(value.version),
                exists=bool(value.exists),
            )
            for oid in self._catalog
        }

    def _snapshot_threadsafe(self) -> Dict[str, SnapshotValue]:
        with self._lock:
            return self._snapshot_locked()

    def _resolve_cc_module(self, strategy: str, txn: AgentTransaction) -> tuple[Any, str, str]:
        resolution = self.cc_registry.resolve(strategy, txn)
        return (
            resolution.module,
            resolution.requested_strategy,
            resolution.selected_strategy,
        )

    def _select_adaptive_cc(self, txn: AgentTransaction) -> str:
        return self.cc_registry.resolve("adaptive", txn).selected_strategy

    def _operation_policy_decisions(self, txn: AgentTransaction):
        return self.cc_registry.operation_policy_decisions(txn)

    def _pessimistic_operation_targets(self, txn: AgentTransaction) -> tuple[List[str], List[Dict[str, Any]]]:
        return self.cc_registry.pessimistic_operation_targets(txn)

    def _object_lock_scope(self, branches: List[Any], cc_module: Any, targets: Optional[List[str]] = None):
        return self.object_locks.scope_for_branches(branches, cc_module, targets)

    def commit(
        self,
        txn: AgentTransaction,
        *,
        strategy: str = "cast",
        regenerator: Optional[Callable[[AgentTransaction], None]] = None,
        max_regenerations: int = 1,
    ) -> TransactionResult:
        with txn._lifecycle_lock:
            try:
                return self._commit_locked(
                    txn,
                    strategy=strategy,
                    regenerator=regenerator,
                    max_regenerations=max_regenerations,
                )
            finally:
                txn._release_prelocks()

    def _commit_locked(
        self,
        txn: AgentTransaction,
        *,
        strategy: str = "cast",
        regenerator: Optional[Callable[[AgentTransaction], None]] = None,
        max_regenerations: int = 1,
    ) -> TransactionResult:
        return self.commit_protocol.commit(
            txn,
            strategy=strategy,
            regenerator=regenerator,
            max_regenerations=max_regenerations,
            refresh_snapshot=self._snapshot_threadsafe,
            record=self._record,
        )

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
