"""Concurrency-control plugin registry and ATCC selectors."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, List, Optional, Tuple

from agent.native import load_cast_core
from agent.runtime.adaptive import (
    AdaptivePolicyTable,
    OperationPolicyDecision,
    OperationPolicyTable,
    profile_agent_operations,
)

cc = load_cast_core()


@dataclasses.dataclass(frozen=True)
class CCResolution:
    module: Any
    requested_strategy: str
    selected_strategy: str


class ConcurrencyControlRegistry:
    """Registry for concrete CC modules and adaptive selectors."""

    full_traditional_cc_names = frozenset(
        {"2pl-nowait", "2pl-wait-die", "mvcc-full", "silo-full", "tictoc-full"}
    )
    adaptive_cc_names = frozenset({"adaptive", "atcc", "atcc-table"})
    operation_adaptive_cc_names = frozenset(
        {"adaptive-op", "atcc-op", "operation-atcc"}
    )
    strict_operation_adaptive_cc_names = frozenset(
        {"adaptive-op-strict", "atcc-op-strict", "operation-atcc-strict"}
    )
    pre_snapshot_2pl_names = frozenset(
        {"2pl-pre", "pre-snapshot-2pl", "strict-2pl-pre"}
    )

    def __init__(
        self,
        *,
        adaptive_policy: Optional[AdaptivePolicyTable] = None,
        operation_policy: Optional[OperationPolicyTable] = None,
    ):
        self._lock = threading.RLock()
        self._cc_modules: Dict[str, Any] = {}
        self._cc_metadata: Dict[str, Dict[str, Any]] = {}
        self._adaptive_policy = adaptive_policy or AdaptivePolicyTable.default()
        self._operation_policy = operation_policy or OperationPolicyTable.default()
        self._install_builtin_cc_modules()

    @staticmethod
    def normalize_name(name: str) -> str:
        return str(name).strip().lower()

    def _install_builtin_cc_modules(self) -> None:
        semantic = cc.SemanticConcurrencyControl()
        self.register_cc(
            "semantic",
            semantic,
            aliases=("cast", "semantic-v2"),
            source="ASTRA",
        )
        self.register_cc(
            "occ",
            cc.StrictOccConcurrencyControl(),
            aliases=("strict", "strict-occ", "dbx1000-occ"),
            source="agent",
        )

        traditional = (
            ("mvcc", "DBx1000 MVCC-style strict version validation"),
            ("silo", "DBx1000 Silo-style strict version validation"),
            ("tictoc", "DBx1000 TicToc-style strict version validation"),
        )
        for name, description in traditional:
            self.register_cc(
                name,
                cc.StrictValidationConcurrencyControl(
                    name,
                    "dbx1000-traditional",
                    False,
                    description,
                ),
                aliases=(f"dbx1000-{name}",),
                source="DBx1000-inspired",
            )

        self.register_cc(
            "2pl",
            cc.StrictValidationConcurrencyControl(
                "2pl",
                "pessimistic",
                True,
                "Strict validation with agent-level object locks during commit",
            ),
            aliases=("two-phase-locking", "two_phase_locking", "no-wait", "wait-die"),
            source="DBx1000-inspired",
        )
        self._register_adaptive_cc_metadata()
        self._register_operation_adaptive_cc_metadata()
        self._register_full_traditional_cc_metadata()

    def _register_full_traditional_cc_metadata(self) -> None:
        rows = {
            "2pl-nowait": {
                "family": "pessimistic",
                "description": (
                    "Full transaction-level strict 2PL using no-wait deadlock prevention"
                ),
                "requires_object_locks": True,
                "lock_phase": "transaction",
            },
            "2pl-wait-die": {
                "family": "pessimistic",
                "description": (
                    "Full transaction-level strict 2PL using wait-die deadlock prevention"
                ),
                "requires_object_locks": True,
                "lock_phase": "transaction",
            },
            "mvcc-full": {
                "family": "mvcc",
                "description": "Full snapshot-isolation MVCC over agent transaction read/write sets",
                "requires_object_locks": False,
                "lock_phase": "",
            },
            "silo-full": {
                "family": "silo",
                "description": "Silo-style OCC with write-set locking and read-set TID validation",
                "requires_object_locks": True,
                "lock_phase": "commit",
            },
            "tictoc-full": {
                "family": "tictoc",
                "description": "TicToc-style timestamp validation with read/write timestamp metadata",
                "requires_object_locks": True,
                "lock_phase": "commit",
            },
        }
        for name, metadata in rows.items():
            self._cc_metadata[name] = {
                "canonical_name": name,
                "module_name": name,
                "family": metadata["family"],
                "description": metadata["description"],
                "allows_semantic_rebase": False,
                "requires_object_locks": bool(metadata["requires_object_locks"]),
                "source": "agent-full",
                "aliases": [],
                "selector": "transaction_protocol",
                "lock_phase": metadata["lock_phase"],
                "lookup_name": name,
            }

    def _register_adaptive_cc_metadata(self) -> None:
        metadata = {
            "canonical_name": "adaptive",
            "module_name": "adaptive",
            "family": "adaptive",
            "description": (
                "ATCC-style policy table selector over agent transaction profiles"
            ),
            "allows_semantic_rebase": True,
            "requires_object_locks": False,
            "source": "agent",
            "aliases": sorted(self.adaptive_cc_names - {"adaptive"}),
            "selector": "policy_table",
            "policy_table": self._adaptive_policy.to_dict(),
        }
        for name in self.adaptive_cc_names:
            alias_metadata = dict(metadata)
            alias_metadata["lookup_name"] = name
            self._cc_metadata[name] = alias_metadata

    def _register_operation_adaptive_cc_metadata(self) -> None:
        metadata = {
            "canonical_name": "adaptive-op",
            "module_name": "operation-atcc",
            "family": "adaptive-operation",
            "description": (
                "ATCC-style per-read/write operation policy table over agent traces"
            ),
            "allows_semantic_rebase": True,
            "requires_object_locks": True,
            "source": "agent",
            "aliases": sorted(self.operation_adaptive_cc_names - {"adaptive-op"}),
            "selector": "operation_policy_table",
            "operation_policy_table": self._operation_policy.to_dict(),
        }
        for name in self.operation_adaptive_cc_names:
            alias_metadata = dict(metadata)
            alias_metadata["lookup_name"] = name
            self._cc_metadata[name] = alias_metadata

        strict_metadata = {
            "canonical_name": "adaptive-op-strict",
            "module_name": "operation-atcc-strict",
            "family": "adaptive-operation",
            "description": (
                "Traditional operation ATCC with OCC for optimistic operations "
                "and pre-snapshot locks for pessimistic operations"
            ),
            "allows_semantic_rebase": False,
            "requires_object_locks": True,
            "source": "agent",
            "aliases": sorted(
                self.strict_operation_adaptive_cc_names - {"adaptive-op-strict"}
            ),
            "selector": "operation_policy_table",
            "base_strategy": "occ",
            "lock_phase": "pre_snapshot",
            "operation_policy_table": self._operation_policy.to_dict(),
        }
        for name in self.strict_operation_adaptive_cc_names:
            alias_metadata = dict(strict_metadata)
            alias_metadata["lookup_name"] = name
            self._cc_metadata[name] = alias_metadata

        full_2pl_metadata = {
            "canonical_name": "2pl-pre",
            "module_name": "pre-snapshot-2pl",
            "family": "pessimistic",
            "description": (
                "Acquire all task operation locks before snapshot and hold "
                "them through strict commit validation"
            ),
            "allows_semantic_rebase": False,
            "requires_object_locks": True,
            "source": "agent",
            "aliases": sorted(self.pre_snapshot_2pl_names - {"2pl-pre"}),
            "selector": "all_operations_pessimistic",
            "base_strategy": "occ",
            "lock_phase": "pre_snapshot",
        }
        for name in self.pre_snapshot_2pl_names:
            alias_metadata = dict(full_2pl_metadata)
            alias_metadata["lookup_name"] = name
            self._cc_metadata[name] = alias_metadata

    def register_cc(
        self,
        name: str,
        module: Any,
        *,
        aliases: Tuple[str, ...] = (),
        source: str = "custom",
    ) -> None:
        normalized = self.normalize_name(name)
        if not normalized:
            raise ValueError("CC module name must not be empty")
        if not isinstance(module, cc.ConcurrencyControl):
            raise TypeError("module must implement cast_core.ConcurrencyControl")
        alias_names = tuple(self.normalize_name(alias) for alias in aliases)
        if any(not alias for alias in alias_names):
            raise ValueError("CC module aliases must not be empty")
        metadata = {
            "canonical_name": normalized,
            "module_name": module.name,
            "family": module.family,
            "description": module.description,
            "allows_semantic_rebase": bool(module.allows_semantic_rebase),
            "requires_object_locks": bool(module.requires_object_locks),
            "source": source,
            "aliases": list(alias_names),
            "selector": "module",
        }
        with self._lock:
            for registered_name in (normalized, *alias_names):
                registered_metadata = dict(metadata)
                registered_metadata["lookup_name"] = registered_name
                self._cc_modules[registered_name] = module
                self._cc_metadata[registered_name] = registered_metadata

    def strategies(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                name: dict(metadata)
                for name, metadata in sorted(self._cc_metadata.items())
            }

    def strategy(self, name: str) -> Dict[str, Any]:
        normalized = self.normalize_name(name)
        with self._lock:
            if normalized not in self._cc_metadata:
                raise ValueError(f"unknown CC module: {name}")
            return dict(self._cc_metadata[normalized])

    def adaptive_policy(self) -> Dict[str, Any]:
        with self._lock:
            return self._adaptive_policy.to_dict()

    def operation_policy(self) -> Dict[str, Any]:
        with self._lock:
            return self._operation_policy.to_dict()

    def set_adaptive_policy(self, policy: AdaptivePolicyTable) -> None:
        if not isinstance(policy, AdaptivePolicyTable):
            raise TypeError("policy must be an AdaptivePolicyTable")
        with self._lock:
            self._adaptive_policy = policy
            self._register_adaptive_cc_metadata()

    def set_operation_policy(self, policy: OperationPolicyTable) -> None:
        if not isinstance(policy, OperationPolicyTable):
            raise TypeError("policy must be an OperationPolicyTable")
        with self._lock:
            self._operation_policy = policy
            self._register_operation_adaptive_cc_metadata()

    def resolve(self, strategy: str, txn: Any) -> CCResolution:
        requested = self.normalize_name(strategy)
        with self._lock:
            if requested in self.full_traditional_cc_names:
                return CCResolution(self._cc_modules["occ"], requested, requested)
            if requested in (
                self.strict_operation_adaptive_cc_names
                | self.pre_snapshot_2pl_names
            ):
                selected = (
                    "adaptive-op-strict"
                    if requested in self.strict_operation_adaptive_cc_names
                    else "2pl-pre"
                )
                return CCResolution(self._cc_modules["occ"], requested, selected)
            if requested in self.operation_adaptive_cc_names:
                return CCResolution(self._cc_modules["semantic"], requested, "adaptive-op")
            if requested in self.adaptive_cc_names:
                selected = self._adaptive_policy.select(
                    txn.candidates,
                    read_count=len(txn.read_set),
                    metadata=txn.metadata,
                    available_strategies=self._cc_modules,
                )
                return CCResolution(self._cc_modules[selected], requested, selected)
            cc_module = self._cc_modules.get(requested)
            if cc_module is None:
                raise ValueError(f"unknown CC module: {strategy}")
            return CCResolution(cc_module, requested, requested)

    def is_operation_adaptive(self, strategy: str) -> bool:
        normalized = self.normalize_name(strategy)
        return normalized in (
            self.operation_adaptive_cc_names
            | self.strict_operation_adaptive_cc_names
            | self.pre_snapshot_2pl_names
        )

    def is_full_traditional(self, strategy: str) -> bool:
        return self.normalize_name(strategy) in self.full_traditional_cc_names

    def requires_pre_snapshot_locks(self, strategy: str) -> bool:
        normalized = self.normalize_name(strategy)
        return normalized in (
            self.strict_operation_adaptive_cc_names
            | self.pre_snapshot_2pl_names
        )

    def records_operation_feedback(self, strategy: str) -> bool:
        return self.normalize_name(strategy) in self.strict_operation_adaptive_cc_names

    def pre_snapshot_operation_plan(
        self,
        strategy: str,
        candidates: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Tuple[OperationPolicyDecision, ...]]:
        normalized = self.normalize_name(strategy)
        if normalized in self.strict_operation_adaptive_cc_names:
            decisions = self._operation_policy.select_agent_operations(
                candidates, metadata=metadata
            )
        elif normalized in self.pre_snapshot_2pl_names:
            decisions = tuple(
                OperationPolicyDecision(
                    object_id=profile.object_id,
                    access_kind=profile.access_kind,
                    intent_name=profile.intent_name,
                    policy="pessimistic",
                    rule="pre-snapshot-2pl-all-operations",
                )
                for profile in profile_agent_operations(candidates, metadata=metadata)
            )
        else:
            return [], ()
        targets = sorted(
            {
                decision.object_id
                for decision in decisions
                if decision.policy == "pessimistic"
            }
        )
        return targets, decisions

    def operation_policy_decisions(self, txn: Any):
        precomputed = getattr(txn, "precomputed_operation_policy_decisions", ())
        if precomputed:
            return tuple(precomputed)
        with self._lock:
            return self._operation_policy.select(
                txn.candidates,
                read_object_ids=txn.read_set,
                metadata=txn.metadata,
            )

    def pessimistic_operation_targets(
        self, txn: Any
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        decisions = self.operation_policy_decisions(txn)
        targets = sorted(
            {
                decision.object_id
                for decision in decisions
                if decision.policy == "pessimistic"
            }
        )
        return targets, [decision.to_dict() for decision in decisions]

    def observe_operation_feedback(
        self,
        strategy: str,
        txn: Any,
        result: Any,
        *,
        conflict_object_ids: Tuple[str, ...] = (),
    ) -> None:
        if not self.records_operation_feedback(strategy):
            return
        decisions = self.operation_policy_decisions(txn)
        if not decisions:
            return
        state = getattr(result, "state", "")
        state_value = getattr(state, "value", state)
        conflict_abort = (
            str(state_value) == "aborted"
            and getattr(result, "reason", "") != "traditional_k_loser"
            and getattr(result, "action", "") in {"regenerate_required", "abort"}
        )
        with self._lock:
            self._operation_policy.observe_result(
                decisions,
                committed=bool(getattr(result, "committed", False)),
                rejected=bool(getattr(result, "rejected", False)),
                conflict_abort=bool(conflict_abort),
                conflict_object_ids=conflict_object_ids,
                lock_wait_s=float(getattr(txn, "prelock_wait_s", 0.0)),
                lock_wait_by_object=dict(
                    getattr(txn, "prelock_target_wait_s", {}) or {}
                ),
                lock_queue_by_object=dict(
                    getattr(txn, "prelock_target_queue_depth", {}) or {}
                ),
                lock_handoff_by_object=dict(
                    getattr(txn, "prelock_target_handoff_count", {}) or {}
                ),
                committing_count=float(
                    getattr(txn, "prelock_committing_target_count", 0.0) or 0.0
                ),
                latency_s=float(getattr(result, "elapsed_s", 0.0)),
            )
