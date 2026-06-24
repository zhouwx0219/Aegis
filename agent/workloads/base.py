"""Provider-neutral data model for agent-style transactional workloads."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from agent.runtime import (
    AgentTransaction,
    AgentTransactionManager,
    TransactionResult,
    TransactionState,
)


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    initial_value: str
    kind: str = "generic"
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class WorkloadManifest:
    name: str
    benchmark_family: str
    source_system: str
    source_files: Tuple[str, ...]
    preserved_semantics: Tuple[str, ...]
    agent_adaptations: Tuple[str, ...]
    workload_layer: str = "semantic"
    canonical_name: str = ""
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "benchmark_family": self.benchmark_family,
            "source_system": self.source_system,
            "source_files": list(self.source_files),
            "preserved_semantics": list(self.preserved_semantics),
            "agent_adaptations": list(self.agent_adaptations),
            "workload_layer": self.workload_layer,
            "canonical_name": self.canonical_name or self.name,
            "config": dict(self.config),
        }


@dataclasses.dataclass(frozen=True)
class AgentOperation:
    kind: str
    object_id: str
    value: str = ""
    payload: str = ""
    expected: str = ""
    amount: int = 0
    constrained: bool = False
    lower_bound: int = 0
    commutative: bool = False
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def read(cls, object_id: str, **metadata: Any) -> "AgentOperation":
        return cls("read", object_id, metadata=metadata)

    @classmethod
    def overwrite(
        cls, object_id: str, value: str, **metadata: Any
    ) -> "AgentOperation":
        return cls("overwrite", object_id, value=str(value), metadata=metadata)

    @classmethod
    def append(
        cls, object_id: str, payload: str, *, commutative: bool = False, **metadata: Any
    ) -> "AgentOperation":
        return cls(
            "append",
            object_id,
            payload=str(payload),
            commutative=commutative,
            metadata=metadata,
        )

    @classmethod
    def delta(
        cls,
        object_id: str,
        amount: int,
        *,
        constrained: bool = False,
        lower_bound: int = 0,
        **metadata: Any,
    ) -> "AgentOperation":
        return cls(
            "delta",
            object_id,
            amount=int(amount),
            constrained=constrained,
            lower_bound=int(lower_bound),
            metadata=metadata,
        )

    @classmethod
    def cas(
        cls, object_id: str, expected: str, value: str, **metadata: Any
    ) -> "AgentOperation":
        return cls(
            "cas",
            object_id,
            value=str(value),
            expected=str(expected),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AgentCandidate:
    candidate_id: str
    quality: float
    operations: Tuple[AgentOperation, ...]
    generation_cost: float = 0.0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        write_targets = [
            operation.object_id
            for operation in self.operations
            if operation.kind != "read"
        ]
        if len(write_targets) != len(set(write_targets)):
            raise ValueError("candidate contains duplicate write targets")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "quality": self.quality,
            "generation_cost": self.generation_cost,
            "metadata": dict(self.metadata),
            "operations": [operation.to_dict() for operation in self.operations],
        }


@dataclasses.dataclass(frozen=True)
class AgentTask:
    task_id: str
    workload: str
    task_type: str
    request: str
    candidates: Tuple[AgentCandidate, ...]
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload": self.workload,
            "task_type": self.task_type,
            "request": self.request,
            "context": dict(self.context),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class AgentWorkload(ABC):
    name: str

    def manifest(self) -> WorkloadManifest:
        return WorkloadManifest(
            name=self.name,
            benchmark_family=self.name,
            source_system="custom",
            source_files=(),
            preserved_semantics=(),
            agent_adaptations=(),
        )

    @abstractmethod
    def objects(self) -> Iterable[ObjectSpec]:
        raise NotImplementedError

    @abstractmethod
    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        raise NotImplementedError


def register_workload(
    manager: AgentTransactionManager, workload: AgentWorkload
) -> None:
    for spec in workload.objects():
        manager.register_object(spec.object_id, spec.initial_value, kind=spec.kind)


def execute_task(
    manager: AgentTransactionManager,
    task: AgentTask,
    *,
    cc: str = "semantic",
    regenerator: Optional[Any] = None,
) -> TransactionResult:
    transaction = prepare_task_transaction(manager, task, strategy=cc)
    return transaction.commit(strategy=cc, regenerator=regenerator)


def prepare_task_transaction(
    manager: AgentTransactionManager,
    task: AgentTask,
    *,
    strategy: Optional[str] = None,
    runtime_context: Optional[Mapping[str, Any]] = None,
) -> AgentTransaction:
    """Create a transaction, optionally locking selected targets before snapshot."""

    context = {**dict(task.context), **dict(runtime_context or {})}
    metadata = {
        "workload": task.workload,
        "task_type": task.task_type,
        "request": task.request,
        "context": context,
        "retry_count": int(context.get("retry_count", 0) or 0),
        "agent_interval_s": float(
            context.get("agent_interval_s", context.get("interaction_latency_s", 0.0))
            or 0.0
        ),
        "agent_phase": str(context.get("agent_phase", "") or ""),
    }
    prelock_targets = []
    operation_policy_decisions = ()
    if strategy is not None:
        prelock_targets, operation_policy_decisions = (
            manager.cc_registry.pre_snapshot_operation_plan(
                strategy,
                task.candidates,
                metadata=metadata,
            )
        )
    transaction = manager.begin(
        task.task_id,
        metadata,
        prelock_targets=prelock_targets,
        operation_policy_decisions=operation_policy_decisions,
    )
    populate_task_transaction(transaction, task)
    return transaction


def populate_task_transaction(
    transaction: AgentTransaction,
    task: AgentTask,
) -> AgentTransaction:
    """Replay an agent task into an active transaction snapshot."""

    read_ids = {
        operation.object_id
        for candidate in task.candidates
        for operation in candidate.operations
        if operation.kind == "read"
    }
    try:
        for object_id in sorted(read_ids):
            if transaction.state != TransactionState.ACTIVE:
                return transaction
            transaction.read(object_id)

        for plan in task.candidates:
            if transaction.state != TransactionState.ACTIVE:
                return transaction
            candidate = transaction.add_candidate(
                plan.candidate_id,
                quality=plan.quality,
                gen_cost=plan.generation_cost,
                metadata={**dict(plan.metadata), "workload": task.workload},
            )
            for operation in plan.operations:
                if transaction.state != TransactionState.ACTIVE:
                    return transaction
                if operation.kind == "read":
                    continue
                if operation.kind == "overwrite":
                    candidate.overwrite(operation.object_id, operation.value)
                elif operation.kind == "append":
                    candidate.append(
                        operation.object_id,
                        operation.payload,
                        commutative=operation.commutative,
                    )
                elif operation.kind == "delta":
                    candidate.delta(
                        operation.object_id,
                        operation.amount,
                        constrained=operation.constrained,
                        lower_bound=operation.lower_bound,
                    )
                elif operation.kind == "cas":
                    candidate.cas(
                        operation.object_id, operation.expected, operation.value
                    )
                else:
                    raise ValueError(f"unsupported workload operation: {operation.kind}")
    except RuntimeError:
        if transaction.state != TransactionState.ACTIVE:
            return transaction
        raise
    return transaction
