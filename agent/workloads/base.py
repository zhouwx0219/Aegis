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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentOperation":
        return cls(
            kind=str(data.get("kind", "")),
            object_id=str(data.get("object_id", "")),
            value=str(data.get("value", "")),
            payload=str(data.get("payload", "")),
            expected=str(data.get("expected", "")),
            amount=int(data.get("amount", 0) or 0),
            constrained=bool(data.get("constrained", False)),
            lower_bound=int(data.get("lower_bound", 0) or 0),
            commutative=bool(data.get("commutative", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclasses.dataclass(frozen=True)
class AgentStage:
    phase: str
    operations: Tuple[AgentOperation, ...]
    delay_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "delay_weight": float(self.delay_weight),
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentStage":
        return cls(
            phase=str(data.get("phase", "")),
            operations=tuple(
                AgentOperation.from_dict(operation)
                for operation in data.get("operations", ()) or ()
            ),
            delay_weight=float(data.get("delay_weight", 1.0) or 1.0),
        )


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


def task_agent_stages(task: AgentTask) -> Tuple[AgentStage, ...]:
    raw_stages = task.context.get("agent_stages", ())
    stages = []
    for row in raw_stages or ():
        if isinstance(row, AgentStage):
            stages.append(row)
        elif isinstance(row, Mapping):
            stages.append(AgentStage.from_dict(row))
    if stages:
        return tuple(stages)
    return tuple(
        AgentStage(str(phase), (), 1.0)
        for phase in task.context.get("agent_phase_sequence", ()) or ()
    )


def stage_operations(task: AgentTask, phase: str) -> Tuple[AgentOperation, ...]:
    requested = str(phase)
    return tuple(
        operation
        for stage in task_agent_stages(task)
        if stage.phase == requested
        for operation in stage.operations
    )


def task_stage_view(task: AgentTask, phase: str) -> AgentTask:
    """Return a task view containing only operations declared for one stage."""

    requested = str(phase)
    allowed = {_operation_key(operation) for operation in stage_operations(task, requested)}
    candidates = []
    for candidate in task.candidates:
        operations = tuple(
            operation
            for operation in candidate.operations
            if _operation_key(operation) in allowed
        )
        if not operations:
            continue
        candidates.append(
            AgentCandidate(
                candidate_id=candidate.candidate_id,
                quality=candidate.quality,
                operations=operations,
                generation_cost=candidate.generation_cost,
                metadata=dict(candidate.metadata),
            )
        )
    context = dict(task.context)
    context["agent_phase"] = requested
    context["agent_stage_local"] = True
    context["agent_phase_sequence"] = (requested,)
    context["agent_stages"] = [
        AgentStage(requested, tuple(stage_operations(task, requested))).to_dict()
    ]
    return AgentTask(
        task_id=task.task_id,
        workload=task.workload,
        task_type=task.task_type,
        request=task.request,
        candidates=tuple(candidates),
        context=context,
    )


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
    populate: bool = True,
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
    if populate:
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

        _populate_candidate_writes(transaction, task)
    except RuntimeError:
        if transaction.state != TransactionState.ACTIVE:
            return transaction
        raise
    return transaction


def populate_task_stage(
    transaction: AgentTransaction,
    task: AgentTask,
    phase: str,
) -> AgentTransaction:
    """Replay one workload-declared agent phase into an active transaction."""

    operations = stage_operations(task, phase)
    read_ids = {
        operation.object_id for operation in operations if operation.kind == "read"
    }
    try:
        for object_id in sorted(read_ids):
            if transaction.state != TransactionState.ACTIVE:
                return transaction
            transaction.read(object_id)
        if str(phase) == "commit" or any(
            operation.kind != "read" for operation in operations
        ):
            _populate_candidate_writes(
                transaction,
                task,
                allowed_operations=operations,
            )
    except RuntimeError:
        if transaction.state != TransactionState.ACTIVE:
            return transaction
        raise
    return transaction


def _populate_candidate_writes(
    transaction: AgentTransaction,
    task: AgentTask,
    *,
    allowed_operations: Optional[Iterable[AgentOperation]] = None,
) -> None:
    allowed = None
    if allowed_operations is not None:
        allowed = {
            _operation_key(operation)
            for operation in allowed_operations
            if operation.kind != "read"
        }
    existing = {
        str(getattr(candidate, "branch_id", ""))
        for candidate in getattr(transaction, "candidates", ())
    }
    for plan in task.candidates:
        if transaction.state != TransactionState.ACTIVE:
            return
        if plan.candidate_id in existing:
            continue
        candidate = transaction.add_candidate(
            plan.candidate_id,
            quality=plan.quality,
            gen_cost=plan.generation_cost,
            metadata={**dict(plan.metadata), "workload": task.workload},
        )
        existing.add(plan.candidate_id)
        for operation in plan.operations:
            if transaction.state != TransactionState.ACTIVE:
                return
            if operation.kind == "read":
                continue
            if allowed is not None and _operation_key(operation) not in allowed:
                continue
            _apply_candidate_operation(candidate, operation)


def _operation_key(operation: AgentOperation) -> Tuple[Any, ...]:
    return (
        operation.kind,
        operation.object_id,
        operation.value,
        operation.payload,
        operation.expected,
        int(operation.amount),
        bool(operation.constrained),
        int(operation.lower_bound),
        bool(operation.commutative),
    )


def _apply_candidate_operation(candidate: Any, operation: AgentOperation) -> None:
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
        candidate.cas(operation.object_id, operation.expected, operation.value)
    else:
        raise ValueError(f"unsupported workload operation: {operation.kind}")
