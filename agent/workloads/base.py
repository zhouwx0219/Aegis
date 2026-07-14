"""Simple workload model for single-plan agent transactions."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from agent.runtime import AgentTransaction, AgentTransactionManager, TransactionResult


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    initial_value: str
    kind: str = "generic"
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class AgentOperation:
    kind: str
    object_id: str
    value: str = ""
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def read(cls, object_id: str, **metadata: Any) -> "AgentOperation":
        return cls("read", str(object_id), metadata=metadata)

    @classmethod
    def write(cls, object_id: str, value: Any, **metadata: Any) -> "AgentOperation":
        return cls("write", str(object_id), value=str(value), metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AgentTask:
    task_id: str
    workload: str
    task_type: str
    operations: Tuple[AgentOperation, ...]
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload": self.workload,
            "task_type": self.task_type,
            "context": dict(self.context),
            "operations": [operation.to_dict() for operation in self.operations],
        }


class AgentWorkload(ABC):
    name: str
    family: str
    level: str

    @abstractmethod
    def objects(self) -> Iterable[ObjectSpec]:
        raise NotImplementedError

    @abstractmethod
    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        raise NotImplementedError


def register_workload(manager: AgentTransactionManager, workload: AgentWorkload) -> None:
    for spec in workload.objects():
        manager.register_object(spec.object_id, spec.initial_value, kind=spec.kind)


def prepare_task_transaction(
    manager: AgentTransactionManager,
    task: AgentTask,
    *,
    runtime_context: Optional[Mapping[str, Any]] = None,
    cc: str = "occ",
) -> AgentTransaction:
    context = {**dict(task.context), **dict(runtime_context or {})}
    txn = manager.begin(
        task.task_id,
        {
            "workload": task.workload,
            "task_type": task.task_type,
            "context": context,
            "retry_count": int(context.get("retry_count", 0) or 0),
            "cc": str(cc),
            "planned_write_targets": [
                operation.object_id
                for operation in task.operations
                if operation.kind == "write"
            ],
        },
        strategy=str(cc),
    )
    populate_task_transaction(txn, task)
    return txn


def execute_task(
    manager: AgentTransactionManager,
    task: AgentTask,
    *,
    cc: str = "occ",
    runtime_context: Optional[Mapping[str, Any]] = None,
) -> TransactionResult:
    txn = prepare_task_transaction(manager, task, runtime_context=runtime_context, cc=cc)
    return txn.commit(strategy=cc)


def populate_task_transaction(
    transaction: AgentTransaction,
    task: AgentTask,
) -> AgentTransaction:
    for operation in task.operations:
        apply_operation(transaction, operation)
    return transaction


def apply_operation(transaction: AgentTransaction, operation: AgentOperation) -> None:
    if operation.kind == "read":
        transaction.read(operation.object_id)
    elif operation.kind == "write":
        transaction.write(operation.object_id, operation.value)
    else:
        raise ValueError(f"unsupported workload operation: {operation.kind}")
