"""Feature extraction for ATCC variants."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Sequence, Tuple

from agent.cc.base import unique_targets


@dataclasses.dataclass(frozen=True)
class ATCCFeatures:
    workload: str
    task_type: str
    level: str
    read_count: int
    write_count: int
    hot_write_count: int
    retry_count: int
    hot_targets: Tuple[str, ...] = ()
    hot_read_targets: Tuple[str, ...] = ()
    write_targets: Tuple[str, ...] = ()
    read_targets: Tuple[str, ...] = ()
    phase_count: int = 0
    reasoning_delay_ms: int = 0
    retry_delay_ms: int = 0

    @property
    def state_key(self) -> str:
        return (
            f"workload={self.workload}|task={self.task_type}|level={self.level}|"
            f"contention={contention_bucket(self.hot_write_count, self.write_count)}|"
            f"hot_reads={hot_read_bucket(self.hot_read_count)}|"
            f"agent_cost={agent_cost_bucket(self.reasoning_delay_ms)}|"
            f"read_set={read_set_bucket(self.read_count)}|"
            f"write_set={write_set_bucket(self.write_count)}|"
            f"retry={retry_stage(self.retry_count)}"
        )

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)

    @property
    def hot_read_count(self) -> int:
        return len(self.hot_read_targets)

    @property
    def hot_access_count(self) -> int:
        return len(set(self.hot_targets) | set(self.hot_read_targets))


def extract_features(txn: Any) -> ATCCFeatures:
    metadata = dict(getattr(txn, "metadata", {}) or {})
    context = dict(metadata.get("context", {}) or {})
    agentic = dict(metadata.get("agentic", {}) or {})
    read_targets = unique_targets(getattr(txn, "read_set", {}).keys())
    write_targets = unique_targets(getattr(txn, "write_set", {}).keys())
    hot_targets = hot_write_targets(txn)
    hot_reads = hot_read_targets(txn)
    return ATCCFeatures(
        workload=str(metadata.get("workload", "")),
        task_type=str(metadata.get("task_type", "")),
        level=str(context.get("level", "")),
        read_count=len(getattr(txn, "read_set", {}) or {}),
        write_count=len(getattr(txn, "write_set", {}) or {}),
        hot_write_count=len(hot_targets),
        retry_count=int(metadata.get("retry_count", context.get("retry_count", 0)) or 0),
        phase_count=int(agentic.get("phase_count", 0) or 0),
        reasoning_delay_ms=int(agentic.get("reasoning_delay_ms", 0) or 0),
        retry_delay_ms=int(agentic.get("retry_delay_ms", 0) or 0),
        hot_targets=hot_targets,
        hot_read_targets=hot_reads,
        write_targets=write_targets,
        read_targets=read_targets,
    )


def extract_task_features(
    task: Any,
    *,
    retry_count: int = 0,
    agentic: Dict[str, Any] | None = None,
) -> ATCCFeatures:
    context = dict(getattr(task, "context", {}) or {})
    agentic = dict(agentic or {})
    operations = tuple(getattr(task, "operations", ()) or ())
    read_targets = unique_targets(
        operation.object_id
        for operation in operations
        if str(getattr(operation, "kind", "")) == "read"
    )
    write_targets = unique_targets(
        operation.object_id
        for operation in operations
        if str(getattr(operation, "kind", "")) != "read"
    )
    hot_targets = hot_task_targets(task, operations=operations)
    hot_reads = hot_task_read_targets(task, operations=operations)
    return ATCCFeatures(
        workload=str(getattr(task, "workload", "")),
        task_type=str(getattr(task, "task_type", "")),
        level=str(context.get("level", "")),
        read_count=sum(1 for operation in operations if str(operation.kind) == "read"),
        write_count=len(write_targets),
        hot_write_count=len(hot_targets),
        retry_count=int(retry_count),
        phase_count=int(agentic.get("phase_count", 0) or 0),
        reasoning_delay_ms=int(agentic.get("reasoning_delay_ms", 0) or 0),
        retry_delay_ms=int(agentic.get("retry_delay_ms", 0) or 0),
        hot_targets=hot_targets,
        hot_read_targets=hot_reads,
        write_targets=write_targets,
        read_targets=read_targets,
    )


def hot_write_targets(txn: Any) -> Tuple[str, ...]:
    targets = []
    metadata = dict(getattr(txn, "metadata", {}) or {})
    context = dict(metadata.get("context", {}) or {})
    for object_id, write in getattr(txn, "write_set", {}).items():
        text = str(object_id)
        if "next_order_id" in text:
            targets.append(text)
        elif text.endswith(":orders"):
            targets.append(text)
        elif ":stock:" in text:
            targets.append(text)
        elif context.get("hot_record_count") and ":record:" in text:
            parts = text.split(":")
            try:
                record_index = parts.index("record") + 1
                if int(parts[record_index]) < int(context.get("hot_record_count") or 0):
                    targets.append(text)
            except (ValueError, IndexError):
                continue
    return unique_targets(targets)


def hot_read_targets(txn: Any) -> Tuple[str, ...]:
    targets = []
    metadata = dict(getattr(txn, "metadata", {}) or {})
    context = dict(metadata.get("context", {}) or {})
    for object_id in getattr(txn, "read_set", {}).keys():
        if is_hot_object_id(str(object_id), context):
            targets.append(str(object_id))
    return unique_targets(targets)


def hot_task_targets(task: Any, *, operations: Sequence[Any] | None = None) -> Tuple[str, ...]:
    targets = []
    context = dict(getattr(task, "context", {}) or {})
    for operation in tuple(operations if operations is not None else getattr(task, "operations", ()) or ()):
        if str(getattr(operation, "kind", "")) == "read":
            continue
        object_id = str(getattr(operation, "object_id", ""))
        if is_hot_object_id(object_id, context):
            targets.append(object_id)
    return unique_targets(targets)


def hot_task_read_targets(task: Any, *, operations: Sequence[Any] | None = None) -> Tuple[str, ...]:
    targets = []
    context = dict(getattr(task, "context", {}) or {})
    for operation in tuple(operations if operations is not None else getattr(task, "operations", ()) or ()):
        if str(getattr(operation, "kind", "")) != "read":
            continue
        object_id = str(getattr(operation, "object_id", ""))
        if is_hot_object_id(object_id, context):
            targets.append(object_id)
    return unique_targets(targets)


def is_hot_object_id(object_id: str, context: Dict[str, Any]) -> bool:
    text = str(object_id)
    if "next_order_id" in text:
        return True
    if text.endswith(":orders"):
        return True
    if ":stock:" in text:
        return True
    if context.get("hot_record_count") and ":record:" in text:
        parts = text.split(":")
        try:
            record_index = parts.index("record") + 1
            return int(parts[record_index]) < int(context.get("hot_record_count") or 0)
        except (ValueError, IndexError):
            return False
    return False


def contention_bucket(hot_write_count: int, write_count: int) -> str:
    hot = int(hot_write_count)
    writes = max(0, int(write_count))
    if hot <= 0:
        return "cold"
    if hot == 1 and writes <= 3:
        return "warm"
    if hot <= 3:
        return "hot"
    return "extreme"


def agent_cost_bucket(reasoning_delay_ms: int) -> str:
    delay = int(reasoning_delay_ms)
    if delay < 10:
        return "short"
    if delay < 50:
        return "medium"
    if delay < 200:
        return "long"
    return "very-long"


def write_set_bucket(write_count: int) -> str:
    count = int(write_count)
    if count <= 1:
        return "tiny"
    if count <= 4:
        return "small"
    return "large"


def read_set_bucket(read_count: int) -> str:
    count = int(read_count)
    if count <= 0:
        return "none"
    if count <= 2:
        return "small"
    if count <= 6:
        return "medium"
    return "large"


def hot_read_bucket(hot_read_count: int) -> str:
    count = int(hot_read_count)
    if count <= 0:
        return "none"
    if count <= 2:
        return "some"
    return "many"


def retry_stage(retry_count: int) -> str:
    return "first" if int(retry_count) <= 0 else "retry"
