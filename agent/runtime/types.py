"""Shared DTOs for the single-plan agent transaction runtime."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, Tuple


class TransactionState(str, enum.Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
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


@dataclasses.dataclass(frozen=True)
class ReadRecord:
    object_id: str
    version: int


@dataclasses.dataclass(frozen=True)
class WriteRecord:
    object_id: str
    base_value: str
    base_version: int
    value: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TransactionResult:
    task_id: str
    state: TransactionState
    strategy: str
    committed: bool
    action: str
    reason: str
    elapsed_s: float
    read_count: int
    write_count: int
    conflict_object_ids: Tuple[str, ...] = ()
    lock_wait_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["state"] = self.state.value
        row["conflict_object_ids"] = list(self.conflict_object_ids)
        return row
