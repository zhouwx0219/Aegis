"""Shared transaction DTOs for the agent runtime."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict


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
