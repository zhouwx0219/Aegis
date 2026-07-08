"""Reward calculation for dynamic ATCC policy learning."""

from __future__ import annotations

import dataclasses
from typing import Dict, Mapping


@dataclasses.dataclass(frozen=True)
class ATCCRewardConfig:
    commit_value: float = 100.0
    abort_penalty: float = 80.0
    reasoning_weight: float = 1.0
    lock_wait_weight: float = 0.5
    latency_weight: float = 0.1
    lock_hold_weight: float = 0.05
    background_abort_weight: float = 2.0
    background_tps_loss_weight: float = 0.1

    def reward(
        self,
        *,
        committed: bool,
        elapsed_ms: float,
        lock_wait_ms: float,
        wasted_reasoning_ms: float,
        lock_hold_ms: float = 0.0,
        background_aborts: float = 0.0,
        background_tps_loss: float = 0.0,
    ) -> float:
        value = self.commit_value if committed else -self.abort_penalty
        return (
            float(value)
            - float(wasted_reasoning_ms) * self.reasoning_weight
            - float(lock_wait_ms) * self.lock_wait_weight
            - float(elapsed_ms) * self.latency_weight
            - float(lock_hold_ms) * self.lock_hold_weight
            - float(background_aborts) * self.background_abort_weight
            - float(background_tps_loss) * self.background_tps_loss_weight
        )

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "ATCCRewardConfig":
        row = dict(data or {})
        return cls(
            commit_value=float(row.get("commit_value", 100.0) or 100.0),
            abort_penalty=float(row.get("abort_penalty", 80.0) or 80.0),
            reasoning_weight=float(row.get("reasoning_weight", 1.0) or 1.0),
            lock_wait_weight=float(row.get("lock_wait_weight", 0.5) or 0.5),
            latency_weight=float(row.get("latency_weight", 0.1) or 0.1),
            lock_hold_weight=float(row.get("lock_hold_weight", 0.05) or 0.05),
            background_abort_weight=float(row.get("background_abort_weight", 2.0) or 2.0),
            background_tps_loss_weight=float(row.get("background_tps_loss_weight", 0.1) or 0.1),
        )
