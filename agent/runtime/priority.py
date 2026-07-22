"""Abort-cost-aware dynamic priority from the Aegis paper."""

from __future__ import annotations

import dataclasses
import math

from .context import TransactionContext


@dataclasses.dataclass(frozen=True)
class PriorityConfig:
    # Kept under its original public name for CLI compatibility.  It is the
    # paper's Delta_o (completed-operation quantum), not a wall-clock unit.
    sql_weight: float = 1.0
    blocked_weight: float = 1.0
    retry_weight: float = 1.0
    interval_weight: float = 1.0
    sql_quantum_ms: float = 10.0
    blocked_quantum_ms: float = 100.0
    interval_quantum_ms: float = 10.0
    max_priority: int = 1_000_000


class PriorityManager:
    def __init__(self, config: PriorityConfig | None = None):
        self.config = config or PriorityConfig()

    def compute(self, context: TransactionContext) -> int:
        cfg = self.config
        score = (
            math.floor(
                cfg.sql_weight
                * context.completed_operations
                / max(cfg.sql_quantum_ms, 1e-9)
            )
            + math.floor(cfg.blocked_weight * context.blocked_time_ms / max(cfg.blocked_quantum_ms, 1e-9))
            + math.floor(cfg.retry_weight * context.retry_count)
            + math.floor(
                cfg.interval_weight
                * (context.agent_cost_ms + context.prior_retry_cost_ms)
                / max(cfg.interval_quantum_ms, 1e-9)
            )
        )
        return min(cfg.max_priority, max(0, int(score)))

    def refresh(self, context: TransactionContext, lock_manager: object | None = None) -> int:
        priority = self.compute(context)
        if lock_manager is not None:
            lock_manager.update_priority(context, priority)
        else:
            context.priority = priority
            context.priority_epoch += 1
        return priority
