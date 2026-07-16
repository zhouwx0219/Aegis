"""Incremental paper state features independent of benchmark labels."""

from __future__ import annotations

import dataclasses
from typing import Iterable

from .context import TransactionContext


@dataclasses.dataclass(frozen=True)
class PhaseAwareState:
    phase: str
    inter_round_interval_ms: float
    read_set_size: int
    write_set_size: int
    read_set_growth: int
    write_set_growth: int
    access_overlap_ratio: float
    completed_rounds: int
    completed_operations: int
    recent_write_ratio: float
    hotspot_access_ratio: float
    blocked_time_ms: float
    retry_count: int
    current_action: int
    priority: int
    recent_conflict_kind: str = "none"
    global_active_transactions: int = 0
    global_waiter_count: int = 0
    global_abort_rate: float = 0.0
    global_throughput: float = 0.0
    global_avg_latency_ms: float = 0.0
    global_tail_latency_ms: float = 0.0
    global_agent_task_throughput: float = 0.0
    global_agent_task_avg_latency_ms: float = 0.0
    global_agent_task_tail_latency_ms: float = 0.0
    global_conflict_abort_rate: float = 0.0
    global_background_throughput: float = 0.0
    global_background_abort_rate: float = 0.0


class StateCollector:
    def __init__(self):
        self._last_sizes: dict[str, tuple[int, int]] = {}
        self._completed_round: dict[str, set[str]] = {}
        self._current_round: dict[str, set[str]] = {}
        self._round_writes: dict[str, int] = {}
        self._round_operations: dict[str, int] = {}
        self._last_interval_ms: dict[str, float] = {}
        self._last_overlap_ratio: dict[str, float] = {}
        self._last_write_ratio: dict[str, float] = {}

    def record_operation(
        self,
        context: TransactionContext,
        object_id: str,
        *,
        write: bool,
        interval_ms: float,
        hot: bool = False,
    ) -> None:
        tid = context.tid
        self._current_round.setdefault(tid, set()).add(str(object_id))
        self._round_operations[tid] = self._round_operations.get(tid, 0) + 1
        if write:
            self._round_writes[tid] = self._round_writes.get(tid, 0) + 1
            if hot:
                context.hot_write_targets.add(str(object_id))
        elif hot:
            context.hot_read_targets.add(str(object_id))
        self._last_interval_ms[tid] = max(0.0, float(interval_ms))

    def record_agent_interval(self, context: TransactionContext, interval_ms: float) -> None:
        self._last_interval_ms[context.tid] = max(0.0, float(interval_ms))

    def finish_round(self, context: TransactionContext) -> None:
        tid = context.tid
        previous = self._completed_round.get(tid, set())
        current = set(self._current_round.get(tid, set()))
        union = previous | current
        self._last_overlap_ratio[tid] = (
            len(previous & current) / len(union) if union else 0.0
        )
        operations = self._round_operations.get(tid, 0)
        writes = self._round_writes.get(tid, 0)
        self._last_write_ratio[tid] = writes / operations if operations else 0.0
        self._completed_round[tid] = current
        self._current_round[tid] = set()
        self._round_writes[tid] = 0
        self._round_operations[tid] = 0

    def snapshot(self, context: TransactionContext) -> PhaseAwareState:
        tid = context.tid
        read_size = len(context.read_versions)
        write_size = len(context.write_targets)
        previous_sizes = self._last_sizes.get(tid, (0, 0))
        self._last_sizes[tid] = (read_size, write_size)
        operations = self._round_operations.get(tid, 0)
        writes = self._round_writes.get(tid, 0)
        previous_round = self._completed_round.get(tid, set())
        current_round = self._current_round.get(tid, set())
        overlap_union = previous_round | current_round
        overlap = (
            len(previous_round & current_round) / len(overlap_union)
            if current_round and overlap_union
            else self._last_overlap_ratio.get(tid, 0.0)
        )
        all_targets = set(context.read_versions) | set(context.write_targets)
        hot_targets = context.hot_read_targets | context.hot_write_targets
        return PhaseAwareState(
            phase=context.phase.value,
            inter_round_interval_ms=self._last_interval_ms.get(tid, 0.0),
            read_set_size=read_size,
            write_set_size=write_size,
            read_set_growth=max(0, read_size - previous_sizes[0]),
            write_set_growth=max(0, write_size - previous_sizes[1]),
            access_overlap_ratio=overlap,
            completed_rounds=context.round_index,
            completed_operations=context.completed_operations,
            recent_write_ratio=(
                writes / operations
                if operations
                else self._last_write_ratio.get(tid, 0.0)
            ),
            hotspot_access_ratio=len(all_targets & hot_targets) / len(all_targets) if all_targets else 0.0,
            blocked_time_ms=context.blocked_time_ms,
            retry_count=context.retry_count,
            current_action=int(context.action.protected),
            priority=context.priority,
            recent_conflict_kind=context.recent_conflict_kind,
        )

    def discard(self, context: TransactionContext) -> None:
        for mapping in (
            self._last_sizes,
            self._completed_round,
            self._current_round,
            self._round_writes,
            self._round_operations,
            self._last_interval_ms,
            self._last_overlap_ratio,
            self._last_write_ratio,
        ):
            mapping.pop(context.tid, None)


def phase_aware_state_from_dict(data: dict[str, object]) -> PhaseAwareState:
    fields = {field.name for field in dataclasses.fields(PhaseAwareState)}
    return PhaseAwareState(
        **{key: value for key, value in dict(data).items() if key in fields}
    )
