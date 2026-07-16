"""Metrics aggregation for concurrent CC benchmarks."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Sequence

from agent.runtime import TransactionResult


@dataclasses.dataclass(frozen=True)
class BenchmarkAttempt:
    task_id: str
    attempt: int
    committed: bool
    reason: str
    elapsed_s: float
    lock_wait_s: float
    conflict_object_ids: tuple[str, ...]
    read_count: int
    write_count: int
    phase_count: int = 0
    reasoning_delay_ms: int = 0
    lock_hold_s: float = 0.0
    early_abort: bool = False
    skipped_reasoning_ms: int = 0
    atcc_action: str = ""

    @classmethod
    def from_result(cls, result: TransactionResult, *, attempt: int) -> "BenchmarkAttempt":
        return cls(
            task_id=result.task_id,
            attempt=int(attempt),
            committed=bool(result.committed),
            reason=str(result.reason),
            elapsed_s=float(result.elapsed_s),
            lock_wait_s=float(result.lock_wait_s),
            conflict_object_ids=tuple(str(value) for value in result.conflict_object_ids),
            read_count=int(result.read_count),
            write_count=int(result.write_count),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempt": self.attempt,
            "committed": self.committed,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "lock_wait_s": self.lock_wait_s,
            "conflict_object_ids": list(self.conflict_object_ids),
            "read_count": self.read_count,
            "write_count": self.write_count,
            "phase_count": self.phase_count,
            "reasoning_delay_ms": self.reasoning_delay_ms,
            "lock_hold_s": self.lock_hold_s,
            "early_abort": self.early_abort,
            "skipped_reasoning_ms": self.skipped_reasoning_ms,
            "atcc_action": self.atcc_action,
        }


@dataclasses.dataclass(frozen=True)
class BenchmarkMetrics:
    cc: str
    tasks: int
    attempts: int
    committed_tasks: int
    committed_attempts: int
    abort_count: int
    retry_count: int
    elapsed_s: float
    throughput: float
    task_commit_rate: float
    attempt_commit_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    avg_lock_wait_ms: float
    avg_lock_hold_ms: float
    avg_reasoning_delay_ms: float
    total_reasoning_delay_ms: int
    wasted_reasoning_ms: int
    wasted_elapsed_ms: float
    skipped_reasoning_ms: int
    early_abort_count: int
    avg_phase_count: float
    action_counts: Dict[str, int]
    conflict_objects: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["conflict_objects"] = list(self.conflict_objects)
        row["commit_rate"] = self.task_commit_rate
        return row


def aggregate_metrics(
    *,
    cc: str,
    task_count: int,
    attempts: Sequence[BenchmarkAttempt],
    elapsed_s: float,
) -> BenchmarkMetrics:
    committed_task_ids = {attempt.task_id for attempt in attempts if attempt.committed}
    committed_attempts = sum(1 for attempt in attempts if attempt.committed)
    latencies_ms = [attempt.elapsed_s * 1000.0 for attempt in attempts]
    lock_waits_ms = [attempt.lock_wait_s * 1000.0 for attempt in attempts]
    lock_holds_ms = [attempt.lock_hold_s * 1000.0 for attempt in attempts]
    reasoning_delays_ms = [attempt.reasoning_delay_ms for attempt in attempts]
    phase_counts = [attempt.phase_count for attempt in attempts]
    aborted_attempts = [attempt for attempt in attempts if not attempt.committed]
    abort_count = len(attempts) - committed_attempts
    retry_count = sum(max(0, attempt.attempt) for attempt in attempts)
    return BenchmarkMetrics(
        cc=str(cc),
        tasks=int(task_count),
        attempts=len(attempts),
        committed_tasks=len(committed_task_ids),
        committed_attempts=committed_attempts,
        abort_count=abort_count,
        retry_count=retry_count,
        elapsed_s=float(elapsed_s),
        throughput=len(committed_task_ids) / elapsed_s if elapsed_s > 0 else 0.0,
        task_commit_rate=len(committed_task_ids) / task_count if task_count else 0.0,
        attempt_commit_rate=committed_attempts / len(attempts) if attempts else 0.0,
        p50_latency_ms=percentile(latencies_ms, 50),
        p95_latency_ms=percentile(latencies_ms, 95),
        avg_latency_ms=average(latencies_ms),
        avg_lock_wait_ms=average(lock_waits_ms),
        avg_lock_hold_ms=average(lock_holds_ms),
        avg_reasoning_delay_ms=average(reasoning_delays_ms),
        total_reasoning_delay_ms=sum(reasoning_delays_ms),
        wasted_reasoning_ms=sum(
            max(0, attempt.reasoning_delay_ms - attempt.skipped_reasoning_ms)
            for attempt in aborted_attempts
        ),
        wasted_elapsed_ms=sum(attempt.elapsed_s * 1000.0 for attempt in aborted_attempts),
        skipped_reasoning_ms=sum(attempt.skipped_reasoning_ms for attempt in attempts),
        early_abort_count=sum(1 for attempt in attempts if attempt.early_abort),
        avg_phase_count=average(phase_counts),
        action_counts=action_counts(attempts),
        conflict_objects=tuple(
            sorted(
                {
                    object_id
                    for attempt in attempts
                    for object_id in attempt.conflict_object_ids
                }
            )
        ),
    )


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def average(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def action_counts(attempts: Sequence[BenchmarkAttempt]) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for attempt in attempts:
        action = str(attempt.atcc_action or "")
        if not action:
            continue
        rows[action] = rows.get(action, 0) + 1
    return rows
