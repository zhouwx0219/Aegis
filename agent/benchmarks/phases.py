"""Phase planning and deterministic agent reasoning delay simulation."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Iterable, Sequence, Tuple

from agent.workloads import AgentOperation, AgentTask


@dataclasses.dataclass(frozen=True)
class ReasoningProfile:
    name: str = "agentic"
    scale: float = 1.0

    def delay_ms(self, *, level: str, phase: str, task_id: str, attempt: int) -> int:
        if self.name in {"none", "off", "disabled"}:
            return 0
        reasoning_level = "high" if self.name in {"agentic", "heavy", "long"} else level
        low, high = delay_range_ms(level=reasoning_level, phase=phase)
        if self.name in {"light", "short"}:
            low, high = low // 2, high // 2
        elif self.name in {"heavy", "long"}:
            low, high = low * 2, high * 2
        low = int(max(0, round(low * self.scale)))
        high = int(max(low, round(high * self.scale)))
        return deterministic_int(
            low,
            high,
            "|".join((self.name, level, phase, task_id, str(attempt))),
        )

    def retry_delay_ms(self, *, level: str, task_id: str, attempt: int) -> int:
        if self.name in {"none", "off", "disabled"} or attempt <= 0:
            return 0
        if self.name in {"agentic", "heavy", "long"}:
            low, high = (500, 5000)
        else:
            low, high = retry_delay_range_ms(level)
        low = int(max(0, round(low * self.scale)))
        high = int(max(low, round(high * self.scale)))
        return deterministic_int(
            low,
            high,
            "|".join((self.name, level, "retry", task_id, str(attempt))),
        )

    def operation_delay_ms(
        self,
        *,
        level: str,
        phase: str,
        task_id: str,
        attempt: int,
        operation_index: int,
    ) -> int:
        if self.name in {"none", "off", "disabled"}:
            return 0
        low, high = (1, 20)
        if self.name in {"light", "short"}:
            low, high = (1, 10)
        elif self.name in {"heavy", "long"}:
            low, high = (2, 40)
        low = int(max(0, round(low * self.scale)))
        high = int(max(low, round(high * self.scale)))
        return deterministic_int(
            low,
            high,
            "|".join(
                (
                    self.name,
                    str(level),
                    str(phase),
                    str(task_id),
                    str(attempt),
                    str(operation_index),
                )
            ),
        )


@dataclasses.dataclass(frozen=True)
class PlannedPhase:
    name: str
    operations: Tuple[AgentOperation, ...]
    reasoning_delay_ms: int = 0
    operation_delays_ms: Tuple[int, ...] = ()

    @property
    def total_reasoning_delay_ms(self) -> int:
        return int(self.reasoning_delay_ms) + sum(int(value) for value in self.operation_delays_ms)


@dataclasses.dataclass(frozen=True)
class PlannedTask:
    task: AgentTask
    phases: Tuple[PlannedPhase, ...]
    retry_delay_ms: int = 0

    @property
    def total_reasoning_delay_ms(self) -> int:
        return int(self.retry_delay_ms) + sum(
            phase.total_reasoning_delay_ms for phase in self.phases
        )

    @property
    def phase_count(self) -> int:
        return len(self.phases)


def plan_task_phases(
    task: AgentTask,
    *,
    attempt: int,
    profile: ReasoningProfile,
) -> PlannedTask:
    context = dict(task.context)
    level = str(context.get("level", "low"))
    operations_by_phase = phase_operations(task)
    operation_indexes = {id(operation): index for index, operation in enumerate(task.operations)}
    paper_timing = str(context.get("profile", "small")).strip().lower() == "paper"
    if paper_timing:
        phases = tuple(
            PlannedPhase(
                phase,
                operations,
                0,
                tuple(
                    profile.operation_delay_ms(
                        level=level,
                        phase=phase,
                        task_id=task.task_id,
                        attempt=attempt,
                        operation_index=operation_indexes[id(operation)],
                    )
                    for operation in operations
                ),
            )
            for phase, operations in operations_by_phase
        )
    else:
        phases = tuple(
            PlannedPhase(
                phase,
                operations,
                agent_delay_ms(
                    profile,
                    context,
                    level=level,
                    phase=phase,
                    task_id=task.task_id,
                    attempt=attempt,
                )
                + (side_effect_delay_ms(profile, context) if phase == "commit" else 0),
            )
            for phase, operations in operations_by_phase
        )
    return PlannedTask(
        task=task,
        phases=tuple(
            phase
            for phase in phases
            if phase.operations or phase.total_reasoning_delay_ms > 0
        ),
        retry_delay_ms=profile.retry_delay_ms(level=level, task_id=task.task_id, attempt=attempt),
    )


def phase_operations(task: AgentTask) -> tuple[tuple[str, Tuple[AgentOperation, ...]], ...]:
    tagged = {
        phase: tuple(
            operation
            for operation in task.operations
            if str(dict(operation.metadata).get("phase", "")).strip().lower() == phase
        )
        for phase in ("explore", "refine", "commit")
    }
    if any(tagged.values()):
        if sum(len(values) for values in tagged.values()) != len(task.operations):
            raise ValueError("phase-tagged tasks must tag every operation")
        return tuple((phase, tagged[phase]) for phase in ("explore", "refine", "commit"))

    reads = tuple(operation for operation in task.operations if operation.kind == "read")
    writes = tuple(operation for operation in task.operations if operation.kind == "write")
    explore_reads, refine_reads = split_reads(reads)
    return (
        ("explore", explore_reads),
        ("refine", refine_reads),
        ("commit", writes),
    )


def sleep_for_reasoning(delay_ms: int) -> None:
    if int(delay_ms) > 0:
        time.sleep(int(delay_ms) / 1000.0)


def split_reads(reads: Sequence[AgentOperation]) -> tuple[Tuple[AgentOperation, ...], Tuple[AgentOperation, ...]]:
    if not reads:
        return (), ()
    pivot = max(1, (len(reads) + 1) // 2)
    return tuple(reads[:pivot]), tuple(reads[pivot:])


def delay_range_ms(*, level: str, phase: str) -> tuple[int, int]:
    level = str(level).strip().lower()
    phase = str(phase).strip().lower()
    ranges = {
        "low": {
            "explore": (1, 3),
            "refine": (1, 3),
            "commit": (0, 1),
        },
        "medium": {
            "explore": (8, 16),
            "refine": (8, 16),
            "commit": (4, 8),
        },
        "high": {
            "explore": (25, 50),
            "refine": (25, 50),
            "commit": (10, 25),
        },
    }
    return ranges.get(level, ranges["low"]).get(phase, (0, 0))


def retry_delay_range_ms(level: str) -> tuple[int, int]:
    ranges = {
        "low": (2, 5),
        "medium": (20, 40),
        "high": (60, 120),
    }
    return ranges.get(str(level).strip().lower(), ranges["low"])


def deterministic_int(low: int, high: int, key: str) -> int:
    if high <= low:
        return int(low)
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    span = high - low + 1
    return int(low + (int.from_bytes(digest[:8], "big") % span))


def agent_delay_ms(
    profile: ReasoningProfile,
    context: dict,
    *,
    level: str,
    phase: str,
    task_id: str,
    attempt: int,
) -> int:
    delay = profile.delay_ms(level=level, phase=phase, task_id=task_id, attempt=attempt)
    cost_class = str(context.get("agent_cost_class", "normal"))
    phase_shape = str(context.get("phase_shape", "multi_stage"))
    multiplier = {
        "cheap": 0.6,
        "normal": 1.0,
        "expensive": 1.8,
    }.get(cost_class, 1.0)
    if phase_shape == "tool_heavy" and phase in {"refine", "commit"}:
        multiplier *= 1.5
    elif phase_shape == "short":
        multiplier *= 0.7
    return int(round(delay * multiplier))


def side_effect_delay_ms(profile: ReasoningProfile, context: dict) -> int:
    if profile.name in {"none", "off", "disabled"}:
        return 0
    return int(context.get("side_effect_cost_ms", 0) or 0)


def total_phase_delay_ms(plans: Iterable[PlannedTask]) -> int:
    return sum(plan.total_reasoning_delay_ms for plan in plans)
