"""YCSB-style single-plan workload."""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable, Sequence

from .base import AgentOperation, AgentTask, AgentWorkload, ObjectSpec


@dataclasses.dataclass(frozen=True)
class YCSBConfig:
    level: str = "low"
    logical_record_count: int = 0
    record_count: int = 128
    field_count: int = 10
    operations_per_task: int = 4
    read_weight: float = 0.9
    update_weight: float = 0.1
    zipf_theta: float = 0.0
    hotspot_fraction: float = 0.0
    hotspot_access_probability: float = 0.0
    access_distribution: str = "hotspot"
    initial_value: str = "0"

    def __post_init__(self) -> None:
        if min(self.record_count, self.field_count, self.operations_per_task) <= 0:
            raise ValueError("YCSB dimensions must be positive")
        if self.logical_record_count < 0:
            raise ValueError("YCSB logical record count must be non-negative")
        if self.read_weight < 0 or self.update_weight < 0:
            raise ValueError("YCSB weights must be non-negative")
        if self.read_weight + self.update_weight <= 0:
            raise ValueError("YCSB needs at least one operation type")
        if self.zipf_theta < 0:
            raise ValueError("YCSB Zipfian theta must be non-negative")
        if self.access_distribution not in {"hotspot", "zipfian"}:
            raise ValueError(f"unsupported YCSB access distribution: {self.access_distribution}")


def ycsb_config(
    level: str,
    profile: str = "small",
    *,
    zipf_theta: float | None = None,
) -> YCSBConfig:
    level = str(level).strip().lower()
    profile = str(profile).strip().lower()
    if profile == "small":
        configs = small_ycsb_configs()
    elif profile == "paper":
        configs = paper_ycsb_configs()
    else:
        raise ValueError(f"unsupported YCSB profile: {profile}")
    if level not in configs:
        raise ValueError(f"unsupported YCSB level: {level}")
    config = configs[level]
    if zipf_theta is not None:
        config = dataclasses.replace(
            config,
            zipf_theta=float(zipf_theta),
            access_distribution="zipfian",
        )
    return config


def small_ycsb_configs() -> dict[str, YCSBConfig]:
    return {
        "low": YCSBConfig(
            level="low",
            record_count=256,
            field_count=10,
            operations_per_task=4,
            read_weight=0.95,
            update_weight=0.05,
        ),
        "medium": YCSBConfig(
            level="medium",
            record_count=96,
            field_count=10,
            operations_per_task=6,
            read_weight=0.8,
            update_weight=0.2,
            zipf_theta=0.7,
            hotspot_fraction=0.1,
            hotspot_access_probability=0.5,
        ),
        "high": YCSBConfig(
            level="high",
            record_count=48,
            field_count=10,
            operations_per_task=8,
            read_weight=0.5,
            update_weight=0.5,
            zipf_theta=0.99,
            hotspot_fraction=0.1,
            hotspot_access_probability=0.8,
        ),
    }


def paper_ycsb_configs() -> dict[str, YCSBConfig]:
    return {
        "low": YCSBConfig(
            level="low",
            logical_record_count=1_000_000,
            record_count=512,
            field_count=10,
            operations_per_task=10,
            read_weight=0.95,
            update_weight=0.05,
            zipf_theta=0.0,
            hotspot_fraction=0.0,
            hotspot_access_probability=0.0,
        ),
        "medium": YCSBConfig(
            level="medium",
            logical_record_count=1_000_000,
            record_count=128,
            field_count=10,
            operations_per_task=10,
            read_weight=0.90,
            update_weight=0.10,
            zipf_theta=0.7,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.50,
        ),
        "high": YCSBConfig(
            level="high",
            logical_record_count=1_000_000,
            record_count=64,
            field_count=10,
            operations_per_task=10,
            read_weight=0.50,
            update_weight=0.50,
            zipf_theta=0.99,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.75,
        ),
    }


class YCSBWorkload(AgentWorkload):
    name = "ycsb"
    family = "ycsb"

    def __init__(self, config: YCSBConfig):
        self.config = config
        self.level = config.level
        self._record_weights = tuple(
            1.0 / ((record + 1) ** config.zipf_theta)
            for record in range(config.record_count)
        )

    @staticmethod
    def object_id(record: int, field: int) -> str:
        return f"ycsb:record:{record}:field:{field}"

    def objects(self) -> Iterable[ObjectSpec]:
        for record in range(self.config.record_count):
            for field in range(self.config.field_count):
                yield ObjectSpec(self.object_id(record, field), self.config.initial_value, "row")

    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        if count < 0:
            raise ValueError("task count must be non-negative")
        rng = random.Random(seed)
        tasks = []
        hot_record_count = self._hot_record_count()
        for task_index in range(count):
            targets = []
            while len(targets) < self.config.operations_per_task:
                target = self._sample_target(rng)
                if target not in targets:
                    targets.append(target)
            operations = []
            for op_index, (record, field) in enumerate(targets):
                object_id = self.object_id(record, field)
                update = rng.random() < (
                    self.config.update_weight
                    / (self.config.read_weight + self.config.update_weight)
                )
                if update:
                    operations.append(
                        AgentOperation.write(
                            object_id,
                            f"task-{task_index}:op-{op_index}",
                        )
                    )
                else:
                    operations.append(AgentOperation.read(object_id))
            tasks.append(
                AgentTask(
                    task_id=f"ycsb-{task_index}",
                    workload=self.name,
                    task_type="read-update",
                    operations=tuple(operations),
                    context={
                        "level": self.level,
                        "hot_record_count": hot_record_count,
                        "record_count": self.config.record_count,
                        "logical_record_count": self.config.logical_record_count or self.config.record_count,
                        "read_weight": self.config.read_weight,
                        "update_weight": self.config.update_weight,
                        "zipf_theta": self.config.zipf_theta,
                        "hotspot_fraction": self.config.hotspot_fraction,
                        "hotspot_access_probability": self.config.hotspot_access_probability,
                        "access_distribution": self.config.access_distribution,
                        "operations_per_task": self.config.operations_per_task,
                        "agent_cost_class": agent_cost_class(self.level, task_index),
                        "phase_shape": "tool_heavy" if self.level == "high" and task_index % 3 == 0 else "multi_stage",
                        "side_effect_cost_ms": side_effect_cost_ms(self.level, task_index),
                    },
                )
            )
        return tasks

    def _hot_record_count(self) -> int:
        if self.config.hotspot_fraction <= 0.0:
            return 0
        return max(1, int(self.config.record_count * self.config.hotspot_fraction))

    def _sample_target(self, rng: random.Random) -> tuple[int, int]:
        if self.config.access_distribution == "zipfian":
            record = rng.choices(
                range(self.config.record_count),
                weights=self._record_weights,
                k=1,
            )[0]
            return record, rng.randrange(self.config.field_count)
        hot_count = self._hot_record_count()
        if hot_count and rng.random() < self.config.hotspot_access_probability:
            record = rng.randrange(hot_count)
        elif hot_count and hot_count < self.config.record_count:
            record = rng.randrange(hot_count, self.config.record_count)
        else:
            record = rng.choices(
                range(self.config.record_count),
                weights=self._record_weights,
                k=1,
            )[0]
        return record, rng.randrange(self.config.field_count)


def agent_cost_class(level: str, task_index: int) -> str:
    level = str(level).strip().lower()
    if level == "high" and task_index % 3 == 0:
        return "expensive"
    if level in {"medium", "high"} and task_index % 2 == 0:
        return "normal"
    return "cheap"


def side_effect_cost_ms(level: str, task_index: int) -> int:
    level = str(level).strip().lower()
    if level == "high" and task_index % 3 == 0:
        return 40
    if level == "medium" and task_index % 4 == 0:
        return 15
    return 0
