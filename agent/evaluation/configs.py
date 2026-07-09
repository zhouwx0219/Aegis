"""Evaluation configuration objects shared by experiment runners."""

from __future__ import annotations

import dataclasses
from typing import Tuple


DEFAULT_TEST_SEEDS: Tuple[int, ...] = (920104, 920105, 920106, 920107, 920108)
DEFAULT_TRAIN_SEEDS: Tuple[int, ...] = (910104, 910105, 910106, 910107, 910108)
DEFAULT_PROFILES: Tuple[str, ...] = ("low", "medium", "high")
DEFAULT_WORKLOADS: Tuple[str, ...] = ("ycsb", "tpcc")


@dataclasses.dataclass(frozen=True)
class RetryExperimentConfig:
    task_count: int = 60
    repeats: int = 1
    workers: int = 24
    agent_slots: int = 4
    planning_delay_s: float = 0.050
    latency_distribution: str = "lognormal"
    latency_cv: float = 0.8
    latency_max_s: float = 0.500
    max_attempts: int = 8
    background_workers: int = 4
    background_interval_s: float = 0.002
    background_strategy: str = "occ"
    object_lock_scheduler: str = "bounded-priority"
    object_lock_priority_burst: int = 2
    prelock_wait_budget_s: float = 0.070
    prelock_wait_budget_mode: str = "object"
    agent_execution_mode: str = "staged"
    snapshot_timing: str = "before-planning"


@dataclasses.dataclass(frozen=True)
class AblationTrainingConfig:
    train_seeds: Tuple[int, ...] = DEFAULT_TRAIN_SEEDS
    train_rounds: int = 4
    train_task_count: int = 60
    train_policy_epsilon: float = 0.05
    freeze_dynamic_policy: bool = True
    priority_cap: int = 1
