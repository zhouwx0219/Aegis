"""Benchmark configuration DTOs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    workload: str = "ycsb"
    level: str = "low"
    workload_profile: str = "small"
    cc: str = "all"
    tasks: int = 10
    workers: int = 8
    retries: int = 0
    seed: int = 920104
    reasoning_profile: str = "agentic"
    reasoning_scale: float = 1.0
    policy_mode: str = "online"
    policy: Optional[Path] = None
    atcc_policy: Optional[Any] = dataclasses.field(default=None, compare=False, repr=False)

    def normalized(self) -> "BenchmarkConfig":
        if self.tasks < 0:
            raise ValueError("task count must be non-negative")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.reasoning_scale < 0:
            raise ValueError("reasoning scale must be non-negative")
        mode = str(self.policy_mode).strip().lower()
        if mode not in {"train", "eval", "online"}:
            raise ValueError(f"unsupported policy mode: {self.policy_mode}")
        return dataclasses.replace(
            self,
            workload=str(self.workload).strip().lower(),
            level=str(self.level).strip().lower(),
            workload_profile=str(self.workload_profile).strip().lower() or "small",
            cc=str(self.cc).strip() or "all",
            reasoning_profile=str(self.reasoning_profile).strip().lower() or "agentic",
            policy_mode=mode,
            workers=min(int(self.workers), max(1, int(self.tasks) or 1)),
        )
