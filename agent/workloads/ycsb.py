"""General agent-style workload derived from DBx1000 YCSB."""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable, Sequence, Tuple

from .base import (
    AgentCandidate,
    AgentOperation,
    AgentStage,
    AgentTask,
    AgentWorkload,
    ObjectSpec,
    WorkloadManifest,
)


@dataclasses.dataclass(frozen=True)
class YCSBConfig:
    record_count: int = 1000
    field_count: int = 10
    requests_per_task: int = 4
    candidates_per_task: int = 3
    read_weight: float = 0.5
    update_weight: float = 0.5
    zipf_theta: float = 0.6
    hotspot_fraction: float = 0.0
    hotspot_access_probability: float = 0.0
    initial_value: str = "0"

    def __post_init__(self) -> None:
        if min(
            self.record_count,
            self.field_count,
            self.requests_per_task,
            self.candidates_per_task,
        ) <= 0:
            raise ValueError("YCSB dimensions must be positive")
        if self.read_weight < 0 or self.update_weight < 0:
            raise ValueError("YCSB operation weights must be non-negative")
        if self.read_weight + self.update_weight <= 0:
            raise ValueError("YCSB needs at least one operation type")
        if self.zipf_theta < 0:
            raise ValueError("zipf_theta must be non-negative")
        if not 0.0 <= self.hotspot_fraction <= 1.0:
            raise ValueError("hotspot_fraction must be in [0, 1]")
        if not 0.0 <= self.hotspot_access_probability <= 1.0:
            raise ValueError("hotspot_access_probability must be in [0, 1]")
        if self.requests_per_task > self.record_count * self.field_count:
            raise ValueError(
                "requests_per_task cannot exceed the number of YCSB fields"
            )


class YCSBAgentWorkload(AgentWorkload):
    name = "agent-ycsb-semantic"
    workload_layer = "semantic"

    def __init__(self, config: YCSBConfig = YCSBConfig()):
        self.config = config
        self._record_weights = tuple(
            1.0 / ((record + 1) ** config.zipf_theta)
            for record in range(config.record_count)
        )

    @staticmethod
    def object_id(record: int, field: int) -> str:
        return f"ycsb:record:{record}:field:{field}"

    def manifest(self) -> WorkloadManifest:
        return WorkloadManifest(
            name=self.name,
            benchmark_family="YCSB",
            source_system="DBx1000",
            source_files=(
                "third_party/dbx1000/benchmarks/ycsb.h",
                "third_party/dbx1000/benchmarks/ycsb_wl.cpp",
                "third_party/dbx1000/benchmarks/ycsb_txn.cpp",
                "third_party/dbx1000/benchmarks/ycsb_query.cpp",
                "third_party/dbx1000/benchmarks/YCSB_schema.txt",
            ),
            preserved_semantics=(
                "record/field key space",
                "read-update request mix",
                "Zipfian record skew",
                "bounded request width without duplicate field targets per candidate",
            ),
            agent_adaptations=(
                "natural-language task envelope",
                "ranked K candidate plans",
                "typed read/overwrite operations over versioned KV objects",
                "deterministic generation without requiring an LLM",
            ),
            workload_layer=self.workload_layer,
            canonical_name=self.name,
            config=dataclasses.asdict(self.config),
        )

    def objects(self) -> Iterable[ObjectSpec]:
        for record in range(self.config.record_count):
            for field in range(self.config.field_count):
                yield ObjectSpec(
                    self.object_id(record, field),
                    self.config.initial_value,
                    kind="row",
                    metadata={"record": record, "field": field},
                )

    def _hot_record_count(self) -> int:
        if self.config.hotspot_fraction <= 0.0:
            return 0
        return max(1, int(self.config.record_count * self.config.hotspot_fraction))

    def _sample_record(self, rng: random.Random) -> int:
        hot_count = self._hot_record_count()
        if hot_count and self.config.hotspot_access_probability > 0.0:
            if rng.random() < self.config.hotspot_access_probability:
                return rng.randrange(hot_count)
            if hot_count < self.config.record_count:
                return rng.randrange(hot_count, self.config.record_count)
        return rng.choices(
            range(self.config.record_count), weights=self._record_weights, k=1
        )[0]

    def _sample_target(self, rng: random.Random) -> Tuple[int, int]:
        record = self._sample_record(rng)
        return record, rng.randrange(self.config.field_count)

    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        if count < 0:
            raise ValueError("task count must be non-negative")
        rng = random.Random(seed)
        tasks = []
        for task_index in range(count):
            candidates = []
            has_read = False
            has_write = False
            for candidate_index in range(self.config.candidates_per_task):
                targets = []
                while len(targets) < self.config.requests_per_task:
                    target = self._sample_target(rng)
                    if target not in targets:
                        targets.append(target)
                operations = []
                for operation_index, (record, field) in enumerate(targets):
                    object_id = self.object_id(record, field)
                    update = rng.random() < (
                        self.config.update_weight
                        / (self.config.read_weight + self.config.update_weight)
                    )
                    if update:
                        has_write = True
                        value = f"task-{task_index}:candidate-{candidate_index}:op-{operation_index}"
                        operations.append(AgentOperation.overwrite(object_id, value))
                    else:
                        has_read = True
                        operations.append(AgentOperation.read(object_id))
                candidates.append(
                    AgentCandidate(
                        candidate_id=f"ycsb-{task_index}-candidate-{candidate_index}",
                        quality=float(self.config.candidates_per_task - candidate_index),
                        operations=tuple(operations),
                        metadata={"source": "DBx1000/YCSB"},
                    )
                )
            stages = _agent_stages_for_candidates(candidates)
            tasks.append(
                AgentTask(
                    task_id=f"ycsb-{task_index}",
                    workload=self.name,
                    task_type="read-update",
                    request="Read or update a valid group of YCSB records.",
                    candidates=tuple(candidates),
                    context={
                        "zipf_theta": self.config.zipf_theta,
                        "hotspot_fraction": self.config.hotspot_fraction,
                        "hotspot_access_probability": (
                            self.config.hotspot_access_probability
                        ),
                        "hot_record_count": self._hot_record_count(),
                        "record_count": self.config.record_count,
                        "requests_per_task": self.config.requests_per_task,
                        "agent_phase_sequence": _agent_phase_sequence(
                            has_read=has_read,
                            has_write=has_write,
                            operation_count=self.config.requests_per_task,
                        ),
                        "agent_stages": [stage.to_dict() for stage in stages],
                    },
                )
            )
        return tasks


def _agent_stages_for_candidates(
    candidates: Sequence[AgentCandidate],
) -> Tuple[AgentStage, ...]:
    reads = [
        operation
        for candidate in candidates
        for operation in candidate.operations
        if operation.kind == "read"
    ]
    writes = [
        operation
        for candidate in candidates
        for operation in candidate.operations
        if operation.kind != "read"
    ]
    explore_count = max(1, int(len(reads) * 0.6)) if reads else 0
    explore = tuple(reads[:explore_count])
    refine = tuple(reads[explore_count:])
    commit = tuple(writes)
    return (
        AgentStage("explore", explore, delay_weight=1.0),
        AgentStage("refine", refine, delay_weight=1.0),
        AgentStage("commit", commit, delay_weight=1.0),
    )


def _agent_phase_sequence(
    *,
    has_read: bool,
    has_write: bool,
    operation_count: int,
) -> Tuple[str, ...]:
    if has_read and not has_write:
        return ("explore", "refine")
    if has_read and has_write:
        return ("explore", "refine", "commit")
    if operation_count >= 3:
        return ("refine", "commit")
    return ("commit",)


class YCSBFaithfulAgentWorkload(YCSBAgentWorkload):
    """Agent-side YCSB layer that preserves DBx1000-style single-plan requests.

    This layer is still executed by the agent transaction runtime over the
    versioned KV store. It intentionally removes K-candidate generation so it can
    be compared against DBx1000 native YCSB without adding semantic re-planning
    as another variable.
    """

    name = "agent-ycsb-faithful"
    workload_layer = "faithful"

    def __init__(self, config: YCSBConfig = YCSBConfig()):
        super().__init__(dataclasses.replace(config, candidates_per_task=1))

    def manifest(self) -> WorkloadManifest:
        manifest = super().manifest()
        return WorkloadManifest(
            name=self.name,
            benchmark_family=manifest.benchmark_family,
            source_system=manifest.source_system,
            source_files=manifest.source_files,
            preserved_semantics=manifest.preserved_semantics
            + ("single candidate per request for native comparability",),
            agent_adaptations=(
                "agent transaction envelope over DBx1000-derived YCSB keys",
                "typed read/overwrite operations over versioned KV objects",
                "deterministic generation without requiring an LLM",
            ),
            workload_layer=self.workload_layer,
            canonical_name=self.name,
            config=dataclasses.asdict(self.config),
        )
