"""Compiled phase-aware policy table for the paper ATCC runtime."""

from __future__ import annotations

import dataclasses
import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .context import LockAction, LockClass
from .state_collector import PhaseAwareState


@dataclasses.dataclass(frozen=True)
class CompiledPolicyEntry:
    phase: str
    action: int
    inter_round_bucket: int = 0
    read_set_bucket: int = 0
    write_set_bucket: int = 0
    read_growth_bucket: int = 0
    write_growth_bucket: int = 0
    overlap_bucket: int = 0
    completed_rounds_bucket: int = 0
    completed_operations_bucket: int = 0
    recent_write_bucket: int = 0
    hotspot_bucket: int = 0
    blocked_bucket: int = 0
    retry_bucket: int = 0
    current_action: int = 0
    priority_bucket: int = 0
    recent_conflict_kind: str = "none"
    active_bucket: int = 0
    waiter_bucket: int = 0
    abort_bucket: int = 0
    throughput_bucket: int = 0
    average_latency_bucket: int = 0
    tail_latency_bucket: int = 0
    agent_task_throughput_bucket: int = 0
    agent_task_average_latency_bucket: int = 0
    agent_task_tail_latency_bucket: int = 0
    conflict_abort_bucket: int = 0
    background_throughput_bucket: int = 0
    background_abort_bucket: int = 0
    support_weight: int = 1

    def distance(self, state: PhaseAwareState) -> float:
        phase_penalty = 1000.0 if self.phase != state.phase else 0.0
        return phase_penalty + sum(
            weight * abs(left - right)
            for left, right, weight in (
                (self.inter_round_bucket, bucket_log(state.inter_round_interval_ms), 1.0),
                (self.read_set_bucket, bucket_size(state.read_set_size), 1.0),
                (self.write_set_bucket, bucket_size(state.write_set_size), 1.0),
                (self.read_growth_bucket, bucket_size(state.read_set_growth), 1.0),
                (self.write_growth_bucket, bucket_size(state.write_set_growth), 1.0),
                (self.overlap_bucket, bucket_ratio(state.access_overlap_ratio), 1.0),
                (self.completed_rounds_bucket, bucket_size(state.completed_rounds), 1.0),
                (self.completed_operations_bucket, bucket_size(state.completed_operations), 1.0),
                (self.recent_write_bucket, bucket_ratio(state.recent_write_ratio), 1.0),
                (self.hotspot_bucket, bucket_ratio(state.hotspot_access_ratio), 2.0),
                (self.blocked_bucket, bucket_log(state.blocked_time_ms), 1.0),
                (self.retry_bucket, min(3, state.retry_count), 2.0),
                (self.current_action, int(state.current_action), 4.0),
                (
                    0 if self.recent_conflict_kind == normalize_conflict_kind(state.recent_conflict_kind) else 1,
                    0,
                    2.0,
                ),
                (self.active_bucket, bucket_size(state.global_active_transactions), 1.0),
                (self.waiter_bucket, bucket_size(state.global_waiter_count), 1.0),
                (self.abort_bucket, bucket_ratio(state.global_abort_rate), 2.0),
                (self.throughput_bucket, bucket_log(state.global_throughput), 0.25),
                (self.average_latency_bucket, bucket_log(state.global_avg_latency_ms), 0.25),
                (self.tail_latency_bucket, bucket_log(state.global_tail_latency_ms), 0.25),
                (
                    self.agent_task_throughput_bucket,
                    bucket_log(state.global_agent_task_throughput),
                    1.0,
                ),
                (
                    self.agent_task_average_latency_bucket,
                    bucket_log(state.global_agent_task_avg_latency_ms),
                    0.5,
                ),
                (
                    self.agent_task_tail_latency_bucket,
                    bucket_log(state.global_agent_task_tail_latency_ms),
                    1.0,
                ),
                (
                    self.conflict_abort_bucket,
                    bucket_ratio(state.global_conflict_abort_rate),
                    2.0,
                ),
                (
                    self.background_throughput_bucket,
                    bucket_log(state.global_background_throughput),
                    0.5,
                ),
                (
                    self.background_abort_bucket,
                    bucket_ratio(state.global_background_abort_rate),
                    2.0,
                ),
            )
        )


class CompiledPhasePolicy:
    def __init__(
        self,
        entries: Iterable[CompiledPolicyEntry] = (),
        *,
        generation: int = 0,
        medoids_per_group: int = 1,
        refinement_actor: dict[str, object] | None = None,
        occ_cold_start_guard: bool = False,
    ):
        self.entries = tuple(entries)
        self.generation = int(generation)
        self.medoids_per_group = max(1, int(medoids_per_group))
        self.refinement_actor = dict(refinement_actor or {})
        self.occ_cold_start_guard = bool(occ_cold_start_guard)
        exact = {}
        indexed = defaultdict(list)
        by_phase = defaultdict(list)
        for entry in self.entries:
            key = _entry_key(entry)
            incumbent = exact.get(key)
            if incumbent is None or entry.support_weight > incumbent.support_weight:
                exact[key] = entry
            indexed[(entry.phase, int(entry.current_action))].append(entry)
            by_phase[entry.phase].append(entry)
        self._exact = exact
        self._indexed = {key: tuple(values) for key, values in indexed.items()}
        self._by_phase = {key: tuple(values) for key, values in by_phase.items()}
        self._fallback_cache: dict[tuple[object, ...], CompiledPolicyEntry] = {}

    def select(self, state: PhaseAwareState) -> LockAction:
        if self.occ_cold_start_guard and should_keep_occ(state):
            return LockAction()
        if not self.entries:
            return self._select_refinement(state) or LockAction()
        key = _state_key(state)
        entry = self._exact.get(key)
        if entry is None:
            entry = self._fallback_cache.get(key)
            if entry is None:
                candidates = self._indexed.get((state.phase, int(state.current_action)))
                if not candidates:
                    candidates = tuple(
                        candidate
                        for candidate in self._by_phase.get(state.phase, ())
                        if (int(candidate.action) | int(state.current_action))
                        == int(candidate.action)
                    )
                if not candidates:
                    return self._select_refinement(state) or LockAction(
                        LockClass(int(state.current_action))
                    )
                entry = min(
                    candidates,
                    key=lambda candidate: (
                        candidate.distance(state),
                        -max(1, int(candidate.support_weight)),
                        int(candidate.action),
                    ),
                )
                self._fallback_cache.setdefault(key, entry)
        threshold = float(self.refinement_actor.get("distance_threshold", -1.0) or 0.0)
        if self.refinement_actor and entry.distance(state) > threshold:
            refined = self._select_refinement(state)
            if refined is not None:
                return refined
        return LockAction(LockClass(int(entry.action)))

    def _select_refinement(self, state: PhaseAwareState) -> LockAction | None:
        if not self.refinement_actor:
            return None
        from agent.cc.atcc.ppo import dot, policy_group_key, state_features, valid_actions

        actor_weights = self.refinement_actor.get("actor_weights", ())
        if not actor_weights:
            return None
        group_logits = dict(self.refinement_actor.get("group_logits", {}) or {}).get(
            policy_group_key(state),
            (),
        )
        features = state_features(state)
        if self.refinement_actor.get("type") == "factorized-bernoulli-lock-bits":
            action = int(state.current_action) & 0xF
            for bit_index, bit in enumerate((1, 2, 4, 8)):
                if action & bit:
                    continue
                logit = dot(actor_weights[bit_index], features) + (
                    float(group_logits[bit_index])
                    if len(group_logits) > bit_index else 0.0
                )
                if logit > 0.0:
                    action |= bit
            return LockAction(LockClass(action))
        action = max(
            valid_actions(state.current_action),
            key=lambda candidate: (
                dot(actor_weights[candidate], features)
                + (float(group_logits[candidate]) if len(group_logits) > candidate else 0.0),
                -candidate,
            ),
        )
        return LockAction(LockClass(int(action)))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "cast-das-paper-atcc-policy",
            "version": 4,
            "generation": self.generation,
            "action_space": "lock_protection_mask_4bit",
            "action_dimensions": ["hot_read", "cold_read", "hot_write", "cold_write"],
            "priority_control": "transaction_manager_formula",
            "priority_is_policy_action": False,
            "state_source": "observed_execution_only",
            "future_access_plan_features": False,
            "compaction": "weighted_k_medoids_per_phase_current_action_action",
            "medoids_per_group": self.medoids_per_group,
            "selective_refinement": bool(self.refinement_actor),
            "refinement_actor": self.refinement_actor,
            "occ_cold_start_guard": self.occ_cold_start_guard,
            "entries": [dataclasses.asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CompiledPhasePolicy":
        fields = {field.name for field in dataclasses.fields(CompiledPolicyEntry)}
        return cls(
            (
                CompiledPolicyEntry(
                    **{key: value for key, value in dict(row).items() if key in fields}
                )
                for row in data.get("entries", [])
            ),
            generation=int(data.get("generation", 0) or 0),
            medoids_per_group=int(data.get("medoids_per_group", 1) or 1),
            refinement_actor=dict(data.get("refinement_actor", {}) or {}),
            occ_cold_start_guard=bool(data.get("occ_cold_start_guard", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompiledPhasePolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class StaticThresholdPhasePolicy:
    """Naive one-signal threshold baseline for the switching ablation.

    Unlike the learned policy, this baseline does not inspect transaction
    shape, reasoning cost, retry state, hotspot ratio, or queue state.  A
    fixed global conflict-abort threshold switches observed hot writes from
    OCC to pessimistic protection.  The action remains monotonic.
    """

    CONFLICT_ABORT_THRESHOLD = 0.20

    def __init__(
        self,
        conflict_abort_threshold: float | None = None,
        protection_mask: int = int(LockClass.HOT_WRITE),
    ):
        threshold = (
            self.CONFLICT_ABORT_THRESHOLD
            if conflict_abort_threshold is None
            else float(conflict_abort_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("static conflict threshold must be in [0, 1]")
        self.conflict_abort_threshold = threshold
        self.protection_mask = LockClass(int(protection_mask) & 0xF)

    def select(self, state: PhaseAwareState) -> LockAction:
        current = LockClass(int(state.current_action) & 0xF)
        selected = current
        if (
            str(state.phase) in {"refine", "commit"}
            and float(state.global_conflict_abort_rate)
            >= self.conflict_abort_threshold
        ):
            selected |= self.protection_mask
        return LockAction(selected)


class AtomicPolicyManager:
    def __init__(self, policy: object | None = None):
        self._lock = threading.RLock()
        self._policy = policy or CompiledPhasePolicy()

    def snapshot(self) -> object:
        with self._lock:
            return self._policy

    def install(self, policy: CompiledPhasePolicy) -> None:
        with self._lock:
            if policy.generation <= self._policy.generation:
                raise ValueError("policy generation must increase")
            self._policy = policy


def bucket_size(value: int) -> int:
    value = max(0, int(value))
    if value == 0:
        return 0
    if value <= 2:
        return 1
    if value <= 8:
        return 2
    return 3


def bucket_ratio(value: float) -> int:
    value = max(0.0, min(1.0, float(value)))
    if value == 0:
        return 0
    if value < 0.25:
        return 1
    if value < 0.75:
        return 2
    return 3


def bucket_log(value: float) -> int:
    value = max(0.0, float(value))
    if value <= 0.0:
        return 0
    bucket = 1
    boundary = 1.0
    while value > boundary and bucket < 15:
        boundary *= 2.0
        bucket += 1
    return bucket


def _entry_key(entry: CompiledPolicyEntry) -> tuple[object, ...]:
    return (
        entry.phase,
        entry.inter_round_bucket,
        entry.read_set_bucket,
        entry.write_set_bucket,
        entry.read_growth_bucket,
        entry.write_growth_bucket,
        entry.overlap_bucket,
        entry.completed_rounds_bucket,
        entry.completed_operations_bucket,
        entry.recent_write_bucket,
        entry.hotspot_bucket,
        entry.blocked_bucket,
        entry.retry_bucket,
        entry.current_action,
        entry.recent_conflict_kind,
        entry.active_bucket,
        entry.waiter_bucket,
        entry.abort_bucket,
        entry.throughput_bucket,
        entry.average_latency_bucket,
        entry.tail_latency_bucket,
        entry.agent_task_throughput_bucket,
        entry.agent_task_average_latency_bucket,
        entry.agent_task_tail_latency_bucket,
        entry.conflict_abort_bucket,
        entry.background_throughput_bucket,
        entry.background_abort_bucket,
    )


def _state_key(state: PhaseAwareState) -> tuple[object, ...]:
    return (
        state.phase,
        bucket_log(state.inter_round_interval_ms),
        bucket_size(state.read_set_size),
        bucket_size(state.write_set_size),
        bucket_size(state.read_set_growth),
        bucket_size(state.write_set_growth),
        bucket_ratio(state.access_overlap_ratio),
        bucket_size(state.completed_rounds),
        bucket_size(state.completed_operations),
        bucket_ratio(state.recent_write_ratio),
        bucket_ratio(state.hotspot_access_ratio),
        bucket_log(state.blocked_time_ms),
        min(3, state.retry_count),
        int(state.current_action),
        normalize_conflict_kind(state.recent_conflict_kind),
        bucket_size(state.global_active_transactions),
        bucket_size(state.global_waiter_count),
        bucket_ratio(state.global_abort_rate),
        bucket_log(state.global_throughput),
        bucket_log(state.global_avg_latency_ms),
        bucket_log(state.global_tail_latency_ms),
        bucket_log(state.global_agent_task_throughput),
        bucket_log(state.global_agent_task_avg_latency_ms),
        bucket_log(state.global_agent_task_tail_latency_ms),
        bucket_ratio(state.global_conflict_abort_rate),
        bucket_log(state.global_background_throughput),
        bucket_ratio(state.global_background_abort_rate),
    )


def should_keep_occ(state: PhaseAwareState) -> bool:
    """Avoid speculative locking until the runtime has observed conflict pressure."""
    low_conflict_rate = 0.05
    return (
        int(state.current_action) == 0
        and int(state.retry_count) == 0
        and normalize_conflict_kind(state.recent_conflict_kind) == "none"
        and int(state.global_waiter_count) == 0
        and float(state.global_abort_rate) <= low_conflict_rate
        and float(state.global_conflict_abort_rate) <= low_conflict_rate
        and float(state.global_background_abort_rate) <= low_conflict_rate
    )


def normalize_conflict_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized else "none"
