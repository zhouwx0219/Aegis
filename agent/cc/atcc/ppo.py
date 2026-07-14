"""Dependency-free discrete PPO for paper ATCC trajectories."""

from __future__ import annotations

import dataclasses
import math
import random
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from agent.runtime.context import LockClass
from agent.runtime.paper_policy import (
    CompiledPhasePolicy,
    CompiledPolicyEntry,
    bucket_log,
    bucket_ratio,
    bucket_size,
)
from agent.runtime.state_collector import PhaseAwareState
from agent.runtime.trajectory import PolicyTransition


ALL_ACTIONS = tuple(range(16))
LOCK_BITS = (1, 2, 4, 8)


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.003
    group_learning_rate: float = 0.03
    critic_learning_rate: float = 0.005
    clip_ratio: float = 0.20
    discount: float = 0.99
    entropy_weight: float = 0.01
    epochs: int = 8
    min_group_samples: int = 16
    min_group_actions: int = 2


class DiscretePPOPolicy:
    def __init__(self, *, seed: int = 0, stay_probability: float = 0.0):
        self.prototypes: Dict[str, PhaseAwareState] = {}
        self.prototype_counts: Dict[str, int] = defaultdict(int)
        self.actor_weights = [[0.0] * FEATURE_COUNT for _ in LOCK_BITS]
        self.actor_group_logits: Dict[str, List[float]] = defaultdict(
            lambda: [0.0] * len(LOCK_BITS)
        )
        self.observed_group_actions: Dict[str, set[int]] = defaultdict(set)
        self.critic_weights = [0.0] * FEATURE_COUNT
        self.rng = random.Random(int(seed))
        self.stay_probability = max(0.0, min(1.0, float(stay_probability)))
        self._lock = threading.RLock()

    def probabilities(self, state: PhaseAwareState) -> list[float]:
        with self._lock:
            key = state_key(state)
            self.prototypes.setdefault(key, state)
            features = state_features(state)
            group_logits = self.actor_group_logits[policy_group_key(state)]
            valid = valid_actions(state.current_action)
            bit_probabilities = []
            for bit_index, bit in enumerate(LOCK_BITS):
                if int(state.current_action) & bit:
                    bit_probabilities.append(1.0)
                    continue
                logit = dot(self.actor_weights[bit_index], features) + group_logits[bit_index]
                bit_probabilities.append(sigmoid(logit))
            probabilities = [0.0] * len(ALL_ACTIONS)
            for action in valid:
                probability = 1.0
                for bit_index, bit in enumerate(LOCK_BITS):
                    if int(state.current_action) & bit:
                        continue
                    selected = bool(action & bit)
                    bit_probability = bit_probabilities[bit_index]
                    probability *= bit_probability if selected else 1.0 - bit_probability
                probabilities[action] = probability
            supported = set(self.observed_group_actions.get(policy_group_key(state), ()))
            if supported:
                supported.add(int(state.current_action) & 0xF)
                allowed = set(valid) & supported
                probabilities = [
                    probability if action in allowed else 0.0
                    for action, probability in enumerate(probabilities)
                ]
                total = sum(probabilities)
                if total > 0.0:
                    probabilities = [probability / total for probability in probabilities]
                else:
                    probabilities = [
                        1.0 / len(allowed) if action in allowed else 0.0
                        for action in ALL_ACTIONS
                    ]
            if self.stay_probability > 0.0:
                retain = self.stay_probability
                probabilities = [
                    (1.0 - retain) * probability
                    for probability in probabilities
                ]
                probabilities[int(state.current_action) & 0xF] += retain
            return probabilities

    def value(self, state: PhaseAwareState) -> float:
        with self._lock:
            return dot(self.critic_weights, state_features(state))

    def sample(self, state: PhaseAwareState) -> tuple[int, float]:
        with self._lock:
            probabilities = self.probabilities(state)
            action = sample_action(self.rng, probabilities, state.current_action)
            return action, probabilities[action]

    def select_with_probability(self, state: PhaseAwareState) -> tuple[object, float]:
        from agent.runtime.context import LockAction

        action, probability = self.sample(state)
        return LockAction(LockClass(action)), probability

    def select_with_distribution(
        self,
        state: PhaseAwareState,
    ) -> tuple[object, tuple[float, ...]]:
        from agent.runtime.context import LockAction

        with self._lock:
            probabilities = tuple(self.probabilities(state))
            action = sample_action(self.rng, probabilities, state.current_action)
            return LockAction(LockClass(action)), probabilities

    def compile(
        self,
        *,
        generation: int,
        refinement_distance_threshold: float | None = None,
        occ_cold_start_guard: bool = False,
    ) -> CompiledPhasePolicy:
        grouped: Dict[tuple[str, int, int], list[tuple[CompiledPolicyEntry, PhaseAwareState, int]]] = defaultdict(list)
        for key, state in sorted(self.prototypes.items()):
            probabilities = self.probabilities(state)
            action = max(valid_actions(state.current_action), key=lambda candidate: probabilities[candidate])
            entry = compiled_entry(state, action)
            grouped[(state.phase, int(state.current_action), action)].append(
                (entry, state, max(1, int(self.prototype_counts.get(key, 0))))
            )
        entries = [weighted_one_medoid(rows) for _group, rows in sorted(grouped.items())]
        refinement_actor = {}
        if refinement_distance_threshold is not None:
            refinement_actor = {
                "type": "factorized-bernoulli-lock-bits",
                "distance_threshold": max(0.0, float(refinement_distance_threshold)),
                "actor_weights": [list(row) for row in self.actor_weights],
                "group_logits": {
                    key: list(values)
                    for key, values in sorted(self.actor_group_logits.items())
                },
                "group_action_support": {
                    key: sorted(actions)
                    for key, actions in sorted(self.observed_group_actions.items())
                },
                "state_source": "observed_execution_only",
                "workload_labels": False,
            }
        return CompiledPhasePolicy(
            entries,
            generation=generation,
            refinement_actor=refinement_actor,
            occ_cold_start_guard=occ_cold_start_guard,
        )


class EpsilonGreedyPolicy:
    """Explore around a deployed policy with an exact categorical propensity."""

    def __init__(self, base_policy: object, *, seed: int, epsilon: float = 0.2):
        self.base_policy = base_policy
        self.rng = random.Random(int(seed))
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self._lock = threading.RLock()

    def probabilities(self, state: PhaseAwareState) -> tuple[float, ...]:
        valid = valid_actions(state.current_action)
        base_action = int(self.base_policy.select(state).protected)
        if base_action not in valid:
            base_action = int(state.current_action) & 0xF
        exploration = self.epsilon / len(valid)
        probabilities = [0.0] * len(ALL_ACTIONS)
        for action in valid:
            probabilities[action] = exploration
        probabilities[base_action] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_with_distribution(
        self,
        state: PhaseAwareState,
    ) -> tuple[object, tuple[float, ...]]:
        from agent.runtime.context import LockAction

        with self._lock:
            probabilities = self.probabilities(state)
            action = sample_action(self.rng, probabilities, state.current_action)
            return LockAction(LockClass(action)), probabilities

    def select_with_probability(self, state: PhaseAwareState) -> tuple[object, float]:
        action, probabilities = self.select_with_distribution(state)
        return action, probabilities[int(action.protected)]


class DiscretePPOTrainer:
    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()

    def train(self, policy: DiscretePPOPolicy, transitions: Sequence[PolicyTransition]) -> dict[str, object]:
        if not transitions:
            return {"transitions": 0.0, "mean_return": 0.0, "mean_entropy": 0.0}
        returns = discounted_returns(transitions, self.config.discount)
        samples = [
            (transition, value)
            for transition, value in zip(transitions, returns)
            if not excluded_policy_transition(transition)
        ]
        exclusion_counts: Dict[str, int] = defaultdict(int)
        for transition in transitions:
            reason = policy_transition_exclusion_reason(transition)
            if reason:
                exclusion_counts[reason] += 1
        for transition, _value in samples:
            key = state_key(transition.state)
            policy.prototypes.setdefault(key, transition.state)
            policy.prototype_counts[key] += 1
            policy.observed_group_actions[policy_group_key(transition.state)].add(
                int(transition.action)
            )
        if not samples:
            return {
                "input_transitions": float(len(transitions)),
                "transitions": 0.0,
                "mean_return": 0.0,
                "mean_entropy": 0.0,
                "excluded_transitions": dict(sorted(exclusion_counts.items())),
                "epochs": [],
            }
        behavior_initialization = initialize_group_logits_from_behavior(
            policy, [transition for transition, _value in samples]
        )
        group_sample_counts: Dict[str, int] = defaultdict(int)
        group_action_support: Dict[str, set[int]] = defaultdict(set)
        for transition, _value in samples:
            group_key = policy_group_key(transition.state)
            group_sample_counts[group_key] += 1
            group_action_support[group_key].add(int(transition.action))
        trainable_group_keys = {
            key
            for key, count in group_sample_counts.items()
            if count >= max(1, int(self.config.min_group_samples))
            and len(group_action_support[key]) >= max(1, int(self.config.min_group_actions))
        }
        raw_returns = [value for _transition, value in samples]
        normalized_returns, normalization = normalize_returns_by_source(samples)
        return_mean = sum(raw_returns) / len(raw_returns)
        return_scale = math.sqrt(
            sum((value - return_mean) ** 2 for value in raw_returns) / len(raw_returns)
        )
        critic_order = list(range(len(samples)))
        critic_rng = random.Random(0xC71C)
        for _epoch in range(4):
            critic_rng.shuffle(critic_order)
            for sample_index in critic_order:
                transition, _value = samples[sample_index]
                features = state_features(transition.state)
                group_logits = policy.actor_group_logits[policy_group_key(transition.state)]
                error = normalized_returns[sample_index] - dot(
                    policy.critic_weights, features
                )
                add_scaled(
                    policy.critic_weights,
                    features,
                    self.config.critic_learning_rate * error,
                )
        advantages = [
            target - policy.value(transition.state)
            for (transition, _value), target in zip(samples, normalized_returns)
        ]
        advantage_mean = sum(advantages) / len(advantages)
        advantage_scale = max(
            1e-6,
            math.sqrt(
                sum((value - advantage_mean) ** 2 for value in advantages) / len(advantages)
            ),
        )
        advantages = [
            (value - advantage_mean) / advantage_scale for value in advantages
        ]
        entropy_total = 0.0
        updates = 0
        epoch_reports = []
        sample_order = list(range(len(samples)))
        order_rng = random.Random(0xA73C)
        for epoch in range(max(1, int(self.config.epochs))):
            order_rng.shuffle(sample_order)
            epoch_entropy = 0.0
            epoch_updates = 0
            epoch_clipped = 0
            epoch_ratio_total = 0.0
            for sample_index in sample_order:
                transition, _value = samples[sample_index]
                advantage = advantages[sample_index]
                normalized_return = normalized_returns[sample_index]
                group_key = policy_group_key(transition.state)
                group_logits = policy.actor_group_logits[group_key]
                probabilities = policy.probabilities(transition.state)
                action = int(transition.action)
                if action not in valid_actions(transition.state.current_action):
                    continue
                behavior = max(1e-9, float(getattr(transition, "behavior_probability", 1.0)))
                ratio = probabilities[action] / behavior
                clipped_ratio = max(1.0 - self.config.clip_ratio, min(1.0 + self.config.clip_ratio, ratio))
                if clipped_ratio != ratio:
                    epoch_clipped += 1
                clipped_high = advantage >= 0.0 and ratio > 1.0 + self.config.clip_ratio
                clipped_low = advantage < 0.0 and ratio < 1.0 - self.config.clip_ratio
                coefficient = 0.0 if clipped_high or clipped_low else advantage * ratio
                features = state_features(transition.state)
                bit_probabilities = actor_bit_probabilities(policy, transition.state)
                entropy = 0.0
                for bit_index, bit in enumerate(LOCK_BITS):
                    if int(transition.state.current_action) & bit:
                        continue
                    probability = max(
                        1e-9,
                        min(1.0 - 1e-9, bit_probabilities[bit_index]),
                    )
                    entropy += -probability * math.log(probability) - (
                        1.0 - probability
                    ) * math.log(1.0 - probability)
                    gradient = (1.0 if action & bit else 0.0) - probability
                    entropy_gradient = probability * (1.0 - probability) * math.log(
                        (1.0 - probability) / probability
                    )
                    update = self.config.learning_rate * (
                        coefficient * gradient
                        + self.config.entropy_weight * entropy_gradient
                    )
                    add_scaled(policy.actor_weights[bit_index], features, update)
                    if group_key in trainable_group_keys:
                        group_logits[bit_index] += self.config.group_learning_rate * (
                            coefficient * gradient
                            + self.config.entropy_weight * entropy_gradient
                        )
                value_error = normalized_return - dot(policy.critic_weights, features)
                add_scaled(
                    policy.critic_weights,
                    features,
                    self.config.critic_learning_rate * value_error,
                )
                entropy_total += entropy
                epoch_entropy += entropy
                epoch_ratio_total += ratio
                epoch_updates += 1
                updates += 1
            epoch_reports.append(
                {
                    "epoch": epoch + 1,
                    "updates": epoch_updates,
                    "mean_entropy": epoch_entropy / epoch_updates if epoch_updates else 0.0,
                    "mean_importance_ratio": epoch_ratio_total / epoch_updates if epoch_updates else 0.0,
                    "clip_fraction": epoch_clipped / epoch_updates if epoch_updates else 0.0,
                }
            )
        return {
            "input_transitions": float(len(transitions)),
            "transitions": float(len(samples)),
            "mean_return": sum(value for _transition, value in samples) / len(samples),
            "return_normalization_mean": return_mean,
            "return_normalization_scale": return_scale,
            "return_normalization": "per-source-run",
            "source_run_count": len(normalization),
            "excluded_transitions": dict(sorted(exclusion_counts.items())),
            "behavior_initialization": behavior_initialization,
            "group_residual_training": {
                "min_samples": max(1, int(self.config.min_group_samples)),
                "min_actions": max(1, int(self.config.min_group_actions)),
                "eligible_groups": len(trainable_group_keys),
                "total_groups": len(group_sample_counts),
            },
            "mean_entropy": entropy_total / updates if updates else 0.0,
            "epochs": epoch_reports,
        }


def state_key(state: PhaseAwareState) -> str:
    return "|".join(
        (
            f"phase={state.phase}",
            f"interval={bucket_log(state.inter_round_interval_ms)}",
            f"rs={bucket_size(state.read_set_size)}",
            f"ws={bucket_size(state.write_set_size)}",
            f"rsg={bucket_size(state.read_set_growth)}",
            f"wsg={bucket_size(state.write_set_growth)}",
            f"overlap={bucket_ratio(state.access_overlap_ratio)}",
            f"rounds={bucket_size(state.completed_rounds)}",
            f"ops={bucket_size(state.completed_operations)}",
            f"recent_write={bucket_ratio(state.recent_write_ratio)}",
            f"hot={bucket_ratio(state.hotspot_access_ratio)}",
            f"blocked={bucket_log(state.blocked_time_ms)}",
            f"retry={min(3, state.retry_count)}",
            f"action={int(state.current_action)}",
            f"priority={bucket_log(state.priority)}",
            f"conflict={normalize_conflict_kind(state.recent_conflict_kind)}",
            f"active={bucket_size(state.global_active_transactions)}",
            f"waiters={bucket_size(state.global_waiter_count)}",
            f"abort={bucket_ratio(state.global_abort_rate)}",
            f"throughput={bucket_log(state.global_throughput)}",
            f"avg_latency={bucket_log(state.global_avg_latency_ms)}",
            f"tail={bucket_log(state.global_tail_latency_ms)}",
            f"agent_tps={bucket_log(state.global_agent_task_throughput)}",
            f"agent_avg_latency={bucket_log(state.global_agent_task_avg_latency_ms)}",
            f"agent_tail={bucket_log(state.global_agent_task_tail_latency_ms)}",
            f"conflict_abort={bucket_ratio(state.global_conflict_abort_rate)}",
            f"background_tps={bucket_log(state.global_background_throughput)}",
            f"background_abort={bucket_ratio(state.global_background_abort_rate)}",
        )
    )


def policy_group_key(state: PhaseAwareState) -> str:
    """Stable paper-state aggregation used by the compiled-table PPO actor."""
    return "|".join(
        (
            f"phase={state.phase}",
            f"action={int(state.current_action)}",
            f"active={bucket_size(state.global_active_transactions)}",
            f"waiters={bucket_size(state.global_waiter_count)}",
            f"abort={bucket_ratio(state.global_abort_rate)}",
            f"throughput={bucket_log(state.global_throughput)}",
            f"tail={bucket_log(state.global_tail_latency_ms)}",
            f"agent_tps={bucket_log(state.global_agent_task_throughput)}",
            f"agent_tail={bucket_log(state.global_agent_task_tail_latency_ms)}",
            f"conflict_abort={bucket_ratio(state.global_conflict_abort_rate)}",
            f"background_tps={bucket_log(state.global_background_throughput)}",
            f"background_abort={bucket_ratio(state.global_background_abort_rate)}",
            f"hot={bucket_ratio(state.hotspot_access_ratio)}",
            f"retry={min(3, state.retry_count)}",
            f"conflict={normalize_conflict_kind(state.recent_conflict_kind)}",
            f"recent_write={bucket_ratio(state.recent_write_ratio)}",
            f"priority={bucket_log(state.priority)}",
            f"rounds={bucket_size(state.completed_rounds)}",
        )
    )


def state_features(state: PhaseAwareState) -> tuple[float, ...]:
    phase = str(state.phase)
    current = int(state.current_action)
    return (
        1.0,
        float(phase == "explore"),
        float(phase == "refine"),
        float(phase == "commit"),
        math.log1p(max(0, state.read_set_size)) / 5.0,
        math.log1p(max(0, state.write_set_size)) / 5.0,
        math.log1p(max(0, state.read_set_growth)) / 4.0,
        math.log1p(max(0, state.write_set_growth)) / 4.0,
        max(0.0, min(1.0, state.access_overlap_ratio)),
        max(0.0, min(1.0, state.hotspot_access_ratio)),
        max(0.0, min(1.0, state.recent_write_ratio)),
        min(1.0, max(0.0, state.retry_count / 3.0)),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "none"),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "version-conflict"),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "lock-preempted"),
        float(normalize_conflict_kind(state.recent_conflict_kind) in {"lock-timeout", "lock-conflict"}),
        float(bool(current & 1)),
        float(bool(current & 2)),
        float(bool(current & 4)),
        float(bool(current & 8)),
        math.log1p(max(0, state.global_active_transactions)) / 4.0,
        math.log1p(max(0, state.global_waiter_count)) / 4.0,
        max(0.0, min(1.0, state.global_abort_rate)),
        math.log1p(max(0.0, state.global_throughput)) / 8.0,
        math.log1p(max(0.0, state.global_avg_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.global_tail_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.blocked_time_ms)) / 10.0,
        math.log1p(max(0.0, state.inter_round_interval_ms)) / 10.0,
        math.log1p(max(0, state.completed_rounds)) / 4.0,
        math.log1p(max(0, state.completed_operations)) / 5.0,
        math.log1p(max(0, state.priority)) / 10.0,
        math.log1p(max(0.0, state.global_agent_task_throughput)) / 8.0,
        math.log1p(max(0.0, state.global_agent_task_avg_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.global_agent_task_tail_latency_ms)) / 10.0,
        max(0.0, min(1.0, state.global_conflict_abort_rate)),
        math.log1p(max(0.0, state.global_background_throughput)) / 8.0,
        max(0.0, min(1.0, state.global_background_abort_rate)),
    )


FEATURE_COUNT = 37


def compiled_entry(state: PhaseAwareState, action: int) -> CompiledPolicyEntry:
    return CompiledPolicyEntry(
        phase=state.phase,
        action=int(action),
        inter_round_bucket=bucket_log(state.inter_round_interval_ms),
        read_set_bucket=bucket_size(state.read_set_size),
        write_set_bucket=bucket_size(state.write_set_size),
        read_growth_bucket=bucket_size(state.read_set_growth),
        write_growth_bucket=bucket_size(state.write_set_growth),
        overlap_bucket=bucket_ratio(state.access_overlap_ratio),
        completed_rounds_bucket=bucket_size(state.completed_rounds),
        completed_operations_bucket=bucket_size(state.completed_operations),
        recent_write_bucket=bucket_ratio(state.recent_write_ratio),
        hotspot_bucket=bucket_ratio(state.hotspot_access_ratio),
        blocked_bucket=bucket_log(state.blocked_time_ms),
        retry_bucket=min(3, state.retry_count),
        current_action=int(state.current_action),
        priority_bucket=bucket_log(state.priority),
        recent_conflict_kind=normalize_conflict_kind(state.recent_conflict_kind),
        active_bucket=bucket_size(state.global_active_transactions),
        waiter_bucket=bucket_size(state.global_waiter_count),
        abort_bucket=bucket_ratio(state.global_abort_rate),
        throughput_bucket=bucket_log(state.global_throughput),
        average_latency_bucket=bucket_log(state.global_avg_latency_ms),
        tail_latency_bucket=bucket_log(state.global_tail_latency_ms),
        agent_task_throughput_bucket=bucket_log(state.global_agent_task_throughput),
        agent_task_average_latency_bucket=bucket_log(
            state.global_agent_task_avg_latency_ms
        ),
        agent_task_tail_latency_bucket=bucket_log(
            state.global_agent_task_tail_latency_ms
        ),
        conflict_abort_bucket=bucket_ratio(state.global_conflict_abort_rate),
        background_throughput_bucket=bucket_log(state.global_background_throughput),
        background_abort_bucket=bucket_ratio(state.global_background_abort_rate),
    )


def normalize_conflict_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized else "none"


def weighted_one_medoid(
    rows: Sequence[tuple[CompiledPolicyEntry, PhaseAwareState, int]],
) -> CompiledPolicyEntry:
    support_weight = sum(max(1, int(weight)) for _entry, _state, weight in rows)
    if len(rows) == 1:
        return dataclasses.replace(rows[0][0], support_weight=support_weight)
    candidates = sorted(rows, key=lambda row: (-row[2], state_key(row[1])))[:128]
    selected = min(
        candidates,
        key=lambda candidate: sum(
            max(1, int(weight)) * candidate[0].distance(state)
            for _entry, state, weight in rows
        ),
    )[0]
    return dataclasses.replace(selected, support_weight=support_weight)


def actor_bit_probabilities(
    policy: DiscretePPOPolicy,
    state: PhaseAwareState,
) -> tuple[float, ...]:
    features = state_features(state)
    group_logits = policy.actor_group_logits[policy_group_key(state)]
    values = []
    for bit_index, bit in enumerate(LOCK_BITS):
        if int(state.current_action) & bit:
            values.append(1.0)
        else:
            values.append(
                sigmoid(
                    dot(policy.actor_weights[bit_index], features)
                    + group_logits[bit_index]
                )
            )
    return tuple(values)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        scale = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + scale)
    scale = math.exp(max(-60.0, value))
    return scale / (1.0 + scale)


def sample_action(
    rng: random.Random,
    probabilities: Sequence[float],
    current_action: int,
) -> int:
    draw = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw <= cumulative:
            return action
    return valid_actions(current_action)[-1]


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def add_scaled(weights: List[float], features: Sequence[float], scale: float) -> None:
    for index, feature in enumerate(features):
        weights[index] += float(scale) * feature


def valid_actions(current_action: int) -> tuple[int, ...]:
    current = int(current_action) & 0xF
    return tuple(action for action in ALL_ACTIONS if (action | current) == action)


def discounted_returns(transitions: Sequence[PolicyTransition], discount: float) -> list[float]:
    values = [0.0] * len(transitions)
    running_by_txn: Dict[tuple[str, str], float] = defaultdict(float)
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        txn_key = (str(transition.source_id), transition.txn_id)
        if transition.done:
            running_by_txn[txn_key] = 0.0
        running = float(transition.reward) + float(discount) * running_by_txn[txn_key]
        running_by_txn[txn_key] = running
        values[index] = running
    return values


def normalize_returns_by_source(
    samples: Sequence[tuple[PolicyTransition, float]],
) -> tuple[list[float], dict[str, tuple[float, float]]]:
    """Give each exploration run equal reward scale without exposing its workload label."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for transition, value in samples:
        grouped[str(transition.source_id)].append(float(value))
    statistics: dict[str, tuple[float, float]] = {}
    for source_id, values in grouped.items():
        mean = sum(values) / len(values)
        scale = max(
            1.0,
            math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        )
        statistics[source_id] = (mean, scale)
    return [
        (float(value) - statistics[str(transition.source_id)][0])
        / statistics[str(transition.source_id)][1]
        for transition, value in samples
    ], statistics


def initialize_group_logits_from_behavior(
    policy: DiscretePPOPolicy,
    transitions: Sequence[PolicyTransition],
) -> dict[str, int]:
    """Initialize actor residuals from the full behavior support when recoverable."""
    grouped: Dict[str, List[PolicyTransition]] = defaultdict(list)
    prototypes: Dict[str, PhaseAwareState] = {}
    for transition in transitions:
        key = policy_group_key(transition.state)
        prototypes.setdefault(key, transition.state)
        grouped[key].append(transition)
    exact_groups = 0
    fallback_groups = 0
    for key, rows in grouped.items():
        valid = valid_actions(prototypes[key].current_action)
        reconstructed = [reconstruct_behavior_distribution(row) for row in rows]
        has_explicit_distributions = all(
            len(
                tuple(
                    getattr(row, "behavior_action_probabilities", ()) or ()
                )
            )
            == len(ALL_ACTIONS)
            for row in rows
        )
        if all(distribution is not None for distribution in reconstructed) and (
            has_explicit_distributions
            or behavior_distributions_consistent(reconstructed)
        ):
            exact_groups += 1
            raw = {
                action: sum(distribution[action] for distribution in reconstructed) / len(rows)
                for action in valid
            }
        else:
            fallback_groups += 1
            observed: Dict[int, List[float]] = defaultdict(list)
            for row in rows:
                observed[int(row.action)].append(float(row.behavior_probability))
            raw = {
                action: (
                    sum(observed[action]) / len(observed[action])
                    if observed.get(action)
                    else 1e-6
                )
                for action in valid
            }
        total = sum(raw.values())
        normalized = {action: raw[action] / total for action in valid}
        logits = policy.actor_group_logits[key]
        for bit_index, bit in enumerate(LOCK_BITS):
            if int(prototypes[key].current_action) & bit:
                logits[bit_index] = 0.0
                continue
            bit_probability = sum(
                probability
                for action, probability in normalized.items()
                if action & bit
            )
            bit_probability = max(1e-6, min(1.0 - 1e-6, bit_probability))
            logits[bit_index] = math.log(bit_probability / (1.0 - bit_probability))
    return {
        "state_groups": len(grouped),
        "full_support_groups": exact_groups,
        "fallback_groups": fallback_groups,
    }


def behavior_distributions_consistent(
    distributions: Sequence[dict[int, float] | None],
    *,
    tolerance: float = 1e-6,
) -> bool:
    rows = [distribution for distribution in distributions if distribution is not None]
    if not rows:
        return False
    reference = rows[0]
    return all(
        set(distribution) == set(reference)
        and all(
            abs(distribution[action] - reference[action]) <= tolerance
            for action in reference
        )
        for distribution in rows[1:]
    )


def reconstruct_behavior_distribution(
    transition: PolicyTransition,
) -> dict[int, float] | None:
    """Recover DiscretePPOPolicy's stay-mixture from one recorded propensity."""
    valid = valid_actions(transition.state.current_action)
    if not valid:
        return None
    if len(valid) == 1:
        return {valid[0]: 1.0}
    action = int(transition.action)
    if action not in valid:
        return None
    recorded = tuple(
        float(value)
        for value in getattr(transition, "behavior_action_probabilities", ()) or ()
    )
    if recorded:
        if len(recorded) != len(ALL_ACTIONS):
            return None
        if any(value < 0.0 or value > 1.0 for value in recorded):
            return None
        if any(recorded[candidate] > 1e-12 for candidate in ALL_ACTIONS if candidate not in valid):
            return None
        if abs(sum(recorded) - 1.0) > 1e-6:
            return None
        return {candidate: recorded[candidate] for candidate in valid}
    selected_probability = float(transition.behavior_probability)
    if not 0.0 < selected_probability <= 1.0:
        return None
    current = int(transition.state.current_action) & 0xF
    if action == current:
        other_probability = (1.0 - selected_probability) / (len(valid) - 1)
        distribution = {candidate: other_probability for candidate in valid}
        distribution[current] = selected_probability
    else:
        other_probability = selected_probability
        current_probability = 1.0 - (len(valid) - 1) * other_probability
        if current_probability < -1e-9 or current_probability > 1.0 + 1e-9:
            return None
        distribution = {candidate: other_probability for candidate in valid}
        distribution[current] = max(0.0, min(1.0, current_probability))
    if any(value < -1e-9 or value > 1.0 + 1e-9 for value in distribution.values()):
        return None
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        return None
    return distribution


def excluded_policy_transition(transition: PolicyTransition) -> bool:
    return bool(policy_transition_exclusion_reason(transition))


def policy_transition_exclusion_reason(transition: PolicyTransition) -> str:
    if fixed_initial_occ_transition(transition):
        return "fixed-initial-occ"
    if len(valid_actions(transition.state.current_action)) <= 1:
        return "no-action-choice"
    if float(transition.behavior_probability) >= 1.0 - 1e-12:
        return "deterministic-without-support"
    return ""


def audit_policy(
    policy: DiscretePPOPolicy,
    compiled: CompiledPhasePolicy,
    transitions: Sequence[PolicyTransition],
    *,
    min_action_support: int = 3,
    discount: float = 0.99,
) -> dict[str, object]:
    """Compare actor and compiled lookup decisions with empirical group returns."""
    returns = discounted_returns(transitions, float(discount))
    samples = [
        (transition, value)
        for transition, value in zip(transitions, returns)
        if not excluded_policy_transition(transition)
    ]
    if not samples:
        return {
            "audited_state_groups": 0,
            "actor_top_action_agreement": 0.0,
            "compiled_action_agreement": 0.0,
        }
    normalized_returns, _normalization = normalize_returns_by_source(samples)
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    representatives: Dict[str, PhaseAwareState] = {}
    for (transition, _raw_return), normalized_return in zip(samples, normalized_returns):
        key = policy_group_key(transition.state)
        representatives.setdefault(key, transition.state)
        grouped[key][int(transition.action)].append(float(normalized_return))

    actor_agreement = 0
    compiled_agreement = 0
    weighted_actor_agreement = 0
    weighted_compiled_agreement = 0
    weighted_actor_regret = 0.0
    weighted_compiled_regret = 0.0
    actor_regret_weight = 0
    compiled_regret_weight = 0
    total_weight = 0
    audited_groups = 0
    actor_unobserved = 0
    compiled_unobserved = 0
    empirical_actions: Dict[int, int] = defaultdict(int)
    actor_actions: Dict[int, int] = defaultdict(int)
    compiled_actions: Dict[int, int] = defaultdict(int)
    for key, by_action in grouped.items():
        eligible = {
            action: values
            for action, values in by_action.items()
            if len(values) >= max(1, int(min_action_support))
        }
        if len(eligible) < 2:
            continue
        means = {
            action: sum(values) / len(values)
            for action, values in eligible.items()
        }
        best_action = min(
            means,
            key=lambda action: (-means[action], action),
        )
        state = representatives[key]
        probabilities = policy.probabilities(state)
        actor_action = max(
            valid_actions(state.current_action),
            key=lambda action: (probabilities[action], -action),
        )
        compiled_action = int(compiled.select(state).protected)
        weight = sum(len(values) for values in eligible.values())
        audited_groups += 1
        total_weight += weight
        actor_agreement += int(actor_action == best_action)
        compiled_agreement += int(compiled_action == best_action)
        weighted_actor_agreement += int(actor_action == best_action) * weight
        weighted_compiled_agreement += int(compiled_action == best_action) * weight
        empirical_actions[best_action] += weight
        actor_actions[actor_action] += weight
        compiled_actions[compiled_action] += weight
        if actor_action in means:
            weighted_actor_regret += (means[best_action] - means[actor_action]) * weight
            actor_regret_weight += weight
        else:
            actor_unobserved += 1
        if compiled_action in means:
            weighted_compiled_regret += (
                means[best_action] - means[compiled_action]
            ) * weight
            compiled_regret_weight += weight
        else:
            compiled_unobserved += 1

    def action_counts(values: Dict[int, int]) -> dict[str, int]:
        return {str(action): values[action] for action in sorted(values)}

    return {
        "min_action_support": max(1, int(min_action_support)),
        "policy_control_point_transitions": len(samples),
        "audited_state_groups": audited_groups,
        "audited_transition_weight": total_weight,
        "actor_top_action_agreement": actor_agreement / audited_groups if audited_groups else 0.0,
        "compiled_action_agreement": compiled_agreement / audited_groups if audited_groups else 0.0,
        "actor_weighted_agreement": (
            weighted_actor_agreement / total_weight if total_weight else 0.0
        ),
        "compiled_weighted_agreement": (
            weighted_compiled_agreement / total_weight if total_weight else 0.0
        ),
        "actor_weighted_regret": (
            weighted_actor_regret / actor_regret_weight if actor_regret_weight else 0.0
        ),
        "compiled_weighted_regret": (
            weighted_compiled_regret / compiled_regret_weight
            if compiled_regret_weight else 0.0
        ),
        "actor_unobserved_action_groups": actor_unobserved,
        "compiled_unobserved_action_groups": compiled_unobserved,
        "empirical_best_action_counts": action_counts(empirical_actions),
        "actor_top_action_counts": action_counts(actor_actions),
        "compiled_action_counts": action_counts(compiled_actions),
    }


def fixed_initial_occ_transition(transition: PolicyTransition) -> bool:
    state = transition.state
    return bool(
        state.phase == "explore"
        and state.completed_operations == 0
        and state.current_action == 0
        and int(transition.action) == 0
        and abs(float(transition.behavior_probability) - 1.0) < 1e-12
    )
