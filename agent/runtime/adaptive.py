"""ATCC-style policy-table selection for agent-side CC modules."""

from __future__ import annotations

import dataclasses
import random
import threading
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from agent.native import load_cast_core
from agent.runtime.atcc import (
    ATCCRuntimeStats,
    PhaseAwareATCCDecision,
    PhaseAwareATCCModule,
)

cc = load_cast_core()


_INTENT_NAMES = {
    cc.IntentType.kRead: "read",
    cc.IntentType.kOverwrite: "overwrite",
    cc.IntentType.kAppend: "append",
    cc.IntentType.kDelta: "delta",
    cc.IntentType.kCas: "cas",
}


@dataclasses.dataclass(frozen=True)
class AdaptiveTransactionProfile:
    candidate_count: int
    read_count: int
    write_count: int
    intent_counts: Mapping[str, int]
    distinct_write_targets: int
    task_type: str = ""
    workload: str = ""

    @property
    def has_semantic_intent(self) -> bool:
        return any(self.intent_counts.get(name, 0) for name in ("append", "delta", "cas"))

    @property
    def overwrite_only(self) -> bool:
        return self.write_count > 0 and self.intent_counts == {"overwrite": self.write_count}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "intent_counts": dict(sorted(self.intent_counts.items())),
            "distinct_write_targets": self.distinct_write_targets,
            "task_type": self.task_type,
            "workload": self.workload,
            "has_semantic_intent": self.has_semantic_intent,
            "overwrite_only": self.overwrite_only,
        }


@dataclasses.dataclass(frozen=True)
class AdaptivePolicyRule:
    name: str
    target_strategy: str
    description: str = ""
    task_types: Tuple[str, ...] = ()
    workloads: Tuple[str, ...] = ()
    any_intent: Tuple[str, ...] = ()
    min_reads: Optional[int] = None
    min_writes: Optional[int] = None
    min_distinct_write_targets: Optional[int] = None
    overwrite_only: Optional[bool] = None

    def matches(self, profile: AdaptiveTransactionProfile) -> bool:
        if self.task_types and profile.task_type not in self.task_types:
            return False
        if self.workloads and profile.workload not in self.workloads:
            return False
        if self.any_intent and not any(
            profile.intent_counts.get(intent, 0) for intent in self.any_intent
        ):
            return False
        if self.min_reads is not None and profile.read_count < self.min_reads:
            return False
        if self.min_writes is not None and profile.write_count < self.min_writes:
            return False
        if (
            self.min_distinct_write_targets is not None
            and profile.distinct_write_targets < self.min_distinct_write_targets
        ):
            return False
        if self.overwrite_only is not None and profile.overwrite_only != self.overwrite_only:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_strategy": self.target_strategy,
            "description": self.description,
            "task_types": list(self.task_types),
            "workloads": list(self.workloads),
            "any_intent": list(self.any_intent),
            "min_reads": self.min_reads,
            "min_writes": self.min_writes,
            "min_distinct_write_targets": self.min_distinct_write_targets,
            "overwrite_only": self.overwrite_only,
        }


@dataclasses.dataclass(frozen=True)
class AdaptivePolicyTable:
    rules: Tuple[AdaptivePolicyRule, ...]
    fallback_strategy: str = "occ"
    name: str = "default-atcc-table"

    @classmethod
    def default(cls) -> "AdaptivePolicyTable":
        return cls(
            rules=(
                AdaptivePolicyRule(
                    name="semantic-intents",
                    target_strategy="semantic",
                    description="Use semantic CC for rebaseable agent intents.",
                    any_intent=("append", "delta", "cas"),
                ),
                AdaptivePolicyRule(
                    name="wide-strict-read-write",
                    target_strategy="2pl",
                    description=(
                        "Use pessimistic validation for strict plans with a wide "
                        "read/write footprint."
                    ),
                    min_reads=2,
                    min_writes=4,
                    overwrite_only=True,
                ),
            )
        )

    @classmethod
    def new_order(cls, *, wide_write_threshold: int = 8) -> "AdaptivePolicyTable":
        """ATCC-style table specialized for TPC-C NewOrder.

        Small NewOrder tasks benefit from semantic optimistic rebase because
        stock/ytd deltas and order appends can merge. Wide NewOrder tasks touch
        many stock rows and create larger validation windows, so the table
        switches to the agent-level pessimistic strategy before falling back to
        OCC for strict non-semantic tasks.
        """

        if wide_write_threshold <= 0:
            raise ValueError("wide_write_threshold must be positive")
        return cls(
            name="tpcc-new-order-atcc-table",
            rules=(
                AdaptivePolicyRule(
                    name="wide-new-order-to-2pl",
                    target_strategy="2pl",
                    description=(
                        "Use pessimistic object locking for wide NewOrder "
                        "requests whose validation footprint is large."
                    ),
                    task_types=("new_order",),
                    min_distinct_write_targets=int(wide_write_threshold),
                ),
                AdaptivePolicyRule(
                    name="new-order-semantic-rebase",
                    target_strategy="semantic",
                    description=(
                        "Use semantic optimistic rebase for narrow NewOrder "
                        "delta/append intents."
                    ),
                    task_types=("new_order",),
                    any_intent=("append", "delta"),
                ),
                AdaptivePolicyRule(
                    name="semantic-intents",
                    target_strategy="semantic",
                    description="Use semantic CC for other rebaseable agent intents.",
                    any_intent=("append", "delta", "cas"),
                ),
            ),
            fallback_strategy="occ",
        )

    def select(
        self,
        candidates: Sequence[Any],
        *,
        read_count: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        available_strategies: Optional[Iterable[str]] = None,
    ) -> str:
        available = (
            {str(strategy).strip().lower() for strategy in available_strategies}
            if available_strategies is not None
            else None
        )
        profile = profile_candidates(candidates, read_count=read_count, metadata=metadata)
        for rule in self.rules:
            target = rule.target_strategy.strip().lower()
            if rule.matches(profile) and (available is None or target in available):
                return target
        fallback = self.fallback_strategy.strip().lower()
        if available is not None and fallback not in available:
            raise ValueError(f"adaptive fallback strategy is not registered: {fallback}")
        return fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fallback_strategy": self.fallback_strategy,
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclasses.dataclass(frozen=True)
class OperationPolicyProfile:
    object_id: str
    access_kind: str
    intent_name: str = "read"
    task_type: str = ""
    workload: str = ""
    candidate_count: int = 1
    operation_count_for_object: int = 1
    total_writes: int = 0
    retry_count: int = 0
    agent_interval_s: float = 0.0
    agent_phase: str = ""

    @property
    def is_semantic_write(self) -> bool:
        return self.intent_name in {"append", "delta", "cas"}

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class OperationPolicyRule:
    name: str
    target_policy: str
    description: str = ""
    access_kinds: Tuple[str, ...] = ()
    task_types: Tuple[str, ...] = ()
    workloads: Tuple[str, ...] = ()
    intent_names: Tuple[str, ...] = ()
    object_id_contains: Tuple[str, ...] = ()
    min_operation_count_for_object: Optional[int] = None
    min_total_writes: Optional[int] = None

    def matches(self, profile: OperationPolicyProfile) -> bool:
        if self.access_kinds and profile.access_kind not in self.access_kinds:
            return False
        if self.task_types and profile.task_type not in self.task_types:
            return False
        if self.workloads and profile.workload not in self.workloads:
            return False
        if self.intent_names and profile.intent_name not in self.intent_names:
            return False
        if self.object_id_contains and not any(
            needle in profile.object_id for needle in self.object_id_contains
        ):
            return False
        if (
            self.min_operation_count_for_object is not None
            and profile.operation_count_for_object < self.min_operation_count_for_object
        ):
            return False
        if self.min_total_writes is not None and profile.total_writes < self.min_total_writes:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_policy": self.target_policy,
            "description": self.description,
            "access_kinds": list(self.access_kinds),
            "task_types": list(self.task_types),
            "workloads": list(self.workloads),
            "intent_names": list(self.intent_names),
            "object_id_contains": list(self.object_id_contains),
            "min_operation_count_for_object": self.min_operation_count_for_object,
            "min_total_writes": self.min_total_writes,
        }


@dataclasses.dataclass(frozen=True)
class OperationPolicyDecision:
    object_id: str
    access_kind: str
    intent_name: str
    policy: str
    rule: str
    task_type: str = ""
    workload: str = ""
    candidate_count: int = 0
    operation_count_for_object: int = 0
    total_writes: int = 0
    retry_count: int = 0
    agent_interval_s: float = 0.0
    agent_phase: str = ""
    object_class: str = ""
    profile_key: str = ""
    exact_key: str = ""
    optimistic_cost: float = 0.0
    pessimistic_cost: float = 0.0
    telemetry_observations: int = 0
    rl_state_key: str = ""
    rl_explore: bool = False
    rl_q_optimistic: float = 0.0
    rl_q_pessimistic: float = 0.0
    atcc_state_key: str = ""
    atcc_action: str = ""
    atcc_phase: str = ""
    atcc_priority: int = 0
    atcc_explore: bool = False
    atcc_q_values: Mapping[str, float] = dataclasses.field(default_factory=dict)
    atcc_global_abort_rate: float = 0.0
    atcc_global_lock_wait_s: float = 0.0
    atcc_global_latency_s: float = 0.0
    atcc_global_lock_queue_depth: float = 0.0
    atcc_global_lock_handoff_count: float = 0.0
    atcc_global_committing_count: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class OperationPolicyStats:
    observations: int = 0
    optimistic_observations: int = 0
    pessimistic_observations: int = 0
    optimistic_conflicts: int = 0
    committed: int = 0
    aborted: int = 0
    rejected: int = 0
    ewma_conflict_rate: float = 0.0
    ewma_lock_wait_s: float = 0.0
    ewma_lock_queue_depth: float = 0.0

    def observe(
        self,
        *,
        policy: str,
        conflict_abort: bool,
        committed: bool,
        rejected: bool,
        lock_wait_s: float,
        lock_queue_depth: float = 0.0,
        alpha: float,
    ) -> None:
        self.observations += 1
        if committed:
            self.committed += 1
        elif rejected:
            self.rejected += 1
        else:
            self.aborted += 1

        if policy == "optimistic":
            self.optimistic_observations += 1
            if conflict_abort:
                self.optimistic_conflicts += 1
            sample = 1.0 if conflict_abort else 0.0
            self.ewma_conflict_rate = _ewma(
                self.ewma_conflict_rate,
                sample,
                alpha,
                self.optimistic_observations,
            )
            return

        if policy == "pessimistic":
            self.pessimistic_observations += 1
            self.ewma_lock_wait_s = _ewma(
                self.ewma_lock_wait_s,
                max(0.0, float(lock_wait_s)),
                alpha,
                self.pessimistic_observations,
            )
            self.ewma_lock_queue_depth = _ewma(
                self.ewma_lock_queue_depth,
                max(0.0, float(lock_queue_depth)),
                alpha,
                self.pessimistic_observations,
            )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperationPolicyStats":
        return cls(
            observations=int(data.get("observations", 0)),
            optimistic_observations=int(data.get("optimistic_observations", 0)),
            pessimistic_observations=int(data.get("pessimistic_observations", 0)),
            optimistic_conflicts=int(data.get("optimistic_conflicts", 0)),
            committed=int(data.get("committed", 0)),
            aborted=int(data.get("aborted", 0)),
            rejected=int(data.get("rejected", 0)),
            ewma_conflict_rate=float(data.get("ewma_conflict_rate", 0.0)),
            ewma_lock_wait_s=float(data.get("ewma_lock_wait_s", 0.0)),
            ewma_lock_queue_depth=float(data.get("ewma_lock_queue_depth", 0.0)),
        )


class OperationPolicyTelemetry:
    """Online ATCC feedback shared by one manager run."""

    def __init__(self, *, alpha: float = 0.25):
        if not 0 < alpha <= 1:
            raise ValueError("telemetry alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._lock = threading.RLock()
        self._stats: Dict[str, OperationPolicyStats] = {}

    def stats_for(self, key: str) -> OperationPolicyStats:
        with self._lock:
            stats = self._stats.get(key)
            if stats is None:
                return OperationPolicyStats()
            return dataclasses.replace(stats)

    def observe(
        self,
        keys: Iterable[str],
        *,
        policy: str,
        conflict_abort: bool,
        committed: bool,
        rejected: bool,
        lock_wait_s: float,
        lock_queue_depth: float = 0.0,
    ) -> None:
        normalized = str(policy).strip().lower()
        with self._lock:
            for key in keys:
                if not key:
                    continue
                stats = self._stats.setdefault(str(key), OperationPolicyStats())
                stats.observe(
                    policy=normalized,
                    conflict_abort=conflict_abort,
                    committed=committed,
                    rejected=rejected,
                    lock_wait_s=lock_wait_s,
                    lock_queue_depth=lock_queue_depth,
                    alpha=self.alpha,
                )

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                key: stats.to_dict()
                for key, stats in sorted(self._stats.items())
            }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        alpha: float = 0.25,
    ) -> "OperationPolicyTelemetry":
        telemetry = cls(alpha=alpha)
        with telemetry._lock:
            telemetry._stats = {
                str(key): OperationPolicyStats.from_dict(value)
                for key, value in dict(data or {}).items()
                if isinstance(value, Mapping)
            }
        return telemetry


class OperationPolicyQLearner:
    """Tabular epsilon-greedy Q-learning for operation-level ATCC."""

    actions = ("optimistic", "pessimistic")

    def __init__(
        self,
        *,
        learning_rate: float = 0.25,
        discount: float = 0.0,
        epsilon: float = 0.20,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.999,
        seed: int = 0,
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= min_epsilon <= epsilon <= 1:
            raise ValueError("epsilon must satisfy 0 <= min_epsilon <= epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self._rng = random.Random(int(seed))
        self._lock = threading.RLock()
        self._q: Dict[str, Dict[str, float]] = {}
        self._visits: Counter[Tuple[str, str]] = Counter()
        self._updates = 0

    def select(self, state_key: str, *, prior_action: str) -> Tuple[str, bool, float, float]:
        prior = _normalize_policy(prior_action)
        with self._lock:
            values = self._q.setdefault(state_key, {action: 0.0 for action in self.actions})
            explore = self._rng.random() < self.epsilon
            if explore:
                action = self._rng.choice(self.actions)
            else:
                action = max(
                    self.actions,
                    key=lambda candidate: (
                        values.get(candidate, 0.0),
                        1 if candidate == prior else 0,
                    ),
                )
            return (
                action,
                bool(explore),
                float(values.get("optimistic", 0.0)),
                float(values.get("pessimistic", 0.0)),
            )

    def update(self, state_key: str, action: str, reward: float, *, next_state_key: str = "") -> None:
        normalized = _normalize_policy(action)
        if normalized not in self.actions:
            return
        with self._lock:
            values = self._q.setdefault(state_key, {name: 0.0 for name in self.actions})
            current = float(values.get(normalized, 0.0))
            next_best = 0.0
            if next_state_key:
                next_values = self._q.setdefault(
                    next_state_key, {name: 0.0 for name in self.actions}
                )
                next_best = max(float(next_values.get(name, 0.0)) for name in self.actions)
            target = float(reward) + self.discount * next_best
            values[normalized] = current + self.learning_rate * (target - current)
            self._visits[(state_key, normalized)] += 1
            self._updates += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "learning_rate": self.learning_rate,
                "discount": self.discount,
                "epsilon": self.epsilon,
                "min_epsilon": self.min_epsilon,
                "epsilon_decay": self.epsilon_decay,
                "updates": self._updates,
                "q_values": {
                    state: dict(sorted(values.items()))
                    for state, values in sorted(self._q.items())
                },
                "visits": {
                    f"{state}|{action}": count
                    for (state, action), count in sorted(self._visits.items())
                },
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperationPolicyQLearner":
        min_epsilon = float(data.get("min_epsilon", 0.0))
        learner = cls(
            learning_rate=float(data.get("learning_rate", 0.25)),
            discount=float(data.get("discount", 0.0)),
            epsilon=float(data.get("epsilon", min_epsilon)),
            min_epsilon=min_epsilon,
            epsilon_decay=float(data.get("epsilon_decay", 1.0)),
        )
        with learner._lock:
            learner._q = {
                str(state): {
                    action: float(values.get(action, 0.0))
                    for action in learner.actions
                }
                for state, values in dict(data.get("q_values", {})).items()
                if isinstance(values, Mapping)
            }
            learner._visits = Counter()
            for key, count in dict(data.get("visits", {})).items():
                state, separator, action = str(key).rpartition("|")
                if separator and action in learner.actions:
                    learner._visits[(state, action)] = int(count)
            learner._updates = int(data.get("updates", sum(learner._visits.values())))
        return learner


@dataclasses.dataclass(frozen=True)
class OperationPolicyTable:
    rules: Tuple[OperationPolicyRule, ...]
    fallback_policy: str = "optimistic"
    name: str = "operation-atcc-table"
    online_feedback: bool = False
    min_feedback_observations: int = 6
    exact_key_min_observations: int = 3
    conflict_abort_cost: float = 1.0
    lock_wait_cost_per_s: float = 50.0
    lock_queue_depth_cost: float = 0.05
    lock_overhead_cost: float = 0.02
    hysteresis: float = 1.15
    pinned_rules: Tuple[str, ...] = ()
    rl_enabled: bool = False
    rl_conflict_penalty: float = 2.0
    rl_commit_reward: float = 1.0
    rl_reject_penalty: float = 0.5
    rl_lock_wait_cost_per_s: float = 50.0
    rl_pessimistic_action_cost: float = 0.02
    rl_learner: OperationPolicyQLearner = dataclasses.field(
        default_factory=OperationPolicyQLearner, compare=False
    )
    atcc_module: Optional[PhaseAwareATCCModule] = dataclasses.field(
        default=None, compare=False
    )
    atcc_runtime_stats: ATCCRuntimeStats = dataclasses.field(
        default_factory=ATCCRuntimeStats, compare=False
    )
    atcc_cold_occ_fast_path: bool = False
    atcc_fast_path_max_retry_count: int = 0
    atcc_fast_path_max_agent_interval_s: float = 0.120
    atcc_fast_path_max_global_abort_rate: float = 0.03
    atcc_fast_path_max_lock_queue_depth: float = 0.0
    atcc_fast_path_max_total_writes: int = 1
    telemetry: OperationPolicyTelemetry = dataclasses.field(
        default_factory=OperationPolicyTelemetry, compare=False
    )

    @classmethod
    def default(cls) -> "OperationPolicyTable":
        return cls(
            rules=(
                OperationPolicyRule(
                    name="hot-overwrite-to-pessimistic",
                    target_policy="pessimistic",
                    description="Lock repeatedly written overwrite objects.",
                    access_kinds=("write",),
                    intent_names=("overwrite",),
                    min_operation_count_for_object=2,
                ),
                OperationPolicyRule(
                    name="semantic-writes-stay-optimistic",
                    target_policy="optimistic",
                    description="Let semantic writes rebase optimistically.",
                    access_kinds=("write",),
                    intent_names=("append", "delta", "cas"),
                ),
            )
        )

    @classmethod
    def feedback_atcc(cls) -> "OperationPolicyTable":
        """Generic online ATCC table.

        Static rules only provide a safe prior. Runtime feedback can override
        them when observed conflict cost exceeds expected lock cost, or when
        lock waiting becomes more expensive than optimistic validation.
        """

        return cls(
            name="online-operation-atcc-table",
            rules=(
                OperationPolicyRule(
                    name="read-optimistic-prior",
                    target_policy="optimistic",
                    description="Reads do not need pessimistic operation locks.",
                    access_kinds=("read",),
                ),
                OperationPolicyRule(
                    name="shared-overwrite-pessimistic-prior",
                    target_policy="pessimistic",
                    description="Repeated overwrite targets are likely strict conflicts.",
                    access_kinds=("write",),
                    intent_names=("overwrite",),
                    min_operation_count_for_object=2,
                ),
            ),
            fallback_policy="optimistic",
            online_feedback=True,
        )

    @classmethod
    def tpcc_new_order(cls, *, hot_object_threshold: int = 2) -> "OperationPolicyTable":
        if hot_object_threshold <= 0:
            raise ValueError("hot_object_threshold must be positive")
        return cls(
            name="tpcc-new-order-operation-atcc-table",
            rules=(
                OperationPolicyRule(
                    name="new-order-district-counter-pessimistic",
                    target_policy="pessimistic",
                    description=(
                        "Lock hot district order counters; each NewOrder competes "
                        "for next_order_id."
                    ),
                    access_kinds=("write",),
                    task_types=("new_order",),
                    object_id_contains=("next_order_id",),
                    min_operation_count_for_object=hot_object_threshold,
                ),
                OperationPolicyRule(
                    name="new-order-stock-delta-optimistic",
                    target_policy="optimistic",
                    description="Stock quantity/ytd deltas use semantic optimistic rebase.",
                    access_kinds=("write",),
                    task_types=("new_order",),
                    intent_names=("delta", "append"),
                ),
                OperationPolicyRule(
                    name="new-order-read-optimistic",
                    target_policy="optimistic",
                    access_kinds=("read",),
                    task_types=("new_order",),
                ),
            ),
            fallback_policy="optimistic",
        )

    @classmethod
    def tpcc_atcc(cls) -> "OperationPolicyTable":
        return cls(
            name="tpcc-online-operation-atcc-table",
            rules=(
                OperationPolicyRule(
                    name="tpcc-read-optimistic-prior",
                    target_policy="optimistic",
                    access_kinds=("read",),
                    task_types=(
                        "new_order",
                        "payment",
                        "order_status",
                        "delivery",
                        "stock_level",
                    ),
                ),
                OperationPolicyRule(
                    name="tpcc-order-counter-risk-prior",
                    target_policy="pessimistic",
                    description=(
                        "District order counters serialize NewOrder id allocation; "
                        "runtime feedback may relax this if lock waiting dominates."
                    ),
                    access_kinds=("write",),
                    task_types=("new_order",),
                    object_id_contains=("next_order_id",),
                ),
                OperationPolicyRule(
                    name="tpcc-commutative-delta-optimistic-prior",
                    target_policy="optimistic",
                    description=(
                        "Stock, warehouse, district, and customer deltas start "
                        "optimistic, then move only if feedback shows conflict cost."
                    ),
                    access_kinds=("write",),
                    intent_names=("delta", "append", "cas"),
                ),
            ),
            fallback_policy="optimistic",
            online_feedback=True,
            min_feedback_observations=4,
            exact_key_min_observations=3,
            conflict_abort_cost=1.0,
            lock_wait_cost_per_s=50.0,
            lock_overhead_cost=0.02,
            hysteresis=1.10,
            pinned_rules=("tpcc-order-counter-risk-prior",),
            atcc_cold_occ_fast_path=True,
            atcc_fast_path_max_agent_interval_s=0.080,
            atcc_fast_path_max_global_abort_rate=0.02,
        )

    @classmethod
    def tpcc_rl_atcc(cls) -> "OperationPolicyTable":
        base = cls.tpcc_atcc()
        return dataclasses.replace(
            base,
            name="tpcc-rl-operation-atcc-table",
            rl_enabled=True,
            rl_conflict_penalty=2.5,
            rl_commit_reward=1.0,
            rl_reject_penalty=0.5,
            rl_lock_wait_cost_per_s=80.0,
            rl_pessimistic_action_cost=0.03,
            rl_learner=OperationPolicyQLearner(
                learning_rate=0.30,
                discount=0.0,
                epsilon=0.18,
                min_epsilon=0.02,
                epsilon_decay=0.9995,
                seed=2027,
            ),
        )

    @classmethod
    def tpcc_phase_rl_atcc(cls) -> "OperationPolicyTable":
        base = cls.tpcc_atcc()
        return dataclasses.replace(
            base,
            name="tpcc-phase-aware-operation-atcc-table",
            atcc_module=PhaseAwareATCCModule.tpcc(),
            rl_enabled=False,
        )

    @classmethod
    def ycsb_atcc(cls) -> "OperationPolicyTable":
        return cls(
            name="ycsb-online-operation-atcc-table",
            rules=(
                OperationPolicyRule(
                    name="ycsb-read-optimistic-prior",
                    target_policy="optimistic",
                    access_kinds=("read",),
                    task_types=("read-update",),
                ),
                OperationPolicyRule(
                    name="ycsb-shared-overwrite-pessimistic-prior",
                    target_policy="pessimistic",
                    access_kinds=("write",),
                    task_types=("read-update",),
                    intent_names=("overwrite",),
                    min_operation_count_for_object=2,
                ),
            ),
            fallback_policy="optimistic",
            online_feedback=True,
            min_feedback_observations=16,
            exact_key_min_observations=8,
            conflict_abort_cost=1.0,
            lock_wait_cost_per_s=200.0,
            lock_overhead_cost=0.05,
            hysteresis=1.10,
        )

    @classmethod
    def ycsb_rl_atcc(cls) -> "OperationPolicyTable":
        base = cls.ycsb_atcc()
        return dataclasses.replace(
            base,
            name="ycsb-rl-operation-atcc-table",
            rl_enabled=True,
            rl_conflict_penalty=1.2,
            rl_commit_reward=1.0,
            rl_reject_penalty=0.5,
            rl_lock_wait_cost_per_s=250.0,
            rl_pessimistic_action_cost=0.08,
            rl_learner=OperationPolicyQLearner(
                learning_rate=0.25,
                discount=0.0,
                epsilon=0.15,
                min_epsilon=0.01,
                epsilon_decay=0.999,
                seed=3031,
            ),
        )

    @classmethod
    def ycsb_phase_rl_atcc(cls) -> "OperationPolicyTable":
        base = cls.ycsb_atcc()
        return dataclasses.replace(
            base,
            name="ycsb-phase-aware-operation-atcc-table",
            atcc_module=PhaseAwareATCCModule.ycsb(),
            rl_enabled=False,
        )

    @classmethod
    def ycsb_strict_tuned_atcc(cls) -> "OperationPolicyTable":
        base = cls.ycsb_atcc()
        return dataclasses.replace(
            base,
            name="ycsb-strict-tuned-operation-atcc-table",
            atcc_module=PhaseAwareATCCModule.ycsb_strict_tuned(),
            rl_enabled=False,
            min_feedback_observations=8,
            exact_key_min_observations=3,
            conflict_abort_cost=1.4,
            lock_wait_cost_per_s=220.0,
            lock_queue_depth_cost=0.08,
            lock_overhead_cost=0.08,
            hysteresis=1.25,
            atcc_cold_occ_fast_path=False,
            atcc_fast_path_max_agent_interval_s=0.120,
            atcc_fast_path_max_global_abort_rate=0.03,
            atcc_fast_path_max_total_writes=1,
        )

    def with_learned_state(
        self,
        artifact: Mapping[str, Any],
        *,
        policy_epsilon: Optional[float] = None,
        load_runtime_stats: bool = False,
    ) -> "OperationPolicyTable":
        """Return this policy table with trained Q tables and telemetry loaded.

        Training artifacts are intentionally treated as learned state, not as a
        replacement for the local rule schema. That keeps the module compatible
        with the current data-agent runtime while applying the policy-table
        values and hot-object statistics learned offline.  Runtime EWMA signals
        default to a fresh window because ATCC's global abort/latency/lock-wait
        metrics are supposed to describe the current workload, not the training
        profile that produced the Q table.
        """

        table_data = dict(artifact.get("operation_policy_table", artifact))
        telemetry_alpha = float(getattr(self.telemetry, "alpha", 0.25))
        telemetry = OperationPolicyTelemetry.from_dict(
            table_data.get("telemetry", {}),
            alpha=telemetry_alpha,
        )
        rl_learner = self.rl_learner
        if isinstance(table_data.get("rl_learner"), Mapping):
            rl_learner = OperationPolicyQLearner.from_dict(table_data["rl_learner"])
        atcc_module = self.atcc_module
        module_data = table_data.get("atcc_module") or artifact.get("atcc_module")
        if isinstance(module_data, Mapping):
            atcc_module = PhaseAwareATCCModule.from_dict(module_data)
        atcc_runtime_stats = self.atcc_runtime_stats
        if load_runtime_stats:
            runtime_stats_data = (
                table_data.get("atcc_runtime_stats")
                or artifact.get("atcc_runtime_stats")
                or {}
            )
            atcc_runtime_stats = (
                ATCCRuntimeStats.from_dict(runtime_stats_data)
                if isinstance(runtime_stats_data, Mapping)
                else self.atcc_runtime_stats
            )
        if policy_epsilon is not None:
            epsilon = max(0.0, min(1.0, float(policy_epsilon)))
            rl_learner.epsilon = epsilon
            rl_learner.min_epsilon = epsilon
            if atcc_module is not None:
                atcc_module.learner.epsilon = epsilon
                atcc_module.learner.min_epsilon = epsilon
        return dataclasses.replace(
            self,
            telemetry=telemetry,
            rl_learner=rl_learner,
            atcc_module=atcc_module,
            atcc_runtime_stats=atcc_runtime_stats,
        )

    def select(
        self,
        candidates: Sequence[Any],
        *,
        read_object_ids: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[OperationPolicyDecision, ...]:
        profiles = profile_operations(
            candidates, read_object_ids=read_object_ids, metadata=metadata
        )
        return self.select_profiles(profiles)

    def select_agent_operations(
        self,
        candidates: Sequence[Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[OperationPolicyDecision, ...]:
        return self.select_profiles(
            profile_agent_operations(candidates, metadata=metadata)
        )

    def select_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        if self.atcc_module is not None:
            return self._select_phase_aware_profiles(profiles)
        decisions = []
        for profile in profiles:
            policy, rule_name = self._prior_decision(profile)
            object_class = operation_object_class(profile.object_id)
            profile_key = operation_profile_key(profile, object_class=object_class)
            exact_key = operation_profile_key(
                profile, object_class=object_class, exact_object=True
            )
            rl_state_key = self._rl_state_key_for_profile(
                profile,
                object_class=object_class,
                profile_key=profile_key,
                exact_key=exact_key,
            )
            rl_explore = False
            rl_q_optimistic = 0.0
            rl_q_pessimistic = 0.0
            optimistic_cost = 0.0
            pessimistic_cost = 0.0
            observations = 0
            if self.online_feedback:
                (
                    policy,
                    rule_name,
                    optimistic_cost,
                    pessimistic_cost,
                    observations,
                ) = self._feedback_decision(
                    profile,
                    policy,
                    rule_name,
                    profile_key,
                    exact_key,
                )
            if self.rl_enabled and rule_name not in self.pinned_rules:
                (
                    policy,
                    rl_explore,
                    rl_q_optimistic,
                    rl_q_pessimistic,
                ) = self.rl_learner.select(rl_state_key, prior_action=policy)
                rule_name = (
                    "rl-explore-" + policy if rl_explore else "rl-q-" + policy
                )
            decisions.append(
                OperationPolicyDecision(
                    object_id=profile.object_id,
                    access_kind=profile.access_kind,
                    intent_name=profile.intent_name,
                    policy=policy,
                    rule=rule_name,
                    task_type=profile.task_type,
                    workload=profile.workload,
                    candidate_count=profile.candidate_count,
                    operation_count_for_object=profile.operation_count_for_object,
                    total_writes=profile.total_writes,
                    retry_count=profile.retry_count,
                    agent_interval_s=profile.agent_interval_s,
                    agent_phase=profile.agent_phase,
                    object_class=object_class,
                    profile_key=profile_key,
                    exact_key=exact_key,
                    optimistic_cost=optimistic_cost,
                    pessimistic_cost=pessimistic_cost,
                    telemetry_observations=observations,
                    rl_state_key=rl_state_key,
                    rl_explore=rl_explore,
                    rl_q_optimistic=rl_q_optimistic,
                    rl_q_pessimistic=rl_q_pessimistic,
                )
            )
        return tuple(decisions)

    def _select_phase_aware_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        module = self.atcc_module
        if module is None:
            return ()
        fast_path = self._phase_atcc_fast_path_decisions(profiles)
        if fast_path:
            return fast_path
        decisions_by_class: Dict[str, PhaseAwareATCCDecision] = {}
        profiles_by_class: Dict[str, list[OperationPolicyProfile]] = {}
        for profile in profiles:
            profiles_by_class.setdefault(
                operation_object_class(profile.object_id),
                [],
            ).append(profile)
        for object_class, grouped_profiles in sorted(profiles_by_class.items()):
            decisions_by_class[object_class] = module.select(
                grouped_profiles,
                stats_for=self.telemetry.stats_for,
                object_class_for=operation_object_class,
                profile_key_for=operation_profile_key,
                runtime_stats=self.atcc_runtime_stats,
            )
        decisions = []
        for profile in profiles:
            object_class = operation_object_class(profile.object_id)
            class_decision = decisions_by_class[object_class]
            profile_key = operation_profile_key(profile, object_class=object_class)
            exact_key = operation_profile_key(
                profile,
                object_class=object_class,
                exact_object=True,
            )
            class_stats = self.telemetry.stats_for(profile_key)
            exact_stats = self.telemetry.stats_for(exact_key)
            policy = (
                "pessimistic"
                if module.should_lock(
                    profile,
                    class_decision,
                    class_stats=class_stats,
                    exact_stats=exact_stats,
                    object_class=object_class,
                )
                else "optimistic"
            )
            stats = exact_stats if exact_stats.observations else class_stats
            optimistic_cost = stats.ewma_conflict_rate * float(self.conflict_abort_cost)
            pessimistic_cost = (
                stats.ewma_lock_wait_s * float(self.lock_wait_cost_per_s)
                + stats.ewma_lock_queue_depth * float(self.lock_queue_depth_cost)
                + float(self.lock_overhead_cost)
            )
            decisions.append(
                OperationPolicyDecision(
                    object_id=profile.object_id,
                    access_kind=profile.access_kind,
                    intent_name=profile.intent_name,
                    policy=policy,
                    rule=f"phase-atcc-{class_decision.phase}-{class_decision.action}",
                    task_type=profile.task_type,
                    workload=profile.workload,
                    candidate_count=profile.candidate_count,
                    operation_count_for_object=profile.operation_count_for_object,
                    total_writes=profile.total_writes,
                    retry_count=profile.retry_count,
                    agent_interval_s=profile.agent_interval_s,
                    agent_phase=profile.agent_phase,
                    object_class=object_class,
                    profile_key=profile_key,
                    exact_key=exact_key,
                    optimistic_cost=optimistic_cost,
                    pessimistic_cost=pessimistic_cost,
                    telemetry_observations=stats.observations,
                    atcc_state_key=class_decision.state_key,
                    atcc_action=class_decision.action,
                    atcc_phase=class_decision.phase,
                    atcc_priority=class_decision.priority,
                    atcc_explore=class_decision.explore,
                    atcc_q_values=dict(class_decision.q_values),
                    atcc_global_abort_rate=class_decision.global_abort_rate,
                    atcc_global_lock_wait_s=class_decision.global_lock_wait_s,
                    atcc_global_latency_s=class_decision.global_latency_s,
                    atcc_global_lock_queue_depth=(
                        class_decision.global_lock_queue_depth
                    ),
                    atcc_global_lock_handoff_count=(
                        class_decision.global_lock_handoff_count
                    ),
                    atcc_global_committing_count=(
                        class_decision.global_committing_count
                    ),
                )
            )
        return tuple(decisions)

    def _phase_atcc_fast_path_decisions(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        if not self.atcc_cold_occ_fast_path or self.atcc_module is None:
            return ()
        rows = tuple(profiles)
        if not rows:
            return ()
        max_retry = max(int(profile.retry_count) for profile in rows)
        if max_retry > int(self.atcc_fast_path_max_retry_count):
            return ()
        max_interval_s = max(float(profile.agent_interval_s) for profile in rows)
        if max_interval_s > float(self.atcc_fast_path_max_agent_interval_s):
            return ()
        max_total_writes = max(int(profile.total_writes) for profile in rows)
        if max_total_writes > int(self.atcc_fast_path_max_total_writes):
            return ()
        runtime_stats = self.atcc_runtime_stats
        if (
            int(runtime_stats.observations) > 0
            and float(runtime_stats.ewma_abort_rate)
            > float(self.atcc_fast_path_max_global_abort_rate)
        ):
            return ()

        decisions: list[OperationPolicyDecision] = []
        for profile in rows:
            object_class = operation_object_class(profile.object_id)
            profile_key = operation_profile_key(profile, object_class=object_class)
            exact_key = operation_profile_key(
                profile,
                object_class=object_class,
                exact_object=True,
            )
            class_stats = self.telemetry.stats_for(profile_key)
            exact_stats = self.telemetry.stats_for(exact_key)
            if self._phase_atcc_fast_path_risk_seen(class_stats):
                return ()
            if self._phase_atcc_fast_path_risk_seen(exact_stats):
                return ()
            stats = exact_stats if exact_stats.observations else class_stats
            optimistic_cost = stats.ewma_conflict_rate * float(self.conflict_abort_cost)
            pessimistic_cost = (
                stats.ewma_lock_wait_s * float(self.lock_wait_cost_per_s)
                + stats.ewma_lock_queue_depth * float(self.lock_queue_depth_cost)
                + float(self.lock_overhead_cost)
            )
            decisions.append(
                OperationPolicyDecision(
                    object_id=profile.object_id,
                    access_kind=profile.access_kind,
                    intent_name=profile.intent_name,
                    policy="optimistic",
                    rule="phase-atcc-fastpath-occ",
                    task_type=profile.task_type,
                    workload=profile.workload,
                    candidate_count=profile.candidate_count,
                    operation_count_for_object=profile.operation_count_for_object,
                    total_writes=profile.total_writes,
                    retry_count=profile.retry_count,
                    agent_interval_s=profile.agent_interval_s,
                    agent_phase=profile.agent_phase,
                    object_class=object_class,
                    profile_key=profile_key,
                    exact_key=exact_key,
                    optimistic_cost=optimistic_cost,
                    pessimistic_cost=pessimistic_cost,
                    telemetry_observations=stats.observations,
                    atcc_state_key=(
                        f"fastpath=occ|workload={profile.workload}|"
                        f"task={profile.task_type}|class={object_class}|"
                        f"retry={max_retry}"
                    ),
                    atcc_action="occ",
                    atcc_phase="fastpath-occ",
                    atcc_priority=0,
                    atcc_explore=False,
                    atcc_global_abort_rate=float(
                        self.atcc_runtime_stats.ewma_abort_rate
                    ),
                    atcc_global_lock_wait_s=float(
                        self.atcc_runtime_stats.ewma_lock_wait_s
                    ),
                    atcc_global_latency_s=float(
                        self.atcc_runtime_stats.ewma_latency_s
                    ),
                    atcc_global_lock_queue_depth=float(
                        self.atcc_runtime_stats.ewma_lock_queue_depth
                    ),
                    atcc_global_lock_handoff_count=float(
                        self.atcc_runtime_stats.ewma_lock_handoff_count
                    ),
                    atcc_global_committing_count=float(
                        self.atcc_runtime_stats.ewma_committing_count
                    ),
                )
            )
        return tuple(decisions)

    def _phase_atcc_fast_path_risk_seen(self, stats: OperationPolicyStats) -> bool:
        module = self.atcc_module
        if module is None or int(stats.observations) <= 0:
            return False
        if int(stats.observations) < int(module.min_hot_observations):
            return False
        if float(stats.ewma_conflict_rate) >= float(module.hot_conflict_threshold):
            return True
        if float(stats.ewma_lock_wait_s) >= float(module.hot_lock_wait_threshold_s):
            return True
        return float(stats.ewma_lock_queue_depth) > float(
            self.atcc_fast_path_max_lock_queue_depth
        )

    def _rl_state_key_for_profile(
        self,
        profile: OperationPolicyProfile,
        *,
        object_class: str,
        profile_key: str,
        exact_key: str,
    ) -> str:
        source, observations, conflict, lock_wait = self._rl_history_buckets(
            profile_key,
            exact_key,
        )
        return operation_rl_state_key(
            profile,
            object_class=object_class,
            history_source=source,
            observation_bucket=observations,
            conflict_bucket=conflict,
            lock_wait_bucket=lock_wait,
        )

    def _rl_history_buckets(self, profile_key: str, exact_key: str) -> Tuple[str, str, str, str]:
        exact_stats = self.telemetry.stats_for(exact_key)
        class_stats = self.telemetry.stats_for(profile_key)
        if exact_stats.observations > 0:
            source = "exact"
            stats = exact_stats
        else:
            source = "class"
            stats = class_stats
        return (
            source,
            _bucket_count(stats.observations) if stats.observations else "0",
            _bucket_rate(stats.ewma_conflict_rate),
            _bucket_latency_s(stats.ewma_lock_wait_s),
        )

    def _prior_decision(self, profile: OperationPolicyProfile) -> Tuple[str, str]:
        policy = self.fallback_policy
        rule_name = "fallback"
        for rule in self.rules:
            if rule.matches(profile):
                policy = rule.target_policy
                rule_name = rule.name
                break
        return str(policy).strip().lower(), rule_name

    def _feedback_decision(
        self,
        profile: OperationPolicyProfile,
        prior_policy: str,
        prior_rule: str,
        profile_key: str,
        exact_key: str,
    ) -> Tuple[str, str, float, float, int]:
        if profile.access_kind == "read":
            return prior_policy, prior_rule, 0.0, 0.0, 0
        if prior_rule in self.pinned_rules:
            return prior_policy, prior_rule, 0.0, 0.0, 0

        class_stats = self.telemetry.stats_for(profile_key)
        exact_stats = self.telemetry.stats_for(exact_key)
        if exact_stats.observations >= self.exact_key_min_observations:
            stats = exact_stats
        elif prior_policy == "pessimistic":
            stats = class_stats
        else:
            return prior_policy, prior_rule, 0.0, 0.0, exact_stats.observations
        observations = stats.observations
        if observations < self.min_feedback_observations:
            return prior_policy, prior_rule, 0.0, 0.0, observations

        optimistic_cost = (
            stats.ewma_conflict_rate * float(self.conflict_abort_cost)
        )
        pessimistic_cost = (
            stats.ewma_lock_wait_s * float(self.lock_wait_cost_per_s)
            + stats.ewma_lock_queue_depth * float(self.lock_queue_depth_cost)
            + float(self.lock_overhead_cost)
        )
        if optimistic_cost > pessimistic_cost * float(self.hysteresis):
            return (
                "pessimistic",
                "feedback-conflict-risk-pessimistic",
                optimistic_cost,
                pessimistic_cost,
                observations,
            )
        if pessimistic_cost > optimistic_cost * float(self.hysteresis):
            return (
                "optimistic",
                "feedback-lock-cost-optimistic",
                optimistic_cost,
                pessimistic_cost,
                observations,
            )
        return prior_policy, prior_rule, optimistic_cost, pessimistic_cost, observations

    def observe_result(
        self,
        decisions: Sequence[OperationPolicyDecision],
        *,
        committed: bool,
        rejected: bool,
        conflict_abort: bool,
        conflict_object_ids: Iterable[str] = (),
        lock_wait_s: float = 0.0,
        lock_wait_by_object: Optional[Mapping[str, float]] = None,
        lock_queue_by_object: Optional[Mapping[str, float]] = None,
        lock_handoff_by_object: Optional[Mapping[str, float]] = None,
        committing_count: float = 0.0,
        latency_s: float = 0.0,
    ) -> None:
        if not self.online_feedback:
            return
        conflict_targets = {str(object_id) for object_id in conflict_object_ids}
        object_waits = {
            str(object_id): max(0.0, float(wait_s))
            for object_id, wait_s in dict(lock_wait_by_object or {}).items()
        }
        object_queue_depths = {
            str(object_id): max(0.0, float(depth))
            for object_id, depth in dict(lock_queue_by_object or {}).items()
        }
        object_handoff_counts = {
            str(object_id): max(0.0, float(count))
            for object_id, count in dict(lock_handoff_by_object or {}).items()
        }
        pessimistic_count = sum(
            1 for decision in decisions if decision.policy == "pessimistic"
        )
        wait_per_pessimistic = (
            max(0.0, float(lock_wait_s)) / pessimistic_count
            if pessimistic_count
            else 0.0
        )
        phase_updates: Dict[Tuple[str, str], PhaseAwareATCCDecision] = {}
        queue_samples: list[float] = []
        handoff_samples: list[float] = []
        for decision in decisions:
            wait_for_decision = (
                object_waits.get(decision.object_id, wait_per_pessimistic)
                if decision.policy == "pessimistic"
                else 0.0
            )
            queue_for_decision = (
                object_queue_depths.get(decision.object_id, 0.0)
                if decision.policy == "pessimistic"
                else 0.0
            )
            handoff_for_decision = (
                object_handoff_counts.get(decision.object_id, 0.0)
                if decision.policy == "pessimistic"
                else 0.0
            )
            if decision.policy == "pessimistic":
                queue_samples.append(queue_for_decision)
                handoff_samples.append(handoff_for_decision)
            keys = (
                decision.profile_key
                or operation_profile_key(
                    OperationPolicyProfile(
                        object_id=decision.object_id,
                        access_kind=decision.access_kind,
                        intent_name=decision.intent_name,
                    ),
                    object_class=decision.object_class
                    or operation_object_class(decision.object_id),
                ),
                decision.exact_key,
            )
            self.telemetry.observe(
                keys,
                policy=decision.policy,
                conflict_abort=bool(
                    conflict_abort
                    and decision.policy == "optimistic"
                    and (
                        not conflict_targets
                        or decision.object_id in conflict_targets
                    )
                ),
                committed=bool(committed),
                rejected=bool(rejected),
                lock_wait_s=wait_for_decision,
                lock_queue_depth=queue_for_decision,
            )
            if self.rl_enabled and decision.rl_state_key:
                target_conflict = bool(
                    conflict_abort
                    and decision.policy == "optimistic"
                    and (
                        not conflict_targets
                        or decision.object_id in conflict_targets
                    )
                )
                reward = self._rl_reward(
                    decision.policy,
                    committed=bool(committed),
                    rejected=bool(rejected),
                    conflict_abort=target_conflict,
                    lock_wait_s=wait_for_decision,
                )
                self.rl_learner.update(
                    decision.rl_state_key,
                    decision.policy,
                    reward,
                )
                refreshed_profile = OperationPolicyProfile(
                    object_id=decision.object_id,
                    access_kind=decision.access_kind,
                    intent_name=decision.intent_name,
                    task_type=decision.task_type,
                    workload=decision.workload,
                    candidate_count=decision.candidate_count,
                    operation_count_for_object=decision.operation_count_for_object,
                    total_writes=decision.total_writes,
                    retry_count=decision.retry_count,
                    agent_interval_s=decision.agent_interval_s,
                    agent_phase=decision.agent_phase,
                )
                refreshed_state_key = self._rl_state_key_for_profile(
                    refreshed_profile,
                    object_class=decision.object_class
                    or operation_object_class(decision.object_id),
                    profile_key=decision.profile_key,
                    exact_key=decision.exact_key,
                )
                if refreshed_state_key != decision.rl_state_key:
                    self.rl_learner.update(
                        refreshed_state_key,
                        decision.policy,
                        reward,
                    )
            if self.atcc_module is not None and decision.atcc_state_key and decision.atcc_action:
                phase_updates.setdefault(
                    (decision.atcc_state_key, decision.atcc_action),
                    PhaseAwareATCCDecision(
                        state_key=decision.atcc_state_key,
                        action=decision.atcc_action,
                        phase=decision.atcc_phase,
                        priority=decision.atcc_priority,
                        explore=decision.atcc_explore,
                        q_values=dict(decision.atcc_q_values),
                        hot_read_ratio=0.0,
                        hot_write_ratio=0.0,
                        retry_count=int(decision.retry_count),
                        agent_interval_s=float(decision.agent_interval_s),
                        global_abort_rate=float(
                            getattr(decision, "atcc_global_abort_rate", 0.0)
                            or 0.0
                        ),
                        global_lock_wait_s=float(
                            getattr(decision, "atcc_global_lock_wait_s", 0.0)
                            or 0.0
                        ),
                        global_latency_s=float(
                            getattr(decision, "atcc_global_latency_s", 0.0)
                            or 0.0
                        ),
                        global_lock_queue_depth=float(
                            getattr(
                                decision,
                                "atcc_global_lock_queue_depth",
                                0.0,
                            )
                            or 0.0
                        ),
                        global_lock_handoff_count=float(
                            getattr(
                                decision,
                                "atcc_global_lock_handoff_count",
                                0.0,
                            )
                            or 0.0
                        ),
                        global_committing_count=float(
                            getattr(
                                decision,
                                "atcc_global_committing_count",
                                0.0,
                            )
                            or 0.0
                        ),
                    ),
                )
        if self.atcc_module is not None and phase_updates:
            lock_queue_depth = (
                sum(queue_samples) / len(queue_samples)
                if queue_samples
                else 0.0
            )
            lock_handoff_count = (
                sum(handoff_samples) / len(handoff_samples)
                if handoff_samples
                else 0.0
            )
            committing_pressure = max(0.0, float(committing_count))
            self.atcc_runtime_stats.observe(
                committed=bool(committed),
                rejected=bool(rejected),
                conflict_abort=bool(conflict_abort),
                lock_wait_s=max(0.0, float(lock_wait_s)),
                latency_s=max(0.0, float(latency_s)),
                lock_queue_depth=lock_queue_depth,
                lock_handoff_count=lock_handoff_count,
                committing_count=committing_pressure,
            )
            for phase_decision in phase_updates.values():
                self.atcc_module.update(
                    phase_decision,
                    committed=bool(committed),
                    rejected=bool(rejected),
                    conflict_abort=bool(conflict_abort),
                    lock_wait_s=max(0.0, float(lock_wait_s)),
                    operation_count=len(decisions),
                    lock_queue_depth=lock_queue_depth,
                    lock_handoff_count=lock_handoff_count,
                    committing_count=committing_pressure,
                )

    def _rl_reward(
        self,
        policy: str,
        *,
        committed: bool,
        rejected: bool,
        conflict_abort: bool,
        lock_wait_s: float,
    ) -> float:
        reward = float(self.rl_commit_reward) if committed else 0.0
        if rejected:
            reward -= float(self.rl_reject_penalty)
        if conflict_abort:
            reward -= float(self.rl_conflict_penalty)
        if policy == "pessimistic":
            reward -= float(self.rl_pessimistic_action_cost)
            reward -= max(0.0, float(lock_wait_s)) * float(self.rl_lock_wait_cost_per_s)
        return reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fallback_policy": self.fallback_policy,
            "rules": [rule.to_dict() for rule in self.rules],
            "online_feedback": self.online_feedback,
            "min_feedback_observations": self.min_feedback_observations,
            "exact_key_min_observations": self.exact_key_min_observations,
            "conflict_abort_cost": self.conflict_abort_cost,
            "lock_wait_cost_per_s": self.lock_wait_cost_per_s,
            "lock_queue_depth_cost": self.lock_queue_depth_cost,
            "lock_overhead_cost": self.lock_overhead_cost,
            "hysteresis": self.hysteresis,
            "pinned_rules": list(self.pinned_rules),
            "rl_enabled": self.rl_enabled,
            "rl_conflict_penalty": self.rl_conflict_penalty,
            "rl_commit_reward": self.rl_commit_reward,
            "rl_reject_penalty": self.rl_reject_penalty,
            "rl_lock_wait_cost_per_s": self.rl_lock_wait_cost_per_s,
            "rl_pessimistic_action_cost": self.rl_pessimistic_action_cost,
            "rl_learner": self.rl_learner.to_dict(),
            "atcc_module": self.atcc_module.to_dict() if self.atcc_module else None,
            "atcc_runtime_stats": self.atcc_runtime_stats.to_dict(),
            "atcc_cold_occ_fast_path": self.atcc_cold_occ_fast_path,
            "atcc_fast_path_max_retry_count": self.atcc_fast_path_max_retry_count,
            "atcc_fast_path_max_agent_interval_s": (
                self.atcc_fast_path_max_agent_interval_s
            ),
            "atcc_fast_path_max_global_abort_rate": (
                self.atcc_fast_path_max_global_abort_rate
            ),
            "atcc_fast_path_max_lock_queue_depth": (
                self.atcc_fast_path_max_lock_queue_depth
            ),
            "atcc_fast_path_max_total_writes": (
                self.atcc_fast_path_max_total_writes
            ),
            "telemetry": self.telemetry.to_dict(),
        }


def _ewma(previous: float, sample: float, alpha: float, observations: int) -> float:
    if observations <= 1:
        return float(sample)
    return float(alpha) * float(sample) + (1.0 - float(alpha)) * float(previous)


def _max_risk_stats(
    class_stats: OperationPolicyStats,
    exact_stats: OperationPolicyStats,
) -> OperationPolicyStats:
    if exact_stats.ewma_conflict_rate > class_stats.ewma_conflict_rate:
        return exact_stats
    return class_stats


def _normalize_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    return normalized if normalized in {"optimistic", "pessimistic"} else "optimistic"


def operation_object_class(object_id: str) -> str:
    parts = str(object_id).split(":")
    if len(parts) >= 4 and parts[0] == "tpcc":
        family = parts[1]
        field = parts[-1]
        return f"tpcc:{family}:{field}"
    if len(parts) >= 5 and parts[0] == "ycsb" and parts[1] == "record":
        return f"ycsb:field:{parts[-1]}"
    if len(parts) >= 2:
        return ":".join(parts[:2])
    return str(object_id)


def _bucket_count(value: int) -> str:
    count = int(value)
    if count <= 1:
        return "1"
    if count <= 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17+"


def _bucket_rate(value: float) -> str:
    rate = max(0.0, min(1.0, float(value)))
    if rate <= 0.0:
        return "0"
    if rate <= 0.05:
        return "0-5"
    if rate <= 0.20:
        return "5-20"
    if rate <= 0.50:
        return "20-50"
    return "50+"


def _bucket_latency_s(value: float) -> str:
    ms = max(0.0, float(value)) * 1000.0
    if ms <= 0.0:
        return "0ms"
    if ms <= 1.0:
        return "0-1ms"
    if ms <= 10.0:
        return "1-10ms"
    if ms <= 50.0:
        return "10-50ms"
    if ms <= 200.0:
        return "50-200ms"
    return "200ms+"


def operation_profile_key(
    profile: OperationPolicyProfile,
    *,
    object_class: Optional[str] = None,
    exact_object: bool = False,
) -> str:
    target = str(profile.object_id) if exact_object else (
        object_class or operation_object_class(profile.object_id)
    )
    return "|".join(
        (
            str(profile.workload),
            str(profile.task_type),
            str(profile.access_kind),
            str(profile.intent_name),
            target,
        )
    )


def operation_rl_state_key(
    profile: OperationPolicyProfile,
    *,
    object_class: Optional[str] = None,
    history_source: str = "none",
    observation_bucket: str = "0",
    conflict_bucket: str = "0",
    lock_wait_bucket: str = "0ms",
) -> str:
    return "|".join(
        (
            str(profile.workload),
            str(profile.task_type),
            str(profile.access_kind),
            str(profile.intent_name),
            object_class or operation_object_class(profile.object_id),
            "k=" + _bucket_count(profile.candidate_count),
            "op=" + _bucket_count(profile.operation_count_for_object),
            "writes=" + _bucket_count(profile.total_writes),
            "hist=" + str(history_source),
            "obs=" + str(observation_bucket),
            "conflict=" + str(conflict_bucket),
            "lockwait=" + str(lock_wait_bucket),
        )
    )


def profile_candidates(
    candidates: Sequence[Any],
    *,
    read_count: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AdaptiveTransactionProfile:
    metadata = dict(metadata or {})
    intent_counts: Counter[str] = Counter()
    write_targets = set()
    write_count = 0
    for candidate in candidates:
        for write in candidate._writes:
            write_count += 1
            write_targets.add(write.object_id)
            intent_counts[_intent_name(write.intent.intent_type)] += 1
    return AdaptiveTransactionProfile(
        candidate_count=len(candidates),
        read_count=int(read_count),
        write_count=write_count,
        intent_counts=dict(intent_counts),
        distinct_write_targets=len(write_targets),
        task_type=str(metadata.get("task_type", "")),
        workload=str(metadata.get("workload", "")),
    )


def _intent_name(intent_type: Any) -> str:
    return _INTENT_NAMES.get(intent_type, str(intent_type).split(".")[-1].lower())


def profile_operations(
    candidates: Sequence[Any],
    *,
    read_object_ids: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[OperationPolicyProfile, ...]:
    metadata = dict(metadata or {})
    retry_count, agent_interval_s, agent_phase = _agent_runtime_metadata(metadata)
    write_counts: Counter[str] = Counter()
    write_intents: Dict[str, str] = {}
    total_writes = 0
    for candidate in candidates:
        for write in candidate._writes:
            total_writes += 1
            write_counts[write.object_id] += 1
            write_intents.setdefault(write.object_id, _intent_name(write.intent.intent_type))

    profiles = []
    read_ids = set(read_object_ids)
    for object_id in sorted(read_ids - set(write_counts)):
        profiles.append(
            OperationPolicyProfile(
                object_id=object_id,
                access_kind="read",
                intent_name="read",
                task_type=str(metadata.get("task_type", "")),
                workload=str(metadata.get("workload", "")),
                candidate_count=len(candidates),
                operation_count_for_object=1,
                total_writes=total_writes,
                retry_count=retry_count,
                agent_interval_s=agent_interval_s,
                agent_phase=agent_phase,
            )
        )
    for object_id, count in sorted(write_counts.items()):
        profiles.append(
            OperationPolicyProfile(
                object_id=object_id,
                access_kind="write",
                intent_name=write_intents.get(object_id, "write"),
                task_type=str(metadata.get("task_type", "")),
                workload=str(metadata.get("workload", "")),
                candidate_count=len(candidates),
                operation_count_for_object=count,
                total_writes=total_writes,
                retry_count=retry_count,
                agent_interval_s=agent_interval_s,
                agent_phase=agent_phase,
            )
        )
    return tuple(profiles)



def profile_agent_operations(
    candidates: Sequence[Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[OperationPolicyProfile, ...]:
    """Profile workload operations before the transaction snapshot is read."""

    metadata = dict(metadata or {})
    retry_count, agent_interval_s, agent_phase = _agent_runtime_metadata(metadata)
    write_counts: Counter[str] = Counter()
    write_intents: Dict[str, str] = {}
    read_ids = set()
    total_writes = 0
    for candidate in candidates:
        for operation in candidate.operations:
            if operation.kind == "read":
                read_ids.add(operation.object_id)
                continue
            total_writes += 1
            write_counts[operation.object_id] += 1
            write_intents.setdefault(operation.object_id, operation.kind)

    profiles = []
    for object_id in sorted(read_ids - set(write_counts)):
        profiles.append(
            OperationPolicyProfile(
                object_id=object_id,
                access_kind="read",
                intent_name="read",
                task_type=str(metadata.get("task_type", "")),
                workload=str(metadata.get("workload", "")),
                candidate_count=len(candidates),
                operation_count_for_object=1,
                total_writes=total_writes,
                retry_count=retry_count,
                agent_interval_s=agent_interval_s,
                agent_phase=agent_phase,
            )
        )
    for object_id, count in sorted(write_counts.items()):
        profiles.append(
            OperationPolicyProfile(
                object_id=object_id,
                access_kind="write",
                intent_name=write_intents.get(object_id, "write"),
                task_type=str(metadata.get("task_type", "")),
                workload=str(metadata.get("workload", "")),
                candidate_count=len(candidates),
                operation_count_for_object=count,
                total_writes=total_writes,
                retry_count=retry_count,
                agent_interval_s=agent_interval_s,
                agent_phase=agent_phase,
            )
        )
    return tuple(profiles)


def _agent_runtime_metadata(metadata: Mapping[str, Any]) -> Tuple[int, float, str]:
    context = metadata.get("context", {})
    if not isinstance(context, Mapping):
        context = {}
    retry_count = metadata.get("retry_count", context.get("retry_count", 0))
    agent_interval_s = metadata.get(
        "agent_interval_s",
        context.get("agent_interval_s", context.get("interaction_latency_s", 0.0)),
    )
    agent_phase = metadata.get("agent_phase", context.get("agent_phase", ""))
    return int(retry_count or 0), float(agent_interval_s or 0.0), str(agent_phase or "")
