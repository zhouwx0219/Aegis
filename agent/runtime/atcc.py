"""Phase-aware ATCC policy module for data-agent transactions.

The original ATCC paper runs inside openGauss.  This module keeps the same
shape of the mechanism for the data-agent runtime: infer an agentic phase,
bucket contention/cost signals, choose a locking-scope action from a compact
policy table, and train that table from commit feedback.
"""

from __future__ import annotations

import dataclasses
import math
import random
import threading
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class ATCCActionSpec:
    name: str
    lock_reads: bool = False
    lock_writes: bool = False
    hot_only: bool = True
    priority_boost: bool = False
    description: str = ""


@dataclasses.dataclass(frozen=True)
class PhaseAwareATCCDecision:
    state_key: str
    action: str
    phase: str
    priority: int
    explore: bool
    q_values: Mapping[str, float]
    hot_read_ratio: float
    hot_write_ratio: float
    retry_count: int
    agent_interval_s: float
    global_abort_rate: float = 0.0
    global_lock_wait_s: float = 0.0
    global_latency_s: float = 0.0
    global_lock_queue_depth: float = 0.0
    global_lock_handoff_count: float = 0.0
    global_committing_count: float = 0.0


@dataclasses.dataclass
class ATCCRuntimeStats:
    """Global runtime signals used by the phase-aware ATCC policy table."""

    alpha: float = 0.20
    observations: int = 0
    committed: int = 0
    aborted: int = 0
    rejected: int = 0
    ewma_abort_rate: float = 0.0
    ewma_lock_wait_s: float = 0.0
    ewma_latency_s: float = 0.0
    ewma_lock_queue_depth: float = 0.0
    ewma_lock_handoff_count: float = 0.0
    ewma_committing_count: float = 0.0

    def observe(
        self,
        *,
        committed: bool,
        rejected: bool,
        conflict_abort: bool,
        lock_wait_s: float,
        latency_s: float,
        lock_queue_depth: float = 0.0,
        lock_handoff_count: float = 0.0,
        committing_count: float = 0.0,
    ) -> None:
        self.observations += 1
        if committed:
            self.committed += 1
        elif rejected:
            self.rejected += 1
        else:
            self.aborted += 1
        self.ewma_abort_rate = _ewma(
            self.ewma_abort_rate,
            1.0 if conflict_abort else 0.0,
            self.alpha,
            self.observations,
        )
        self.ewma_lock_wait_s = _ewma(
            self.ewma_lock_wait_s,
            max(0.0, float(lock_wait_s)),
            self.alpha,
            self.observations,
        )
        self.ewma_latency_s = _ewma(
            self.ewma_latency_s,
            max(0.0, float(latency_s)),
            self.alpha,
            self.observations,
        )
        self.ewma_lock_queue_depth = _ewma(
            self.ewma_lock_queue_depth,
            max(0.0, float(lock_queue_depth)),
            self.alpha,
            self.observations,
        )
        self.ewma_lock_handoff_count = _ewma(
            self.ewma_lock_handoff_count,
            max(0.0, float(lock_handoff_count)),
            self.alpha,
            self.observations,
        )
        self.ewma_committing_count = _ewma(
            self.ewma_committing_count,
            max(0.0, float(committing_count)),
            self.alpha,
            self.observations,
        )

    def state_buckets(self) -> Tuple[str, str, str, str, str, str, str]:
        return (
            _bucket_count(self.observations),
            _bucket_rate(self.ewma_abort_rate),
            _bucket_latency_s(self.ewma_lock_wait_s),
            _bucket_latency_s(self.ewma_latency_s),
            _bucket_count(int(round(self.ewma_lock_queue_depth))),
            _bucket_count(int(round(self.ewma_lock_handoff_count))),
            _bucket_count(int(round(self.ewma_committing_count))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ATCCRuntimeStats":
        return cls(
            alpha=float(data.get("alpha", 0.20)),
            observations=int(data.get("observations", 0)),
            committed=int(data.get("committed", 0)),
            aborted=int(data.get("aborted", 0)),
            rejected=int(data.get("rejected", 0)),
            ewma_abort_rate=float(data.get("ewma_abort_rate", 0.0)),
            ewma_lock_wait_s=float(data.get("ewma_lock_wait_s", 0.0)),
            ewma_latency_s=float(data.get("ewma_latency_s", 0.0)),
            ewma_lock_queue_depth=float(data.get("ewma_lock_queue_depth", 0.0)),
            ewma_lock_handoff_count=float(
                data.get("ewma_lock_handoff_count", 0.0)
            ),
            ewma_committing_count=float(data.get("ewma_committing_count", 0.0)),
        )


class ATCCPolicyQLearner:
    """Tabular epsilon-greedy Q-learning over ATCC lock-scope actions."""

    def __init__(
        self,
        actions: Sequence[str],
        *,
        learning_rate: float = 0.20,
        discount: float = 0.0,
        epsilon: float = 0.12,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.999,
        seed: int = 0,
    ):
        if not actions:
            raise ValueError("actions must not be empty")
        if len(set(actions)) != len(tuple(actions)):
            raise ValueError("actions must be unique")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= min_epsilon <= epsilon <= 1:
            raise ValueError("epsilon must satisfy 0 <= min_epsilon <= epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.actions = tuple(str(action) for action in actions)
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

    def select(self, state_key: str, *, prior_action: str) -> Tuple[str, bool, Dict[str, float]]:
        prior = str(prior_action)
        with self._lock:
            values = self._q.setdefault(
                state_key,
                {action: 0.0 for action in self.actions},
            )
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
            return action, bool(explore), dict(values)

    def q_values(self, state_key: str) -> Dict[str, float]:
        with self._lock:
            values = self._q.setdefault(
                state_key,
                {action: 0.0 for action in self.actions},
            )
            return dict(values)

    def update(self, state_key: str, action: str, reward: float, *, next_state_key: str = "") -> None:
        normalized = str(action)
        if normalized not in self.actions:
            return
        with self._lock:
            values = self._q.setdefault(
                state_key,
                {name: 0.0 for name in self.actions},
            )
            current = float(values.get(normalized, 0.0))
            next_best = 0.0
            if next_state_key:
                next_values = self._q.setdefault(
                    next_state_key,
                    {name: 0.0 for name in self.actions},
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
                "actions": list(self.actions),
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
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        fallback_actions: Sequence[str] = (),
    ) -> "ATCCPolicyQLearner":
        actions = tuple(str(action) for action in data.get("actions", ()) or fallback_actions)
        min_epsilon = float(data.get("min_epsilon", 0.0))
        learner = cls(
            actions,
            learning_rate=float(data.get("learning_rate", 0.20)),
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


class PhaseAwareATCCModule:
    """ATCC policy-table module adapted from openGauss to data-agent traces."""

    ACTIONS: Tuple[ATCCActionSpec, ...] = (
        ATCCActionSpec(
            "occ",
            description="Remain optimistic; do not acquire pre-snapshot locks.",
        ),
        ATCCActionSpec(
            "lock-hot-writes",
            lock_writes=True,
            hot_only=True,
            description="Lock hot write-set objects only.",
        ),
        ATCCActionSpec(
            "lock-hot-read-write",
            lock_reads=True,
            lock_writes=True,
            hot_only=True,
            priority_boost=True,
            description="Protect hot read/write objects and boost priority.",
        ),
        ATCCActionSpec(
            "lock-write-set",
            lock_writes=True,
            hot_only=False,
            priority_boost=True,
            description="Lock the full write set for high abort-cost commit phases.",
        ),
        ATCCActionSpec(
            "lock-read-write-set",
            lock_reads=True,
            lock_writes=True,
            hot_only=False,
            priority_boost=True,
            description="Lock full read/write set for pathological retry loops.",
        ),
    )

    def __init__(
        self,
        *,
        name: str,
        learner: Optional[ATCCPolicyQLearner] = None,
        hot_conflict_threshold: float = 0.20,
        hot_lock_wait_threshold_s: float = 0.050,
        min_hot_observations: int = 2,
        commit_reward: float = 1.0,
        abort_penalty: float = 2.0,
        reject_penalty: float = 0.5,
        lock_wait_cost_per_s: float = 80.0,
        lock_queue_depth_cost: float = 0.05,
        lock_handoff_cost: float = 0.03,
        committing_count_cost: float = 0.005,
        lock_action_cost: float = 0.02,
        interval_cost_per_s: float = 1.0,
        ycsb_tuned_prior: bool = False,
    ):
        self.name = str(name)
        self.action_specs = {spec.name: spec for spec in self.ACTIONS}
        self.learner = learner or ATCCPolicyQLearner(
            tuple(self.action_specs),
            learning_rate=0.20,
            epsilon=0.12,
            min_epsilon=0.01,
            epsilon_decay=0.999,
            seed=17,
        )
        self.hot_conflict_threshold = float(hot_conflict_threshold)
        self.hot_lock_wait_threshold_s = float(hot_lock_wait_threshold_s)
        self.min_hot_observations = int(min_hot_observations)
        self.commit_reward = float(commit_reward)
        self.abort_penalty = float(abort_penalty)
        self.reject_penalty = float(reject_penalty)
        self.lock_wait_cost_per_s = float(lock_wait_cost_per_s)
        self.lock_queue_depth_cost = float(lock_queue_depth_cost)
        self.lock_handoff_cost = float(lock_handoff_cost)
        self.committing_count_cost = float(committing_count_cost)
        self.lock_action_cost = float(lock_action_cost)
        self.interval_cost_per_s = float(interval_cost_per_s)
        self.ycsb_tuned_prior = bool(ycsb_tuned_prior)

    @classmethod
    def tpcc(cls) -> "PhaseAwareATCCModule":
        return cls(
            name="tpcc-phase-aware-atcc",
            learner=ATCCPolicyQLearner(
                tuple(spec.name for spec in cls.ACTIONS),
                learning_rate=0.25,
                epsilon=0.14,
                min_epsilon=0.02,
                epsilon_decay=0.9995,
                seed=2603,
            ),
            hot_conflict_threshold=0.15,
            hot_lock_wait_threshold_s=0.080,
            min_hot_observations=2,
            abort_penalty=3.0,
            lock_wait_cost_per_s=70.0,
            interval_cost_per_s=2.0,
        )

    @classmethod
    def ycsb(cls) -> "PhaseAwareATCCModule":
        return cls(
            name="ycsb-phase-aware-atcc",
            learner=ATCCPolicyQLearner(
                tuple(spec.name for spec in cls.ACTIONS),
                learning_rate=0.20,
                epsilon=0.10,
                min_epsilon=0.01,
                epsilon_decay=0.999,
                seed=13906,
            ),
            hot_conflict_threshold=0.25,
            hot_lock_wait_threshold_s=0.030,
            min_hot_observations=3,
            abort_penalty=1.5,
            lock_wait_cost_per_s=180.0,
            lock_action_cost=0.05,
            interval_cost_per_s=1.0,
        )

    @classmethod
    def ycsb_strict_tuned(cls) -> "PhaseAwareATCCModule":
        return cls(
            name="ycsb-strict-tuned-phase-aware-atcc",
            learner=ATCCPolicyQLearner(
                tuple(spec.name for spec in cls.ACTIONS),
                learning_rate=0.30,
                epsilon=0.0,
                min_epsilon=0.0,
                epsilon_decay=1.0,
                seed=13907,
            ),
            hot_conflict_threshold=0.12,
            hot_lock_wait_threshold_s=0.020,
            min_hot_observations=1,
            abort_penalty=2.2,
            lock_wait_cost_per_s=220.0,
            lock_queue_depth_cost=0.08,
            lock_handoff_cost=0.04,
            committing_count_cost=0.008,
            lock_action_cost=0.08,
            interval_cost_per_s=1.4,
            ycsb_tuned_prior=True,
        )

    def select(
        self,
        profiles: Sequence[Any],
        *,
        stats_for: Callable[[str], Any],
        object_class_for: Callable[[str], str],
        profile_key_for: Callable[..., str],
        runtime_stats: Optional[ATCCRuntimeStats] = None,
    ) -> PhaseAwareATCCDecision:
        rows = tuple(profiles)
        phase = self.infer_phase(rows)
        retry_count = max((int(getattr(profile, "retry_count", 0)) for profile in rows), default=0)
        interval_s = max(
            (float(getattr(profile, "agent_interval_s", 0.0)) for profile in rows),
            default=0.0,
        )
        hot_reads, reads, hot_writes, writes = self._hot_counts(
            rows,
            stats_for=stats_for,
            object_class_for=object_class_for,
            profile_key_for=profile_key_for,
        )
        object_classes = sorted(
            {
                str(object_class_for(str(getattr(profile, "object_id", ""))))
                for profile in rows
            }
        )
        hot_read_ratio = hot_reads / reads if reads else 0.0
        hot_write_ratio = hot_writes / writes if writes else 0.0
        priority = self.priority_score(
            profiles=rows,
            retry_count=retry_count,
            agent_interval_s=interval_s,
            hot_read_ratio=hot_read_ratio,
            hot_write_ratio=hot_write_ratio,
            global_abort_rate=(
                float(runtime_stats.ewma_abort_rate)
                if runtime_stats is not None
                else 0.0
            ),
        )
        prior = self.prior_action(
            profiles=rows,
            phase=phase,
            retry_count=retry_count,
            agent_interval_s=interval_s,
            hot_read_ratio=hot_read_ratio,
            hot_write_ratio=hot_write_ratio,
            global_abort_rate=(
                float(runtime_stats.ewma_abort_rate)
                if runtime_stats is not None
                else 0.0
            ),
        )
        state_key = self.state_key(
            profiles=rows,
            phase=phase,
            retry_count=retry_count,
            agent_interval_s=interval_s,
            hot_read_ratio=hot_read_ratio,
            hot_write_ratio=hot_write_ratio,
            priority=priority,
            runtime_stats=runtime_stats,
            object_classes=object_classes,
        )
        if self._should_force_occ_for_low_risk(
            profiles=rows,
            phase=phase,
            retry_count=retry_count,
            agent_interval_s=interval_s,
            hot_read_ratio=hot_read_ratio,
            hot_write_ratio=hot_write_ratio,
            runtime_stats=runtime_stats,
            prior_action=prior,
        ):
            return PhaseAwareATCCDecision(
                state_key=state_key,
                action="occ",
                phase=phase,
                priority=priority,
                explore=False,
                q_values=self.learner.q_values(state_key),
                hot_read_ratio=hot_read_ratio,
                hot_write_ratio=hot_write_ratio,
                retry_count=retry_count,
                agent_interval_s=interval_s,
                global_abort_rate=(
                    float(runtime_stats.ewma_abort_rate)
                    if runtime_stats is not None
                    else 0.0
                ),
                global_lock_wait_s=(
                    float(runtime_stats.ewma_lock_wait_s)
                    if runtime_stats is not None
                    else 0.0
                ),
                global_latency_s=(
                    float(runtime_stats.ewma_latency_s)
                    if runtime_stats is not None
                    else 0.0
                ),
                global_lock_queue_depth=(
                    float(runtime_stats.ewma_lock_queue_depth)
                    if runtime_stats is not None
                    else 0.0
                ),
                global_lock_handoff_count=(
                    float(runtime_stats.ewma_lock_handoff_count)
                    if runtime_stats is not None
                    else 0.0
                ),
                global_committing_count=(
                    float(runtime_stats.ewma_committing_count)
                    if runtime_stats is not None
                    else 0.0
                ),
            )
        action, explore, q_values = self.learner.select(state_key, prior_action=prior)
        if action != "occ" and self._should_override_lock_for_low_abort_pressure(
            phase=phase,
            retry_count=retry_count,
            runtime_stats=runtime_stats,
        ):
            action = "occ"
            explore = False
        return PhaseAwareATCCDecision(
            state_key=state_key,
            action=action,
            phase=phase,
            priority=priority,
            explore=explore,
            q_values=q_values,
            hot_read_ratio=hot_read_ratio,
            hot_write_ratio=hot_write_ratio,
            retry_count=retry_count,
            agent_interval_s=interval_s,
            global_abort_rate=(
                float(runtime_stats.ewma_abort_rate)
                if runtime_stats is not None
                else 0.0
            ),
            global_lock_wait_s=(
                float(runtime_stats.ewma_lock_wait_s)
                if runtime_stats is not None
                else 0.0
            ),
            global_latency_s=(
                float(runtime_stats.ewma_latency_s)
                if runtime_stats is not None
                else 0.0
            ),
            global_lock_queue_depth=(
                float(runtime_stats.ewma_lock_queue_depth)
                if runtime_stats is not None
                else 0.0
            ),
            global_lock_handoff_count=(
                float(runtime_stats.ewma_lock_handoff_count)
                if runtime_stats is not None
                else 0.0
            ),
            global_committing_count=(
                float(runtime_stats.ewma_committing_count)
                if runtime_stats is not None
                else 0.0
            ),
        )

    def should_lock(
        self,
        profile: Any,
        decision: PhaseAwareATCCDecision,
        *,
        class_stats: Any,
        exact_stats: Any,
        object_class: str,
    ) -> bool:
        spec = self.action_specs.get(decision.action, self.action_specs["occ"])
        access_kind = str(getattr(profile, "access_kind", ""))
        if access_kind == "read" and not spec.lock_reads:
            return False
        if access_kind == "write" and not spec.lock_writes:
            return False
        if access_kind not in {"read", "write"}:
            return False
        if not spec.hot_only:
            return True
        return self.is_hot_profile(
            profile,
            class_stats=class_stats,
            exact_stats=exact_stats,
            object_class=object_class,
        )

    def update(
        self,
        decision: PhaseAwareATCCDecision,
        *,
        committed: bool,
        rejected: bool,
        conflict_abort: bool,
        lock_wait_s: float,
        operation_count: int,
        lock_queue_depth: float = 0.0,
        lock_handoff_count: float = 0.0,
        committing_count: float = 0.0,
    ) -> float:
        reward = self.reward(
            decision.action,
            committed=committed,
            rejected=rejected,
            conflict_abort=conflict_abort,
            lock_wait_s=lock_wait_s,
            lock_queue_depth=lock_queue_depth,
            lock_handoff_count=lock_handoff_count,
            committing_count=committing_count,
            retry_count=decision.retry_count,
            agent_interval_s=decision.agent_interval_s,
            operation_count=operation_count,
        )
        self.learner.update(decision.state_key, decision.action, reward)
        return reward

    def reward(
        self,
        action: str,
        *,
        committed: bool,
        rejected: bool,
        conflict_abort: bool,
        lock_wait_s: float,
        retry_count: int,
        agent_interval_s: float,
        operation_count: int,
        lock_queue_depth: float = 0.0,
        lock_handoff_count: float = 0.0,
        committing_count: float = 0.0,
    ) -> float:
        value = self.commit_reward if committed else 0.0
        if rejected:
            value -= self.reject_penalty
        if conflict_abort:
            value -= self.abort_penalty * (1.0 + max(0, int(retry_count)))
            value -= max(0.0, float(agent_interval_s)) * self.interval_cost_per_s
        if action != "occ":
            value -= self.lock_action_cost
            value -= max(0.0, float(lock_wait_s)) * self.lock_wait_cost_per_s
            value -= max(0.0, float(lock_queue_depth)) * self.lock_queue_depth_cost
            value -= max(0.0, float(lock_handoff_count)) * self.lock_handoff_cost
            value -= max(0.0, float(committing_count)) * self.committing_count_cost
            value -= 0.001 * max(0, int(operation_count))
        return value

    def infer_phase(self, profiles: Sequence[Any]) -> str:
        explicit = {
            str(getattr(profile, "agent_phase", ""))
            for profile in profiles
            if str(getattr(profile, "agent_phase", ""))
        }
        if explicit:
            return sorted(explicit)[0]
        reads = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "read")
        writes = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "write")
        task_types = {str(getattr(profile, "task_type", "")) for profile in profiles}
        if writes == 0:
            return "explore"
        if task_types & {"new_order", "payment", "delivery"}:
            return "commit"
        if reads > writes * 2:
            return "refine"
        return "commit"

    def prior_action(
        self,
        *,
        profiles: Sequence[Any],
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        hot_read_ratio: float,
        hot_write_ratio: float,
        global_abort_rate: float = 0.0,
    ) -> str:
        if self.ycsb_tuned_prior:
            return self._ycsb_strict_tuned_prior_action(
                profiles=profiles,
                phase=phase,
                retry_count=retry_count,
                agent_interval_s=agent_interval_s,
                hot_read_ratio=hot_read_ratio,
                hot_write_ratio=hot_write_ratio,
                global_abort_rate=global_abort_rate,
            )
        writes = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "write")
        if phase == "explore":
            return "occ"
        if retry_count >= 3 and (hot_read_ratio or hot_write_ratio):
            return "lock-read-write-set"
        if retry_count >= 1 and hot_write_ratio >= 0.25:
            return "lock-write-set"
        if phase == "commit" and global_abort_rate >= 0.20 and hot_write_ratio > 0.0:
            return "lock-hot-writes"
        if agent_interval_s >= 0.10 and hot_read_ratio >= 0.25:
            return "lock-hot-read-write"
        if agent_interval_s >= 0.10 and hot_write_ratio > 0.0:
            return "lock-hot-writes"
        if writes >= 8 and phase == "commit" and global_abort_rate >= 0.10:
            return "lock-hot-writes"
        return "occ"

    def _ycsb_strict_tuned_prior_action(
        self,
        *,
        profiles: Sequence[Any],
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        hot_read_ratio: float,
        hot_write_ratio: float,
        global_abort_rate: float = 0.0,
    ) -> str:
        writes = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "write")
        reads = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "read")
        total_writes = max(
            (int(getattr(profile, "total_writes", 0) or 0) for profile in profiles),
            default=writes,
        )
        if phase == "explore":
            return "occ"
        if int(retry_count) <= 0:
            if hot_write_ratio >= 0.75 and max(0.0, float(global_abort_rate)) >= 0.25:
                return "lock-hot-writes"
            return "occ"
        if int(retry_count) >= 3 and (hot_read_ratio > 0.0 or hot_write_ratio > 0.0):
            return "lock-write-set"
        if int(retry_count) >= 2 and hot_write_ratio >= 0.50:
            return "lock-write-set"
        if hot_write_ratio > 0.0:
            return "lock-hot-writes"
        if (
            int(retry_count) >= 2
            and hot_read_ratio >= 0.50
            and (reads >= writes or max(0.0, float(agent_interval_s)) >= 0.10)
        ):
            return "lock-hot-read-write"
        if writes >= 8 and max(0.0, float(global_abort_rate)) >= 0.20:
            return "lock-hot-writes"
        return "occ"

    def priority_score(
        self,
        *,
        profiles: Sequence[Any],
        retry_count: int,
        agent_interval_s: float,
        hot_read_ratio: float,
        hot_write_ratio: float,
        global_abort_rate: float = 0.0,
    ) -> int:
        operation_count = len(tuple(profiles))
        interval_ms = max(0.0, float(agent_interval_s)) * 1000.0
        return int(
            operation_count // 4
            + max(0, int(retry_count)) * 5
            + min(10, int(interval_ms // 25))
            + math.ceil((hot_read_ratio + hot_write_ratio) * 5)
            + math.ceil(max(0.0, float(global_abort_rate)) * 5)
        )

    def _should_force_occ_for_low_risk(
        self,
        *,
        profiles: Sequence[Any],
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        hot_read_ratio: float,
        hot_write_ratio: float,
        runtime_stats: Optional[ATCCRuntimeStats],
        prior_action: str,
    ) -> bool:
        """Keep truly low-risk operations optimistic even with a stale Q table.

        Offline policy artifacts may be trained on high-contention profiles and
        then reused in a low-contention run.  The original ATCC design treats
        global runtime metrics as online signals; they should reflect the
        current workload window, not permanently force pessimistic actions from
        the training environment.  This guard preserves the paper's fast OCC
        path for cold, first-attempt, low-abort phases.
        """

        if str(prior_action) != "occ":
            return False
        if int(retry_count) > 0:
            return False
        if hot_read_ratio > 0.0 or hot_write_ratio > 0.0:
            return False
        global_abort = (
            float(runtime_stats.ewma_abort_rate)
            if runtime_stats is not None
            else 0.0
        )
        if global_abort >= 0.05:
            return False
        writes = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "write")
        reads = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "read")
        if phase == "explore":
            return True
        if max(0.0, float(agent_interval_s)) < 0.050:
            return True
        return writes == 0 and reads > 0

    def _should_override_lock_for_low_abort_pressure(
        self,
        *,
        phase: str,
        retry_count: int,
        runtime_stats: Optional[ATCCRuntimeStats],
    ) -> bool:
        if phase not in {"explore", "refine"}:
            return False
        if int(retry_count) > 0:
            return False
        if runtime_stats is None or int(runtime_stats.observations) < 4:
            return False
        if float(runtime_stats.ewma_abort_rate) >= 0.05:
            return False
        return (
            float(runtime_stats.ewma_lock_wait_s) >= 0.010
            or float(runtime_stats.ewma_lock_queue_depth) >= 1.0
            or float(runtime_stats.ewma_lock_handoff_count) >= 1.0
            or float(runtime_stats.ewma_committing_count) >= 1.0
        )

    def state_key(
        self,
        *,
        profiles: Sequence[Any],
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        hot_read_ratio: float,
        hot_write_ratio: float,
        priority: int,
        runtime_stats: Optional[ATCCRuntimeStats] = None,
        object_classes: Sequence[str] = (),
    ) -> str:
        workloads = sorted({str(getattr(profile, "workload", "")) for profile in profiles})
        task_types = sorted({str(getattr(profile, "task_type", "")) for profile in profiles})
        classes = tuple(str(value) for value in object_classes if str(value))
        reads = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "read")
        writes = sum(1 for profile in profiles if getattr(profile, "access_kind", "") == "write")
        intents = Counter(str(getattr(profile, "intent_name", "")) for profile in profiles)
        (
            global_obs,
            global_abort,
            global_lock_wait,
            global_latency,
            global_queue_depth,
            global_handoff,
            global_committing,
        ) = (
            runtime_stats.state_buckets()
            if runtime_stats is not None
            else ("0", "0", "0ms", "0ms", "0", "0", "0")
        )
        parts = [
            "workload=" + ",".join(workloads),
            "task=" + ",".join(task_types),
            "class=" + ",".join(classes),
            "phase=" + str(phase),
            "reads=" + _bucket_count(reads),
            "writes=" + _bucket_count(writes),
            "hotR=" + _bucket_rate(hot_read_ratio),
            "hotW=" + _bucket_rate(hot_write_ratio),
            "retry=" + _bucket_count(retry_count),
            "interval=" + _bucket_latency_s(agent_interval_s),
            "priority=" + _bucket_count(priority),
            "globalObs=" + global_obs,
            "globalAbort=" + global_abort,
            "globalLockWait=" + global_lock_wait,
            "globalLatency=" + global_latency,
        ]
        if (
            global_queue_depth != "0"
            or global_handoff != "0"
            or global_committing != "0"
        ):
            parts.extend(
                (
                    "globalQueueDepth=" + global_queue_depth,
                    "globalHandoff=" + global_handoff,
                    "globalCommitting=" + global_committing,
                )
            )
        parts.append(
            "intent="
            + ",".join(
                f"{name}:{_bucket_count(count)}"
                for name, count in sorted(intents.items())
            )
        )
        return "|".join(parts)

    def is_hot_profile(
        self,
        profile: Any,
        *,
        class_stats: Any,
        exact_stats: Any,
        object_class: str,
    ) -> bool:
        object_id = str(getattr(profile, "object_id", ""))
        if "next_order_id" in object_id:
            if float(getattr(profile, "agent_interval_s", 0.0) or 0.0) >= 0.050:
                return True
            if int(getattr(profile, "retry_count", 0) or 0) > 0:
                return True
        if str(getattr(profile, "agent_phase", "")) == "commit" and int(
            getattr(profile, "retry_count", 0) or 0
        ) >= 2:
            return True
        for stats in (exact_stats, class_stats):
            observations = int(getattr(stats, "observations", 0))
            conflict = float(getattr(stats, "ewma_conflict_rate", 0.0))
            lock_wait = float(getattr(stats, "ewma_lock_wait_s", 0.0))
            if (
                observations >= self.min_hot_observations
                and conflict >= self.hot_conflict_threshold
            ):
                return True
            if (
                observations >= self.min_hot_observations
                and lock_wait >= self.hot_lock_wait_threshold_s
                and conflict >= self.hot_conflict_threshold * 0.5
            ):
                return True
        if str(getattr(profile, "access_kind", "")) == "write" and object_class.startswith("ycsb:"):
            return int(getattr(exact_stats, "optimistic_conflicts", 0)) > 0
        return False

    def _hot_counts(
        self,
        profiles: Sequence[Any],
        *,
        stats_for: Callable[[str], Any],
        object_class_for: Callable[[str], str],
        profile_key_for: Callable[..., str],
    ) -> Tuple[int, int, int, int]:
        hot_reads = reads = hot_writes = writes = 0
        for profile in profiles:
            object_class = object_class_for(str(getattr(profile, "object_id", "")))
            profile_key = profile_key_for(profile, object_class=object_class)
            exact_key = profile_key_for(profile, object_class=object_class, exact_object=True)
            class_stats = stats_for(profile_key)
            exact_stats = stats_for(exact_key)
            hot = self.is_hot_profile(
                profile,
                class_stats=class_stats,
                exact_stats=exact_stats,
                object_class=object_class,
            )
            if getattr(profile, "access_kind", "") == "read":
                reads += 1
                hot_reads += int(hot)
            elif getattr(profile, "access_kind", "") == "write":
                writes += 1
                hot_writes += int(hot)
        return hot_reads, reads, hot_writes, writes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actions": [dataclasses.asdict(spec) for spec in self.ACTIONS],
            "hot_conflict_threshold": self.hot_conflict_threshold,
            "hot_lock_wait_threshold_s": self.hot_lock_wait_threshold_s,
            "min_hot_observations": self.min_hot_observations,
            "commit_reward": self.commit_reward,
            "abort_penalty": self.abort_penalty,
            "reject_penalty": self.reject_penalty,
            "lock_wait_cost_per_s": self.lock_wait_cost_per_s,
            "lock_queue_depth_cost": self.lock_queue_depth_cost,
            "lock_handoff_cost": self.lock_handoff_cost,
            "committing_count_cost": self.committing_count_cost,
            "lock_action_cost": self.lock_action_cost,
            "interval_cost_per_s": self.interval_cost_per_s,
            "ycsb_tuned_prior": self.ycsb_tuned_prior,
            "learner": self.learner.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseAwareATCCModule":
        action_names = tuple(spec.name for spec in cls.ACTIONS)
        learner_data = dict(data.get("learner", {}))
        learner = ATCCPolicyQLearner.from_dict(
            learner_data,
            fallback_actions=action_names,
        )
        return cls(
            name=str(data.get("name", "phase-aware-atcc")),
            learner=learner,
            hot_conflict_threshold=float(data.get("hot_conflict_threshold", 0.20)),
            hot_lock_wait_threshold_s=float(
                data.get("hot_lock_wait_threshold_s", 0.050)
            ),
            min_hot_observations=int(data.get("min_hot_observations", 2)),
            commit_reward=float(data.get("commit_reward", 1.0)),
            abort_penalty=float(data.get("abort_penalty", 2.0)),
            reject_penalty=float(data.get("reject_penalty", 0.5)),
            lock_wait_cost_per_s=float(data.get("lock_wait_cost_per_s", 80.0)),
            lock_queue_depth_cost=float(data.get("lock_queue_depth_cost", 0.05)),
            lock_handoff_cost=float(data.get("lock_handoff_cost", 0.03)),
            committing_count_cost=float(data.get("committing_count_cost", 0.005)),
            lock_action_cost=float(data.get("lock_action_cost", 0.02)),
            interval_cost_per_s=float(data.get("interval_cost_per_s", 1.0)),
            ycsb_tuned_prior=bool(data.get("ycsb_tuned_prior", False)),
        )


def _bucket_count(value: int) -> str:
    count = int(value)
    if count <= 0:
        return "0"
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
    if ms <= 10.0:
        return "0-10ms"
    if ms <= 50.0:
        return "10-50ms"
    if ms <= 200.0:
        return "50-200ms"
    if ms <= 1000.0:
        return "200ms-1s"
    return "1s+"


def _ewma(previous: float, sample: float, alpha: float, observations: int) -> float:
    if observations <= 1:
        return float(sample)
    return float(alpha) * float(sample) + (1.0 - float(alpha)) * float(previous)
