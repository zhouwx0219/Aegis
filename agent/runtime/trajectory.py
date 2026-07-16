"""Transaction-internal trajectories for offline ATCC policy training."""

from __future__ import annotations

import dataclasses
import math
import threading
from typing import Dict, List, Sequence

from .state_collector import PhaseAwareState, phase_aware_state_from_dict


@dataclasses.dataclass(frozen=True)
class PolicyTransition:
    txn_id: str
    state: PhaseAwareState
    action: int
    reward: float
    next_state: PhaseAwareState
    done: bool
    behavior_probability: float = 1.0
    source_id: str = ""
    commit_indicator: int = 0
    abort_indicator: int = 0
    operation_cost_ms: float = 0.0
    agent_cost_ms: float = 0.0
    retry_cost_ms: float = 0.0
    lock_wait_ms: float = 0.0
    new_lock_count: int = 0
    system_delta: float = 0.0
    behavior_action_probabilities: tuple[float, ...] = ()
    background_blocked_ms_caused: float = 0.0
    background_aborts_caused: int = 0
    agent_blocked_ms_caused: float = 0.0
    agent_aborts_caused: int = 0


@dataclasses.dataclass(frozen=True)
class PaperRewardConfig:
    commit_reward: float = 100.0
    abort_weight: float = 80.0
    retry_weight: float = 1.0
    lock_weight: float = 5.0
    system_weight: float = 10.0


@dataclasses.dataclass(frozen=True)
class _PendingDecision:
    state: PhaseAwareState
    action: int
    behavior_probability: float
    blocked_time_ms: float
    lock_count: int
    behavior_action_probabilities: tuple[float, ...] = ()
    background_blocked_ms_caused: float = 0.0
    background_aborts_caused: int = 0
    agent_blocked_ms_caused: float = 0.0
    agent_aborts_caused: int = 0


class TrajectoryCollector:
    def __init__(self):
        self._lock = threading.RLock()
        self._pending: Dict[str, _PendingDecision] = {}
        self._transitions: List[PolicyTransition] = []

    def decision(
        self,
        txn_id: str,
        state: PhaseAwareState,
        action: int,
        behavior_probability: float = 1.0,
        *,
        blocked_time_ms: float = 0.0,
        lock_count: int = 0,
        behavior_action_probabilities: Sequence[float] = (),
        background_blocked_ms_caused: float = 0.0,
        background_aborts_caused: int = 0,
        agent_blocked_ms_caused: float = 0.0,
        agent_aborts_caused: int = 0,
    ) -> None:
        with self._lock:
            previous = self._pending.get(str(txn_id))
            if previous is not None:
                system_delta = system_performance_delta(previous.state, state)
                self._transitions.append(
                    PolicyTransition(
                        str(txn_id),
                        previous.state,
                        previous.action,
                        system_delta,
                        state,
                        False,
                        previous.behavior_probability,
                        lock_wait_ms=max(0.0, float(blocked_time_ms) - previous.blocked_time_ms),
                        new_lock_count=max(0, int(lock_count) - previous.lock_count),
                        system_delta=system_delta,
                        behavior_action_probabilities=previous.behavior_action_probabilities,
                        background_blocked_ms_caused=max(
                            0.0,
                            float(background_blocked_ms_caused)
                            - previous.background_blocked_ms_caused,
                        ),
                        background_aborts_caused=max(
                            0,
                            int(background_aborts_caused) - previous.background_aborts_caused,
                        ),
                        agent_blocked_ms_caused=max(
                            0.0,
                            float(agent_blocked_ms_caused) - previous.agent_blocked_ms_caused,
                        ),
                        agent_aborts_caused=max(
                            0,
                            int(agent_aborts_caused) - previous.agent_aborts_caused,
                        ),
                    )
                )
            probability = float(behavior_probability)
            if not 0.0 < probability <= 1.0:
                raise ValueError("behavior probability must be in (0, 1]")
            self._pending[str(txn_id)] = _PendingDecision(
                state,
                int(action),
                probability,
                max(0.0, float(blocked_time_ms)),
                max(0, int(lock_count)),
                tuple(float(value) for value in behavior_action_probabilities),
                max(0.0, float(background_blocked_ms_caused)),
                max(0, int(background_aborts_caused)),
                max(0.0, float(agent_blocked_ms_caused)),
                max(0, int(agent_aborts_caused)),
            )

    def finish(
        self,
        txn_id: str,
        state: PhaseAwareState,
        *,
        committed: bool,
        operation_cost_ms: float,
        agent_cost_ms: float,
        retry_cost_ms: float = 0.0,
        blocked_time_ms: float = 0.0,
        lock_count: int = 0,
        background_blocked_ms_caused: float = 0.0,
        background_aborts_caused: int = 0,
        agent_blocked_ms_caused: float = 0.0,
        agent_aborts_caused: int = 0,
    ) -> None:
        with self._lock:
            previous = self._pending.pop(str(txn_id), None)
            if previous is not None:
                system_delta = system_performance_delta(previous.state, state)
                transition = PolicyTransition(
                    str(txn_id),
                    previous.state,
                    previous.action,
                    0.0,
                    state,
                    True,
                    previous.behavior_probability,
                    commit_indicator=int(bool(committed)),
                    abort_indicator=int(not committed),
                    operation_cost_ms=max(0.0, float(operation_cost_ms)),
                    agent_cost_ms=max(0.0, float(agent_cost_ms)),
                    retry_cost_ms=max(0.0, float(retry_cost_ms)),
                    lock_wait_ms=max(0.0, float(blocked_time_ms) - previous.blocked_time_ms),
                    new_lock_count=max(0, int(lock_count) - previous.lock_count),
                    system_delta=system_delta,
                    behavior_action_probabilities=previous.behavior_action_probabilities,
                    background_blocked_ms_caused=max(
                        0.0,
                        float(background_blocked_ms_caused)
                        - previous.background_blocked_ms_caused,
                    ),
                    background_aborts_caused=max(
                        0,
                        int(background_aborts_caused) - previous.background_aborts_caused,
                    ),
                    agent_blocked_ms_caused=max(
                        0.0,
                        float(agent_blocked_ms_caused) - previous.agent_blocked_ms_caused,
                    ),
                    agent_aborts_caused=max(
                        0,
                        int(agent_aborts_caused) - previous.agent_aborts_caused,
                    ),
                )
                self._transitions.append(
                    dataclasses.replace(
                        transition,
                        reward=paper_reward(transition, reward_normalization((transition,))),
                    )
                )

    def snapshot(self) -> tuple[PolicyTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._transitions.clear()


def system_performance_delta(previous: PhaseAwareState, current: PhaseAwareState) -> float:
    """Paper Delta-Psys term over logical agent tasks and conflict aborts."""
    throughput_scale = max(
        1.0,
        previous.global_agent_task_throughput,
        current.global_agent_task_throughput,
    )
    throughput_delta = (
        current.global_agent_task_throughput - previous.global_agent_task_throughput
    ) / throughput_scale
    latency_scale = max(
        1.0,
        previous.global_agent_task_tail_latency_ms,
        current.global_agent_task_tail_latency_ms,
    )
    latency_delta = (
        previous.global_agent_task_tail_latency_ms
        - current.global_agent_task_tail_latency_ms
    ) / latency_scale
    abort_delta = previous.global_conflict_abort_rate - current.global_conflict_abort_rate
    background_throughput_scale = max(
        1.0,
        previous.global_background_throughput,
        current.global_background_throughput,
    )
    background_throughput_delta = (
        current.global_background_throughput - previous.global_background_throughput
    ) / background_throughput_scale
    background_abort_delta = (
        previous.global_background_abort_rate - current.global_background_abort_rate
    )
    combined = max(
        -3.0,
        min(
            3.0,
            throughput_delta
            + latency_delta
            + abort_delta
            + background_throughput_delta
            + background_abort_delta,
        ),
    )
    return combined


def reward_normalization(
    transitions: Sequence[PolicyTransition],
) -> dict[str, float]:
    return {
        "operation_cost_ms": robust_scale(row.operation_cost_ms for row in transitions),
        "agent_cost_ms": robust_scale(row.agent_cost_ms for row in transitions),
        "retry_cost_ms": robust_scale(row.retry_cost_ms for row in transitions),
        "lock_wait_ms": robust_scale(row.lock_wait_ms for row in transitions),
        "new_lock_count": robust_scale(row.new_lock_count for row in transitions),
        "background_blocked_ms_caused": robust_scale(
            row.background_blocked_ms_caused for row in transitions
        ),
        "background_aborts_caused": robust_scale(
            row.background_aborts_caused for row in transitions
        ),
        "agent_blocked_ms_caused": robust_scale(
            row.agent_blocked_ms_caused for row in transitions
        ),
        "agent_aborts_caused": robust_scale(
            row.agent_aborts_caused for row in transitions
        ),
    }


def apply_paper_rewards(
    transitions: Sequence[PolicyTransition],
    config: PaperRewardConfig | None = None,
) -> tuple[list[PolicyTransition], dict[str, object]]:
    normalization = reward_normalization(transitions)
    reward_config = config or PaperRewardConfig()
    rows = [
        dataclasses.replace(
            transition,
            reward=paper_reward(transition, normalization, reward_config),
        )
        for transition in transitions
    ]
    return rows, {
        "formula": "beta1*commit-beta2*(1+C_restart+beta3*C_retry)*abort-lambda*C_lock+eta*delta_Psys",
        "config": dataclasses.asdict(reward_config),
        "normalization": normalization,
    }


def paper_reward(
    transition: PolicyTransition,
    normalization: dict[str, float],
    config: PaperRewardConfig | None = None,
) -> float:
    cfg = config or PaperRewardConfig()
    restart_cost = (
        transition.operation_cost_ms / normalization["operation_cost_ms"]
        + transition.agent_cost_ms / normalization["agent_cost_ms"]
    )
    retry_cost = transition.retry_cost_ms / normalization["retry_cost_ms"]
    lock_cost = (
        transition.lock_wait_ms / normalization["lock_wait_ms"]
        + transition.new_lock_count / normalization["new_lock_count"]
        + transition.background_blocked_ms_caused
        / normalization["background_blocked_ms_caused"]
        + transition.background_aborts_caused
        / normalization["background_aborts_caused"]
        + transition.agent_blocked_ms_caused
        / normalization["agent_blocked_ms_caused"]
        + transition.agent_aborts_caused
        / normalization["agent_aborts_caused"]
    )
    return (
        cfg.commit_reward * int(transition.commit_indicator)
        - cfg.abort_weight
        * (1.0 + restart_cost + cfg.retry_weight * retry_cost)
        * int(transition.abort_indicator)
        - cfg.lock_weight * lock_cost
        + cfg.system_weight * float(transition.system_delta)
    )


def robust_scale(values) -> float:
    rows = sorted(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
    if not rows:
        return 1.0
    index = max(0, min(len(rows) - 1, math.ceil(0.95 * len(rows)) - 1))
    return max(1.0, rows[index])


def policy_transition_from_dict(
    data: dict[str, object],
    *,
    source_id: str | None = None,
) -> PolicyTransition:
    row = dict(data)
    fields = {field.name for field in dataclasses.fields(PolicyTransition)}
    payload = {
        key: value
        for key, value in row.items()
        if key in fields and key not in {"state", "next_state"}
    }
    if source_id is not None:
        payload["source_id"] = str(source_id)
    if "behavior_action_probabilities" in payload:
        payload["behavior_action_probabilities"] = tuple(
            float(value) for value in payload["behavior_action_probabilities"] or ()
        )
    return PolicyTransition(
        state=phase_aware_state_from_dict(dict(row["state"])),
        next_state=phase_aware_state_from_dict(dict(row["next_state"])),
        **payload,
    )
