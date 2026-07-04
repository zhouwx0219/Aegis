"""ATCC ablation runner for static/dynamic and priority/no-priority variants."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agent.evaluation.aggregation import sample_stddev
from agent.evaluation.atcc_retry_experiment import (
    RetryRunSummary,
    _operation_policy,
    _run_one_retry,
    _transaction_atcc_policy,
    aggregate_retry_runs,
)
from agent.evaluation.reporting import render_atcc_ablation_markdown
from agent.evaluation.strategy_matrix import (
    ABLATION_VARIANTS,
    DEFAULT_ABLATION_BASELINES,
    STATIC_OPERATION_THRESHOLD32_WIDE_OVERWRITE_THRESHOLD,
    STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD,
    STATIC_PRESETS,
    STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD,
    STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD,
    AblationVariantSpec,
    ablation_variant_metadata,
    ablation_variant_spec,
    bucket_count,
    bucket_latency_s,
    coarse_interval_s,
    normalize_static_preset,
    priority_cap_arg,
    profile_name_from_workload,
    select_ablation_variants,
    select_named_values,
    split_csv,
    static_operation_wide_overwrite_threshold,
    static_transaction_wide_write_threshold,
    workload_kind_from_name,
)
from agent.evaluation.workload_factory import (
    build_profile_workload as shared_build_profile_workload,
)
from agent.runtime import (
    ATCCPolicyQLearner,
    ATCCRuntimeStats,
    OperationPolicyDecision,
    OperationPolicyProfile,
    OperationPolicyTable,
    PhaseAwareATCCModule,
    TransactionAwareATCCDecision,
    TransactionAwareATCCModule,
)
from agent.runtime.adaptive import operation_object_class, profile_agent_operations
from agent.workloads import AgentWorkload


DEFAULT_BASELINES: Tuple[str, ...] = DEFAULT_ABLATION_BASELINES
VALIDATION_SELECTION_MARGIN = 1.03
VALIDATION_ATTEMPT_MARGIN = 1.01
VALIDATION_ABORT_RATE_MARGIN = 0.005
PRIORITY_VALIDATION_ATTEMPT_MARGIN = 1.01
PRIORITY_VALIDATION_ABORT_RATE_MARGIN = 0.005
PRIORITY_VALIDATION_PRELOCK_WAIT_MARGIN = 1.05


def _normalize_static_preset(value: str) -> str:
    return normalize_static_preset(value)


def _static_operation_wide_overwrite_threshold(static_preset: str) -> int:
    return static_operation_wide_overwrite_threshold(static_preset)


def _static_transaction_wide_write_threshold(static_preset: str) -> int:
    return static_transaction_wide_write_threshold(static_preset)


class FrozenATCCPolicyQLearner:
    """Read-only ATCC Q table for trained-policy evaluation."""

    def __init__(self, learner: ATCCPolicyQLearner):
        self._learner = learner
        self.actions = tuple(getattr(learner, "actions", ()))
        self.learning_rate = float(getattr(learner, "learning_rate", 0.0))
        self.discount = float(getattr(learner, "discount", 0.0))
        self.epsilon = 0.0
        self.min_epsilon = 0.0
        self.epsilon_decay = 1.0
        if hasattr(learner, "epsilon"):
            learner.epsilon = 0.0
        if hasattr(learner, "min_epsilon"):
            learner.min_epsilon = 0.0

    def select(self, state_key: str, *, prior_action: str) -> Tuple[str, bool, Dict[str, float]]:
        action, _explore, values = self._learner.select(
            state_key,
            prior_action=prior_action,
        )
        return action, False, values

    def q_values(self, state_key: str) -> Dict[str, float]:
        return self._learner.q_values(state_key)

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = self._learner.to_dict()
        data["epsilon"] = 0.0
        data["min_epsilon"] = 0.0
        data["epsilon_decay"] = 1.0
        data["frozen"] = True
        return data


def _cap_priority_score(value: int, priority_cap: Optional[int]) -> int:
    score = max(0, int(value))
    if priority_cap is None:
        return score
    cap = max(0, int(priority_cap))
    if cap <= 0:
        return 0
    return min(score, cap)


def _learner_action_visits(learner: Any, state_key: str, action: str) -> int:
    inner = getattr(learner, "_learner", learner)
    visits = getattr(inner, "_visits", {})
    try:
        return int(visits.get((str(state_key), str(action)), 0))
    except AttributeError:
        return 0


def _operation_decision_value(
    decision: OperationPolicyDecision,
    policy: str,
) -> float:
    normalized = str(policy)
    if decision.atcc_q_values:
        action = "occ" if normalized == "optimistic" else "lock-hot-writes"
        if action not in decision.atcc_q_values and normalized == "pessimistic":
            pessimistic_values = [
                float(value)
                for action_name, value in dict(decision.atcc_q_values).items()
                if action_name != "occ"
            ]
            return max(pessimistic_values) if pessimistic_values else 0.0
        return float(dict(decision.atcc_q_values).get(action, 0.0))
    if normalized == "optimistic":
        return float(decision.rl_q_optimistic)
    if normalized == "pessimistic":
        return float(decision.rl_q_pessimistic)
    return 0.0


def _operation_decision_visits(
    policy: OperationPolicyTable,
    decision: OperationPolicyDecision,
    selected_policy: str,
) -> int:
    if decision.atcc_state_key:
        action = str(decision.atcc_action or "")
        return _learner_action_visits(
            getattr(policy.atcc_module, "learner", None),
            decision.atcc_state_key,
            action,
        )
    if decision.rl_state_key:
        return _learner_action_visits(
            policy.rl_learner,
            decision.rl_state_key,
            str(selected_policy),
        )
    return 1


def _ablation_object_lock_scheduler(
    spec: AblationVariantSpec,
    profile_name: str,
) -> str:
    """Use bounded queue priority only when the variant enables priority.

    `bounded-priority` is now queue ordering only: it no longer wounds the
    current owner.  That makes it appropriate for medium contention too, while
    non-priority variants keep the plain race scheduler as the control.
    """

    if not spec.priority:
        return "race"
    _ = profile_name
    return "bounded-priority"


class StaticOperationATCCPolicy(OperationPolicyTable):
    """Static threshold operation ATCC used only by the ablation runner."""

    def __init__(
        self,
        workload_kind: str,
        *,
        priority_enabled: bool = False,
        priority_cap: Optional[int] = 1,
        static_preset: str = "conservative",
    ):
        workload = str(workload_kind).strip().lower()
        if workload not in {"ycsb", "tpcc"}:
            raise ValueError(f"unsupported workload kind: {workload_kind}")
        preset = _normalize_static_preset(static_preset)
        object.__setattr__(self, "workload_kind", workload)
        object.__setattr__(self, "priority_enabled", bool(priority_enabled))
        object.__setattr__(self, "priority_cap", priority_cap)
        object.__setattr__(self, "static_preset", preset)
        suffix = "priority" if self.priority_enabled else "no-priority"
        super().__init__(
            rules=(),
            fallback_policy="optimistic",
            name=f"{workload}-static-{preset}-{suffix}-operation-atcc-table",
            online_feedback=False,
        )

    def select_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        decisions = []
        for profile in profiles:
            policy, reason = self._static_policy(profile)
            priority = (
                _cap_priority_score(self._priority(profile), self.priority_cap)
                if self.priority_enabled
                else 0
            )
            object_class = operation_object_class(profile.object_id)
            action = "lock-hot-writes" if policy == "pessimistic" else "occ"
            phase = str(profile.agent_phase or "commit")
            decisions.append(
                OperationPolicyDecision(
                    object_id=profile.object_id,
                    access_kind=profile.access_kind,
                    intent_name=profile.intent_name,
                    policy=policy,
                    rule=f"ablation-op-static-{reason}",
                    task_type=profile.task_type,
                    workload=profile.workload,
                    candidate_count=profile.candidate_count,
                    operation_count_for_object=profile.operation_count_for_object,
                    total_writes=profile.total_writes,
                    retry_count=profile.retry_count,
                    agent_interval_s=profile.agent_interval_s,
                    agent_phase=profile.agent_phase,
                    object_class=object_class,
                    profile_key="|".join(
                        (
                            "scope=operation-static",
                            f"preset={self.static_preset}",
                            f"workload={profile.workload}",
                            f"task={profile.task_type}",
                            f"access={profile.access_kind}",
                            f"intent={profile.intent_name}",
                            f"class={object_class}",
                        )
                    ),
                    exact_key=f"object={profile.object_id}",
                    atcc_state_key="|".join(
                        (
                            "scope=operation-static",
                            f"preset={self.static_preset}",
                            f"workload={profile.workload}",
                            f"task={profile.task_type}",
                            f"phase={phase}",
                            f"writes={_bucket_count(profile.total_writes)}",
                            f"retry={_bucket_count(profile.retry_count)}",
                            f"priority={_bucket_count(priority)}",
                        )
                    ),
                    atcc_action=action,
                    atcc_phase=phase,
                    atcc_priority=priority,
                    atcc_explore=False,
                )
            )
        return tuple(decisions)

    def _static_policy(self, profile: OperationPolicyProfile) -> Tuple[str, str]:
        if profile.access_kind == "read":
            return "optimistic", "read-optimistic"
        if self.static_preset == "naive":
            if (
                profile.total_writes
                >= _static_operation_wide_overwrite_threshold(self.static_preset)
            ):
                return "pessimistic", "naive-wide-write-set"
            return "optimistic", "naive-cold-write-optimistic"
        if profile.operation_count_for_object >= 2:
            return "pessimistic", "repeated-object"
        if profile.retry_count > 0:
            return "pessimistic", "retry"
        if profile.intent_name in {"append", "delta", "cas"}:
            return "optimistic", "semantic-write-optimistic"
        if (
            profile.intent_name == "overwrite"
            and profile.total_writes
            >= _static_operation_wide_overwrite_threshold(self.static_preset)
        ):
            return "pessimistic", "wide-overwrite-set"
        return "optimistic", "cold-write-optimistic"

    @staticmethod
    def _priority(profile: OperationPolicyProfile) -> int:
        interval_ms = max(0.0, float(profile.agent_interval_s)) * 1000.0
        return int(
            max(0, int(profile.total_writes)) // 4
            + max(0, int(profile.retry_count)) * 5
            + min(10, int(interval_ms // 25))
            + max(0, int(profile.operation_count_for_object) - 1)
        )


class _NoPriorityPhaseAwareATCCModule(PhaseAwareATCCModule):
    def priority_score(self, **_kwargs: Any) -> int:
        return 0


class _CompactStatePhaseAwareATCCModule(PhaseAwareATCCModule):
    """Ablation-only operation module with a coarser train/test state key."""

    def state_key(self, *args: Any, **kwargs: Any) -> str:
        profiles = tuple(kwargs.get("profiles", ()) or ())
        workload = ",".join(
            sorted({str(getattr(profile, "workload", "")) for profile in profiles})
        )
        task = ",".join(
            sorted({str(getattr(profile, "task_type", "")) for profile in profiles})
        )
        object_groups = sorted(
            {
                _compact_operation_group(str(getattr(profile, "object_id", "")))
                for profile in profiles
            }
        )
        reads = sum(
            1 for profile in profiles if str(getattr(profile, "access_kind", "")) == "read"
        )
        writes = len(profiles) - reads
        runtime_stats = kwargs.get("runtime_stats")
        abort_rate = (
            float(getattr(runtime_stats, "ewma_abort_rate", 0.0) or 0.0)
            if runtime_stats is not None
            else 0.0
        )
        lock_wait_s = (
            float(getattr(runtime_stats, "ewma_lock_wait_s", 0.0) or 0.0)
            if runtime_stats is not None
            else 0.0
        )
        queue_depth = (
            float(getattr(runtime_stats, "ewma_lock_queue_depth", 0.0) or 0.0)
            if runtime_stats is not None
            else 0.0
        )
        return "|".join(
            (
                "scope=operation-ablation-dynamic-compact",
                f"workload={workload}",
                f"task={task}",
                "group=" + ",".join(object_groups),
                "phase=" + str(kwargs.get("phase", "")),
                f"reads={_bucket_count(reads)}",
                f"writes={_bucket_count(writes)}",
                "hotR=" + _compact_hot_bucket(kwargs.get("hot_read_ratio", 0.0)),
                "hotW=" + _compact_hot_bucket(kwargs.get("hot_write_ratio", 0.0)),
                f"retry={_bucket_count(int(kwargs.get('retry_count', 0) or 0))}",
                "pressure="
                + _compact_pressure_bucket(
                    abort_rate=abort_rate,
                    lock_wait_s=lock_wait_s,
                    queue_depth=queue_depth,
                ),
            )
        )


class _NoPriorityCompactStatePhaseAwareATCCModule(
    _CompactStatePhaseAwareATCCModule
):
    def priority_score(self, **_kwargs: Any) -> int:
        return 0


class _CappedPriorityCompactStatePhaseAwareATCCModule(
    _CompactStatePhaseAwareATCCModule
):
    def __init__(self, *args: Any, priority_cap: Optional[int] = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.priority_cap = priority_cap

    def state_key(self, *args: Any, **kwargs: Any) -> str:
        kwargs["priority"] = 0
        return super().state_key(*args, **kwargs)

    def priority_score(self, **kwargs: Any) -> int:
        retry_count = int(kwargs.get("retry_count", 0) or 0)
        global_abort_rate = max(0.0, float(kwargs.get("global_abort_rate", 0.0) or 0.0))
        hot_read_ratio = max(0.0, float(kwargs.get("hot_read_ratio", 0.0) or 0.0))
        hot_write_ratio = max(0.0, float(kwargs.get("hot_write_ratio", 0.0) or 0.0))
        profiles = tuple(kwargs.get("profiles", ()) or ())
        operation_count = len(profiles)
        total_writes = max(
            (int(getattr(profile, "total_writes", 0) or 0) for profile in profiles),
            default=0,
        )
        has_named_hotspot = any(
            "next_order_id" in str(getattr(profile, "object_id", ""))
            for profile in profiles
        )
        if (
            retry_count <= 0
            and global_abort_rate < 0.15
            and hot_read_ratio <= 0.0
            and hot_write_ratio <= 0.0
            and operation_count < 8
            and total_writes < 8
            and not has_named_hotspot
        ):
            return 0
        return _cap_priority_score(
            super().priority_score(**kwargs),
            self.priority_cap,
        )


class _CappedPriorityPhaseAwareATCCModule(PhaseAwareATCCModule):
    def __init__(self, *args: Any, priority_cap: Optional[int] = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.priority_cap = priority_cap

    def state_key(self, *args: Any, **kwargs: Any) -> str:
        kwargs["priority"] = 0
        return super().state_key(*args, **kwargs)

    def priority_score(self, **kwargs: Any) -> int:
        retry_count = int(kwargs.get("retry_count", 0) or 0)
        global_abort_rate = max(0.0, float(kwargs.get("global_abort_rate", 0.0) or 0.0))
        hot_read_ratio = max(0.0, float(kwargs.get("hot_read_ratio", 0.0) or 0.0))
        hot_write_ratio = max(0.0, float(kwargs.get("hot_write_ratio", 0.0) or 0.0))
        profiles = tuple(kwargs.get("profiles", ()) or ())
        operation_count = len(profiles)
        total_writes = max(
            (int(getattr(profile, "total_writes", 0) or 0) for profile in profiles),
            default=0,
        )
        has_named_hotspot = any(
            "next_order_id" in str(getattr(profile, "object_id", ""))
            for profile in profiles
        )
        if (
            retry_count <= 0
            and global_abort_rate < 0.15
            and hot_read_ratio <= 0.0
            and hot_write_ratio <= 0.0
            and operation_count < 8
            and total_writes < 8
            and not has_named_hotspot
        ):
            return 0
        return _cap_priority_score(
            super().priority_score(**kwargs),
            self.priority_cap,
        )


class _NoPriorityTransactionAwareATCCModule(TransactionAwareATCCModule):
    def _priority_score(self, **_kwargs: Any) -> int:
        return 0


class _CappedPriorityTransactionAwareATCCModule(TransactionAwareATCCModule):
    def __init__(self, *args: Any, priority_cap: Optional[int] = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.priority_cap = priority_cap

    def _priority_score(self, **kwargs: Any) -> int:
        return _cap_priority_score(
            super()._priority_score(**kwargs),
            self.priority_cap,
        )


def _compact_hot_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number <= 0.0:
        return "0"
    if number >= 1.0:
        return "some"
    return "some"


def _compact_operation_group(object_id: str) -> str:
    text = str(object_id)
    if "next_order_id" in text:
        return "tpcc-order-counter"
    if text.startswith("tpcc:stock:"):
        return "tpcc-stock"
    if text.startswith("tpcc:"):
        return "tpcc-other"
    if text.startswith("ycsb:record:"):
        return "ycsb-record"
    if text.startswith("ycsb:field:"):
        return "ycsb-field"
    return text.split(":", 1)[0] or "object"


def _compact_pressure_bucket(
    *,
    abort_rate: float = 0.0,
    lock_wait_s: float = 0.0,
    queue_depth: float = 0.0,
) -> str:
    abort = max(0.0, float(abort_rate))
    wait = max(0.0, float(lock_wait_s))
    queue = max(0.0, float(queue_depth))
    if abort >= 0.20:
        return "abort-high"
    if abort >= 0.05:
        return "abort-low"
    if wait >= 0.050 or queue >= 1.0:
        return "wait-high"
    if wait >= 0.010:
        return "wait-low"
    return "cold"


class CompactStateOperationATCCPolicy(OperationPolicyTable):
    """Operation dynamic policy with ablation-only compact Q state keys."""

    def __init__(
        self,
        base_policy: OperationPolicyTable,
        workload_kind: str,
    ):
        object.__setattr__(self, "workload_kind", str(workload_kind).strip().lower())
        super().__init__(
            rules=base_policy.rules,
            fallback_policy=base_policy.fallback_policy,
            name=base_policy.name + "-compact-state",
            online_feedback=base_policy.online_feedback,
            min_feedback_observations=base_policy.min_feedback_observations,
            exact_key_min_observations=base_policy.exact_key_min_observations,
            conflict_abort_cost=base_policy.conflict_abort_cost,
            lock_wait_cost_per_s=base_policy.lock_wait_cost_per_s,
            lock_queue_depth_cost=base_policy.lock_queue_depth_cost,
            lock_overhead_cost=base_policy.lock_overhead_cost,
            hysteresis=base_policy.hysteresis,
            pinned_rules=base_policy.pinned_rules,
            rl_enabled=base_policy.rl_enabled,
            rl_conflict_penalty=base_policy.rl_conflict_penalty,
            rl_commit_reward=base_policy.rl_commit_reward,
            rl_reject_penalty=base_policy.rl_reject_penalty,
            rl_lock_wait_cost_per_s=base_policy.rl_lock_wait_cost_per_s,
            rl_pessimistic_action_cost=base_policy.rl_pessimistic_action_cost,
            rl_learner=base_policy.rl_learner,
            atcc_module=base_policy.atcc_module,
            atcc_runtime_stats=base_policy.atcc_runtime_stats,
            atcc_cold_occ_fast_path=base_policy.atcc_cold_occ_fast_path,
            atcc_fast_path_max_retry_count=base_policy.atcc_fast_path_max_retry_count,
            atcc_fast_path_max_agent_interval_s=(
                base_policy.atcc_fast_path_max_agent_interval_s
            ),
            atcc_fast_path_max_global_abort_rate=(
                base_policy.atcc_fast_path_max_global_abort_rate
            ),
            atcc_fast_path_max_lock_queue_depth=(
                base_policy.atcc_fast_path_max_lock_queue_depth
            ),
            atcc_fast_path_max_total_writes=(
                base_policy.atcc_fast_path_max_total_writes
            ),
            telemetry=base_policy.telemetry,
        )

    def select_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        return super().select_profiles(profiles)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["name"] = self.name
        data["compact_state"] = {
            "scope": "operation-ablation-dynamic",
            "workload_kind": self.workload_kind,
        }
        return data

class _AblationDynamicTransactionATCCModule(TransactionAwareATCCModule):
    """Transaction ATCC variant with a compact train/test state space.

    The production transaction ATCC table keeps several runtime EWMA buckets in
    the state key.  That is useful online, but it fragments the offline
    ablation artifact enough that most TPCC states have only one or two visits.
    The ablation runner uses this compact wrapper only for dynamic ablation
    variants so trained Q values can transfer to frozen test seeds.
    """

    def __init__(
        self,
        *args: Any,
        static_preset: str = "naive",
        static_prior_enabled: bool = False,
        profile_name: str = "",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.static_preset = _normalize_static_preset(static_preset)
        self.static_prior_enabled = bool(static_prior_enabled)
        self.profile_name = str(profile_name).strip().lower()

    def select_transaction(
        self,
        profiles: Sequence[Any],
        *,
        runtime_stats: Optional[ATCCRuntimeStats] = None,
    ) -> TransactionAwareATCCDecision:
        decision = super().select_transaction(
            profiles,
            runtime_stats=runtime_stats,
        )
        fallback_action = self._static_fallback_action(decision)
        if self.static_prior_enabled and self._should_use_static_fallback(
            decision,
            fallback_action,
        ):
            decision = self._replace_action(decision, fallback_action)
        if (
            decision.phase == "commit"
            and int(decision.retry_count) >= 1
            and len(decision.write_set) >= 16
            and decision.action in {"occ", "lock-hot-writes", "lock-hot-read-write"}
        ):
            action = "lock-write-set"
            return dataclasses.replace(
                decision,
                action=action,
                fast_path=False,
                prelock_targets=self._prelock_targets(
                    action,
                    read_set=decision.read_set,
                    write_set=decision.write_set,
                    hot_read_set=decision.hot_read_set,
                    hot_write_set=decision.hot_write_set,
                ),
            )
        return decision

    def _static_fallback_action(
        self,
        decision: TransactionAwareATCCDecision,
    ) -> str:
        if len(decision.write_set) >= _static_transaction_wide_write_threshold(
            self.static_preset
        ):
            return "lock-write-set"
        return "occ"

    def _should_use_static_fallback(
        self,
        decision: TransactionAwareATCCDecision,
        fallback_action: str,
    ) -> bool:
        if decision.action == fallback_action:
            return False
        if decision.explore:
            return False
        q_values = dict(decision.q_values or {})
        selected_value = float(q_values.get(decision.action, 0.0))
        fallback_value = float(q_values.get(fallback_action, 0.0))
        selected_visits = _learner_action_visits(
            self.learner,
            decision.state_key,
            decision.action,
        )
        fallback_visits = _learner_action_visits(
            self.learner,
            decision.state_key,
            fallback_action,
        )
        if (
            fallback_action == "lock-write-set"
            and decision.action in {"occ", "lock-hot-writes", "lock-hot-read-write"}
            and int(decision.retry_count) <= 0
            and float(decision.global_abort_rate) < 0.20
        ):
            if selected_visits < 8:
                return True
            if selected_value < fallback_value + 0.25:
                return True
        if selected_visits < 2:
            return True
        if selected_value < fallback_value + 0.10:
            return True
        if (
            fallback_action == "occ"
            and decision.action != "occ"
            and int(decision.retry_count) <= 0
            and float(decision.global_abort_rate) < 0.05
            and fallback_visits >= selected_visits
        ):
            return True
        return False

    def _replace_action(
        self,
        decision: TransactionAwareATCCDecision,
        action: str,
    ) -> TransactionAwareATCCDecision:
        return dataclasses.replace(
            decision,
            action=action,
            fast_path=action == "occ",
            prelock_targets=self._prelock_targets(
                action,
                read_set=decision.read_set,
                write_set=decision.write_set,
                hot_read_set=decision.hot_read_set,
                hot_write_set=decision.hot_write_set,
            ),
        )

    def _prior_action(
        self,
        *,
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        read_count: int,
        write_count: int,
        hot_read_count: int,
        hot_write_count: int,
        global_abort_rate: float,
    ) -> str:
        if phase == "commit":
            if int(retry_count) >= 1 and int(write_count) >= 16:
                return "lock-write-set"
            if (
                int(write_count) >= 16
                and max(0.0, float(global_abort_rate)) >= 0.20
            ):
                return "lock-write-set"
        return super()._prior_action(
            phase=phase,
            retry_count=retry_count,
            agent_interval_s=agent_interval_s,
            read_count=read_count,
            write_count=write_count,
            hot_read_count=hot_read_count,
            hot_write_count=hot_write_count,
            global_abort_rate=global_abort_rate,
        )

    def _state_key(
        self,
        profiles: Sequence[Any],
        *,
        phase: str,
        retry_count: int,
        agent_interval_s: float,
        priority: int,
        read_count: int,
        write_count: int,
        hot_read_count: int,
        hot_write_count: int,
        runtime_stats: ATCCRuntimeStats,
    ) -> str:
        workloads = sorted({str(getattr(profile, "workload", "")) for profile in profiles})
        task_types = sorted({str(getattr(profile, "task_type", "")) for profile in profiles})
        pressure = self._pressure_bucket(runtime_stats)
        return "|".join(
            (
                "scope=transaction-ablation-dynamic",
                "workload=" + ",".join(workloads),
                "task=" + ",".join(task_types),
                "phase=" + str(phase),
                "reads=" + _bucket_count(read_count),
                "writes=" + _bucket_count(write_count),
                "hotR=" + _bucket_count(hot_read_count),
                "hotW=" + _bucket_count(hot_write_count),
                "retry=" + _bucket_count(retry_count),
                "interval=" + _coarse_interval_s(agent_interval_s),
                "priority=" + _bucket_count(priority),
                "pressure=" + pressure,
            )
        )

    @staticmethod
    def _pressure_bucket(runtime_stats: ATCCRuntimeStats) -> str:
        return _compact_pressure_bucket(
            abort_rate=float(runtime_stats.ewma_abort_rate),
            lock_wait_s=float(runtime_stats.ewma_lock_wait_s),
            queue_depth=float(getattr(runtime_stats, "ewma_lock_queue_depth", 0.0)),
        )


class _AblationNoPriorityTransactionATCCModule(
    _AblationDynamicTransactionATCCModule
):
    def _priority_score(self, **_kwargs: Any) -> int:
        return 0


class _AblationCappedPriorityTransactionATCCModule(
    _AblationDynamicTransactionATCCModule
):
    def __init__(self, *args: Any, priority_cap: Optional[int] = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.priority_cap = priority_cap

    def select_transaction(
        self,
        profiles: Sequence[Any],
        *,
        runtime_stats: Optional[ATCCRuntimeStats] = None,
    ) -> TransactionAwareATCCDecision:
        return super().select_transaction(
            profiles,
            runtime_stats=runtime_stats,
        )

    def _state_key(self, *args: Any, **kwargs: Any) -> str:
        kwargs["priority"] = 0
        return super()._state_key(*args, **kwargs)

    def _priority_score(self, **kwargs: Any) -> int:
        retry_count = int(kwargs.get("retry_count", 0) or 0)
        global_abort_rate = max(0.0, float(kwargs.get("global_abort_rate", 0.0) or 0.0))
        hot_read_count = int(kwargs.get("hot_read_count", 0) or 0)
        hot_write_count = int(kwargs.get("hot_write_count", 0) or 0)
        operation_count = int(kwargs.get("operation_count", 0) or 0)
        if retry_count <= 0 and global_abort_rate < 0.15:
            return 0
        if (
            retry_count <= 0
            and global_abort_rate < 0.15
            and operation_count < 8
        ):
            return 0
        if (
            retry_count <= 0
            and operation_count < 8
            and not (hot_read_count or hot_write_count)
        ):
            return 0
        return _cap_priority_score(
            super()._priority_score(**kwargs),
            self.priority_cap,
        )


class StaticTransactionATCCModule(TransactionAwareATCCModule):
    """Static threshold transaction ATCC used only by the ablation runner."""

    def __init__(
        self,
        workload_kind: str,
        *,
        priority_enabled: bool = False,
        priority_cap: Optional[int] = 1,
        static_preset: str = "conservative",
    ):
        workload = str(workload_kind).strip().lower()
        if workload not in {"ycsb", "tpcc"}:
            raise ValueError(f"unsupported workload kind: {workload_kind}")
        preset = _normalize_static_preset(static_preset)
        self.workload_kind = workload
        self.priority_enabled = bool(priority_enabled)
        self.priority_cap = priority_cap
        self.static_preset = preset
        suffix = "priority" if self.priority_enabled else "no-priority"
        super().__init__(
            name=f"{workload}-static-{preset}-{suffix}-transaction-atcc",
            learner=ATCCPolicyQLearner(
                tuple(spec.name for spec in self.ACTIONS),
                epsilon=0.0,
                min_epsilon=0.0,
                epsilon_decay=1.0,
                seed=9107,
            ),
        )

    def select_transaction(
        self,
        profiles: Sequence[Any],
        *,
        runtime_stats: Optional[ATCCRuntimeStats] = None,
    ) -> TransactionAwareATCCDecision:
        rows = tuple(profiles)
        phase = self._infer_phase(rows)
        retry_count = max(
            (int(getattr(profile, "retry_count", 0) or 0) for profile in rows),
            default=0,
        )
        interval_s = max(
            (
                float(getattr(profile, "agent_interval_s", 0.0) or 0.0)
                for profile in rows
            ),
            default=0.0,
        )
        read_set, write_set, hot_read_set, hot_write_set = self._partition_sets(rows)
        cold_read_set = tuple(oid for oid in read_set if oid not in hot_read_set)
        cold_write_set = tuple(oid for oid in write_set if oid not in hot_write_set)
        if len(write_set) >= _static_transaction_wide_write_threshold(
            self.static_preset
        ):
            action = "lock-write-set"
        elif self.static_preset != "naive" and retry_count >= 3 and (
            hot_read_set or hot_write_set
        ):
            action = "lock-read-write-set"
        elif self.static_preset != "naive" and retry_count > 0:
            action = "lock-write-set"
        else:
            action = "occ"
        priority = (
            _cap_priority_score(
                self._priority_score(
                    operation_count=len(rows),
                    retry_count=retry_count,
                    agent_interval_s=interval_s,
                    hot_read_count=len(hot_read_set),
                    hot_write_count=len(hot_write_set),
                    global_abort_rate=0.0,
                ),
                self.priority_cap,
            )
            if self.priority_enabled
            else 0
        )
        state_key = "|".join(
            (
                "scope=transaction-static",
                f"preset={self.static_preset}",
                "workload=" + ",".join(
                    sorted({str(getattr(profile, "workload", "")) for profile in rows})
                ),
                "task=" + ",".join(
                    sorted({str(getattr(profile, "task_type", "")) for profile in rows})
                ),
                f"phase={phase}",
                f"reads={_bucket_count(len(read_set))}",
                f"writes={_bucket_count(len(write_set))}",
                f"hotR={_bucket_count(len(hot_read_set))}",
                f"hotW={_bucket_count(len(hot_write_set))}",
                f"retry={_bucket_count(retry_count)}",
                f"interval={_bucket_latency_s(interval_s)}",
                f"priority={_bucket_count(priority)}",
            )
        )
        return TransactionAwareATCCDecision(
            state_key=state_key,
            action=action,
            phase=phase,
            priority=priority,
            fast_path=action == "occ",
            explore=False,
            q_values={spec.name: 0.0 for spec in self.ACTIONS},
            read_set=read_set,
            write_set=write_set,
            hot_read_set=hot_read_set,
            hot_write_set=hot_write_set,
            cold_read_set=cold_read_set,
            cold_write_set=cold_write_set,
            prelock_targets=self._prelock_targets(
                action,
                read_set=read_set,
                write_set=write_set,
                hot_read_set=hot_read_set,
                hot_write_set=hot_write_set,
            ),
            retry_count=retry_count,
            agent_interval_s=interval_s,
            global_abort_rate=0.0,
            global_lock_wait_s=0.0,
            global_latency_s=0.0,
        )

    def observe_result(self, *_args: Any, **_kwargs: Any) -> float:
        return 0.0


class StaticPriorOperationATCCPolicy(OperationPolicyTable):
    """Operation dynamic policy with a conservative static threshold fallback."""

    def __init__(
        self,
        base_policy: OperationPolicyTable,
        workload_kind: str,
        *,
        static_preset: str = "naive",
        profile_name: str = "",
    ):
        object.__setattr__(self, "workload_kind", str(workload_kind).strip().lower())
        object.__setattr__(self, "static_preset", _normalize_static_preset(static_preset))
        object.__setattr__(self, "profile_name", str(profile_name).strip().lower())
        object.__setattr__(
            self,
            "priority_cap",
            getattr(base_policy, "priority_cap", None),
        )
        super().__init__(
            rules=base_policy.rules,
            fallback_policy=base_policy.fallback_policy,
            name=base_policy.name + f"-static-prior-{self.static_preset}",
            online_feedback=base_policy.online_feedback,
            min_feedback_observations=base_policy.min_feedback_observations,
            exact_key_min_observations=base_policy.exact_key_min_observations,
            conflict_abort_cost=base_policy.conflict_abort_cost,
            lock_wait_cost_per_s=base_policy.lock_wait_cost_per_s,
            lock_queue_depth_cost=base_policy.lock_queue_depth_cost,
            lock_overhead_cost=base_policy.lock_overhead_cost,
            hysteresis=base_policy.hysteresis,
            pinned_rules=base_policy.pinned_rules,
            rl_enabled=base_policy.rl_enabled,
            rl_conflict_penalty=base_policy.rl_conflict_penalty,
            rl_commit_reward=base_policy.rl_commit_reward,
            rl_reject_penalty=base_policy.rl_reject_penalty,
            rl_lock_wait_cost_per_s=base_policy.rl_lock_wait_cost_per_s,
            rl_pessimistic_action_cost=base_policy.rl_pessimistic_action_cost,
            rl_learner=base_policy.rl_learner,
            atcc_module=base_policy.atcc_module,
            atcc_runtime_stats=base_policy.atcc_runtime_stats,
            atcc_cold_occ_fast_path=base_policy.atcc_cold_occ_fast_path,
            atcc_fast_path_max_retry_count=base_policy.atcc_fast_path_max_retry_count,
            atcc_fast_path_max_agent_interval_s=(
                base_policy.atcc_fast_path_max_agent_interval_s
            ),
            atcc_fast_path_max_global_abort_rate=(
                base_policy.atcc_fast_path_max_global_abort_rate
            ),
            atcc_fast_path_max_lock_queue_depth=(
                base_policy.atcc_fast_path_max_lock_queue_depth
            ),
            atcc_fast_path_max_total_writes=(
                base_policy.atcc_fast_path_max_total_writes
            ),
            telemetry=base_policy.telemetry,
        )

    def select_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        dynamic = super().select_profiles(profiles)
        rows = tuple(profiles)
        return tuple(
            self._with_priority(self._with_static_prior(decision, profile), profile)
            for decision, profile in zip(dynamic, rows)
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["name"] = self.name
        data["static_prior"] = {
            "workload_kind": self.workload_kind,
            "static_preset": self.static_preset,
            "profile": self.profile_name,
            "priority_cap": self.priority_cap,
        }
        return data

    def _with_static_prior(
        self,
        decision: OperationPolicyDecision,
        profile: OperationPolicyProfile,
    ) -> OperationPolicyDecision:
        fallback_policy, reason = self._static_policy(profile)
        if decision.policy == fallback_policy:
            return decision
        if not self._should_use_static_prior(decision, fallback_policy):
            return decision
        action = "lock-write-set" if fallback_policy == "pessimistic" else "occ"
        return dataclasses.replace(
            decision,
            policy=fallback_policy,
            rule=f"{decision.rule}|static-prior-{reason}",
            atcc_action=action if decision.atcc_action else decision.atcc_action,
        )

    def _should_use_static_prior(
        self,
        decision: OperationPolicyDecision,
        fallback_policy: str,
    ) -> bool:
        if decision.atcc_explore or decision.rl_explore:
            return False
        if not decision.atcc_state_key and not decision.rl_state_key:
            if int(decision.telemetry_observations) <= 0:
                return True
            optimistic_cost = float(decision.optimistic_cost)
            pessimistic_cost = float(decision.pessimistic_cost)
            margin = 1.02
            if decision.policy == "pessimistic":
                return not (optimistic_cost > pessimistic_cost * margin)
            if decision.policy == "optimistic":
                return not (pessimistic_cost > optimistic_cost * margin)
            return True
        dynamic_value = _operation_decision_value(decision, decision.policy)
        fallback_value = _operation_decision_value(decision, fallback_policy)
        visit_count = _operation_decision_visits(
            self,
            decision,
            decision.policy,
        )
        if visit_count < 2:
            return True
        if dynamic_value < fallback_value + 0.10:
            return True
        if (
            fallback_policy == "optimistic"
            and decision.policy == "pessimistic"
            and int(decision.retry_count) <= 0
            and float(decision.atcc_global_abort_rate) < 0.05
        ):
            return True
        return False

    def _static_policy(self, profile: OperationPolicyProfile) -> Tuple[str, str]:
        if profile.access_kind == "read":
            return "optimistic", "read"
        if (
            profile.total_writes
            >= _static_operation_wide_overwrite_threshold(self.static_preset)
        ):
            return "pessimistic", "wide-write-set"
        return "optimistic", "cold-write"

    def _with_priority(
        self,
        decision: OperationPolicyDecision,
        profile: OperationPolicyProfile,
    ) -> OperationPolicyDecision:
        if decision.policy != "pessimistic":
            return decision
        priority = PriorityOperationATCCPolicy.priority_for_decision(
            decision,
            profile,
            priority_cap=self.priority_cap,
            profile_name=self.profile_name,
        )
        if priority <= 0:
            return dataclasses.replace(decision, atcc_priority=0)
        return dataclasses.replace(
            decision,
            atcc_priority=priority,
            atcc_state_key=decision.atcc_state_key
            or decision.rl_state_key
            or decision.profile_key,
            atcc_action=decision.atcc_action
            or ("lock-write-set" if decision.policy == "pessimistic" else "occ"),
        )


class PriorityOperationATCCPolicy(OperationPolicyTable):
    """Add pressure-gated priority metadata without changing dynamic actions."""

    def __init__(
        self,
        base_policy: OperationPolicyTable,
        *,
        priority_cap: Optional[int] = 1,
        profile_name: str = "",
    ):
        self.priority_cap = priority_cap
        self.profile_name = str(profile_name).strip().lower()
        super().__init__(
            rules=base_policy.rules,
            fallback_policy=base_policy.fallback_policy,
            name=base_policy.name + f"-priority-cap-{priority_cap}",
            online_feedback=base_policy.online_feedback,
            min_feedback_observations=base_policy.min_feedback_observations,
            exact_key_min_observations=base_policy.exact_key_min_observations,
            conflict_abort_cost=base_policy.conflict_abort_cost,
            lock_wait_cost_per_s=base_policy.lock_wait_cost_per_s,
            lock_queue_depth_cost=base_policy.lock_queue_depth_cost,
            lock_overhead_cost=base_policy.lock_overhead_cost,
            hysteresis=base_policy.hysteresis,
            pinned_rules=base_policy.pinned_rules,
            rl_enabled=base_policy.rl_enabled,
            rl_conflict_penalty=base_policy.rl_conflict_penalty,
            rl_commit_reward=base_policy.rl_commit_reward,
            rl_reject_penalty=base_policy.rl_reject_penalty,
            rl_lock_wait_cost_per_s=base_policy.rl_lock_wait_cost_per_s,
            rl_pessimistic_action_cost=base_policy.rl_pessimistic_action_cost,
            rl_learner=base_policy.rl_learner,
            atcc_module=base_policy.atcc_module,
            atcc_runtime_stats=base_policy.atcc_runtime_stats,
            atcc_cold_occ_fast_path=base_policy.atcc_cold_occ_fast_path,
            atcc_fast_path_max_retry_count=base_policy.atcc_fast_path_max_retry_count,
            atcc_fast_path_max_agent_interval_s=(
                base_policy.atcc_fast_path_max_agent_interval_s
            ),
            atcc_fast_path_max_global_abort_rate=(
                base_policy.atcc_fast_path_max_global_abort_rate
            ),
            atcc_fast_path_max_lock_queue_depth=(
                base_policy.atcc_fast_path_max_lock_queue_depth
            ),
            atcc_fast_path_max_total_writes=(
                base_policy.atcc_fast_path_max_total_writes
            ),
            telemetry=base_policy.telemetry,
        )

    def select_profiles(
        self, profiles: Sequence[OperationPolicyProfile]
    ) -> Tuple[OperationPolicyDecision, ...]:
        return tuple(
            self._with_priority(decision, profile)
            for decision, profile in zip(super().select_profiles(profiles), tuple(profiles))
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["name"] = self.name
        data["priority_cap"] = self.priority_cap
        data["priority_profile"] = self.profile_name
        return data

    def _with_priority(
        self,
        decision: OperationPolicyDecision,
        profile: OperationPolicyProfile,
    ) -> OperationPolicyDecision:
        if decision.policy != "pessimistic":
            return decision
        priority = self._priority_score(decision, profile)
        if priority <= 0:
            return dataclasses.replace(decision, atcc_priority=0)
        return dataclasses.replace(
            decision,
            atcc_priority=priority,
            atcc_state_key=decision.atcc_state_key
            or decision.rl_state_key
            or decision.profile_key,
            atcc_action=decision.atcc_action
            or ("lock-write-set" if decision.policy == "pessimistic" else "occ"),
        )

    def _priority_score(
        self,
        decision: OperationPolicyDecision,
        profile: OperationPolicyProfile,
    ) -> int:
        return self.priority_for_decision(
            decision,
            profile,
            priority_cap=self.priority_cap,
            profile_name=self.profile_name,
        )

    @staticmethod
    def priority_for_decision(
        decision: OperationPolicyDecision,
        profile: OperationPolicyProfile,
        *,
        priority_cap: Optional[int],
        profile_name: str = "",
    ) -> int:
        interval_ms = max(0.0, float(profile.agent_interval_s)) * 1000.0
        has_named_hotspot = "next_order_id" in str(profile.object_id)
        has_retry = int(profile.retry_count) > 0
        active_profile = str(profile_name or "").strip().lower()
        global_abort = max(0.0, float(decision.atcc_global_abort_rate))
        telemetry_pressure = (
            int(decision.telemetry_observations) > 0
            and float(decision.optimistic_cost) > float(decision.pessimistic_cost)
        )
        if active_profile != "high" and not has_retry and not telemetry_pressure:
            return 0
        if not (
            has_retry
            or has_named_hotspot
            or global_abort >= 0.15
            or telemetry_pressure
            or int(profile.total_writes) >= 8
            or interval_ms >= 75.0
        ):
            return 0
        raw = (
            max(0, int(profile.retry_count)) * 5
            + max(0, int(profile.total_writes)) // 4
            + max(0, int(profile.operation_count_for_object) - 1)
            + min(10, int(interval_ms // 25))
            + (3 if has_named_hotspot else 0)
            + int(global_abort * 10)
        )
        return _cap_priority_score(raw, priority_cap)


def run_ablation_suite(
    *,
    workloads: Sequence[str],
    profiles: Sequence[str],
    variants: Sequence[str],
    seeds: Sequence[int],
    task_count: int,
    workers: int,
    agent_slots: int,
    planning_delay_s: float,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    background_workers: int,
    background_interval_s: float,
    background_strategy: str,
    prelock_wait_budget_s: float,
    prelock_wait_budget_mode: str,
    prelock_lease_mode_ycsb: str,
    prelock_lease_mode_tpcc: str,
    agent_execution_mode: str,
    snapshot_timing: str,
    train_seeds: Sequence[int] = (),
    train_rounds: int = 0,
    train_task_count: int = 0,
    train_policy_epsilon: float = 0.15,
    validation_seeds: Sequence[int] = (),
    validation_task_count: int = 0,
    priority_cap: Optional[int] = 1,
    freeze_dynamic_policy: bool = True,
    static_preset: str = "conservative",
    include_baselines: bool = True,
    baseline_strategies: Sequence[str] = DEFAULT_BASELINES,
    pretrained_artifacts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    pretrained_artifacts_path: str = "",
) -> Dict[str, Any]:
    normalized_static_preset = _normalize_static_preset(static_preset)
    selected_variants = tuple(_variant_spec(name) for name in variants)
    loaded_artifacts = {
        str(key): dict(value)
        for key, value in dict(pretrained_artifacts or {}).items()
        if isinstance(value, Mapping)
    }
    runs: List[RetryRunSummary] = []
    training_runs: List[RetryRunSummary] = []
    validation_runs: List[RetryRunSummary] = []
    validation_metrics_all: List[Dict[str, Any]] = []
    validation_selections: Dict[str, Dict[str, str]] = {}
    training_artifacts: Dict[str, Dict[str, Any]] = {}
    effective_train_task_count = int(train_task_count) if int(train_task_count) > 0 else int(task_count)
    effective_validation_task_count = (
        int(validation_task_count) if int(validation_task_count) > 0 else int(task_count)
    )
    for workload_kind in workloads:
        for profile_name in profiles:
            workload = build_profile_workload(workload_kind, profile_name)
            profile_initial_artifacts = {
                key: loaded_artifacts[key]
                for key in (
                    _training_artifact_key(workload_kind, profile_name, spec.name)
                    for spec in selected_variants
                    if spec.dynamic
                )
                if key in loaded_artifacts
            }
            profile_artifacts, profile_training_runs = _train_dynamic_variants(
                workload=workload,
                workload_kind=workload_kind,
                profile_name=profile_name,
                specs=selected_variants,
                train_seeds=train_seeds,
                train_rounds=train_rounds,
                train_task_count=effective_train_task_count,
                train_policy_epsilon=train_policy_epsilon,
                priority_cap=priority_cap,
                workers=workers,
                agent_slots=agent_slots,
                planning_delay_s=planning_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
                prelock_wait_budget_s=prelock_wait_budget_s,
                prelock_wait_budget_mode=prelock_wait_budget_mode,
                prelock_lease_mode=_lease_mode(
                    workload_kind,
                    prelock_lease_mode_ycsb,
                    prelock_lease_mode_tpcc,
                ),
                agent_execution_mode=agent_execution_mode,
                snapshot_timing=snapshot_timing,
                initial_artifacts=profile_initial_artifacts,
            )
            training_artifacts.update(profile_artifacts)
            training_runs.extend(profile_training_runs)
            profile_validation_runs = _run_validation_variants(
                workload=workload,
                workload_kind=workload_kind,
                profile_name=profile_name,
                specs=selected_variants,
                artifacts=profile_artifacts,
                validation_seeds=validation_seeds,
                validation_task_count=effective_validation_task_count,
                priority_cap=priority_cap,
                static_preset=normalized_static_preset,
                freeze_dynamic_policy=freeze_dynamic_policy,
                workers=workers,
                agent_slots=agent_slots,
                planning_delay_s=planning_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
                prelock_wait_budget_s=prelock_wait_budget_s,
                prelock_wait_budget_mode=prelock_wait_budget_mode,
                prelock_lease_mode=_lease_mode(
                    workload_kind,
                    prelock_lease_mode_ycsb,
                    prelock_lease_mode_tpcc,
                ),
                agent_execution_mode=agent_execution_mode,
                snapshot_timing=snapshot_timing,
            )
            validation_runs.extend(profile_validation_runs)
            validation_metrics = _metric_rows(_aggregate_ablation_runs(profile_validation_runs))
            validation_metrics_all.extend(validation_metrics)
            validation_selections.update(
                _selection_map_from_validation(
                    validation_metrics,
                    workload_kind=workload_kind,
                    profile_name=profile_name,
                )
            )
            for seed in seeds:
                tasks = tuple(workload.generate_tasks(task_count, seed=int(seed)))
                if include_baselines:
                    for baseline in baseline_strategies:
                        runs.append(
                            _with_profile(
                                _run_one_retry(
                                workload,
                                tasks,
                                str(baseline),
                                workload_kind=workload_kind,
                                policy_variant="baseline",
                                seed=int(seed),
                                workers=workers,
                                agent_slots=agent_slots,
                                agent_admission_mode="before-begin",
                                planning_delay_s=planning_delay_s,
                                latency_distribution=latency_distribution,
                                latency_cv=latency_cv,
                                latency_max_s=latency_max_s,
                                max_attempts=max_attempts,
                                operation_policy=_default_operation_policy(
                                    workload_kind
                                ),
                                background_workers=background_workers,
                                background_interval_s=background_interval_s,
                                background_strategy=background_strategy,
                                object_lock_scheduler="race",
                                object_lock_priority_burst=2,
                                prelock_wait_budget_s=prelock_wait_budget_s,
                                prelock_wait_budget_mode=prelock_wait_budget_mode,
                                prelock_lease_mode=_lease_mode(
                                    workload_kind,
                                    prelock_lease_mode_ycsb,
                                    prelock_lease_mode_tpcc,
                                ),
                                agent_execution_mode=agent_execution_mode,
                                    snapshot_timing=snapshot_timing,
                                ),
                                profile_name,
                            )
                        )
                for spec in selected_variants:
                    artifact = profile_artifacts.get(
                        _training_artifact_key(workload_kind, profile_name, spec.name)
                    )
                    operation_policy = _operation_policy_for_variant(
                        workload_kind,
                        spec,
                        learned_artifact=artifact,
                        priority_cap=priority_cap,
                        static_preset=normalized_static_preset,
                        profile_name=profile_name,
                        freeze_learning=freeze_dynamic_policy and artifact is not None,
                    )
                    transaction_policy = _transaction_policy_for_variant(
                        workload_kind,
                        spec,
                        learned_artifact=artifact,
                        priority_cap=priority_cap,
                        static_preset=normalized_static_preset,
                        profile_name=profile_name,
                        freeze_learning=freeze_dynamic_policy and artifact is not None,
                    )
                    if operation_policy is None:
                        operation_policy = _default_operation_policy(workload_kind)
                    runs.append(
                        _with_profile(
                            _run_one_retry(
                            workload,
                            tasks,
                            spec.strategy,
                            workload_kind=workload_kind,
                            policy_variant=spec.name,
                            randomization_key=_ablation_randomization_key(spec),
                            seed=int(seed),
                            workers=workers,
                            agent_slots=agent_slots,
                            agent_admission_mode="before-begin",
                            planning_delay_s=planning_delay_s,
                            latency_distribution=latency_distribution,
                            latency_cv=latency_cv,
                            latency_max_s=latency_max_s,
                            max_attempts=max_attempts,
                            operation_policy=operation_policy,
                            transaction_atcc_policy=transaction_policy,
                            background_workers=background_workers,
                            background_interval_s=background_interval_s,
                            background_strategy=background_strategy,
                            object_lock_scheduler=_ablation_object_lock_scheduler(
                                spec,
                                profile_name,
                            ),
                            object_lock_priority_burst=2,
                            prelock_wait_budget_s=prelock_wait_budget_s,
                            prelock_wait_budget_mode=prelock_wait_budget_mode,
                            prelock_lease_mode=_lease_mode(
                                workload_kind,
                                prelock_lease_mode_ycsb,
                                prelock_lease_mode_tpcc,
                            ),
                            agent_execution_mode=agent_execution_mode,
                                snapshot_timing=snapshot_timing,
                            ),
                            profile_name,
                        )
                    )
    aggregates = _aggregate_ablation_runs(runs)
    raw_metrics = _metric_rows(aggregates)
    metrics = _with_selected_metric_rows(raw_metrics, validation_selections)
    ratios = _ratio_rows(metrics)
    return {
        "mode": "atcc-ablation",
        "workloads": list(workloads),
        "profiles": list(profiles),
        "variants": list(variants),
        "seeds": list(seeds),
        "task_count": int(task_count),
        "workers": int(workers),
        "agent_slots": int(agent_slots),
        "planning_delay_s": float(planning_delay_s),
        "max_attempts": int(max_attempts),
        "background_workers": int(background_workers),
        "train_seeds": [int(seed) for seed in train_seeds],
        "train_rounds": int(train_rounds),
        "train_task_count": (
            effective_train_task_count
            if int(train_rounds) > 0 and tuple(train_seeds)
            else 0
        ),
        "train_policy_epsilon": float(train_policy_epsilon),
        "validation_seeds": [int(seed) for seed in validation_seeds],
        "validation_task_count": (
            effective_validation_task_count if tuple(validation_seeds) else 0
        ),
        "validation_metrics": sorted(
            validation_metrics_all,
            key=lambda row: (
                row["workload_kind"],
                row["profile"],
                row["scope"],
                row["variant"],
            ),
        ),
        "validation_selections": validation_selections,
        "priority_cap": priority_cap,
        "freeze_dynamic_policy": bool(freeze_dynamic_policy),
        "pretrained_artifacts_path": str(pretrained_artifacts_path or ""),
        "pretrained_artifact_keys": sorted(loaded_artifacts),
        "static_preset": normalized_static_preset,
        "static_operation_wide_overwrite_threshold": (
            _static_operation_wide_overwrite_threshold(normalized_static_preset)
        ),
        "static_transaction_wide_write_threshold": (
            _static_transaction_wide_write_threshold(normalized_static_preset)
        ),
        "training_runs": [run.to_dict() for run in training_runs],
        "validation_runs": [run.to_dict() for run in validation_runs],
        "training_artifacts": training_artifacts,
        "runs": [run.to_dict() for run in runs],
        "aggregates": aggregates,
        "metrics": metrics,
        "ratios": ratios,
    }


def _train_dynamic_variants(
    *,
    workload: AgentWorkload,
    workload_kind: str,
    profile_name: str,
    specs: Sequence[AblationVariantSpec],
    train_seeds: Sequence[int],
    train_rounds: int,
    train_task_count: int,
    train_policy_epsilon: float,
    priority_cap: Optional[int],
    workers: int,
    agent_slots: int,
    planning_delay_s: float,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    background_workers: int,
    background_interval_s: float,
    background_strategy: str,
    prelock_wait_budget_s: float,
    prelock_wait_budget_mode: str,
    prelock_lease_mode: str,
    agent_execution_mode: str,
    snapshot_timing: str,
    initial_artifacts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[RetryRunSummary]]:
    seeds = tuple(int(seed) for seed in train_seeds)
    rounds = max(0, int(train_rounds))
    task_count = max(0, int(train_task_count))
    dynamic_specs = tuple(spec for spec in specs if spec.dynamic)
    initial = {
        str(key): dict(value)
        for key, value in dict(initial_artifacts or {}).items()
        if isinstance(value, Mapping)
    }
    if not dynamic_specs or not seeds or rounds <= 0 or task_count <= 0:
        return initial, []

    artifacts: Dict[str, Dict[str, Any]] = {}
    training_runs: List[RetryRunSummary] = []
    for spec in dynamic_specs:
        artifact_key = _training_artifact_key(workload_kind, profile_name, spec.name)
        if artifact_key in artifacts:
            continue
        initial_artifact = initial.get(artifact_key)
        training_spec = _training_spec_for_artifact(spec)
        operation_policy = (
            _operation_policy_for_variant(
                workload_kind,
                training_spec,
                learned_artifact=initial_artifact,
                priority_cap=priority_cap,
                profile_name=profile_name,
            )
            if spec.scope == "op"
            else _default_operation_policy(workload_kind)
        )
        transaction_policy = (
            _transaction_policy_for_variant(
                workload_kind,
                training_spec,
                learned_artifact=initial_artifact,
                priority_cap=priority_cap,
                profile_name=profile_name,
            )
            if spec.scope == "tx"
            else None
        )
        _set_training_epsilon(
            operation_policy=operation_policy,
            transaction_policy=transaction_policy,
            epsilon=train_policy_epsilon,
        )
        for round_index in range(rounds):
            for seed in seeds:
                training_seed = int(seed) + (round_index * 1_000_003)
                tasks = tuple(workload.generate_tasks(task_count, seed=training_seed))
                run = _with_profile(
                    _run_one_retry(
                        workload,
                        tasks,
                        training_spec.strategy,
                        workload_kind=workload_kind,
                        policy_variant=training_spec.name,
                        randomization_key=_ablation_randomization_key(training_spec),
                        seed=training_seed,
                        workers=workers,
                        agent_slots=agent_slots,
                        agent_admission_mode="before-begin",
                        planning_delay_s=planning_delay_s,
                        latency_distribution=latency_distribution,
                        latency_cv=latency_cv,
                        latency_max_s=latency_max_s,
                        max_attempts=max_attempts,
                        operation_policy=operation_policy,
                        transaction_atcc_policy=transaction_policy,
                        background_workers=background_workers,
                        background_interval_s=background_interval_s,
                        background_strategy=background_strategy,
                        object_lock_scheduler=_ablation_object_lock_scheduler(
                            training_spec,
                            profile_name,
                        ),
                        object_lock_priority_burst=2,
                        prelock_wait_budget_s=prelock_wait_budget_s,
                        prelock_wait_budget_mode=prelock_wait_budget_mode,
                        prelock_lease_mode=prelock_lease_mode,
                        agent_execution_mode=agent_execution_mode,
                        snapshot_timing=snapshot_timing,
                    ),
                    profile_name,
                )
                training_runs.append(run)
        artifact = _dynamic_policy_artifact(
            workload_kind=workload_kind,
            profile_name=profile_name,
            spec=training_spec,
            train_seeds=seeds,
            train_rounds=rounds,
            train_task_count=task_count,
            train_policy_epsilon=train_policy_epsilon,
            priority_cap=priority_cap,
            operation_policy=operation_policy if spec.scope == "op" else None,
            transaction_policy=transaction_policy if spec.scope == "tx" else None,
        )
        artifacts[artifact_key] = artifact
    return artifacts, training_runs


def _run_validation_variants(
    *,
    workload: AgentWorkload,
    workload_kind: str,
    profile_name: str,
    specs: Sequence[AblationVariantSpec],
    artifacts: Mapping[str, Mapping[str, Any]],
    validation_seeds: Sequence[int],
    validation_task_count: int,
    priority_cap: Optional[int],
    static_preset: str,
    freeze_dynamic_policy: bool,
    workers: int,
    agent_slots: int,
    planning_delay_s: float,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    background_workers: int,
    background_interval_s: float,
    background_strategy: str,
    prelock_wait_budget_s: float,
    prelock_wait_budget_mode: str,
    prelock_lease_mode: str,
    agent_execution_mode: str,
    snapshot_timing: str,
) -> List[RetryRunSummary]:
    seeds = tuple(int(seed) for seed in validation_seeds)
    task_count = max(0, int(validation_task_count))
    if not seeds or task_count <= 0:
        return []
    rows: List[RetryRunSummary] = []
    for seed in seeds:
        tasks = tuple(workload.generate_tasks(task_count, seed=int(seed)))
        for spec in specs:
            artifact = artifacts.get(
                _training_artifact_key(workload_kind, profile_name, spec.name)
            )
            operation_policy = _operation_policy_for_variant(
                workload_kind,
                spec,
                learned_artifact=artifact,
                priority_cap=priority_cap,
                static_preset=static_preset,
                profile_name=profile_name,
                freeze_learning=freeze_dynamic_policy and artifact is not None,
            )
            transaction_policy = _transaction_policy_for_variant(
                workload_kind,
                spec,
                learned_artifact=artifact,
                priority_cap=priority_cap,
                static_preset=static_preset,
                profile_name=profile_name,
                freeze_learning=freeze_dynamic_policy and artifact is not None,
            )
            if operation_policy is None:
                operation_policy = _default_operation_policy(workload_kind)
            rows.append(
                _with_profile(
                    _run_one_retry(
                        workload,
                        tasks,
                        spec.strategy,
                        workload_kind=workload_kind,
                        policy_variant=spec.name,
                        randomization_key=_ablation_randomization_key(spec),
                        seed=int(seed),
                        workers=workers,
                        agent_slots=agent_slots,
                        agent_admission_mode="before-begin",
                        planning_delay_s=planning_delay_s,
                        latency_distribution=latency_distribution,
                        latency_cv=latency_cv,
                        latency_max_s=latency_max_s,
                        max_attempts=max_attempts,
                        operation_policy=operation_policy,
                        transaction_atcc_policy=transaction_policy,
                        background_workers=background_workers,
                        background_interval_s=background_interval_s,
                        background_strategy=background_strategy,
                        object_lock_scheduler=_ablation_object_lock_scheduler(
                            spec,
                            profile_name,
                        ),
                        object_lock_priority_burst=2,
                        prelock_wait_budget_s=prelock_wait_budget_s,
                        prelock_wait_budget_mode=prelock_wait_budget_mode,
                        prelock_lease_mode=prelock_lease_mode,
                        agent_execution_mode=agent_execution_mode,
                        snapshot_timing=snapshot_timing,
                    ),
                    profile_name,
                )
            )
    return rows


def _dynamic_policy_artifact(
    *,
    workload_kind: str,
    profile_name: str,
    spec: AblationVariantSpec,
    train_seeds: Sequence[int],
    train_rounds: int,
    train_task_count: int,
    train_policy_epsilon: float,
    priority_cap: Optional[int],
    operation_policy: Optional[OperationPolicyTable],
    transaction_policy: Optional[TransactionAwareATCCModule],
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "artifact_type": "atcc-ablation-trained-policy",
        "artifact_version": 1,
        "workload_kind": str(workload_kind),
        "profile": str(profile_name),
        "variant": spec.name,
        "scope": spec.scope,
        "priority": spec.priority,
        "train_seeds": [int(seed) for seed in train_seeds],
        "train_rounds": int(train_rounds),
        "train_task_count": int(train_task_count),
        "train_policy_epsilon": float(train_policy_epsilon),
        "priority_cap": priority_cap,
    }
    if operation_policy is not None:
        artifact["operation_policy_table"] = operation_policy.to_dict()
    if transaction_policy is not None:
        artifact["transaction_atcc_module"] = transaction_policy.to_dict()
    return artifact


def _training_artifact_key(workload_kind: str, profile_name: str, variant: str) -> str:
    spec = _training_spec_for_artifact(_variant_spec(variant))
    return f"{str(workload_kind).lower()}:{str(profile_name).lower()}:{spec.name}"


def _training_spec_for_artifact(spec: AblationVariantSpec) -> AblationVariantSpec:
    if spec.dynamic and spec.priority:
        return _variant_spec(f"{spec.scope}-dynamic")
    return spec


def _ablation_randomization_key(spec: AblationVariantSpec) -> str:
    if spec.scope in {"op", "tx"}:
        return f"{spec.scope}-ablation-paired"
    return spec.name


def _set_training_epsilon(
    *,
    operation_policy: Optional[OperationPolicyTable],
    transaction_policy: Optional[TransactionAwareATCCModule],
    epsilon: float,
) -> None:
    value = max(0.0, min(1.0, float(epsilon)))
    if operation_policy is not None:
        module = operation_policy.atcc_module
        if module is not None:
            module.learner.epsilon = value
        if getattr(operation_policy, "rl_enabled", False):
            operation_policy.rl_learner.epsilon = value
    if transaction_policy is not None:
        transaction_policy.learner.epsilon = value


def build_profile_workload(workload_kind: str, profile_name: str) -> AgentWorkload:
    return shared_build_profile_workload(workload_kind, profile_name)


def _with_profile(run: RetryRunSummary, profile_name: str) -> RetryRunSummary:
    return dataclasses.replace(run, workload=f"{run.workload}:{profile_name}")


def write_ablation_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "atcc_ablation.json", report)
    if report.get("training_artifacts"):
        _write_json(
            output_dir / "atcc_ablation_policy_artifacts.json",
            report["training_artifacts"],
        )
    _write_csv(output_dir / "summary.csv", report["aggregates"])
    _write_csv(output_dir / "atcc_ablation_metrics.csv", report["metrics"])
    _write_csv(output_dir / "atcc_ablation_ratios.csv", report["ratios"])
    (output_dir / "atcc_ablation_report.md").write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    return render_atcc_ablation_markdown(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="all", choices=("all", "ycsb", "tpcc"))
    parser.add_argument("--profile", default="all", choices=("all", "low", "medium", "high"))
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="920104,920105,920106,920107,920108")
    parser.add_argument("--task-count", type=int, default=60)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--agent-slots", type=int, default=4)
    parser.add_argument("--planning-delay-ms", type=float, default=50.0)
    parser.add_argument("--latency-distribution", choices=("fixed", "lognormal", "pareto"), default="lognormal")
    parser.add_argument("--latency-cv", type=float, default=0.8)
    parser.add_argument("--latency-max-ms", type=float, default=500.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--background-workers", type=int, default=4)
    parser.add_argument("--background-interval-ms", type=float, default=2.0)
    parser.add_argument("--background-strategy", default="occ")
    parser.add_argument("--prelock-wait-budget-ms", type=float, default=70.0)
    parser.add_argument("--prelock-wait-budget-mode", choices=("transaction", "object"), default="object")
    parser.add_argument("--prelock-lease-mode-ycsb", default="yield-refresh-regenerate")
    parser.add_argument("--prelock-lease-mode-tpcc", default="hold")
    parser.add_argument("--agent-execution-mode", choices=("legacy", "staged", "staged-local"), default="staged")
    parser.add_argument("--snapshot-timing", choices=("before-planning", "after-planning"), default="before-planning")
    parser.add_argument("--train-seeds", default="910104,910105,910106,910107,910108")
    parser.add_argument("--train-rounds", type=int, default=4)
    parser.add_argument(
        "--train-task-count",
        type=int,
        default=0,
        help="Training tasks per train seed. 0 reuses --task-count.",
    )
    parser.add_argument("--train-policy-epsilon", type=float, default=0.05)
    parser.add_argument("--validation-seeds", default="930104,930105")
    parser.add_argument(
        "--validation-task-count",
        type=int,
        default=0,
        help="Validation tasks per validation seed. 0 reuses --task-count.",
    )
    parser.add_argument(
        "--priority-cap",
        type=int,
        default=1,
        help="Cap positive ATCC priority scores in ablation variants. Use -1 for uncapped existing scores.",
    )
    parser.add_argument(
        "--no-freeze-dynamic-policy",
        action="store_true",
        help="Let dynamic variants keep updating Q tables during test runs.",
    )
    parser.add_argument(
        "--static-preset",
        choices=STATIC_PRESETS,
        default="conservative",
        help=(
            "Static ablation threshold preset. conservative uses wider first-attempt "
            "thresholds; threshold32 reproduces the previous formal matrix."
        ),
    )
    parser.add_argument("--baselines", default="occ,mvcc-full,tictoc-full")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument(
        "--pretrained-artifacts",
        type=Path,
        help=(
            "Load dynamic ablation policy artifacts before optional additional "
            "training and frozen test evaluation. Accepts either "
            "atcc_ablation_policy_artifacts.json or a full atcc_ablation.json report."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or Path(
        "results",
        "atcc_ablation_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    report = run_ablation_suite(
        workloads=_select(args.workload, ("ycsb", "tpcc")),
        profiles=_select(args.profile, ("low", "medium", "high")),
        variants=_select_variants(args.variants),
        seeds=tuple(int(seed) for seed in _split_csv(args.seeds)),
        task_count=args.task_count,
        workers=args.workers,
        agent_slots=args.agent_slots,
        planning_delay_s=args.planning_delay_ms / 1000.0,
        latency_distribution=args.latency_distribution,
        latency_cv=args.latency_cv,
        latency_max_s=args.latency_max_ms / 1000.0,
        max_attempts=args.max_attempts,
        background_workers=args.background_workers,
        background_interval_s=args.background_interval_ms / 1000.0,
        background_strategy=args.background_strategy,
        prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
        prelock_wait_budget_mode=args.prelock_wait_budget_mode,
        prelock_lease_mode_ycsb=args.prelock_lease_mode_ycsb,
        prelock_lease_mode_tpcc=args.prelock_lease_mode_tpcc,
        agent_execution_mode=args.agent_execution_mode,
        snapshot_timing=args.snapshot_timing,
        train_seeds=tuple(int(seed) for seed in _split_csv(args.train_seeds)),
        train_rounds=args.train_rounds,
        train_task_count=args.train_task_count,
        train_policy_epsilon=args.train_policy_epsilon,
        validation_seeds=tuple(int(seed) for seed in _split_csv(args.validation_seeds)),
        validation_task_count=args.validation_task_count,
        priority_cap=_priority_cap_arg(args.priority_cap),
        freeze_dynamic_policy=not args.no_freeze_dynamic_policy,
        static_preset=args.static_preset,
        include_baselines=not args.no_baselines,
        baseline_strategies=tuple(_split_csv(args.baselines)),
        pretrained_artifacts=_load_pretrained_artifacts(args.pretrained_artifacts),
        pretrained_artifacts_path=(
            str(args.pretrained_artifacts) if args.pretrained_artifacts else ""
        ),
    )
    write_ablation_outputs(report, output_dir)
    print(str(output_dir))
    return 0


def _operation_policy_for_variant(
    workload_kind: str,
    spec: AblationVariantSpec,
    *,
    learned_artifact: Optional[Mapping[str, Any]] = None,
    priority_cap: Optional[int] = 1,
    static_preset: str = "conservative",
    profile_name: str = "",
    freeze_learning: bool = False,
) -> Optional[OperationPolicyTable]:
    if spec.scope != "op":
        return None
    if not spec.dynamic:
        return StaticOperationATCCPolicy(
            workload_kind,
            priority_enabled=spec.priority,
            priority_cap=priority_cap,
            static_preset=static_preset,
        )
    variant = "ycsb-strict-tuned" if workload_kind == "ycsb" else "default"
    policy = _operation_policy(
        workload_kind,
        variant,
        policy_artifact=learned_artifact,
        policy_epsilon=0.0 if learned_artifact is not None else None,
    )
    module = policy.atcc_module
    if not spec.priority and module is not None:
        policy = dataclasses.replace(
            policy,
            name=policy.name + "-no-priority",
            atcc_module=_clone_phase_module_without_priority(module),
        )
    elif spec.priority and module is not None:
        policy = dataclasses.replace(
            policy,
            name=policy.name + f"-priority-cap-{priority_cap}",
            atcc_module=_clone_phase_module_with_priority_cap(
                module,
                priority_cap=priority_cap,
            ),
        )
    elif spec.priority:
        policy = PriorityOperationATCCPolicy(
            policy,
            priority_cap=priority_cap,
            profile_name=profile_name,
        )
    policy = CompactStateOperationATCCPolicy(policy, workload_kind)
    if freeze_learning:
        _freeze_operation_policy_learning(policy)
        policy = StaticPriorOperationATCCPolicy(
            policy,
            workload_kind,
            static_preset=static_preset,
            profile_name=profile_name,
        )
    return policy


def _default_operation_policy(workload_kind: str) -> OperationPolicyTable:
    variant = "ycsb-strict-tuned" if str(workload_kind).lower() == "ycsb" else "default"
    return _operation_policy(workload_kind, variant)


def _transaction_policy_for_variant(
    workload_kind: str,
    spec: AblationVariantSpec,
    *,
    learned_artifact: Optional[Mapping[str, Any]] = None,
    priority_cap: Optional[int] = 1,
    static_preset: str = "conservative",
    profile_name: str = "",
    freeze_learning: bool = False,
) -> Optional[TransactionAwareATCCModule]:
    if spec.scope != "tx":
        return None
    if not spec.dynamic:
        return StaticTransactionATCCModule(
            workload_kind,
            priority_enabled=spec.priority,
            priority_cap=priority_cap,
            static_preset=static_preset,
        )
    module_data = (
        dict(learned_artifact.get("transaction_atcc_module", {}))
        if isinstance(learned_artifact, Mapping)
        else {}
    )
    policy = (
        TransactionAwareATCCModule.from_dict(module_data)
        if module_data
        else _transaction_atcc_policy(workload_kind)
    )
    if module_data:
        policy.runtime_stats = ATCCRuntimeStats()
    policy = _clone_transaction_module_for_dynamic_ablation(
        policy,
        priority_enabled=spec.priority,
        priority_cap=priority_cap,
        static_preset=static_preset,
        static_prior_enabled=freeze_learning,
        profile_name=profile_name,
    )
    if freeze_learning:
        _freeze_transaction_policy_learning(policy)
    return policy


def _freeze_operation_policy_learning(policy: OperationPolicyTable) -> None:
    module = policy.atcc_module
    if module is not None:
        module.learner = FrozenATCCPolicyQLearner(module.learner)
    if hasattr(policy.rl_learner, "epsilon"):
        policy.rl_learner.epsilon = 0.0
    if hasattr(policy.rl_learner, "min_epsilon"):
        policy.rl_learner.min_epsilon = 0.0


def _freeze_transaction_policy_learning(policy: TransactionAwareATCCModule) -> None:
    policy.learner = FrozenATCCPolicyQLearner(policy.learner)


def _clone_phase_module_without_priority(
    module: PhaseAwareATCCModule,
) -> PhaseAwareATCCModule:
    return _NoPriorityCompactStatePhaseAwareATCCModule(
        name=module.name + "-no-priority",
        learner=module.learner,
        hot_conflict_threshold=module.hot_conflict_threshold,
        hot_lock_wait_threshold_s=module.hot_lock_wait_threshold_s,
        min_hot_observations=module.min_hot_observations,
        commit_reward=module.commit_reward,
        abort_penalty=module.abort_penalty,
        reject_penalty=module.reject_penalty,
        lock_wait_cost_per_s=module.lock_wait_cost_per_s,
        lock_queue_depth_cost=module.lock_queue_depth_cost,
        lock_handoff_cost=module.lock_handoff_cost,
        committing_count_cost=module.committing_count_cost,
        lock_action_cost=module.lock_action_cost,
        interval_cost_per_s=module.interval_cost_per_s,
        ycsb_tuned_prior=module.ycsb_tuned_prior,
    )


def _clone_phase_module_with_priority_cap(
    module: PhaseAwareATCCModule,
    *,
    priority_cap: Optional[int],
) -> PhaseAwareATCCModule:
    return _CappedPriorityCompactStatePhaseAwareATCCModule(
        name=module.name + f"-priority-cap-{priority_cap}",
        learner=module.learner,
        hot_conflict_threshold=module.hot_conflict_threshold,
        hot_lock_wait_threshold_s=module.hot_lock_wait_threshold_s,
        min_hot_observations=module.min_hot_observations,
        commit_reward=module.commit_reward,
        abort_penalty=module.abort_penalty,
        reject_penalty=module.reject_penalty,
        lock_wait_cost_per_s=module.lock_wait_cost_per_s,
        lock_queue_depth_cost=module.lock_queue_depth_cost,
        lock_handoff_cost=module.lock_handoff_cost,
        committing_count_cost=module.committing_count_cost,
        lock_action_cost=module.lock_action_cost,
        interval_cost_per_s=module.interval_cost_per_s,
        ycsb_tuned_prior=module.ycsb_tuned_prior,
        priority_cap=priority_cap,
    )


def _clone_transaction_module_without_priority(
    module: TransactionAwareATCCModule,
) -> TransactionAwareATCCModule:
    clone = _NoPriorityTransactionAwareATCCModule(
        name=module.name + "-no-priority",
        learner=module.learner,
        hot_conflict_threshold=module.hot_conflict_threshold,
        hot_lock_wait_threshold_s=module.hot_lock_wait_threshold_s,
        min_hot_observations=module.min_hot_observations,
        commit_reward=module.commit_reward,
        abort_penalty=module.abort_penalty,
        reject_penalty=module.reject_penalty,
        lock_wait_cost_per_s=module.lock_wait_cost_per_s,
        lock_action_cost=module.lock_action_cost,
        interval_cost_per_s=module.interval_cost_per_s,
    )
    clone.runtime_stats = module.runtime_stats
    return clone


def _clone_transaction_module_with_priority_cap(
    module: TransactionAwareATCCModule,
    *,
    priority_cap: Optional[int],
) -> TransactionAwareATCCModule:
    clone = _CappedPriorityTransactionAwareATCCModule(
        name=module.name + f"-priority-cap-{priority_cap}",
        learner=module.learner,
        hot_conflict_threshold=module.hot_conflict_threshold,
        hot_lock_wait_threshold_s=module.hot_lock_wait_threshold_s,
        min_hot_observations=module.min_hot_observations,
        commit_reward=module.commit_reward,
        abort_penalty=module.abort_penalty,
        reject_penalty=module.reject_penalty,
        lock_wait_cost_per_s=module.lock_wait_cost_per_s,
        lock_action_cost=module.lock_action_cost,
        interval_cost_per_s=module.interval_cost_per_s,
        priority_cap=priority_cap,
    )
    clone.runtime_stats = module.runtime_stats
    return clone


def _clone_transaction_module_for_dynamic_ablation(
    module: TransactionAwareATCCModule,
    *,
    priority_enabled: bool,
    priority_cap: Optional[int],
    static_preset: str = "naive",
    static_prior_enabled: bool = False,
    profile_name: str = "",
) -> TransactionAwareATCCModule:
    cls: Any
    kwargs: Dict[str, Any] = {}
    if not priority_enabled:
        cls = _AblationNoPriorityTransactionATCCModule
    else:
        cls = _AblationCappedPriorityTransactionATCCModule
        kwargs["priority_cap"] = priority_cap
    clone = cls(
        name=module.name + "-ablation-dynamic",
        learner=module.learner,
        hot_conflict_threshold=module.hot_conflict_threshold,
        hot_lock_wait_threshold_s=module.hot_lock_wait_threshold_s,
        min_hot_observations=module.min_hot_observations,
        commit_reward=module.commit_reward,
        abort_penalty=module.abort_penalty,
        reject_penalty=module.reject_penalty,
        lock_wait_cost_per_s=module.lock_wait_cost_per_s,
        lock_action_cost=module.lock_action_cost,
        interval_cost_per_s=module.interval_cost_per_s,
        static_preset=static_preset,
        static_prior_enabled=static_prior_enabled,
        profile_name=profile_name,
        **kwargs,
    )
    clone.runtime_stats = module.runtime_stats
    return clone


def _aggregate_ablation_runs(runs: Sequence[RetryRunSummary]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[RetryRunSummary]] = defaultdict(list)
    for run in runs:
        grouped[(run.workload, run.strategy, run.policy_variant)].append(run)
    aggregates: List[Dict[str, Any]] = []
    for (workload, _strategy, _variant), group in sorted(grouped.items()):
        row = aggregate_retry_runs(group)[0]
        throughputs = [run.committed_throughput for run in group]
        commit_rates = [run.commit_rate for run in group]
        row["workload"] = workload
        row["workload_kind"] = _workload_kind(workload)
        row["profile"] = _profile_name(workload)
        row["throughput_stddev"] = _stddev(throughputs)
        row["commit_rate_stddev"] = _stddev(commit_rates)
        row.update(_variant_metadata(row["policy_variant"], row["strategy"]))
        aggregates.append(row)
    return aggregates


def _metric_rows(aggregates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in aggregates:
        rows.append(
            {
                "workload_kind": _workload_kind(row.get("workload", "")),
                "profile": str(row.get("profile") or _profile_name(row.get("workload", ""))),
                "strategy": row.get("strategy", ""),
                "variant": row.get("policy_variant", ""),
                "scope": row.get("ablation_scope", ""),
                "mechanism": row.get("ablation_mechanism", ""),
                "priority_enabled": row.get("ablation_priority", ""),
                "committed_throughput": float(row.get("committed_throughput", 0.0)),
                "background_throughput": float(row.get("background_throughput", 0.0)),
                "total_throughput": float(row.get("total_throughput", 0.0)),
                "throughput_stddev": float(row.get("throughput_stddev", 0.0)),
                "commit_rate": float(row.get("commit_rate", 0.0)),
                "commit_rate_stddev": float(row.get("commit_rate_stddev", 0.0)),
                "attempts_per_task": float(row.get("attempts_per_task", 0.0)),
                "agent_latency_p95_s": float(row.get("agent_latency_p95_s", 0.0)),
                "agent_latency_p99_s": float(row.get("agent_latency_p99_s", 0.0)),
                "agent_latency_p999_s": float(row.get("agent_latency_p999_s", 0.0)),
                "agent_latency_p9999_s": float(row.get("agent_latency_p9999_s", 0.0)),
                "conflict_aborts": int(row.get("conflict_aborts", 0)),
                "conflict_abort_rate": float(row.get("conflict_abort_rate", 0.0)),
                "estimated_tokens_per_task": float(
                    row.get("estimated_tokens_per_task", 0.0)
                ),
                "estimated_wasted_tokens_per_task": float(
                    row.get("estimated_wasted_tokens_per_task", 0.0)
                ),
                "prelock_wait_per_task_s": float(
                    row.get("prelock_wait_per_task_s", 0.0)
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["workload_kind"],
            row["profile"],
            row["scope"],
            row["variant"],
        ),
    )


def _selection_map_from_validation(
    validation_metrics: Sequence[Mapping[str, Any]],
    *,
    workload_kind: str,
    profile_name: str,
) -> Dict[str, Dict[str, str]]:
    by_key = {
        (
            str(row["workload_kind"]),
            str(row["profile"]),
            str(row["variant"]),
        ): row
        for row in validation_metrics
    }
    workload = str(workload_kind).strip().lower()
    profile = str(profile_name).strip().lower()
    selections: Dict[str, Dict[str, str]] = {}
    for scope in ("op", "tx"):
        static_variant = f"{scope}-static"
        dynamic_variant = f"{scope}-dynamic"
        priority_variant = f"{scope}-dynamic-priority"
        static = by_key.get((workload, profile, static_variant))
        dynamic = by_key.get((workload, profile, dynamic_variant))
        dynamic_source = dynamic_variant
        if static and dynamic:
            dynamic_source = (
                dynamic_variant
                if _dynamic_validation_passes(dynamic, static)
                else static_variant
            )
            selections[f"{workload}:{profile}:{scope}-dynamic-selected"] = {
                "source": dynamic_source,
                "validated_against": static_variant,
                "criterion": f"validation_committed_throughput_margin_{VALIDATION_SELECTION_MARGIN:.2f}",
            }
        priority = by_key.get((workload, profile, priority_variant))
        base = by_key.get((workload, profile, dynamic_source))
        if priority and base:
            priority_source = (
                priority_variant
                if _priority_validation_passes(priority, base)
                else dynamic_source
            )
            selections[f"{workload}:{profile}:{scope}-dynamic-priority-selected"] = {
                "source": priority_source,
                "validated_against": dynamic_source,
                "criterion": (
                    "validation_committed_throughput_margin_"
                    f"{VALIDATION_SELECTION_MARGIN:.2f}_and_retry_abort_guard"
                ),
            }
    return selections


def _dynamic_validation_passes(
    dynamic: Mapping[str, Any],
    static: Mapping[str, Any],
) -> bool:
    dynamic_tput = float(dynamic.get("committed_throughput", 0.0))
    static_tput = float(static.get("committed_throughput", 0.0))
    if dynamic_tput < static_tput * VALIDATION_SELECTION_MARGIN:
        return False
    dynamic_attempts = float(dynamic.get("attempts_per_task", 0.0))
    static_attempts = float(static.get("attempts_per_task", 0.0))
    if static_attempts > 0.0 and dynamic_attempts > static_attempts * VALIDATION_ATTEMPT_MARGIN:
        return False
    dynamic_abort_rate = float(dynamic.get("conflict_abort_rate", 0.0))
    static_abort_rate = float(static.get("conflict_abort_rate", 0.0))
    if dynamic_abort_rate > static_abort_rate + VALIDATION_ABORT_RATE_MARGIN:
        return False
    return True


def _priority_validation_passes(
    priority: Mapping[str, Any],
    base: Mapping[str, Any],
) -> bool:
    priority_tput = float(priority.get("committed_throughput", 0.0))
    base_tput = float(base.get("committed_throughput", 0.0))
    if priority_tput < base_tput * VALIDATION_SELECTION_MARGIN:
        return False
    priority_attempts = float(priority.get("attempts_per_task", 0.0))
    base_attempts = float(base.get("attempts_per_task", 0.0))
    if (
        base_attempts > 0.0
        and priority_attempts
        > base_attempts * PRIORITY_VALIDATION_ATTEMPT_MARGIN
    ):
        return False
    priority_abort_rate = float(priority.get("conflict_abort_rate", 0.0))
    base_abort_rate = float(base.get("conflict_abort_rate", 0.0))
    if priority_abort_rate > base_abort_rate + PRIORITY_VALIDATION_ABORT_RATE_MARGIN:
        return False
    priority_prelock_wait = float(priority.get("prelock_wait_per_task_s", 0.0))
    base_prelock_wait = float(base.get("prelock_wait_per_task_s", 0.0))
    if (
        base_prelock_wait > 0.0
        and priority_prelock_wait
        > base_prelock_wait * PRIORITY_VALIDATION_PRELOCK_WAIT_MARGIN
    ):
        return False
    return True


def _with_selected_metric_rows(
    metrics: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in metrics]
    by_key = {
        (
            str(row["workload_kind"]),
            str(row["profile"]),
            str(row["variant"]),
        ): dict(row)
        for row in rows
    }
    for key, selection in sorted(dict(selections).items()):
        try:
            workload, profile, selected_variant = str(key).split(":", 2)
        except ValueError:
            continue
        source_variant = str(dict(selection).get("source", ""))
        source = by_key.get((workload, profile, source_variant))
        if source is None:
            continue
        selected = dict(source)
        selected["selected_from"] = source_variant
        selected["validated_against"] = str(
            dict(selection).get("validated_against", "")
        )
        selected["selection_criterion"] = str(
            dict(selection).get("criterion", "")
        )
        selected["variant"] = selected_variant
        selected["mechanism"] = selected_variant.split("-", 1)[1]
        selected["priority_enabled"] = source_variant.endswith("-priority")
        rows.append(selected)
        by_key[(workload, profile, selected_variant)] = selected
    return sorted(
        rows,
        key=lambda row: (
            row["workload_kind"],
            row["profile"],
            row.get("scope", ""),
            row["variant"],
        ),
    )


def _ratio_rows(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    by_key = {
        (
            row["workload_kind"],
            row["profile"],
            row["variant"],
        ): row
        for row in metrics
    }
    for workload in sorted({row["workload_kind"] for row in metrics}):
        for profile in sorted({row["profile"] for row in metrics if row["workload_kind"] == workload}):
            for scope in ("op", "tx"):
                best = by_key.get((workload, profile, f"{scope}-dynamic-priority"))
                if not best:
                    continue
                for baseline in (
                    f"{scope}-static",
                    f"{scope}-static-priority",
                    f"{scope}-dynamic",
                ):
                    base = by_key.get((workload, profile, baseline))
                    if base:
                        rows.append(
                            _ratio_row(workload, profile, f"{best['variant']}_vs_{baseline}", best, base)
                        )
                for baseline in DEFAULT_BASELINES:
                    base = by_key.get((workload, profile, "baseline:" + baseline))
                    if base:
                        rows.append(
                            _ratio_row(workload, profile, f"{best['variant']}_vs_{baseline}", best, base)
                        )
            tx = by_key.get((workload, profile, "tx-dynamic-priority"))
            op = by_key.get((workload, profile, "op-dynamic-priority"))
            if tx and op:
                rows.append(
                    _ratio_row(workload, profile, "tx-dynamic-priority_vs_op-dynamic-priority", tx, op)
                )
            for scope in ("op", "tx"):
                selected_dynamic = by_key.get(
                    (workload, profile, f"{scope}-dynamic-selected")
                )
                selected_priority = by_key.get(
                    (workload, profile, f"{scope}-dynamic-priority-selected")
                )
                static = by_key.get((workload, profile, f"{scope}-static"))
                raw_dynamic = by_key.get((workload, profile, f"{scope}-dynamic"))
                if selected_dynamic and static:
                    rows.append(
                        _ratio_row(
                            workload,
                            profile,
                            f"{scope}-dynamic-selected_vs_{scope}-static",
                            selected_dynamic,
                            static,
                        )
                    )
                if selected_priority and (selected_dynamic or raw_dynamic):
                    rows.append(
                        _ratio_row(
                            workload,
                            profile,
                            f"{scope}-dynamic-priority-selected_vs_{scope}-dynamic-selected",
                            selected_priority,
                            selected_dynamic or raw_dynamic,
                        )
                    )
    return rows


def _ratio_row(
    workload: str,
    profile: str,
    comparison: str,
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
) -> Dict[str, Any]:
    base = float(denominator.get("committed_throughput", 0.0))
    value = float(numerator.get("committed_throughput", 0.0))
    if base <= 0:
        return {
            "workload_kind": workload,
            "profile": profile,
            "comparison": comparison,
            "throughput_ratio": "",
            "note": "baseline_zero",
        }
    return {
        "workload_kind": workload,
        "profile": profile,
        "comparison": comparison,
        "throughput_ratio": value / base,
        "note": "",
    }


def _variant_spec(name: str) -> AblationVariantSpec:
    return ablation_variant_spec(name)


def _variant_metadata(variant: str, strategy: str) -> Dict[str, Any]:
    return ablation_variant_metadata(variant, strategy)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select(value: str, all_values: Sequence[str]) -> Tuple[str, ...]:
    return select_named_values(value, all_values)


def _select_variants(value: str) -> Tuple[str, ...]:
    return select_ablation_variants(value)


def _priority_cap_arg(value: int) -> Optional[int]:
    return priority_cap_arg(value)


def _load_pretrained_artifacts(
    path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping) and "training_artifacts" in data:
        data = data.get("training_artifacts", {})
    if not isinstance(data, Mapping):
        raise ValueError("pretrained artifacts must be a JSON object")
    artifacts: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            raise ValueError("pretrained artifact values must be JSON objects")
        artifacts[str(key)] = dict(value)
    return artifacts


def _split_csv(value: str) -> Tuple[str, ...]:
    return split_csv(value)


def _lease_mode(workload_kind: str, ycsb_mode: str, tpcc_mode: str) -> str:
    return str(ycsb_mode if str(workload_kind).lower() == "ycsb" else tpcc_mode)


def _workload_kind(workload_name: str) -> str:
    return workload_kind_from_name(workload_name)


def _profile_name(workload_name: str) -> str:
    return profile_name_from_workload(workload_name)


def _bucket_count(value: int) -> str:
    return bucket_count(value)


def _bucket_latency_s(value: float) -> str:
    return bucket_latency_s(value)


def _coarse_interval_s(value: float) -> str:
    return coarse_interval_s(value)


def _stddev(values: Sequence[float]) -> float:
    return sample_stddev(values)


if __name__ == "__main__":
    raise SystemExit(main())
