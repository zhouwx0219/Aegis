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


class _CappedPriorityPhaseAwareATCCModule(PhaseAwareATCCModule):
    def __init__(self, *args: Any, priority_cap: Optional[int] = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.priority_cap = priority_cap

    def priority_score(self, **kwargs: Any) -> int:
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


class _AblationDynamicTransactionATCCModule(TransactionAwareATCCModule):
    """Transaction ATCC variant with a compact train/test state space.

    The production transaction ATCC table keeps several runtime EWMA buckets in
    the state key.  That is useful online, but it fragments the offline
    ablation artifact enough that most TPCC states have only one or two visits.
    The ablation runner uses this compact wrapper only for dynamic ablation
    variants so trained Q values can transfer to frozen test seeds.
    """

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
        abort_rate = max(0.0, float(runtime_stats.ewma_abort_rate))
        wait_s = max(0.0, float(runtime_stats.ewma_lock_wait_s))
        if abort_rate >= 0.50:
            return "abort-high"
        if abort_rate >= 0.20:
            return "abort-medium"
        if wait_s >= 0.050:
            return "wait-high"
        if abort_rate >= 0.05 or wait_s >= 0.010:
            return "pressure-low"
        return "cold"


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

    def _priority_score(self, **kwargs: Any) -> int:
        retry_count = int(kwargs.get("retry_count", 0) or 0)
        global_abort_rate = max(0.0, float(kwargs.get("global_abort_rate", 0.0) or 0.0))
        hot_read_count = int(kwargs.get("hot_read_count", 0) or 0)
        hot_write_count = int(kwargs.get("hot_write_count", 0) or 0)
        if retry_count <= 0 and global_abort_rate < 0.20:
            return 0
        if retry_count <= 0 and not (hot_read_count or hot_write_count):
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
        if retry_count >= 3 and (hot_read_set or hot_write_set):
            action = "lock-read-write-set"
        elif retry_count > 0:
            action = "lock-write-set"
        elif len(write_set) >= _static_transaction_wide_write_threshold(
            self.static_preset
        ):
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
    training_artifacts: Dict[str, Dict[str, Any]] = {}
    effective_train_task_count = int(train_task_count) if int(train_task_count) > 0 else int(task_count)
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
                        freeze_learning=freeze_dynamic_policy and artifact is not None,
                    )
                    transaction_policy = _transaction_policy_for_variant(
                        workload_kind,
                        spec,
                        learned_artifact=artifact,
                        priority_cap=priority_cap,
                        static_preset=normalized_static_preset,
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
                            object_lock_scheduler=(
                                "bounded-priority" if spec.priority else "race"
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
    metrics = _metric_rows(aggregates)
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
        initial_artifact = initial.get(artifact_key)
        operation_policy = (
            _operation_policy_for_variant(
                workload_kind,
                spec,
                learned_artifact=initial_artifact,
                priority_cap=priority_cap,
            )
            if spec.scope == "op"
            else _default_operation_policy(workload_kind)
        )
        transaction_policy = (
            _transaction_policy_for_variant(
                workload_kind,
                spec,
                learned_artifact=initial_artifact,
                priority_cap=priority_cap,
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
                        spec.strategy,
                        workload_kind=workload_kind,
                        policy_variant=spec.name,
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
                        object_lock_scheduler=(
                            "bounded-priority" if spec.priority else "race"
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
            spec=spec,
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
    return f"{str(workload_kind).lower()}:{str(profile_name).lower()}:{str(variant).lower()}"


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
    elif spec.priority and module is not None and priority_cap is not None:
        policy = dataclasses.replace(
            policy,
            name=policy.name + f"-priority-cap-{priority_cap}",
            atcc_module=_clone_phase_module_with_priority_cap(
                module,
                priority_cap=priority_cap,
            ),
        )
    if freeze_learning:
        _freeze_operation_policy_learning(policy)
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
    return _NoPriorityPhaseAwareATCCModule(
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
    return _CappedPriorityPhaseAwareATCCModule(
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
) -> TransactionAwareATCCModule:
    cls: Any
    kwargs: Dict[str, Any] = {}
    if not priority_enabled:
        cls = _AblationNoPriorityTransactionATCCModule
    elif priority_cap is not None:
        cls = _AblationCappedPriorityTransactionATCCModule
        kwargs["priority_cap"] = priority_cap
    else:
        cls = _AblationDynamicTransactionATCCModule
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
                "throughput_stddev": float(row.get("throughput_stddev", 0.0)),
                "commit_rate": float(row.get("commit_rate", 0.0)),
                "commit_rate_stddev": float(row.get("commit_rate_stddev", 0.0)),
                "attempts_per_task": float(row.get("attempts_per_task", 0.0)),
                "agent_latency_p95_s": float(row.get("agent_latency_p95_s", 0.0)),
                "agent_latency_p99_s": float(row.get("agent_latency_p99_s", 0.0)),
                "conflict_aborts": int(row.get("conflict_aborts", 0)),
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
