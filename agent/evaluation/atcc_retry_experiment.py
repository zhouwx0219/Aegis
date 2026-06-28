"""Retry-aware ATCC training and evaluation.

This runner measures data-agent tasks as logical requests that may retry after
OCC conflicts. It is intentionally focused on operation ATCC and traditional
OCC/2PL baselines, without semantic merge as a main variable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

from agent.evaluation.atcc_schema import atcc_artifact_schema_status
from agent.runtime import (
    AgentTransactionManager,
    OperationPolicyTable,
    TransactionAwareATCCModule,
    TransactionState,
)
from agent.runtime.adaptive import operation_object_class
from agent.runtime.hybrid import ATCCFamilyPolicyTable, is_atcc_family_strategy
from agent.workloads import (
    AgentTask,
    AgentWorkload,
    TPCCConfig,
    YCSBConfig,
    build_agent_workload,
    populate_task_stage,
    populate_task_transaction,
    prepare_task_transaction,
    register_workload,
    stage_operations,
    task_agent_stages,
    task_stage_view,
)


@dataclasses.dataclass(frozen=True)
class RetryTaskOutcome:
    results: Tuple[Any, ...]
    latency_s: float
    operation_count: int
    wasted_attempts: int
    estimated_tokens: float
    estimated_wasted_tokens: float
    selected_strategy: str


@dataclasses.dataclass(frozen=True)
class RetryRunSummary:
    workload: str
    strategy: str
    policy_variant: str
    seed: int
    task_count: int
    workers: int
    agent_slots: int
    agent_admission_mode: str
    max_attempts: int
    planning_delay_s: float
    latency_distribution: str
    committed_tasks: int
    final_failed_tasks: int
    rejected_tasks: int
    total_attempts: int
    conflict_aborts: int
    conflict_object_counts: Mapping[str, int]
    conflict_object_class_counts: Mapping[str, int]
    operation_policy_counts: Mapping[str, int]
    operation_rule_counts: Mapping[str, int]
    action_counts: Mapping[str, int]
    prelock_wait_s: float
    elapsed_s: float
    task_latencies_s: Tuple[float, ...] = ()
    task_operation_counts: Tuple[int, ...] = ()
    wasted_attempts: int = 0
    tokens_per_operation: float = 0.0
    estimated_tokens: float = 0.0
    estimated_wasted_tokens: float = 0.0
    background_workers: int = 0
    background_commits: int = 0
    background_aborts: int = 0
    lease_refresh_regenerations: int = 0
    lease_refresh_replayed_operations: int = 0
    lease_refresh_rebased_writes: int = 0
    prelock_queue_depth_sum: float = 0.0
    prelock_queue_depth_observations: int = 0
    prelock_queue_depth_max: int = 0
    prelock_handoff_count: int = 0
    prelock_committing_enters: int = 0
    prelock_committing_exits: int = 0
    object_lock_scheduler: str = "race"
    object_lock_priority_burst: int = 2
    prelock_wait_budget_s: float = 0.0
    prelock_wait_budget_mode: str = "transaction"
    prelock_lease_mode: str = "hold"
    agent_execution_mode: str = "legacy"
    snapshot_timing: str = "before-planning"
    abort_retry_delay_s: float = 0.0
    stage_phase_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    selected_strategy_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    fast_through_strategy: str = ""

    @property
    def commit_rate(self) -> float:
        return self.committed_tasks / self.task_count if self.task_count else 0.0

    @property
    def committed_throughput(self) -> float:
        return self.committed_tasks / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def attempts_per_task(self) -> float:
        return self.total_attempts / self.task_count if self.task_count else 0.0

    @property
    def attempts_per_commit(self) -> float:
        return self.total_attempts / self.committed_tasks if self.committed_tasks else 0.0

    @property
    def wasted_attempts_per_task(self) -> float:
        return self.wasted_attempts / self.task_count if self.task_count else 0.0

    @property
    def estimated_tokens_per_task(self) -> float:
        return self.estimated_tokens / self.task_count if self.task_count else 0.0

    @property
    def estimated_wasted_tokens_per_task(self) -> float:
        return self.estimated_wasted_tokens / self.task_count if self.task_count else 0.0

    @property
    def agent_latency_avg_s(self) -> float:
        return _mean(self.task_latencies_s)

    @property
    def agent_latency_p50_s(self) -> float:
        return _percentile(self.task_latencies_s, 50.0)

    @property
    def agent_latency_p95_s(self) -> float:
        return _percentile(self.task_latencies_s, 95.0)

    @property
    def agent_latency_p99_s(self) -> float:
        return _percentile(self.task_latencies_s, 99.0)

    @property
    def agent_latency_max_s(self) -> float:
        return max(self.task_latencies_s) if self.task_latencies_s else 0.0

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["commit_rate"] = self.commit_rate
        row["committed_throughput"] = self.committed_throughput
        row["attempts_per_task"] = self.attempts_per_task
        row["attempts_per_commit"] = self.attempts_per_commit
        row["wasted_attempts_per_task"] = self.wasted_attempts_per_task
        row["estimated_tokens_per_task"] = self.estimated_tokens_per_task
        row["estimated_wasted_tokens_per_task"] = self.estimated_wasted_tokens_per_task
        row["agent_latency_avg_s"] = self.agent_latency_avg_s
        row["agent_latency_p50_s"] = self.agent_latency_p50_s
        row["agent_latency_p95_s"] = self.agent_latency_p95_s
        row["agent_latency_p99_s"] = self.agent_latency_p99_s
        row["agent_latency_max_s"] = self.agent_latency_max_s
        row["prelock_wait_per_task_s"] = (
            self.prelock_wait_s / self.task_count if self.task_count else 0.0
        )
        row["prelock_queue_depth_avg"] = (
            self.prelock_queue_depth_sum / self.prelock_queue_depth_observations
            if self.prelock_queue_depth_observations
            else 0.0
        )
        row["prelock_handoff_per_task"] = (
            self.prelock_handoff_count / self.task_count if self.task_count else 0.0
        )
        return row


def run_retry_matrix(
    workload: AgentWorkload,
    strategies: Iterable[str],
    *,
    workload_kind: str,
    policy_variant: str,
    task_count: int,
    seed: int,
    repeats: int,
    workers: int,
    agent_slots: int,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
    strategy_order: str = "given",
    interleave_blocks: int = 1,
    hybrid_fast_through: bool = False,
    hybrid_selected_fast_through: bool = False,
) -> Tuple[RetryRunSummary, ...]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if interleave_blocks <= 0:
        raise ValueError("interleave_blocks must be positive")
    admission = str(agent_admission_mode or "planning-only").strip().lower()
    if admission not in {"planning-only", "before-begin"}:
        raise ValueError(f"unsupported agent admission mode: {agent_admission_mode}")
    execution_mode = str(agent_execution_mode or "legacy").strip().lower()
    if execution_mode not in {"legacy", "staged", "staged-local"}:
        raise ValueError(f"unsupported agent execution mode: {agent_execution_mode}")
    timing = str(snapshot_timing or "before-planning").strip().lower()
    if timing not in {"before-planning", "after-planning"}:
        raise ValueError(f"unsupported snapshot timing: {snapshot_timing}")

    rows: List[RetryRunSummary] = []
    strategy_list = tuple(str(strategy) for strategy in strategies)
    for offset in range(repeats):
        run_seed = int(seed) + offset
        tasks = tuple(workload.generate_tasks(task_count, seed=run_seed))
        selected_baseline_strategy = _selected_baseline_strategy_for_tasks(
            strategy_list,
            tasks,
            workload_kind=workload_kind,
            policy_artifact=policy_artifact,
        )
        summaries_by_strategy: Dict[str, List[RetryRunSummary]] = {}
        summary_order: List[str] = []
        for strategy, task_start, task_end in _strategy_execution_blocks(
            strategy_list,
            repeat_index=offset,
            strategy_order=strategy_order,
            selected_baseline_strategy=selected_baseline_strategy,
            task_count=len(tasks),
            interleave_blocks=interleave_blocks,
        ):
            block_tasks = tasks[task_start:task_end]
            if not block_tasks:
                continue
            requested_strategy = str(strategy)
            execution_strategy = _fast_through_strategy(
                requested_strategy,
                selected_baseline_strategy=selected_baseline_strategy,
                hybrid_fast_through=hybrid_fast_through,
                hybrid_selected_fast_through=hybrid_selected_fast_through,
            )
            block_seed = run_seed if len(block_tasks) == len(tasks) else run_seed + int(task_start)
            summary = _run_one_retry(
                workload,
                block_tasks,
                execution_strategy,
                workload_kind=workload_kind,
                policy_variant=policy_variant,
                seed=block_seed,
                workers=workers,
                agent_slots=agent_slots,
                agent_admission_mode=admission,
                planning_delay_s=planning_delay_s,
                abort_retry_delay_s=abort_retry_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                tokens_per_operation=tokens_per_operation,
                policy_artifact=policy_artifact,
                policy_epsilon=policy_epsilon,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
                object_lock_scheduler=object_lock_scheduler,
                object_lock_priority_burst=object_lock_priority_burst,
                prelock_wait_budget_s=prelock_wait_budget_s,
                prelock_wait_budget_mode=prelock_wait_budget_mode,
                prelock_lease_mode=prelock_lease_mode,
                agent_execution_mode=execution_mode,
                snapshot_timing=timing,
            )
            if execution_strategy != requested_strategy:
                summary = dataclasses.replace(
                    summary,
                    strategy=requested_strategy,
                    fast_through_strategy=execution_strategy,
                )
            if requested_strategy not in summaries_by_strategy:
                summaries_by_strategy[requested_strategy] = []
                summary_order.append(requested_strategy)
            summaries_by_strategy[requested_strategy].append(summary)
        for requested_strategy in summary_order:
            summaries = summaries_by_strategy[requested_strategy]
            if len(summaries) == 1:
                rows.append(dataclasses.replace(summaries[0], seed=run_seed))
                continue
            rows.append(
                _combine_retry_run_summaries(
                    summaries,
                    strategy=requested_strategy,
                    seed=run_seed,
                )
            )
    return tuple(rows)


def _fast_through_strategy(
    strategy: str,
    *,
    selected_baseline_strategy: Optional[str],
    hybrid_fast_through: bool,
    hybrid_selected_fast_through: bool = False,
) -> str:
    requested = str(strategy)
    if not (bool(hybrid_fast_through) or bool(hybrid_selected_fast_through)):
        return requested
    if not is_atcc_family_strategy(requested):
        return requested
    selected = str(selected_baseline_strategy or "").strip().lower()
    if bool(hybrid_selected_fast_through) and selected:
        return selected
    if selected != "adaptive-op-strict":
        return requested
    return selected


def _strategies_for_repeat(
    strategies: Sequence[str],
    *,
    repeat_index: int,
    strategy_order: str = "given",
    selected_baseline_strategy: Optional[str] = None,
    hybrid_strategy: str = "adaptive-hybrid",
) -> Tuple[str, ...]:
    rows = tuple(str(strategy) for strategy in strategies)
    mode = str(strategy_order or "given").strip().lower()
    if mode == "given" or len(rows) <= 1:
        return rows
    if mode == "rotate":
        offset = int(repeat_index) % len(rows)
        return rows[offset:] + rows[:offset]
    if mode == "pair-selected-baseline":
        selected = str(selected_baseline_strategy or "").strip().lower()
        hybrid = str(hybrid_strategy or "adaptive-hybrid").strip().lower()
        by_name = {str(strategy).strip().lower(): str(strategy) for strategy in rows}
        if not selected or selected == hybrid:
            offset = int(repeat_index) % len(rows)
            return rows[offset:] + rows[:offset]
        if selected not in by_name or hybrid not in by_name:
            offset = int(repeat_index) % len(rows)
            return rows[offset:] + rows[:offset]
        others = tuple(
            strategy
            for strategy in rows
            if str(strategy).strip().lower() not in {selected, hybrid}
        )
        if others:
            offset = int(repeat_index) % len(others)
            ordered_others = others[offset:] + others[:offset]
        else:
            ordered_others = ()
        pair = (
            (by_name[selected], by_name[hybrid])
            if int(repeat_index) % 2 == 0
            else (by_name[hybrid], by_name[selected])
        )
        return ordered_others + pair
    else:
        raise ValueError(f"unsupported strategy order: {strategy_order}")


def _strategy_execution_blocks(
    strategies: Sequence[str],
    *,
    repeat_index: int,
    strategy_order: str = "given",
    selected_baseline_strategy: Optional[str] = None,
    task_count: int,
    interleave_blocks: int = 1,
    hybrid_strategy: str = "adaptive-hybrid",
) -> Tuple[Tuple[str, int, int], ...]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if interleave_blocks <= 0:
        raise ValueError("interleave_blocks must be positive")
    mode = str(strategy_order or "given").strip().lower()
    if mode == "interleave-all-strategies":
        rows = tuple(str(strategy) for strategy in strategies)
        block_count = min(max(1, int(interleave_blocks)), int(task_count))
        execution: List[Tuple[str, int, int]] = []
        for block_index in range(block_count):
            start = int(block_index * int(task_count) / block_count)
            end = int((block_index + 1) * int(task_count) / block_count)
            if start >= end:
                continue
            offset = (int(repeat_index) + block_index) % len(rows)
            ordered = rows[offset:] + rows[:offset]
            execution.extend((strategy, start, end) for strategy in ordered)
        return tuple(execution)

    if mode != "interleave-selected-baseline":
        return tuple(
            (strategy, 0, int(task_count))
            for strategy in _strategies_for_repeat(
                strategies,
                repeat_index=repeat_index,
                strategy_order=strategy_order,
                selected_baseline_strategy=selected_baseline_strategy,
                hybrid_strategy=hybrid_strategy,
            )
        )

    rows = tuple(str(strategy) for strategy in strategies)
    selected = str(selected_baseline_strategy or "").strip().lower()
    hybrid = str(hybrid_strategy or "adaptive-hybrid").strip().lower()
    by_name = {str(strategy).strip().lower(): str(strategy) for strategy in rows}
    if not selected or selected == hybrid or selected not in by_name or hybrid not in by_name:
        return tuple(
            (strategy, 0, int(task_count))
            for strategy in _strategies_for_repeat(
                rows,
                repeat_index=repeat_index,
                strategy_order="rotate",
                selected_baseline_strategy=selected_baseline_strategy,
                hybrid_strategy=hybrid_strategy,
            )
        )

    others = tuple(
        strategy
        for strategy in rows
        if str(strategy).strip().lower() not in {selected, hybrid}
    )
    if others:
        offset = int(repeat_index) % len(others)
        ordered_others = others[offset:] + others[:offset]
    else:
        ordered_others = ()
    execution: List[Tuple[str, int, int]] = [
        (strategy, 0, int(task_count)) for strategy in ordered_others
    ]
    block_count = min(max(1, int(interleave_blocks)), int(task_count))
    for block_index in range(block_count):
        start = int(block_index * int(task_count) / block_count)
        end = int((block_index + 1) * int(task_count) / block_count)
        if start >= end:
            continue
        pair = (
            (by_name[selected], by_name[hybrid])
            if (int(repeat_index) + block_index) % 2 == 0
            else (by_name[hybrid], by_name[selected])
        )
        execution.extend((strategy, start, end) for strategy in pair)
    return tuple(execution)


def _combine_retry_run_summaries(
    summaries: Sequence[RetryRunSummary],
    *,
    strategy: str,
    seed: int,
) -> RetryRunSummary:
    if not summaries:
        raise ValueError("summaries must not be empty")
    first = summaries[0]

    def merged_counter(name: str) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for summary in summaries:
            counter.update(getattr(summary, name))
        return dict(sorted(counter.items()))

    fast_through_values = {
        str(summary.fast_through_strategy)
        for summary in summaries
        if str(summary.fast_through_strategy)
    }
    return dataclasses.replace(
        first,
        strategy=str(strategy),
        seed=int(seed),
        task_count=sum(summary.task_count for summary in summaries),
        committed_tasks=sum(summary.committed_tasks for summary in summaries),
        final_failed_tasks=sum(summary.final_failed_tasks for summary in summaries),
        rejected_tasks=sum(summary.rejected_tasks for summary in summaries),
        total_attempts=sum(summary.total_attempts for summary in summaries),
        conflict_aborts=sum(summary.conflict_aborts for summary in summaries),
        conflict_object_counts=merged_counter("conflict_object_counts"),
        conflict_object_class_counts=merged_counter("conflict_object_class_counts"),
        operation_policy_counts=merged_counter("operation_policy_counts"),
        operation_rule_counts=merged_counter("operation_rule_counts"),
        action_counts=merged_counter("action_counts"),
        prelock_wait_s=sum(summary.prelock_wait_s for summary in summaries),
        elapsed_s=sum(summary.elapsed_s for summary in summaries),
        task_latencies_s=tuple(
            latency for summary in summaries for latency in summary.task_latencies_s
        ),
        task_operation_counts=tuple(
            count for summary in summaries for count in summary.task_operation_counts
        ),
        wasted_attempts=sum(summary.wasted_attempts for summary in summaries),
        estimated_tokens=sum(summary.estimated_tokens for summary in summaries),
        estimated_wasted_tokens=sum(
            summary.estimated_wasted_tokens for summary in summaries
        ),
        background_commits=sum(summary.background_commits for summary in summaries),
        background_aborts=sum(summary.background_aborts for summary in summaries),
        lease_refresh_regenerations=sum(
            summary.lease_refresh_regenerations for summary in summaries
        ),
        lease_refresh_replayed_operations=sum(
            summary.lease_refresh_replayed_operations for summary in summaries
        ),
        lease_refresh_rebased_writes=sum(
            summary.lease_refresh_rebased_writes for summary in summaries
        ),
        prelock_queue_depth_sum=sum(
            summary.prelock_queue_depth_sum for summary in summaries
        ),
        prelock_queue_depth_observations=sum(
            summary.prelock_queue_depth_observations for summary in summaries
        ),
        prelock_queue_depth_max=max(
            summary.prelock_queue_depth_max for summary in summaries
        ),
        prelock_handoff_count=sum(
            summary.prelock_handoff_count for summary in summaries
        ),
        prelock_committing_enters=sum(
            summary.prelock_committing_enters for summary in summaries
        ),
        prelock_committing_exits=sum(
            summary.prelock_committing_exits for summary in summaries
        ),
        stage_phase_counts=merged_counter("stage_phase_counts"),
        selected_strategy_counts=merged_counter("selected_strategy_counts"),
        fast_through_strategy=next(iter(fast_through_values))
        if len(fast_through_values) == 1
        else "",
    )


def _selected_baseline_strategy_for_tasks(
    strategies: Sequence[str],
    tasks: Sequence[AgentTask],
    *,
    workload_kind: str,
    policy_artifact: Optional[Mapping[str, Any]] = None,
    hybrid_strategy: str = "adaptive-hybrid",
) -> Optional[str]:
    strategy_names = {
        str(strategy).strip().lower() for strategy in strategies if str(strategy).strip()
    }
    normalized_hybrid = str(hybrid_strategy or "adaptive-hybrid").strip().lower()
    if normalized_hybrid not in strategy_names:
        return None
    family_table = (
        dict(policy_artifact.get("family_policy_table", {}) or {})
        if isinstance(policy_artifact, Mapping)
        else {}
    )
    family_policy = (
        ATCCFamilyPolicyTable.from_dict(family_table)
        if family_table
        else ATCCFamilyPolicyTable.default()
    ).resolve_for_task_window(tasks, workload_kind=workload_kind)
    selected = {
        family_policy.select_task(task, workload_kind=workload_kind).selected_strategy
        for task in tasks
    }
    normalized = {str(strategy).strip().lower() for strategy in selected}
    if len(normalized) != 1:
        return None
    selected_strategy = next(iter(normalized))
    if selected_strategy not in strategy_names:
        return None
    return selected_strategy


def search_policy_variants(
    workload: AgentWorkload,
    *,
    workload_kind: str,
    variants: Iterable[str],
    task_count: int,
    seed: int,
    repeats: int,
    workers: int,
    agent_slots: int,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
) -> Dict[str, Any]:
    candidates = []
    for variant in variants:
        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind=workload_kind,
            policy_variant=str(variant),
            task_count=task_count,
            seed=seed,
            repeats=repeats,
            workers=workers,
            agent_slots=agent_slots,
            agent_admission_mode=agent_admission_mode,
            planning_delay_s=planning_delay_s,
            abort_retry_delay_s=abort_retry_delay_s,
            latency_distribution=latency_distribution,
            latency_cv=latency_cv,
            latency_max_s=latency_max_s,
            max_attempts=max_attempts,
            tokens_per_operation=tokens_per_operation,
            policy_artifact=policy_artifact,
            policy_epsilon=policy_epsilon,
            background_workers=background_workers,
            background_interval_s=background_interval_s,
            background_strategy=background_strategy,
            object_lock_scheduler=object_lock_scheduler,
            object_lock_priority_burst=object_lock_priority_burst,
            prelock_wait_budget_s=prelock_wait_budget_s,
            prelock_wait_budget_mode=prelock_wait_budget_mode,
            prelock_lease_mode=prelock_lease_mode,
            agent_execution_mode=agent_execution_mode,
            snapshot_timing=snapshot_timing,
        )
        aggregate = aggregate_retry_runs(runs)[0]
        score = (
            aggregate["committed_throughput"]
            * max(0.01, aggregate["commit_rate"])
            - aggregate["prelock_wait_per_task_s"] * 10.0
        )
        candidates.append(
            {
                "policy_variant": str(variant),
                "score": score,
                "aggregate": aggregate,
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["score"],
            row["aggregate"]["commit_rate"],
            row["aggregate"]["committed_throughput"],
        ),
    )
    return {
        "training_method": "retry-aware-offline-policy-variant-search",
        "selection_metric": (
            "committed_throughput * commit_rate - 10 * prelock_wait_per_task_s"
        ),
        "best_policy_variant": best["policy_variant"],
        "candidates": candidates,
    }


def search_family_policy_variants(
    workload: AgentWorkload,
    *,
    workload_kind: str,
    read_heavy_strategies: Iterable[str],
    task_count: int,
    seed: int,
    repeats: int,
    workers: int,
    agent_slots: int,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    policy_variant: str = "default",
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
) -> Dict[str, Any]:
    strategies = tuple(dict.fromkeys(str(item).strip().lower() for item in read_heavy_strategies if str(item).strip()))
    if not strategies:
        raise ValueError("family search needs at least one read-heavy strategy")
    candidates = []
    for read_heavy_strategy in strategies:
        artifact = _family_policy_artifact(
            read_heavy_strategy=read_heavy_strategy,
            base_artifact=policy_artifact,
        )
        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid",),
            workload_kind=workload_kind,
            policy_variant=policy_variant,
            task_count=task_count,
            seed=seed,
            repeats=repeats,
            workers=workers,
            agent_slots=agent_slots,
            agent_admission_mode=agent_admission_mode,
            planning_delay_s=planning_delay_s,
            abort_retry_delay_s=abort_retry_delay_s,
            latency_distribution=latency_distribution,
            latency_cv=latency_cv,
            latency_max_s=latency_max_s,
            max_attempts=max_attempts,
            tokens_per_operation=tokens_per_operation,
            policy_artifact=artifact,
            policy_epsilon=policy_epsilon,
            background_workers=background_workers,
            background_interval_s=background_interval_s,
            background_strategy=background_strategy,
            object_lock_scheduler=object_lock_scheduler,
            object_lock_priority_burst=object_lock_priority_burst,
            prelock_wait_budget_s=prelock_wait_budget_s,
            prelock_wait_budget_mode=prelock_wait_budget_mode,
            prelock_lease_mode=prelock_lease_mode,
            agent_execution_mode=agent_execution_mode,
            snapshot_timing=snapshot_timing,
        )
        aggregate = aggregate_retry_runs(runs)[0]
        score = _family_policy_score(aggregate)
        candidates.append(
            {
                "read_heavy_strategy": read_heavy_strategy,
                "score": score,
                "aggregate": aggregate,
                "artifact": artifact,
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["score"],
            row["aggregate"]["commit_rate"],
            row["aggregate"]["committed_throughput"],
            -row["aggregate"]["agent_latency_p99_s"],
        ),
    )
    return {
        "training_method": "offline-family-policy-search",
        "selection_metric": (
            "committed_throughput * commit_rate - 0.10 * p99_latency "
            "- wasted_tokens_per_task / 100000"
        ),
        "best_read_heavy_strategy": best["read_heavy_strategy"],
        "best_artifact": best["artifact"],
        "candidates": candidates,
    }


def search_family_policy_profiles(
    profile_names: Iterable[str],
    *,
    read_heavy_strategies: Iterable[str],
    cold_read_heavy_strategies: Iterable[str],
    hot_write_strategies: Iterable[str],
    fallback_strategies: Iterable[str],
    hot_write_ratio_thresholds: Iterable[float],
    hotspot_probability_thresholds: Iterable[float],
    prelock_wait_budget_candidates_s: Iterable[float],
    prelock_lease_mode_candidates: Iterable[str],
    agent_execution_mode_candidates: Iterable[str],
    snapshot_timing_candidates: Iterable[str],
    object_lock_scheduler_candidates: Iterable[str],
    baseline_strategies: Iterable[str],
    score_mode: str,
    task_count: int,
    seed: int,
    repeats: int,
    workers: int,
    agent_slots: int,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    policy_variant: str = "default",
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
) -> Dict[str, Any]:
    profiles = _family_search_profiles(profile_names)
    strategies = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in read_heavy_strategies
            if str(item).strip()
        )
    )
    if not strategies:
        raise ValueError("family search needs at least one read-heavy strategy")
    cold_strategies = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in cold_read_heavy_strategies
            if str(item).strip()
        )
    ) or strategies
    hot_strategies = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in hot_write_strategies
            if str(item).strip()
        )
    ) or ("adaptive-op-strict",)
    fallback_candidates = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in fallback_strategies
            if str(item).strip()
        )
    ) or ("tictoc-full",)
    hot_write_thresholds = tuple(
        dict.fromkeys(float(item) for item in hot_write_ratio_thresholds)
    ) or (0.30,)
    hotspot_thresholds = tuple(
        dict.fromkeys(float(item) for item in hotspot_probability_thresholds)
    ) or (0.70,)
    prelock_budget_values = tuple(
        dict.fromkeys(float(item) for item in prelock_wait_budget_candidates_s)
    ) or (float(prelock_wait_budget_s),)
    prelock_lease_modes = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in prelock_lease_mode_candidates
            if str(item).strip()
        )
    ) or (str(prelock_lease_mode).strip().lower(),)
    execution_modes = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in agent_execution_mode_candidates
            if str(item).strip()
        )
    ) or (str(agent_execution_mode).strip().lower(),)
    snapshot_timings = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in snapshot_timing_candidates
            if str(item).strip()
        )
    ) or (str(snapshot_timing).strip().lower(),)
    object_lock_schedulers = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in object_lock_scheduler_candidates
            if str(item).strip()
        )
    ) or (str(object_lock_scheduler).strip().lower(),)
    baseline_names = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in baseline_strategies
            if str(item).strip()
        )
    )
    mode = str(score_mode or "absolute").strip().lower()
    if mode not in {"absolute", "baseline-relative", "baseline-balanced"}:
        raise ValueError(f"unsupported family search score mode: {score_mode}")
    if mode in {"baseline-relative", "baseline-balanced"} and not baseline_names:
        raise ValueError("baseline-relative family search needs baseline strategies")

    baselines: Dict[str, Dict[str, Any]] = {}
    if baseline_names:
        for profile_name, workload_kind, workload, _workload_config in profiles:
            runs = run_retry_matrix(
                workload,
                baseline_names,
                workload_kind=workload_kind,
                policy_variant=policy_variant,
                task_count=task_count,
                seed=seed,
                repeats=repeats,
                workers=workers,
                agent_slots=agent_slots,
                agent_admission_mode=agent_admission_mode,
                planning_delay_s=planning_delay_s,
                abort_retry_delay_s=abort_retry_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                tokens_per_operation=tokens_per_operation,
                policy_artifact=policy_artifact,
                policy_epsilon=policy_epsilon,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
                object_lock_scheduler=object_lock_scheduler,
                object_lock_priority_burst=object_lock_priority_burst,
                prelock_wait_budget_s=prelock_wait_budget_s,
                prelock_wait_budget_mode=prelock_wait_budget_mode,
                prelock_lease_mode=prelock_lease_mode,
                agent_execution_mode=agent_execution_mode,
                snapshot_timing=snapshot_timing,
            )
            aggregates = aggregate_retry_runs(runs)
            best_baseline = max(
                aggregates,
                key=lambda aggregate: (
                    _family_policy_score(aggregate),
                    aggregate["commit_rate"],
                    aggregate["committed_throughput"],
                ),
            )
            baselines[profile_name] = {
                "strategy": best_baseline["strategy"],
                "score": _family_policy_score(best_baseline),
                "aggregate": best_baseline,
            }
    candidates = []
    for read_heavy_strategy in strategies:
        for cold_read_heavy_strategy in cold_strategies:
            for hot_write_strategy in hot_strategies:
                for fallback_strategy in fallback_candidates:
                    for hot_write_threshold in hot_write_thresholds:
                        for hotspot_threshold in hotspot_thresholds:
                            artifact = _family_policy_artifact(
                                read_heavy_strategy=read_heavy_strategy,
                                cold_read_heavy_strategy=cold_read_heavy_strategy,
                                hot_write_strategy=hot_write_strategy,
                                fallback_strategy=fallback_strategy,
                                hot_write_ratio_threshold=hot_write_threshold,
                                hotspot_probability_threshold=hotspot_threshold,
                                base_artifact=policy_artifact,
                            )
                            for budget_s in prelock_budget_values:
                                for lease_mode in prelock_lease_modes:
                                    for execution_mode_candidate in execution_modes:
                                        for snapshot_timing_candidate in snapshot_timings:
                                            for scheduler_candidate in object_lock_schedulers:
                                                profile_aggregates = []
                                                score = 0.0
                                                for profile_name, workload_kind, workload, workload_config in profiles:
                                                    runs = run_retry_matrix(
                                                        workload,
                                                        ("adaptive-hybrid",),
                                                        workload_kind=workload_kind,
                                                        policy_variant=policy_variant,
                                                        task_count=task_count,
                                                        seed=seed,
                                                        repeats=repeats,
                                                        workers=workers,
                                                        agent_slots=agent_slots,
                                                        agent_admission_mode=agent_admission_mode,
                                                        planning_delay_s=planning_delay_s,
                                                        abort_retry_delay_s=abort_retry_delay_s,
                                                        latency_distribution=latency_distribution,
                                                        latency_cv=latency_cv,
                                                        latency_max_s=latency_max_s,
                                                        max_attempts=max_attempts,
                                                        tokens_per_operation=tokens_per_operation,
                                                        policy_artifact=artifact,
                                                        policy_epsilon=policy_epsilon,
                                                        background_workers=background_workers,
                                                        background_interval_s=background_interval_s,
                                                        background_strategy=background_strategy,
                                                        object_lock_scheduler=scheduler_candidate,
                                                        object_lock_priority_burst=object_lock_priority_burst,
                                                        prelock_wait_budget_s=budget_s,
                                                        prelock_wait_budget_mode=prelock_wait_budget_mode,
                                                        prelock_lease_mode=lease_mode,
                                                        agent_execution_mode=execution_mode_candidate,
                                                        snapshot_timing=snapshot_timing_candidate,
                                                    )
                                                    aggregate = aggregate_retry_runs(runs)[0]
                                                    profile_score = _family_policy_score(aggregate)
                                                    baseline = baselines.get(profile_name)
                                                    relative_score = None
                                                    if baseline is not None:
                                                        relative_score = profile_score / max(
                                                            0.000001,
                                                            float(baseline["score"]),
                                                        )
                                                    score += (
                                                        _balanced_relative_family_score(float(relative_score))
                                                        if mode == "baseline-balanced"
                                                        else float(relative_score)
                                                        if mode == "baseline-relative"
                                                        else profile_score
                                                    )
                                                    profile_aggregates.append(
                                                        {
                                                            "profile": profile_name,
                                                            "workload_kind": workload_kind,
                                                            "workload_config": workload_config,
                                                            "aggregate": aggregate,
                                                            **({"baseline": baseline} if baseline is not None else {}),
                                                            **(
                                                                {"relative_score": relative_score}
                                                                if relative_score is not None
                                                                else {}
                                                            ),
                                                        }
                                                    )
                                                candidates.append(
                                                    {
                                                        "read_heavy_strategy": read_heavy_strategy,
                                                        "cold_read_heavy_strategy": cold_read_heavy_strategy,
                                                        "hot_write_strategy": hot_write_strategy,
                                                        "fallback_strategy": fallback_strategy,
                                                        "hot_write_ratio_threshold": hot_write_threshold,
                                                        "hotspot_probability_threshold": hotspot_threshold,
                                                        "prelock_wait_budget_s": budget_s,
                                                        "prelock_lease_mode": lease_mode,
                                                        "agent_execution_mode": execution_mode_candidate,
                                                        "snapshot_timing": snapshot_timing_candidate,
                                                        "object_lock_scheduler": scheduler_candidate,
                                                        "score": score,
                                                        "profile_aggregates": profile_aggregates,
                                                        "artifact": artifact,
                                                    }
                                                )
    best = max(
        candidates,
        key=lambda row: (
            row["score"],
            min(
                item["aggregate"]["commit_rate"]
                for item in row["profile_aggregates"]
            ),
            sum(
                item["aggregate"]["committed_throughput"]
                for item in row["profile_aggregates"]
            ),
        ),
    )
    return {
        "training_method": "offline-family-policy-search",
        "selection_metric": (
            "sum(relative_score - 2.0 * max(0, 1 - relative_score)) across profiles"
            if mode == "baseline-balanced"
            else
            "sum(profile_score / best_baseline_score) across profiles"
            if mode == "baseline-relative"
            else (
                "sum(committed_throughput * commit_rate - 0.10 * p99_latency "
                "- wasted_tokens_per_task / 100000) across profiles"
            )
        ),
        "score_mode": mode,
        "baseline_strategies": baseline_names,
        "profile_count": len(profiles),
        "best_read_heavy_strategy": best["read_heavy_strategy"],
        "best_cold_read_heavy_strategy": best["cold_read_heavy_strategy"],
        "best_hot_write_strategy": best["hot_write_strategy"],
        "best_fallback_strategy": best["fallback_strategy"],
        "best_hot_write_ratio_threshold": best["hot_write_ratio_threshold"],
        "best_hotspot_probability_threshold": best[
            "hotspot_probability_threshold"
        ],
        "best_prelock_wait_budget_s": best["prelock_wait_budget_s"],
        "best_prelock_lease_mode": best["prelock_lease_mode"],
        "best_agent_execution_mode": best["agent_execution_mode"],
        "best_snapshot_timing": best["snapshot_timing"],
        "best_object_lock_scheduler": best["object_lock_scheduler"],
        "best_artifact": best["artifact"],
        "profile_results": best["profile_aggregates"],
        "candidates": candidates,
    }


def _family_policy_score(aggregate: Mapping[str, Any]) -> float:
    return (
        float(aggregate["committed_throughput"])
        * max(0.01, float(aggregate["commit_rate"]))
        - float(aggregate["agent_latency_p99_s"]) * 0.10
        - float(aggregate["estimated_wasted_tokens_per_task"]) / 100000.0
    )


def _balanced_relative_family_score(relative_score: float) -> float:
    return float(relative_score) - 2.0 * max(0.0, 1.0 - float(relative_score))


def _family_search_profiles(
    profile_names: Iterable[str],
) -> Tuple[Tuple[str, str, AgentWorkload, Dict[str, Any]], ...]:
    names = tuple(
        str(name).strip().lower()
        for name in profile_names
        if str(name).strip()
    )
    if not names:
        raise ValueError("family search needs at least one profile")
    profiles = []
    for name in names:
        if name == "ycsb-low":
            config = YCSBConfig(
                record_count=512,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.95,
                update_weight=0.05,
                zipf_theta=0.0,
                hotspot_fraction=0.0,
                hotspot_access_probability=0.0,
            )
            profiles.append(
                (
                    name,
                    "ycsb",
                    build_agent_workload("ycsb", "semantic", ycsb_config=config),
                    dataclasses.asdict(config),
                )
            )
            continue
        if name == "ycsb-medium":
            config = YCSBConfig(
                record_count=128,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            )
            profiles.append(
                (
                    name,
                    "ycsb",
                    build_agent_workload("ycsb", "semantic", ycsb_config=config),
                    dataclasses.asdict(config),
                )
            )
            continue
        if name == "ycsb-high":
            config = YCSBConfig(
                record_count=64,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.50,
                update_weight=0.50,
                zipf_theta=0.99,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.75,
            )
            profiles.append(
                (
                    name,
                    "ycsb",
                    build_agent_workload("ycsb", "semantic", ycsb_config=config),
                    dataclasses.asdict(config),
                )
            )
            continue
        if name == "tpcc-medium":
            config = TPCCConfig(
                warehouses=2,
                districts_per_warehouse=3,
                customers_per_district=60,
                items=200,
                order_lines=8,
                transaction_mix=(("new_order", 1.0),),
            )
            profiles.append(
                (
                    name,
                    "tpcc",
                    build_agent_workload("tpcc", "semantic", tpcc_config=config),
                    dataclasses.asdict(config),
                )
            )
            continue
        if name == "tpcc-high":
            config = TPCCConfig(
                warehouses=1,
                districts_per_warehouse=2,
                customers_per_district=40,
                items=100,
                order_lines=10,
                transaction_mix=(("new_order", 1.0),),
            )
            profiles.append(
                (
                    name,
                    "tpcc",
                    build_agent_workload("tpcc", "semantic", tpcc_config=config),
                    dataclasses.asdict(config),
                )
            )
            continue
        raise ValueError(f"unsupported family-search profile: {name}")
    return tuple(profiles)


def _family_policy_artifact(
    *,
    read_heavy_strategy: str,
    cold_read_heavy_strategy: Optional[str] = None,
    hot_write_strategy: str = "adaptive-op-strict",
    fallback_strategy: str = "tictoc-full",
    hot_write_ratio_threshold: float = 0.30,
    hotspot_probability_threshold: float = 0.70,
    base_artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    read_heavy = str(read_heavy_strategy).strip().lower()
    cold_read_heavy = (
        str(cold_read_heavy_strategy).strip().lower()
        if cold_read_heavy_strategy is not None
        else read_heavy
    )
    hot_write = str(hot_write_strategy).strip().lower()
    fallback = str(fallback_strategy).strip().lower()
    artifact = dict(base_artifact or {})
    artifact.update(
        {
            "artifact_type": "atcc-family-policy-artifact",
            "artifact_version": 1,
            "source_system": "data-agent-runtime",
            "family_policy_table": {
                "read_heavy_strategy": read_heavy,
                "cold_read_heavy_strategy": cold_read_heavy,
                "hot_write_strategy": hot_write,
                "fallback_strategy": fallback,
                "read_heavy_write_ratio": 0.25,
                "cold_hotspot_probability_threshold": 0.01,
                "hot_write_ratio_threshold": float(hot_write_ratio_threshold),
                "hotspot_probability_threshold": float(
                    hotspot_probability_threshold
                ),
            },
        }
    )
    return artifact


def aggregate_retry_runs(runs: Sequence[RetryRunSummary]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[RetryRunSummary]] = {}
    for run in runs:
        grouped.setdefault((run.strategy, run.policy_variant), []).append(run)
    rows = []
    for (strategy, variant), group in sorted(grouped.items()):
        operation_policy_counts: Counter[str] = Counter()
        operation_rule_counts: Counter[str] = Counter()
        conflict_object_counts: Counter[str] = Counter()
        conflict_object_class_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        selected_strategy_counts: Counter[str] = Counter()
        fast_through_strategy_counts: Counter[str] = Counter()
        for run in group:
            operation_policy_counts.update(run.operation_policy_counts)
            operation_rule_counts.update(run.operation_rule_counts)
            conflict_object_counts.update(run.conflict_object_counts)
            conflict_object_class_counts.update(run.conflict_object_class_counts)
            action_counts.update(run.action_counts)
            selected_strategy_counts.update(run.selected_strategy_counts)
            if str(run.fast_through_strategy):
                fast_through_strategy_counts[str(run.fast_through_strategy)] += 1
        task_count = sum(run.task_count for run in group)
        committed = sum(run.committed_tasks for run in group)
        elapsed = sum(run.elapsed_s for run in group)
        attempts = sum(run.total_attempts for run in group)
        prelock_wait_s = sum(run.prelock_wait_s for run in group)
        background_commits = sum(run.background_commits for run in group)
        background_aborts = sum(run.background_aborts for run in group)
        lease_refresh_regenerations = sum(
            run.lease_refresh_regenerations for run in group
        )
        lease_refresh_replayed_operations = sum(
            int(getattr(run, "lease_refresh_replayed_operations", 0) or 0)
            for run in group
        )
        lease_refresh_rebased_writes = sum(
            int(getattr(run, "lease_refresh_rebased_writes", 0) or 0)
            for run in group
        )
        prelock_queue_depth_sum = sum(run.prelock_queue_depth_sum for run in group)
        prelock_queue_depth_observations = sum(
            run.prelock_queue_depth_observations for run in group
        )
        prelock_queue_depth_max = max(
            (run.prelock_queue_depth_max for run in group),
            default=0,
        )
        prelock_handoff_count = sum(run.prelock_handoff_count for run in group)
        prelock_committing_enters = sum(run.prelock_committing_enters for run in group)
        prelock_committing_exits = sum(run.prelock_committing_exits for run in group)
        task_latencies_s = tuple(
            latency
            for run in group
            for latency in run.task_latencies_s
        )
        task_operation_counts = tuple(
            count
            for run in group
            for count in run.task_operation_counts
        )
        wasted_attempts = sum(run.wasted_attempts for run in group)
        estimated_base_tokens = sum(run.estimated_tokens for run in group)
        estimated_base_wasted_tokens = sum(
            run.estimated_wasted_tokens for run in group
        )
        refresh_tokens = _estimated_refresh_tokens(
            lease_refresh_regenerations,
            task_operation_counts,
            group[0].tokens_per_operation,
            replayed_operations=lease_refresh_replayed_operations,
            rebased_writes=lease_refresh_rebased_writes,
        )
        estimated_tokens = estimated_base_tokens + refresh_tokens
        estimated_wasted_tokens = estimated_base_wasted_tokens + refresh_tokens
        rows.append(
            {
                "strategy": strategy,
                "policy_variant": variant,
                "runs": len(group),
                "task_count": task_count,
                "workers": group[0].workers,
                "agent_slots": group[0].agent_slots,
                "agent_admission_mode": group[0].agent_admission_mode,
                "max_attempts": group[0].max_attempts,
                "planning_delay_s": group[0].planning_delay_s,
                "abort_retry_delay_s": group[0].abort_retry_delay_s,
                "committed_tasks": committed,
                "final_failed_tasks": sum(run.final_failed_tasks for run in group),
                "rejected_tasks": sum(run.rejected_tasks for run in group),
                "commit_rate": committed / task_count if task_count else 0.0,
                "total_attempts": attempts,
                "attempts_per_task": attempts / task_count if task_count else 0.0,
                "attempts_per_commit": attempts / committed if committed else 0.0,
                "wasted_attempts": wasted_attempts,
                "wasted_attempts_per_task": wasted_attempts / task_count
                if task_count
                else 0.0,
                "conflict_aborts": sum(run.conflict_aborts for run in group),
                "conflict_object_counts": dict(
                    sorted(conflict_object_counts.items())
                ),
                "conflict_object_class_counts": dict(
                    sorted(conflict_object_class_counts.items())
                ),
                "background_workers": group[0].background_workers,
                "object_lock_scheduler": group[0].object_lock_scheduler,
                "object_lock_priority_burst": group[0].object_lock_priority_burst,
                "prelock_wait_budget_s": group[0].prelock_wait_budget_s,
                "prelock_wait_budget_mode": group[0].prelock_wait_budget_mode,
                "prelock_lease_mode": group[0].prelock_lease_mode,
                "agent_execution_mode": group[0].agent_execution_mode,
                "snapshot_timing": group[0].snapshot_timing,
                "stage_phase_counts": dict(
                    sorted(
                        Counter(
                            phase
                            for run in group
                            for phase, count in run.stage_phase_counts.items()
                            for _ in range(int(count))
                        ).items()
                    )
                ),
                "background_commits": background_commits,
                "background_aborts": background_aborts,
                "lease_refresh_regenerations": lease_refresh_regenerations,
                "lease_refresh_regenerations_per_task": (
                    lease_refresh_regenerations / task_count if task_count else 0.0
                ),
                "lease_refresh_replayed_operations": (
                    lease_refresh_replayed_operations
                ),
                "lease_refresh_replayed_operations_per_task": (
                    lease_refresh_replayed_operations / task_count
                    if task_count
                    else 0.0
                ),
                "lease_refresh_rebased_writes": lease_refresh_rebased_writes,
                "lease_refresh_rebased_writes_per_task": (
                    lease_refresh_rebased_writes / task_count if task_count else 0.0
                ),
                "background_throughput": background_commits / elapsed
                if elapsed > 0
                else 0.0,
                "prelock_wait_s": prelock_wait_s,
                "prelock_wait_per_task_s": prelock_wait_s / task_count
                if task_count
                else 0.0,
                "prelock_queue_depth_sum": prelock_queue_depth_sum,
                "prelock_queue_depth_observations": prelock_queue_depth_observations,
                "prelock_queue_depth_avg": (
                    prelock_queue_depth_sum / prelock_queue_depth_observations
                    if prelock_queue_depth_observations
                    else 0.0
                ),
                "prelock_queue_depth_max": prelock_queue_depth_max,
                "prelock_handoff_count": prelock_handoff_count,
                "prelock_handoff_per_task": (
                    prelock_handoff_count / task_count if task_count else 0.0
                ),
                "prelock_committing_enters": prelock_committing_enters,
                "prelock_committing_exits": prelock_committing_exits,
                "task_latencies_s": list(task_latencies_s),
                "agent_latency_avg_s": _mean(task_latencies_s),
                "agent_latency_p50_s": _percentile(task_latencies_s, 50.0),
                "agent_latency_p95_s": _percentile(task_latencies_s, 95.0),
                "agent_latency_p99_s": _percentile(task_latencies_s, 99.0),
                "agent_latency_max_s": max(task_latencies_s)
                if task_latencies_s
                else 0.0,
                "task_operation_counts": list(task_operation_counts),
                "tokens_per_operation": group[0].tokens_per_operation,
                "estimated_base_tokens": estimated_base_tokens,
                "estimated_base_wasted_tokens": estimated_base_wasted_tokens,
                "estimated_refresh_tokens": refresh_tokens,
                "estimated_refresh_tokens_per_task": refresh_tokens / task_count
                if task_count
                else 0.0,
                "estimated_tokens": estimated_tokens,
                "estimated_tokens_per_task": estimated_tokens / task_count
                if task_count
                else 0.0,
                "estimated_wasted_tokens": estimated_wasted_tokens,
                "estimated_wasted_tokens_per_task": (
                    estimated_wasted_tokens / task_count if task_count else 0.0
                ),
                "elapsed_s": elapsed,
                "committed_throughput": committed / elapsed if elapsed > 0 else 0.0,
                "operation_policy_counts": dict(sorted(operation_policy_counts.items())),
                "operation_rule_counts": dict(sorted(operation_rule_counts.items())),
                "action_counts": dict(sorted(action_counts.items())),
                "selected_strategy_counts": dict(
                    sorted(selected_strategy_counts.items())
                ),
                "fast_through_strategy_counts": dict(
                    sorted(fast_through_strategy_counts.items())
                ),
            }
        )
    return rows


def aggregate_selected_baseline_pairs(
    runs: Sequence[RetryRunSummary],
    *,
    hybrid_strategy: str = "adaptive-hybrid",
) -> Dict[str, Any]:
    """Compare adaptive-hybrid runs with the baseline family they selected.

    The normal aggregate compares each strategy across separate runs.  For the
    family selector, low/medium profiles often intentionally reduce to a
    traditional family such as OCC or TicToc.  This paired view answers the
    narrower question: when hybrid selected one family for a seed, how did it
    compare with that same family on the same seed?
    """

    normalized_hybrid = str(hybrid_strategy).strip().lower()
    by_seed_strategy = {
        (int(run.seed), str(run.strategy).strip().lower()): run for run in runs
    }
    pairs: List[Dict[str, Any]] = []
    missing = 0
    mixed = 0
    for run in runs:
        if str(run.strategy).strip().lower() != normalized_hybrid:
            continue
        selected_counts = {
            str(strategy).strip().lower(): int(count)
            for strategy, count in dict(run.selected_strategy_counts or {}).items()
            if str(strategy).strip() and int(count) > 0
        }
        if len(selected_counts) != 1:
            mixed += 1
            continue
        selected_strategy = next(iter(selected_counts))
        baseline = by_seed_strategy.get((int(run.seed), selected_strategy))
        if baseline is None:
            missing += 1
            continue
        baseline_tps = baseline.committed_throughput
        hybrid_tps = run.committed_throughput
        ratio = hybrid_tps / baseline_tps if baseline_tps > 0 else 0.0
        pairs.append(
            {
                "seed": int(run.seed),
                "selected_strategy": selected_strategy,
                "hybrid_tps": hybrid_tps,
                "baseline_tps": baseline_tps,
                "hybrid_vs_selected_baseline": ratio,
                "hybrid_conflict_aborts": int(run.conflict_aborts),
                "baseline_conflict_aborts": int(baseline.conflict_aborts),
                "hybrid_p99_s": _percentile(run.task_latencies_s, 99.0),
                "baseline_p99_s": _percentile(baseline.task_latencies_s, 99.0),
            }
        )
    ratios = [float(pair["hybrid_vs_selected_baseline"]) for pair in pairs]
    return {
        "hybrid_strategy": normalized_hybrid,
        "paired_runs": len(pairs),
        "missing_baseline_runs": missing,
        "mixed_selection_runs": mixed,
        "mean_hybrid_vs_selected_baseline": _mean(ratios),
        "min_hybrid_vs_selected_baseline": min(ratios) if ratios else 0.0,
        "max_hybrid_vs_selected_baseline": max(ratios) if ratios else 0.0,
        "pairs": pairs,
    }


def _estimated_refresh_tokens(
    refresh_count: int,
    task_operation_counts: Sequence[int],
    tokens_per_operation: float,
    *,
    replayed_operations: int = 0,
    rebased_writes: int = 0,
) -> float:
    if replayed_operations > 0:
        return int(replayed_operations) * max(0.0, float(tokens_per_operation))
    if rebased_writes > 0:
        return 0.0
    if refresh_count <= 0 or not task_operation_counts:
        return 0.0
    operation_count = _mean(task_operation_counts)
    return (
        int(refresh_count)
        * max(0.0, float(operation_count))
        * max(0.0, float(tokens_per_operation))
    )


def _run_one_retry(
    workload: AgentWorkload,
    tasks: Sequence[AgentTask],
    strategy: str,
    *,
    workload_kind: str,
    policy_variant: str,
    seed: int,
    workers: int,
    agent_slots: int,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
    operation_policy: Optional[OperationPolicyTable] = None,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
) -> RetryRunSummary:
    policy = operation_policy or _operation_policy(
        workload_kind,
        policy_variant,
        policy_artifact=policy_artifact,
        policy_epsilon=policy_epsilon,
    )
    manager = AgentTransactionManager(
        operation_policy=policy,
        transaction_atcc_policy=_transaction_atcc_policy(workload_kind),
        object_lock_queue_policy=object_lock_scheduler,
        object_lock_priority_burst=object_lock_priority_burst,
        prelock_wait_budget_s=prelock_wait_budget_s,
        prelock_wait_budget_mode=prelock_wait_budget_mode,
    )
    register_workload(manager, workload)
    interaction_gate = (
        threading.Semaphore(int(agent_slots)) if int(agent_slots) > 0 else None
    )
    background_stop = threading.Event()
    background_counts: Counter[str] = Counter()
    background_lock = threading.Lock()
    background_threads = [
        threading.Thread(
            target=_run_background_worker,
            args=(
                manager,
                workload,
                workload_kind,
                int(index),
                int(seed),
                background_stop,
                background_counts,
                background_lock,
            ),
            kwargs={
                "interval_s": background_interval_s,
                "strategy": background_strategy,
            },
            daemon=True,
        )
        for index in range(max(0, int(background_workers)))
    ]
    execution_mode = str(agent_execution_mode or "legacy").strip().lower()
    timing = str(snapshot_timing or "before-planning").strip().lower()
    abort_retry_delay = max(0.0, float(abort_retry_delay_s))
    family_table = (
        dict(policy_artifact.get("family_policy_table", {}) or {})
        if isinstance(policy_artifact, Mapping)
        else {}
    )
    family_policy = None
    if is_atcc_family_strategy(strategy):
        family_policy = (
            ATCCFamilyPolicyTable.from_dict(family_table)
            if family_table
            else ATCCFamilyPolicyTable.default()
        ).resolve_for_task_window(tasks, workload_kind=workload_kind)
    family_decisions = (
        {
            task.task_id: family_policy.select_task(
                task,
                workload_kind=workload_kind,
            )
            for task in tasks
        }
        if family_policy is not None
        else {}
    )

    def run_task(task: AgentTask) -> RetryTaskOutcome:
        started_at = time.perf_counter()
        results: List[Any] = []
        family_decision = family_decisions.get(task.task_id)
        execution_strategy = (
            family_decision.selected_strategy
            if family_decision is not None
            else str(strategy)
        )
        seed_strategy = execution_strategy if family_decision is not None else strategy
        rng = random.Random(f"{seed}:{seed_strategy}:{policy_variant}:{task.task_id}")
        admit_before_begin = (
            str(agent_admission_mode) == "before-begin"
            and interaction_gate is not None
        )
        for attempt in range(max_attempts):
            strict_atcc = manager.cc_registry.records_operation_feedback(
                execution_strategy
            )
            delay_s = _sample_latency_s(
                rng,
                mean_s=planning_delay_s,
                distribution=latency_distribution,
                cv=latency_cv,
                max_s=latency_max_s,
            )
            defer_prelocks = (
                str(prelock_lease_mode) == "defer-until-after-planning"
                and strict_atcc
                and execution_mode == "legacy"
            )
            admitted = False
            if admit_before_begin:
                interaction_gate.acquire()
                admitted = True
            try:
                if execution_mode in {"staged", "staged-local"}:
                    stage_local_atcc = (
                        execution_mode == "staged-local"
                        and strict_atcc
                    )
                    stage_delays = _stage_delay_plan(task, delay_s)
                    if timing == "after-planning":
                        for _, stage_delay_s in stage_delays:
                            _sleep_agent_delay(
                                stage_delay_s,
                                interaction_gate=interaction_gate,
                                admitted=admitted,
                            )
                    txn = prepare_task_transaction(
                        manager,
                        task,
                        strategy=None if stage_local_atcc else execution_strategy,
                        runtime_context={
                            "retry_count": attempt,
                            "agent_interval_s": delay_s,
                            "agent_phase": (
                                ""
                                if stage_local_atcc
                                else _agent_phase_for_staged_task(task)
                            ),
                            "agent_slots": int(agent_slots),
                            "agent_admission_mode": str(agent_admission_mode),
                            "agent_execution_mode": execution_mode,
                            "snapshot_timing": timing,
                            "operation_candidate_scope": _operation_candidate_scope(
                                task,
                                execution_strategy,
                            ),
                        },
                        populate=False,
                    )
                    stage_policy_decisions = []

                    def apply_stage_local_plan(phase: str) -> bool:
                        if not stage_local_atcc or txn.state != TransactionState.ACTIVE:
                            return bool(txn.prelocked_targets)
                        stage_task = task_stage_view(task, phase)
                        if not stage_task.candidates:
                            txn.precomputed_operation_policy_decisions = tuple(
                                stage_policy_decisions
                            )
                            return False
                        stage_context = {
                            **dict(stage_task.context),
                            "retry_count": attempt,
                            "agent_interval_s": delay_s,
                            "agent_phase": str(phase),
                            "agent_slots": int(agent_slots),
                            "agent_admission_mode": str(agent_admission_mode),
                            "agent_execution_mode": execution_mode,
                            "snapshot_timing": timing,
                            "operation_candidate_scope": _operation_candidate_scope(
                                stage_task,
                                execution_strategy,
                            ),
                        }
                        metadata = {
                            "workload": stage_task.workload,
                            "task_type": stage_task.task_type,
                            "request": stage_task.request,
                            "context": stage_context,
                            "retry_count": attempt,
                            "agent_interval_s": delay_s,
                            "agent_phase": str(phase),
                        }
                        prelock_targets, decisions = (
                            manager.cc_registry.pre_snapshot_operation_plan(
                                execution_strategy,
                                stage_task.candidates,
                                metadata=metadata,
                            )
                        )
                        effective_decisions = txn.replace_prelocks_for_stage(
                            prelock_targets,
                            decisions,
                            reason=f"stage-local-atcc-{phase}",
                        )
                        stage_policy_decisions.extend(effective_decisions)
                        txn.precomputed_operation_policy_decisions = tuple(
                            stage_policy_decisions
                        )
                        txn.metadata["agent_phase"] = str(phase)
                        txn.metadata["context"] = stage_context
                        return bool(txn.prelocked_targets)

                    should_yield_prelocks = (
                        str(prelock_lease_mode)
                        in {"yield-during-planning", "yield-refresh-regenerate"}
                        and strict_atcc
                        and txn.state == TransactionState.ACTIVE
                        and bool(txn.prelocked_targets)
                        and not stage_local_atcc
                    )
                    should_refresh_regenerate = (
                        str(prelock_lease_mode) == "yield-refresh-regenerate"
                        and (
                            should_yield_prelocks
                            or (
                                strict_atcc
                                and _task_has_hotspot_refresh_pressure(task)
                            )
                        )
                    )
                    if should_yield_prelocks and timing == "before-planning":
                        txn.yield_prelocks_for_planning()
                    if timing == "after-planning":
                        _replay_task_stages(txn, task)
                    else:
                        refreshed_and_replayed = False
                        for phase, stage_delay_s in stage_delays:
                            operations = stage_operations(task, phase)
                            if stage_local_atcc and not operations:
                                continue
                            _sleep_agent_delay(
                                stage_delay_s,
                                interaction_gate=interaction_gate,
                                admitted=admitted,
                            )
                            stage_has_prelocks = apply_stage_local_plan(phase)
                            if phase == "commit" and txn.state == TransactionState.ACTIVE:
                                if should_yield_prelocks:
                                    txn.reacquire_yielded_prelocks()
                                stage_refresh_regenerate = (
                                    str(prelock_lease_mode)
                                    == "yield-refresh-regenerate"
                                    and (
                                        should_refresh_regenerate
                                        or (
                                            stage_local_atcc
                                            and (
                                                stage_has_prelocks
                                                or _task_has_hotspot_refresh_pressure(task)
                                            )
                                        )
                                    )
                                )
                                if stage_refresh_regenerate:
                                    if _can_object_refresh_commit_writes(
                                        task,
                                        yielded_prelocks=should_yield_prelocks,
                                    ):
                                        populate_task_stage(txn, task, "commit")
                                        if txn.state != TransactionState.ACTIVE:
                                            refreshed_and_replayed = True
                                            break
                                        txn.refresh_candidate_write_bases(
                                            _object_refresh_commit_targets(txn, task),
                                            reason="ycsb-commit-object-refresh",
                                            clear_read_set=True,
                                            candidate_scope="best",
                                        )
                                    else:
                                        txn.refresh_snapshot_for_regeneration()
                                        replay_phases = _refresh_replay_phases(
                                            task,
                                            yielded_prelocks=should_yield_prelocks,
                                        )
                                        replayed_operations = _replay_task_stages(
                                            txn, task, phases=replay_phases
                                        )
                                        txn.record_refresh_replay(
                                            operation_count=replayed_operations,
                                            phases=replay_phases or (),
                                        )
                                    refreshed_and_replayed = True
                                    break
                            populate_task_stage(txn, task, phase)
                        if (
                            should_yield_prelocks
                            and not refreshed_and_replayed
                            and txn.state == TransactionState.ACTIVE
                        ):
                            txn.reacquire_yielded_prelocks()
                        if (
                            not refreshed_and_replayed
                            and txn.state == TransactionState.ACTIVE
                            and not txn.candidates
                        ):
                            populate_task_stage(txn, task, "commit")
                    if txn.state != TransactionState.ACTIVE:
                        result = txn.result
                        if result is not None:
                            results.append(result)
                            if (
                                result.committed
                                or result.state == TransactionState.REJECTED
                            ):
                                break
                            if attempt + 1 < max_attempts:
                                _sleep_abort_retry_delay(
                                    result,
                                    abort_retry_delay,
                                    interaction_gate=interaction_gate,
                                    admitted=admitted,
                                )
                            continue
                    result = txn.commit(strategy=execution_strategy)
                    results.append(result)
                    if result.committed or result.state == TransactionState.REJECTED:
                        break
                    if attempt + 1 < max_attempts:
                        _sleep_abort_retry_delay(
                            result,
                            abort_retry_delay,
                            interaction_gate=interaction_gate,
                            admitted=admitted,
                        )
                    continue
                if defer_prelocks and delay_s:
                    if interaction_gate is None or admitted:
                        time.sleep(delay_s)
                    else:
                        with interaction_gate:
                            time.sleep(delay_s)
                txn = prepare_task_transaction(
                    manager,
                    task,
                    strategy=execution_strategy,
                    runtime_context={
                        "retry_count": attempt,
                        "agent_interval_s": delay_s,
                        "agent_phase": _agent_phase_for_task(task, attempt),
                        "agent_slots": int(agent_slots),
                        "agent_admission_mode": str(agent_admission_mode),
                        "operation_candidate_scope": _operation_candidate_scope(
                            task,
                            execution_strategy,
                        ),
                    },
                )
                should_yield_prelocks = (
                    str(prelock_lease_mode)
                    in {"yield-during-planning", "yield-refresh-regenerate"}
                    and strict_atcc
                    and txn.state == TransactionState.ACTIVE
                )
                should_refresh_regenerate = (
                    str(prelock_lease_mode) == "yield-refresh-regenerate"
                )
                if should_yield_prelocks:
                    txn.yield_prelocks_for_planning()
                if delay_s:
                    if not defer_prelocks:
                        if interaction_gate is None or admitted:
                            time.sleep(delay_s)
                        else:
                            with interaction_gate:
                                time.sleep(delay_s)
            finally:
                if admitted:
                    interaction_gate.release()
            if should_yield_prelocks and txn.state == TransactionState.ACTIVE:
                txn.reacquire_yielded_prelocks()
                if should_refresh_regenerate:
                    txn.refresh_snapshot_for_regeneration()
                    populate_task_transaction(txn, task)
                    txn.record_refresh_replay(
                        operation_count=_task_operation_count(task),
                    )
            result = txn.commit(strategy=execution_strategy)
            results.append(result)
            if result.committed or result.state == TransactionState.REJECTED:
                break
            if attempt + 1 < max_attempts:
                _sleep_abort_retry_delay(
                    result,
                    abort_retry_delay,
                    interaction_gate=interaction_gate,
                    admitted=admitted,
                )
        operation_count = _task_operation_count(task)
        wasted_attempts = _wasted_attempt_count(results)
        return RetryTaskOutcome(
            results=tuple(results),
            latency_s=time.perf_counter() - started_at,
            operation_count=operation_count,
            wasted_attempts=wasted_attempts,
            estimated_tokens=(
                len(results) * operation_count * max(0.0, float(tokens_per_operation))
            ),
            estimated_wasted_tokens=(
                wasted_attempts
                * operation_count
                * max(0.0, float(tokens_per_operation))
            ),
            selected_strategy=execution_strategy,
        )

    started_at = time.perf_counter()
    for thread in background_threads:
        thread.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_outcomes = list(executor.map(run_task, tasks))
    finally:
        background_stop.set()
        for thread in background_threads:
            thread.join(timeout=2.0)
    elapsed_s = time.perf_counter() - started_at

    grouped_results = [outcome.results for outcome in task_outcomes]
    all_results = [result for group in grouped_results for result in group]
    committed_tasks = sum(1 for group in grouped_results if group[-1].committed)
    rejected_tasks = sum(
        1 for group in grouped_results if group[-1].state == TransactionState.REJECTED
    )
    final_failed_tasks = len(grouped_results) - committed_tasks - rejected_tasks
    conflict_aborts = sum(
        1
        for result in all_results
        if result.state == TransactionState.ABORTED
        and result.reason != "traditional_k_loser"
    )
    operation_policy_counts: Counter[str] = Counter()
    operation_rule_counts: Counter[str] = Counter()
    conflict_object_counts: Counter[str] = Counter()
    conflict_object_class_counts: Counter[str] = Counter()
    lease_refresh_regenerations = 0
    lease_refresh_replayed_operations = 0
    lease_refresh_rebased_writes = 0
    prelock_wait_s = 0.0
    prelock_queue_depth_sum = 0.0
    prelock_queue_depth_observations = 0
    prelock_queue_depth_max = 0
    prelock_handoff_count = 0
    prelock_committing_enters = 0
    prelock_committing_exits = 0
    for trace in manager.traces():
        metadata = dict(trace.get("metadata", {}) or {})
        is_background_trace = str(metadata.get("workload", "")).startswith(
            "background-"
        )
        if not is_background_trace:
            prelock_wait_s += float(trace.get("prelock_wait_s", 0.0) or 0.0)
            for depth in dict(trace.get("prelock_target_queue_depth", {}) or {}).values():
                normalized_depth = max(0, int(depth))
                prelock_queue_depth_sum += normalized_depth
                prelock_queue_depth_observations += 1
                prelock_queue_depth_max = max(
                    prelock_queue_depth_max,
                    normalized_depth,
                )
            prelock_handoff_count += sum(
                max(0, int(count))
                for count in dict(
                    trace.get("prelock_target_handoff_count", {}) or {}
                ).values()
            )
        for event in trace.get("events", ()):
            detail = event.get("detail", {}) or {}
            if event.get("kind") == "finish":
                if is_background_trace:
                    continue
                for object_id in detail.get("conflict_object_ids", ()) or ():
                    normalized = str(object_id)
                    conflict_object_counts[normalized] += 1
                    conflict_object_class_counts[operation_object_class(normalized)] += 1
                continue
            if event.get("kind") == "refresh_regenerate":
                if not is_background_trace:
                    lease_refresh_regenerations += 1
                continue
            if event.get("kind") == "refresh_rebase":
                if not is_background_trace:
                    lease_refresh_regenerations += 1
                    lease_refresh_rebased_writes += max(
                        0,
                        int(detail.get("refreshed_writes", 0) or 0),
                    )
                continue
            if event.get("kind") == "refresh_replay":
                if not is_background_trace:
                    lease_refresh_replayed_operations += max(
                        0,
                        int(detail.get("operation_count", 0) or 0),
                    )
                continue
            if event.get("kind") == "prelock_committing_enter":
                if not is_background_trace:
                    prelock_committing_enters += 1
                continue
            if event.get("kind") == "prelock_committing_exit":
                if not is_background_trace:
                    prelock_committing_exits += 1
                continue
            if event.get("kind") != "validate":
                continue
            for decision in detail.get("operation_policy_decisions", ()):
                policy = decision.get("policy")
                rule = decision.get("rule")
                if policy:
                    operation_policy_counts[str(policy)] += 1
                if rule:
                    operation_rule_counts[str(rule)] += 1
    return RetryRunSummary(
        workload=workload.name,
        strategy=strategy,
        policy_variant=policy_variant,
        seed=seed,
        task_count=len(tasks),
        workers=workers,
        agent_slots=agent_slots,
        agent_admission_mode=str(agent_admission_mode),
        max_attempts=max_attempts,
        planning_delay_s=planning_delay_s,
        abort_retry_delay_s=max(0.0, float(abort_retry_delay_s)),
        latency_distribution=latency_distribution,
        committed_tasks=committed_tasks,
        final_failed_tasks=final_failed_tasks,
        rejected_tasks=rejected_tasks,
        total_attempts=len(all_results),
        conflict_aborts=conflict_aborts,
        conflict_object_counts=dict(sorted(conflict_object_counts.items())),
        conflict_object_class_counts=dict(sorted(conflict_object_class_counts.items())),
        operation_policy_counts=dict(sorted(operation_policy_counts.items())),
        operation_rule_counts=dict(sorted(operation_rule_counts.items())),
        action_counts=dict(sorted(Counter(result.action for result in all_results).items())),
        prelock_wait_s=prelock_wait_s,
        task_latencies_s=tuple(outcome.latency_s for outcome in task_outcomes),
        task_operation_counts=tuple(
            outcome.operation_count for outcome in task_outcomes
        ),
        wasted_attempts=sum(outcome.wasted_attempts for outcome in task_outcomes),
        tokens_per_operation=max(0.0, float(tokens_per_operation)),
        estimated_tokens=sum(outcome.estimated_tokens for outcome in task_outcomes),
        estimated_wasted_tokens=sum(
            outcome.estimated_wasted_tokens for outcome in task_outcomes
        ),
        selected_strategy_counts=dict(
            sorted(
                Counter(
                    outcome.selected_strategy for outcome in task_outcomes
                ).items()
            )
        ),
        background_workers=max(0, int(background_workers)),
        background_commits=int(background_counts.get("committed", 0)),
        background_aborts=int(background_counts.get("aborted", 0)),
        lease_refresh_regenerations=lease_refresh_regenerations,
        lease_refresh_replayed_operations=lease_refresh_replayed_operations,
        lease_refresh_rebased_writes=lease_refresh_rebased_writes,
        prelock_queue_depth_sum=prelock_queue_depth_sum,
        prelock_queue_depth_observations=prelock_queue_depth_observations,
        prelock_queue_depth_max=prelock_queue_depth_max,
        prelock_handoff_count=prelock_handoff_count,
        prelock_committing_enters=prelock_committing_enters,
        prelock_committing_exits=prelock_committing_exits,
        object_lock_scheduler=str(object_lock_scheduler),
        object_lock_priority_burst=int(object_lock_priority_burst),
        prelock_wait_budget_s=float(prelock_wait_budget_s),
        prelock_wait_budget_mode=str(prelock_wait_budget_mode),
        prelock_lease_mode=str(prelock_lease_mode),
        agent_execution_mode=execution_mode,
        snapshot_timing=timing,
        stage_phase_counts=dict(
            sorted(
                _stage_phase_counts(
                    tasks,
                    include_empty=execution_mode != "staged-local",
                ).items()
            )
        ),
        elapsed_s=elapsed_s,
    )


def _operation_policy(
    workload_kind: str,
    variant: str,
    *,
    policy_artifact: Optional[Mapping[str, Any]] = None,
    policy_epsilon: Optional[float] = None,
) -> OperationPolicyTable:
    workload = str(workload_kind).strip().lower()
    normalized = str(variant).strip().lower()
    policy: Optional[OperationPolicyTable] = None
    if normalized in {"optimistic", "occ", "all-optimistic"}:
        policy = OperationPolicyTable(
            rules=(),
            fallback_policy="optimistic",
            name="all-optimistic-operation-atcc-table",
        )
    elif normalized in {"rl", "q-learning", "qlearning"}:
        if workload == "tpcc":
            policy = OperationPolicyTable.tpcc_rl_atcc()
        elif workload == "ycsb":
            policy = OperationPolicyTable.ycsb_rl_atcc()
    elif normalized in {"phase-rl", "phase-aware", "paper-atcc", "original-atcc"}:
        if workload == "tpcc":
            policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        elif workload == "ycsb":
            policy = OperationPolicyTable.ycsb_phase_rl_atcc()
    elif workload == "tpcc":
        base = OperationPolicyTable.tpcc_atcc()
        variants = {
            "default": base,
            "aggressive": dataclasses.replace(
                base,
                min_feedback_observations=2,
                exact_key_min_observations=2,
                lock_wait_cost_per_s=25.0,
                lock_overhead_cost=0.01,
            ),
            "balanced": dataclasses.replace(
                base,
                min_feedback_observations=4,
                exact_key_min_observations=3,
                lock_wait_cost_per_s=50.0,
                lock_overhead_cost=0.02,
            ),
            "conservative": dataclasses.replace(
                base,
                min_feedback_observations=10,
                exact_key_min_observations=6,
                lock_wait_cost_per_s=120.0,
                lock_overhead_cost=0.05,
            ),
        }
        if normalized not in variants:
            raise ValueError(f"unsupported policy variant: {variant}")
        policy = variants[normalized]
    elif workload == "ycsb":
        base = OperationPolicyTable.ycsb_atcc()
        variants = {
            "default": base,
            "strict-tuned": OperationPolicyTable.ycsb_strict_tuned_atcc(),
            "ycsb-strict-tuned": OperationPolicyTable.ycsb_strict_tuned_atcc(),
            "paper-strict-tuned": OperationPolicyTable.ycsb_strict_tuned_atcc(),
            "aggressive": dataclasses.replace(
                base,
                min_feedback_observations=5,
                exact_key_min_observations=3,
                lock_wait_cost_per_s=50.0,
                lock_overhead_cost=0.02,
            ),
            "balanced": dataclasses.replace(
                base,
                min_feedback_observations=12,
                exact_key_min_observations=6,
                lock_wait_cost_per_s=100.0,
                lock_overhead_cost=0.03,
            ),
            "conservative": dataclasses.replace(
                base,
                min_feedback_observations=16,
                exact_key_min_observations=8,
                lock_wait_cost_per_s=200.0,
                lock_overhead_cost=0.05,
            ),
        }
        if normalized not in variants:
            raise ValueError(f"unsupported policy variant: {variant}")
        policy = variants[normalized]
    else:
        raise ValueError(f"unsupported workload kind: {workload_kind}")
    if policy is None:
        raise ValueError(f"unsupported policy variant: {variant}")
    if policy_artifact is not None:
        return policy.with_learned_state(
            policy_artifact,
            policy_epsilon=0.0 if policy_epsilon is None else policy_epsilon,
        )
    if policy_epsilon is not None:
        policy = policy.with_learned_state(
            policy.to_dict(),
            policy_epsilon=policy_epsilon,
        )
    return policy


def _transaction_atcc_policy(workload_kind: str) -> TransactionAwareATCCModule:
    workload = str(workload_kind or "").strip().lower()
    if workload == "tpcc" or "tpcc" in workload:
        return TransactionAwareATCCModule.tpcc()
    return TransactionAwareATCCModule.ycsb()


def _run_background_worker(
    manager: AgentTransactionManager,
    workload: AgentWorkload,
    workload_kind: str,
    worker_index: int,
    seed: int,
    stop_event: threading.Event,
    counts: Counter[str],
    counts_lock: threading.Lock,
    *,
    interval_s: float,
    strategy: str,
) -> None:
    targets = _background_targets(workload, workload_kind)
    if not targets:
        return
    rng = random.Random(f"background:{seed}:{worker_index}")
    attempt = 0
    while not stop_event.is_set():
        target = rng.choice(targets)
        txn = manager.begin(
            f"background-{worker_index}-{attempt}",
            {
                "workload": f"background-{workload_kind}",
                "task_type": "stored-procedure",
                "context": {"background_worker": worker_index},
            },
        )
        candidate = txn.add_candidate("background", quality=1.0, gen_cost=0.0)
        if str(workload_kind).lower() == "tpcc":
            candidate.delta(target, 1)
        else:
            candidate.overwrite(target, f"bg:{worker_index}:{attempt}")
        try:
            with manager.object_locks.acquire((target,), priority=0):
                result = txn.commit(strategy=strategy, max_regenerations=0)
            key = "committed" if result.committed else "aborted"
        except Exception:
            key = "aborted"
        with counts_lock:
            counts[key] += 1
        attempt += 1
        if interval_s > 0:
            stop_event.wait(float(interval_s))


def _background_targets(workload: AgentWorkload, workload_kind: str) -> Tuple[str, ...]:
    kind = str(workload_kind).strip().lower()
    object_ids = [spec.object_id for spec in workload.objects()]
    if kind == "tpcc":
        targets = [object_id for object_id in object_ids if object_id.endswith(":next_order_id")]
        if targets:
            return tuple(targets)
        return tuple(object_id for object_id in object_ids if ":district:" in object_id)
    if kind == "ycsb":
        config = getattr(workload, "config", None)
        hot_record_count = 0
        if config is not None:
            hotspot_fraction = float(getattr(config, "hotspot_fraction", 0.0) or 0.0)
            record_count = int(getattr(config, "record_count", 0) or 0)
            if hotspot_fraction > 0.0 and record_count > 0:
                hot_record_count = max(1, int(record_count * hotspot_fraction))
        if hot_record_count > 0:
            prefixes = tuple(
                f"ycsb:record:{record}:"
                for record in range(hot_record_count)
            )
            hot = [
                object_id
                for object_id in object_ids
                if object_id.startswith(prefixes)
            ]
            if hot:
                return tuple(hot)
        hot = [object_id for object_id in object_ids if object_id.startswith("ycsb:record:0:")]
        return tuple(hot or object_ids[:1])
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry-aware operation ATCC experiments.")
    parser.add_argument("--mode", choices=("matrix", "search", "family-search"), default="matrix")
    parser.add_argument("--workload", choices=("tpcc", "ycsb"), default="tpcc")
    parser.add_argument("--strategies", default="occ,2pl,adaptive-op-strict")
    parser.add_argument("--policy-variant", default="default")
    parser.add_argument(
        "--profile-name",
        default="",
        help=(
            "Optional logical profile name, such as ycsb-high. When provided, "
            "profile eval overrides can be loaded for this run."
        ),
    )
    parser.add_argument(
        "--profile-eval-overrides",
        default="",
        help=(
            "JSON object keyed by profile name. Supports workload_config plus "
            "planning_delay_ms/planning_delay_s and "
            "abort_retry_delay_ms/abort_retry_delay_s."
        ),
    )
    parser.add_argument(
        "--profile-eval-overrides-file",
        type=Path,
        help=(
            "Path to a JSON object keyed by profile name. File overrides are "
            "loaded before --profile-eval-overrides, so inline JSON can "
            "override file defaults."
        ),
    )
    parser.add_argument("--search-variants", default="aggressive,balanced,conservative")
    parser.add_argument(
        "--family-search-read-heavy-strategies",
        default="mvcc-full,tictoc-full",
        help=(
            "Comma-separated CC families to try for read-heavy family-level "
            "ATCC policy search."
        ),
    )
    parser.add_argument(
        "--family-search-profiles",
        default="",
        help=(
            "Comma-separated fixed workload profiles for joint family-level "
            "ATCC policy search. Empty keeps the current workload/config search."
        ),
    )
    parser.add_argument(
        "--family-search-cold-read-heavy-strategies",
        default="",
        help=(
            "Comma-separated CC families to try for low-conflict read-heavy "
            "profiles. Empty reuses --family-search-read-heavy-strategies."
        ),
    )
    parser.add_argument(
        "--family-search-hot-write-strategies",
        default="",
        help=(
            "Comma-separated CC families to try for hot-write/high-conflict "
            "profiles. Empty uses adaptive-op-strict."
        ),
    )
    parser.add_argument(
        "--family-search-fallback-strategies",
        default="",
        help=(
            "Comma-separated CC families to try when no workload-specific "
            "family rule matches. Empty uses tictoc-full."
        ),
    )
    parser.add_argument(
        "--family-search-hot-write-ratio-thresholds",
        default="",
        help=(
            "Comma-separated write-ratio thresholds for routing tasks into "
            "the hot-write family. Empty uses 0.30."
        ),
    )
    parser.add_argument(
        "--family-search-hotspot-probability-thresholds",
        default="",
        help=(
            "Comma-separated hotspot-probability thresholds for routing tasks "
            "into the hot-write family. Empty uses 0.70."
        ),
    )
    parser.add_argument(
        "--family-search-prelock-wait-budget-ms-values",
        default="",
        help=(
            "Comma-separated prelock wait-budget candidates in milliseconds. "
            "Empty uses --prelock-wait-budget-ms."
        ),
    )
    parser.add_argument(
        "--family-search-prelock-lease-modes",
        default="",
        help=(
            "Comma-separated prelock lease mode candidates. Empty uses "
            "--prelock-lease-mode."
        ),
    )
    parser.add_argument(
        "--family-search-agent-execution-modes",
        default="",
        help=(
            "Comma-separated agent execution mode candidates. Empty uses "
            "--agent-execution-mode."
        ),
    )
    parser.add_argument(
        "--family-search-snapshot-timings",
        default="",
        help=(
            "Comma-separated snapshot timing candidates. Empty uses "
            "--snapshot-timing."
        ),
    )
    parser.add_argument(
        "--family-search-object-lock-schedulers",
        default="",
        help=(
            "Comma-separated object lock scheduler candidates. Empty uses "
            "--object-lock-scheduler."
        ),
    )
    parser.add_argument(
        "--family-search-baseline-strategies",
        default="",
        help=(
            "Comma-separated strategies to run once per profile as training "
            "baselines for baseline-relative family policy search."
        ),
    )
    parser.add_argument(
        "--family-search-score-mode",
        choices=("absolute", "baseline-relative", "baseline-balanced"),
        default="absolute",
        help="Family-search objective: absolute profile score or ratio to baseline.",
    )
    parser.add_argument("--task-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--strategy-order",
        choices=(
            "given",
            "rotate",
            "pair-selected-baseline",
            "interleave-selected-baseline",
            "interleave-all-strategies",
        ),
        default="given",
        help=(
            "Order strategies within each repeat. rotate reduces run-order bias; "
            "pair-selected-baseline runs adaptive-hybrid next to the family it "
            "selects when that family is in --strategies; "
            "interleave-selected-baseline splits that selected pair into "
            "alternating task blocks; interleave-all-strategies splits every "
            "strategy into comparable task blocks."
        ),
    )
    parser.add_argument(
        "--interleave-blocks",
        type=int,
        default=4,
        help=(
            "Task blocks per repeat for --strategy-order "
            "interleave-selected-baseline or interleave-all-strategies. Other "
            "order modes ignore this."
        ),
    )
    parser.add_argument(
        "--hybrid-fast-through",
        action="store_true",
        help=(
            "When adaptive-hybrid selects one family for a repeat, execute that "
            "family directly only when it selected adaptive-op-strict, and "
            "report the run as adaptive-hybrid."
        ),
    )
    parser.add_argument(
        "--hybrid-selected-fast-through",
        action="store_true",
        help=(
            "When adaptive-hybrid selects one family for a repeat, execute that "
            "selected family directly, including traditional families, and "
            "report the run as adaptive-hybrid."
        ),
    )
    parser.add_argument(
        "--agent-slots",
        type=int,
        default=0,
        help="Limit concurrent agent/tool interaction sleeps; 0 means unlimited.",
    )
    parser.add_argument(
        "--agent-admission-mode",
        choices=("planning-only", "before-begin"),
        default="planning-only",
        help=(
            "How agent_slots are applied. 'planning-only' preserves the legacy "
            "behavior and may begin/prelock before waiting for a planning slot; "
            "'before-begin' admits a task to a planning slot before transaction "
            "begin/prelock so slot queueing is outside the transaction."
        ),
    )
    parser.add_argument("--planning-delay-ms", type=float, default=2.0)
    parser.add_argument(
        "--abort-retry-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Extra agent-side replanning delay after an abort before retrying. "
            "Default 0 preserves earlier experiments."
        ),
    )
    parser.add_argument(
        "--latency-distribution",
        choices=("fixed", "lognormal", "pareto"),
        default="fixed",
    )
    parser.add_argument("--latency-cv", type=float, default=0.8)
    parser.add_argument("--latency-max-ms", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--tokens-per-operation",
        type=float,
        default=2703.0,
        help=(
            "Estimated LLM tokens per issued operation for retry-waste accounting; "
            "2703 follows the ATCC paper's profiled flight-booking cost model."
        ),
    )
    parser.add_argument("--background-workers", type=int, default=0)
    parser.add_argument("--background-interval-ms", type=float, default=0.0)
    parser.add_argument("--background-strategy", default="occ")
    parser.add_argument(
        "--object-lock-scheduler",
        choices=("race", "priority", "bounded-priority"),
        default="race",
        help=(
            "Object-lock waiter admission policy. 'race' preserves notify-all "
            "thread scheduling; 'priority' grants the highest-priority waiter first; "
            "'bounded-priority' gives low-priority waiters a turn after two "
            "priority grants."
        ),
    )
    parser.add_argument(
        "--object-lock-priority-burst",
        type=int,
        default=2,
        help=(
            "For bounded-priority object locks, grant at most this many "
            "positive-priority waiters before giving a low-priority waiter a turn."
        ),
    )
    parser.add_argument(
        "--prelock-wait-budget-ms",
        type=float,
        default=0.0,
        help=(
            "Optional pre-snapshot lock wait budget. 0 means wait indefinitely; "
            "for adaptive ATCC prelocks only, when exceeded, pessimistic decisions "
            "fall back to optimistic for that attempt. Full 2PL-pre remains a "
            "blocking baseline."
        ),
    )
    parser.add_argument(
        "--prelock-wait-budget-mode",
        choices=("transaction", "object"),
        default="transaction",
        help=(
            "Budget fallback granularity. 'transaction' downgrades all adaptive "
            "prelock decisions on timeout; 'object' downgrades only objects whose "
            "lock wait exceeded the budget."
        ),
    )
    parser.add_argument(
        "--prelock-lease-mode",
        choices=(
            "hold",
            "yield-during-planning",
            "yield-refresh-regenerate",
            "defer-until-after-planning",
        ),
        default="hold",
        help=(
            "Adaptive ATCC prelock lease behavior. 'hold' keeps prelocks through "
            "agent planning; 'yield-during-planning' releases them during the "
            "simulated planning delay and reacquires before commit; "
            "'yield-refresh-regenerate' also refreshes the snapshot and rebuilds "
            "task candidates after reacquiring; "
            "'defer-until-after-planning' starts ATCC prelocking/snapshot after "
            "the simulated planning delay. 2PL-pre is available as an explicit "
            "blocking/oracle baseline but is not part of the default comparison."
        ),
    )
    parser.add_argument(
        "--agent-execution-mode",
        choices=("legacy", "staged", "staged-local"),
        default="legacy",
        help=(
            "Agent task execution model. 'legacy' preserves retry-attempt phase "
            "behavior; 'staged' derives phase from workload-provided "
            "explore/refine/commit stages; 'staged-local' recomputes ATCC "
            "operation policy on each non-empty stage."
        ),
    )
    parser.add_argument(
        "--snapshot-timing",
        choices=("before-planning", "after-planning"),
        default="before-planning",
        help=(
            "Fair begin/snapshot timing applied to every strategy in staged "
            "mode. 'before-planning' begins before simulated planning delay; "
            "'after-planning' begins after it."
        ),
    )
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        help="Load a trained ATCC policy-table artifact before running.",
    )
    parser.add_argument(
        "--policy-epsilon",
        type=float,
        help=(
            "Override exploration probability for loaded/online policy tables; "
            "loaded artifacts default to 0.0."
        ),
    )
    parser.add_argument(
        "--family-policy-output",
        type=Path,
        help="Write the best family-level ATCC policy artifact to this JSON file.",
    )
    parser.add_argument("--output", type=Path)

    ycsb = parser.add_argument_group("YCSB options")
    ycsb.add_argument("--records", type=int, default=64)
    ycsb.add_argument("--fields", type=int, default=4)
    ycsb.add_argument("--requests-per-task", type=int, default=4)
    ycsb.add_argument("--candidates", type=int, default=4)
    ycsb.add_argument("--read-weight", type=float, default=0.2)
    ycsb.add_argument("--update-weight", type=float, default=0.8)
    ycsb.add_argument("--zipf-theta", type=float, default=0.99)
    ycsb.add_argument("--hotspot-fraction", type=float, default=0.0)
    ycsb.add_argument("--hotspot-access-probability", type=float, default=0.0)

    tpcc = parser.add_argument_group("TPC-C options")
    tpcc.add_argument("--warehouses", type=int, default=1)
    tpcc.add_argument("--districts-per-warehouse", type=int, default=2)
    tpcc.add_argument("--customers-per-district", type=int, default=32)
    tpcc.add_argument("--items", type=int, default=128)
    tpcc.add_argument("--initial-stock", type=int, default=1000)
    tpcc.add_argument("--order-lines", type=int, default=8)
    tpcc.add_argument("--transaction-mix", default="new_order:1.0")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    profile_eval_overrides = _profile_eval_override_for_args(args)
    _apply_profile_eval_override(args, profile_eval_overrides)
    workload, workload_config = _build_workload(args)
    policy_artifact = _load_policy_artifact(args.policy_artifact)
    policy_artifact_schema = atcc_artifact_schema_status(policy_artifact)
    effective_policy_epsilon = (
        args.policy_epsilon
        if args.policy_epsilon is not None
        else (0.0 if policy_artifact is not None else None)
    )
    if args.mode == "search":
        report: Dict[str, Any] = {
            "mode": "search",
            "workload": workload.name,
            "workload_kind": args.workload,
            "profile_name": args.profile_name,
            "profile_eval_overrides": profile_eval_overrides,
            "workload_config": workload_config,
            "task_count": args.task_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "workers": args.workers,
            "agent_slots": args.agent_slots,
            "agent_admission_mode": args.agent_admission_mode,
            "planning_delay_s": args.planning_delay_ms / 1000.0,
            "abort_retry_delay_s": args.abort_retry_delay_ms / 1000.0,
            "latency_distribution": args.latency_distribution,
            "latency_cv": args.latency_cv,
            "latency_max_s": args.latency_max_ms / 1000.0,
            "max_attempts": args.max_attempts,
            "tokens_per_operation": args.tokens_per_operation,
            "policy_artifact": str(args.policy_artifact) if args.policy_artifact else "",
            "policy_artifact_schema": policy_artifact_schema,
            "policy_epsilon": effective_policy_epsilon,
            "background_workers": args.background_workers,
            "background_interval_s": args.background_interval_ms / 1000.0,
            "background_strategy": args.background_strategy,
            "object_lock_scheduler": args.object_lock_scheduler,
            "object_lock_priority_burst": args.object_lock_priority_burst,
            "prelock_wait_budget_s": args.prelock_wait_budget_ms / 1000.0,
            "prelock_wait_budget_mode": args.prelock_wait_budget_mode,
            "prelock_lease_mode": args.prelock_lease_mode,
            "agent_execution_mode": args.agent_execution_mode,
            "snapshot_timing": args.snapshot_timing,
            "search": search_policy_variants(
                workload,
                workload_kind=args.workload,
                variants=_split_csv(args.search_variants),
                task_count=args.task_count,
                seed=args.seed,
                repeats=args.repeats,
                workers=args.workers,
                agent_slots=args.agent_slots,
                agent_admission_mode=args.agent_admission_mode,
                planning_delay_s=args.planning_delay_ms / 1000.0,
                abort_retry_delay_s=args.abort_retry_delay_ms / 1000.0,
                latency_distribution=args.latency_distribution,
                latency_cv=args.latency_cv,
                latency_max_s=args.latency_max_ms / 1000.0,
                max_attempts=args.max_attempts,
                tokens_per_operation=args.tokens_per_operation,
                policy_artifact=policy_artifact,
                policy_epsilon=effective_policy_epsilon,
                background_workers=args.background_workers,
                background_interval_s=args.background_interval_ms / 1000.0,
                background_strategy=args.background_strategy,
                object_lock_scheduler=args.object_lock_scheduler,
                object_lock_priority_burst=args.object_lock_priority_burst,
                prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
                prelock_wait_budget_mode=args.prelock_wait_budget_mode,
                prelock_lease_mode=args.prelock_lease_mode,
                agent_execution_mode=args.agent_execution_mode,
                snapshot_timing=args.snapshot_timing,
            ),
        }
    elif args.mode == "family-search":
        family_search_profiles = _split_csv(args.family_search_profiles)
        if family_search_profiles:
            family_search = search_family_policy_profiles(
                family_search_profiles,
                read_heavy_strategies=_split_csv(args.family_search_read_heavy_strategies),
                cold_read_heavy_strategies=_split_csv(
                    args.family_search_cold_read_heavy_strategies
                ),
                hot_write_strategies=_split_csv(
                    args.family_search_hot_write_strategies
                ),
                fallback_strategies=_split_csv(
                    args.family_search_fallback_strategies
                ),
                hot_write_ratio_thresholds=_split_csv_floats(
                    args.family_search_hot_write_ratio_thresholds
                ),
                hotspot_probability_thresholds=_split_csv_floats(
                    args.family_search_hotspot_probability_thresholds
                ),
                prelock_wait_budget_candidates_s=tuple(
                    value / 1000.0
                    for value in _split_csv_floats(
                        args.family_search_prelock_wait_budget_ms_values
                    )
                ),
                prelock_lease_mode_candidates=_split_csv(
                    args.family_search_prelock_lease_modes
                ),
                agent_execution_mode_candidates=_split_csv(
                    args.family_search_agent_execution_modes
                ),
                snapshot_timing_candidates=_split_csv(
                    args.family_search_snapshot_timings
                ),
                object_lock_scheduler_candidates=_split_csv(
                    args.family_search_object_lock_schedulers
                ),
                baseline_strategies=_split_csv(
                    args.family_search_baseline_strategies
                ),
                score_mode=args.family_search_score_mode,
                task_count=args.task_count,
                seed=args.seed,
                repeats=args.repeats,
                workers=args.workers,
                agent_slots=args.agent_slots,
                agent_admission_mode=args.agent_admission_mode,
                planning_delay_s=args.planning_delay_ms / 1000.0,
                abort_retry_delay_s=args.abort_retry_delay_ms / 1000.0,
                latency_distribution=args.latency_distribution,
                latency_cv=args.latency_cv,
                latency_max_s=args.latency_max_ms / 1000.0,
                max_attempts=args.max_attempts,
                tokens_per_operation=args.tokens_per_operation,
                policy_variant=args.policy_variant,
                policy_artifact=policy_artifact,
                policy_epsilon=effective_policy_epsilon,
                background_workers=args.background_workers,
                background_interval_s=args.background_interval_ms / 1000.0,
                background_strategy=args.background_strategy,
                object_lock_scheduler=args.object_lock_scheduler,
                object_lock_priority_burst=args.object_lock_priority_burst,
                prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
                prelock_wait_budget_mode=args.prelock_wait_budget_mode,
                prelock_lease_mode=args.prelock_lease_mode,
                agent_execution_mode=args.agent_execution_mode,
                snapshot_timing=args.snapshot_timing,
            )
        else:
            family_search = search_family_policy_variants(
                workload,
                workload_kind=args.workload,
                read_heavy_strategies=_split_csv(args.family_search_read_heavy_strategies),
                task_count=args.task_count,
                seed=args.seed,
                repeats=args.repeats,
                workers=args.workers,
                agent_slots=args.agent_slots,
                agent_admission_mode=args.agent_admission_mode,
                planning_delay_s=args.planning_delay_ms / 1000.0,
                latency_distribution=args.latency_distribution,
                latency_cv=args.latency_cv,
                latency_max_s=args.latency_max_ms / 1000.0,
                max_attempts=args.max_attempts,
                tokens_per_operation=args.tokens_per_operation,
                policy_variant=args.policy_variant,
                policy_artifact=policy_artifact,
                policy_epsilon=effective_policy_epsilon,
                background_workers=args.background_workers,
                background_interval_s=args.background_interval_ms / 1000.0,
                background_strategy=args.background_strategy,
                object_lock_scheduler=args.object_lock_scheduler,
                object_lock_priority_burst=args.object_lock_priority_burst,
                prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
                prelock_wait_budget_mode=args.prelock_wait_budget_mode,
                prelock_lease_mode=args.prelock_lease_mode,
                agent_execution_mode=args.agent_execution_mode,
                snapshot_timing=args.snapshot_timing,
            )
        if args.family_policy_output is not None:
            args.family_policy_output.parent.mkdir(parents=True, exist_ok=True)
            args.family_policy_output.write_text(
                json.dumps(
                    family_search["best_artifact"],
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        report = {
            "mode": "family-search",
            "workload": workload.name,
            "workload_kind": args.workload,
            "profile_name": args.profile_name,
            "profile_eval_overrides": profile_eval_overrides,
            "workload_config": workload_config,
            "task_count": args.task_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "workers": args.workers,
            "agent_slots": args.agent_slots,
            "agent_admission_mode": args.agent_admission_mode,
            "planning_delay_s": args.planning_delay_ms / 1000.0,
            "abort_retry_delay_s": args.abort_retry_delay_ms / 1000.0,
            "latency_distribution": args.latency_distribution,
            "latency_cv": args.latency_cv,
            "latency_max_s": args.latency_max_ms / 1000.0,
            "max_attempts": args.max_attempts,
            "tokens_per_operation": args.tokens_per_operation,
            "policy_artifact": str(args.policy_artifact) if args.policy_artifact else "",
            "policy_artifact_schema": policy_artifact_schema,
            "policy_epsilon": effective_policy_epsilon,
            "family_search_profiles": family_search_profiles,
            "family_policy_output": (
                str(args.family_policy_output) if args.family_policy_output else ""
            ),
            "background_workers": args.background_workers,
            "background_interval_s": args.background_interval_ms / 1000.0,
            "background_strategy": args.background_strategy,
            "object_lock_scheduler": args.object_lock_scheduler,
            "object_lock_priority_burst": args.object_lock_priority_burst,
            "prelock_wait_budget_s": args.prelock_wait_budget_ms / 1000.0,
            "prelock_wait_budget_mode": args.prelock_wait_budget_mode,
            "prelock_lease_mode": args.prelock_lease_mode,
            "agent_execution_mode": args.agent_execution_mode,
            "snapshot_timing": args.snapshot_timing,
            "family_search": family_search,
        }
    else:
        runs = run_retry_matrix(
            workload,
            _split_csv(args.strategies),
            workload_kind=args.workload,
            policy_variant=args.policy_variant,
            task_count=args.task_count,
            seed=args.seed,
            repeats=args.repeats,
            workers=args.workers,
            strategy_order=args.strategy_order,
            interleave_blocks=args.interleave_blocks,
            hybrid_fast_through=args.hybrid_fast_through,
            hybrid_selected_fast_through=args.hybrid_selected_fast_through,
            agent_slots=args.agent_slots,
            agent_admission_mode=args.agent_admission_mode,
            planning_delay_s=args.planning_delay_ms / 1000.0,
            abort_retry_delay_s=args.abort_retry_delay_ms / 1000.0,
            latency_distribution=args.latency_distribution,
            latency_cv=args.latency_cv,
            latency_max_s=args.latency_max_ms / 1000.0,
            max_attempts=args.max_attempts,
            tokens_per_operation=args.tokens_per_operation,
            policy_artifact=policy_artifact,
            policy_epsilon=effective_policy_epsilon,
            background_workers=args.background_workers,
            background_interval_s=args.background_interval_ms / 1000.0,
            background_strategy=args.background_strategy,
            object_lock_scheduler=args.object_lock_scheduler,
            object_lock_priority_burst=args.object_lock_priority_burst,
            prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
            prelock_wait_budget_mode=args.prelock_wait_budget_mode,
            prelock_lease_mode=args.prelock_lease_mode,
            agent_execution_mode=args.agent_execution_mode,
            snapshot_timing=args.snapshot_timing,
        )
        report = {
            "mode": "matrix",
            "workload": workload.name,
            "workload_kind": args.workload,
            "profile_name": args.profile_name,
            "profile_eval_overrides": profile_eval_overrides,
            "workload_config": workload_config,
            "strategies": list(_split_csv(args.strategies)),
            "policy_variant": args.policy_variant,
            "task_count": args.task_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "workers": args.workers,
            "strategy_order": args.strategy_order,
            "interleave_blocks": args.interleave_blocks,
            "hybrid_fast_through": args.hybrid_fast_through,
            "hybrid_selected_fast_through": args.hybrid_selected_fast_through,
            "agent_slots": args.agent_slots,
            "agent_admission_mode": args.agent_admission_mode,
            "planning_delay_s": args.planning_delay_ms / 1000.0,
            "abort_retry_delay_s": args.abort_retry_delay_ms / 1000.0,
            "latency_distribution": args.latency_distribution,
            "latency_cv": args.latency_cv,
            "latency_max_s": args.latency_max_ms / 1000.0,
            "max_attempts": args.max_attempts,
            "tokens_per_operation": args.tokens_per_operation,
            "policy_artifact": str(args.policy_artifact) if args.policy_artifact else "",
            "policy_artifact_schema": policy_artifact_schema,
            "policy_epsilon": effective_policy_epsilon,
            "background_workers": args.background_workers,
            "background_interval_s": args.background_interval_ms / 1000.0,
            "background_strategy": args.background_strategy,
            "object_lock_scheduler": args.object_lock_scheduler,
            "object_lock_priority_burst": args.object_lock_priority_burst,
            "prelock_wait_budget_s": args.prelock_wait_budget_ms / 1000.0,
            "prelock_wait_budget_mode": args.prelock_wait_budget_mode,
            "prelock_lease_mode": args.prelock_lease_mode,
            "agent_execution_mode": args.agent_execution_mode,
            "snapshot_timing": args.snapshot_timing,
            "runs": [run.to_dict() for run in runs],
            "aggregates": aggregate_retry_runs(runs),
            "selected_baseline_pairs": aggregate_selected_baseline_pairs(runs),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _build_workload(args: argparse.Namespace) -> Tuple[AgentWorkload, Dict[str, Any]]:
    if args.workload == "ycsb":
        config = YCSBConfig(
            record_count=args.records,
            field_count=args.fields,
            requests_per_task=args.requests_per_task,
            candidates_per_task=args.candidates,
            read_weight=args.read_weight,
            update_weight=args.update_weight,
            zipf_theta=args.zipf_theta,
            hotspot_fraction=args.hotspot_fraction,
            hotspot_access_probability=args.hotspot_access_probability,
        )
        return build_agent_workload("ycsb", "semantic", ycsb_config=config), dataclasses.asdict(config)
    config = TPCCConfig(
        warehouses=args.warehouses,
        districts_per_warehouse=args.districts_per_warehouse,
        customers_per_district=args.customers_per_district,
        items=args.items,
        initial_stock=args.initial_stock,
        order_lines=args.order_lines,
        candidates_per_task=args.candidates,
        transaction_mix=_parse_mix(args.transaction_mix),
    )
    return build_agent_workload("tpcc", "semantic", tpcc_config=config), dataclasses.asdict(config)


def _profile_eval_override_for_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = _merge_profile_eval_overrides(
        _load_profile_eval_overrides_file(args.profile_eval_overrides_file),
        _parse_profile_eval_overrides(args.profile_eval_overrides),
    )
    profile_name = str(getattr(args, "profile_name", "") or "").strip()
    if not profile_name:
        return {}
    return dict(
        overrides.get(profile_name)
        or overrides.get(profile_name.lower())
        or {}
    )


def _apply_profile_eval_override(
    args: argparse.Namespace,
    override: Mapping[str, Any],
) -> None:
    if not override:
        return
    if "planning_delay_s" in override:
        args.planning_delay_ms = float(override["planning_delay_s"]) * 1000.0
    if "abort_retry_delay_s" in override:
        args.abort_retry_delay_ms = float(override["abort_retry_delay_s"]) * 1000.0
    if "object_lock_scheduler" in override:
        args.object_lock_scheduler = str(override["object_lock_scheduler"])
    if "prelock_wait_budget_s" in override:
        args.prelock_wait_budget_ms = (
            float(override["prelock_wait_budget_s"]) * 1000.0
        )
    if "prelock_wait_budget_mode" in override:
        args.prelock_wait_budget_mode = str(override["prelock_wait_budget_mode"])
    if "prelock_lease_mode" in override:
        args.prelock_lease_mode = str(override["prelock_lease_mode"])
    config = dict(override.get("workload_config", {}) or {})
    key_to_arg = {
        "record_count": "records",
        "field_count": "fields",
        "candidates_per_task": "candidates",
    }
    for key, value in config.items():
        arg_name = key_to_arg.get(str(key), str(key))
        if hasattr(args, arg_name):
            setattr(args, arg_name, value)


def _load_profile_eval_overrides_file(
    path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    return _parse_profile_eval_overrides(path.read_text(encoding="utf-8"))


def _parse_profile_eval_overrides(raw: str) -> Dict[str, Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("profile eval overrides must be a JSON object")
    parsed: Dict[str, Dict[str, Any]] = {}
    for profile, row in data.items():
        if not isinstance(row, Mapping):
            raise ValueError("profile eval override rows must be JSON objects")
        normalized = dict(row)
        if "planning_delay_ms" in normalized:
            normalized["planning_delay_s"] = (
                float(normalized.pop("planning_delay_ms")) / 1000.0
            )
        if "abort_retry_delay_ms" in normalized:
            normalized["abort_retry_delay_s"] = (
                float(normalized.pop("abort_retry_delay_ms")) / 1000.0
            )
        if "prelock_wait_budget_ms" in normalized:
            normalized["prelock_wait_budget_s"] = (
                float(normalized.pop("prelock_wait_budget_ms")) / 1000.0
            )
        parsed[str(profile)] = normalized
    return parsed


def _merge_profile_eval_overrides(
    *sources: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        for profile, row in dict(source or {}).items():
            target = dict(merged.get(str(profile), {}) or {})
            for key, value in dict(row or {}).items():
                if key == "workload_config":
                    workload_config = dict(target.get("workload_config", {}) or {})
                    workload_config.update(dict(value or {}))
                    target[key] = workload_config
                else:
                    target[key] = value
            merged[str(profile)] = target
    return merged


def _load_policy_artifact(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_mix(value: str) -> Tuple[Tuple[str, float], ...]:
    entries = []
    for item in _split_csv(value):
        name, weight = item.split(":", 1)
        entries.append((name.strip(), float(weight)))
    return tuple(entries)


def _sample_latency_s(
    rng: random.Random,
    *,
    mean_s: float,
    distribution: str,
    cv: float,
    max_s: float,
) -> float:
    mean = max(0.0, float(mean_s))
    if mean <= 0:
        return 0.0
    if distribution == "fixed":
        value = mean
    elif distribution == "lognormal":
        coefficient = max(0.01, float(cv))
        sigma = math.sqrt(math.log(1.0 + coefficient * coefficient))
        mu = math.log(mean) - 0.5 * sigma * sigma
        value = rng.lognormvariate(mu, sigma)
    elif distribution == "pareto":
        alpha = max(1.1, 1.0 + 1.0 / max(0.01, float(cv)))
        xm = mean * (alpha - 1.0) / alpha
        value = xm * (rng.paretovariate(alpha))
    else:
        raise ValueError(f"unsupported latency distribution: {distribution}")
    cap = float(max_s) if max_s and max_s > 0 else mean * 5.0
    return min(value, cap)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(percentile))) / 100.0
    position = rank * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _task_operation_count(task: AgentTask) -> int:
    return sum(len(candidate.operations) for candidate in task.candidates)


def _wasted_attempt_count(results: Sequence[Any]) -> int:
    if not results:
        return 0
    final = results[-1]
    if (
        getattr(final, "committed", False)
        or getattr(final, "state", None) == TransactionState.REJECTED
    ):
        return max(0, len(results) - 1)
    return len(results)


def _agent_phase_for_task(task: AgentTask, attempt: int = 0) -> str:
    sequence = task.context.get("agent_phase_sequence", ())
    if isinstance(sequence, str):
        phases = tuple(part.strip() for part in sequence.split(",") if part.strip())
    else:
        try:
            phases = tuple(str(part) for part in sequence if str(part))
        except TypeError:
            phases = ()
    if phases:
        return phases[min(max(0, int(attempt)), len(phases) - 1)]
    if task.task_type in {"order_status", "stock_level"}:
        return "explore"
    if task.task_type in {"new_order", "payment", "delivery", "read-update"}:
        return "commit"
    return str(task.context.get("agent_phase", "")) or ""


def _agent_phase_for_staged_task(task: AgentTask) -> str:
    stages = task_agent_stages(task)
    phases = {stage.phase for stage in stages}
    for phase in ("commit", "refine", "explore"):
        if phase in phases:
            return phase
    return _agent_phase_for_task(task, 0)


def _stage_delay_plan(task: AgentTask, total_delay_s: float) -> Tuple[Tuple[str, float], ...]:
    stages = task_agent_stages(task)
    if not stages:
        phase = _agent_phase_for_task(task, 0) or "commit"
        return ((phase, max(0.0, float(total_delay_s))),)
    weights = tuple(max(0.0, float(stage.delay_weight)) for stage in stages)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return tuple((stage.phase, 0.0) for stage in stages)
    delay = max(0.0, float(total_delay_s))
    return tuple(
        (stage.phase, delay * weight / total_weight)
        for stage, weight in zip(stages, weights)
    )


def _task_has_hotspot_context(task: AgentTask) -> bool:
    context = dict(getattr(task, "context", {}) or {})
    return (
        int(context.get("hot_record_count", 0) or 0) > 0
        and float(context.get("hotspot_access_probability", 0.0) or 0.0) > 0.0
    )


def _task_has_hotspot_refresh_pressure(task: AgentTask) -> bool:
    if not _task_has_hotspot_context(task):
        return False
    context = dict(getattr(task, "context", {}) or {})
    if float(context.get("hotspot_access_probability", 0.0) or 0.0) >= 0.50:
        return True
    ratios = []
    for candidate in task.candidates:
        operations = tuple(candidate.operations)
        if not operations:
            continue
        writes = sum(1 for operation in operations if operation.kind != "read")
        ratios.append(writes / len(operations))
    return max(ratios or [0.0]) >= 0.40


def _sleep_agent_delay(
    delay_s: float,
    *,
    interaction_gate: Optional[Any],
    admitted: bool,
) -> None:
    if delay_s <= 0.0:
        return
    if interaction_gate is None or admitted:
        time.sleep(delay_s)
        return
    with interaction_gate:
        time.sleep(delay_s)


def _sleep_abort_retry_delay(
    result: Any,
    delay_s: float,
    *,
    interaction_gate: Optional[Any],
    admitted: bool,
) -> None:
    if delay_s <= 0.0:
        return
    state = getattr(result, "state", "")
    state_value = getattr(state, "value", state)
    if str(state_value) != "aborted":
        return
    if getattr(result, "reason", "") == "traditional_k_loser":
        return
    _sleep_agent_delay(
        delay_s,
        interaction_gate=interaction_gate,
        admitted=admitted,
    )


def _refresh_replay_phases(
    task: AgentTask,
    *,
    yielded_prelocks: bool,
) -> Optional[Tuple[str, ...]]:
    if yielded_prelocks:
        return None
    if _task_has_hotspot_refresh_pressure(task):
        return ("commit",)
    return None


def _can_object_refresh_commit_writes(
    task: AgentTask,
    *,
    yielded_prelocks: bool,
) -> bool:
    _ = yielded_prelocks
    if not _task_has_hotspot_refresh_pressure(task):
        return False
    if "ycsb" not in str(getattr(task, "workload", "")).lower():
        return False
    commit_operations = tuple(stage_operations(task, "commit"))
    writes = tuple(operation for operation in commit_operations if operation.kind != "read")
    return bool(writes) and all(operation.kind == "overwrite" for operation in writes)


def _operation_candidate_scope(task: AgentTask, strategy: str) -> str:
    if str(strategy) not in {"adaptive-op-strict", "transaction-atcc-strict"}:
        return "all"
    if not _can_object_refresh_commit_writes(task, yielded_prelocks=False):
        return "all"
    if len(tuple(getattr(task, "candidates", ()) or ())) <= 1:
        return "all"
    return "best"


def _object_refresh_commit_targets(txn: Any, task: AgentTask) -> Tuple[str, ...]:
    write_targets = {
        str(operation.object_id)
        for operation in stage_operations(task, "commit")
        if operation.kind != "read"
    }
    return tuple(sorted(write_targets))


def _replay_task_stages(
    txn: Any,
    task: AgentTask,
    *,
    phases: Optional[Iterable[str]] = None,
) -> int:
    stages = task_agent_stages(task)
    requested = tuple(str(phase) for phase in phases or ())
    if not stages:
        populate_task_transaction(txn, task)
        return _task_operation_count(task)
    operation_count = 0
    for stage in stages:
        if requested and stage.phase not in requested:
            continue
        if txn.state != TransactionState.ACTIVE:
            return operation_count
        operation_count += len(stage.operations)
        populate_task_stage(txn, task, stage.phase)
    if (
        txn.state == TransactionState.ACTIVE
        and not txn.candidates
        and (not requested or "commit" in requested)
    ):
        operation_count += len(stage_operations(task, "commit"))
        populate_task_stage(txn, task, "commit")
    return operation_count


def _stage_phase_counts(
    tasks: Sequence[AgentTask],
    *,
    include_empty: bool = True,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task in tasks:
        for stage in task_agent_stages(task):
            if stage.phase and (include_empty or stage.operations):
                counts[str(stage.phase)] += 1
    return counts


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _split_csv_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(part) for part in _split_csv(value))


if __name__ == "__main__":
    raise SystemExit(main())
