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
from agent.runtime import AgentTransactionManager, OperationPolicyTable, TransactionState
from agent.runtime.adaptive import operation_object_class
from agent.workloads import (
    AgentTask,
    AgentWorkload,
    TPCCConfig,
    YCSBConfig,
    build_agent_workload,
    populate_task_transaction,
    prepare_task_transaction,
    register_workload,
)


@dataclasses.dataclass(frozen=True)
class RetryTaskOutcome:
    results: Tuple[Any, ...]
    latency_s: float
    operation_count: int
    wasted_attempts: int
    estimated_tokens: float
    estimated_wasted_tokens: float


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
    prelock_queue_depth_sum: float = 0.0
    prelock_queue_depth_observations: int = 0
    prelock_queue_depth_max: int = 0
    prelock_handoff_count: int = 0
    prelock_committing_enters: int = 0
    prelock_committing_exits: int = 0

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
) -> Tuple[RetryRunSummary, ...]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    admission = str(agent_admission_mode or "planning-only").strip().lower()
    if admission not in {"planning-only", "before-begin"}:
        raise ValueError(f"unsupported agent admission mode: {agent_admission_mode}")

    rows: List[RetryRunSummary] = []
    for offset in range(repeats):
        run_seed = int(seed) + offset
        tasks = tuple(workload.generate_tasks(task_count, seed=run_seed))
        for strategy in strategies:
            rows.append(
                _run_one_retry(
                    workload,
                    tasks,
                    str(strategy),
                    workload_kind=workload_kind,
                    policy_variant=policy_variant,
                    seed=run_seed,
                    workers=workers,
                    agent_slots=agent_slots,
                    agent_admission_mode=admission,
                    planning_delay_s=planning_delay_s,
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
                )
            )
    return tuple(rows)


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
        for run in group:
            operation_policy_counts.update(run.operation_policy_counts)
            operation_rule_counts.update(run.operation_rule_counts)
            conflict_object_counts.update(run.conflict_object_counts)
            conflict_object_class_counts.update(run.conflict_object_class_counts)
            action_counts.update(run.action_counts)
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
        estimated_tokens = sum(run.estimated_tokens for run in group)
        estimated_wasted_tokens = sum(run.estimated_wasted_tokens for run in group)
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
                "background_commits": background_commits,
                "background_aborts": background_aborts,
                "lease_refresh_regenerations": lease_refresh_regenerations,
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
            }
        )
    return rows


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
) -> RetryRunSummary:
    policy = operation_policy or _operation_policy(
        workload_kind,
        policy_variant,
        policy_artifact=policy_artifact,
        policy_epsilon=policy_epsilon,
    )
    manager = AgentTransactionManager(
        operation_policy=policy,
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

    def run_task(task: AgentTask) -> RetryTaskOutcome:
        started_at = time.perf_counter()
        results: List[Any] = []
        rng = random.Random(f"{seed}:{strategy}:{policy_variant}:{task.task_id}")
        admit_before_begin = (
            str(agent_admission_mode) == "before-begin"
            and interaction_gate is not None
        )
        for attempt in range(max_attempts):
            delay_s = _sample_latency_s(
                rng,
                mean_s=planning_delay_s,
                distribution=latency_distribution,
                cv=latency_cv,
                max_s=latency_max_s,
            )
            defer_prelocks = (
                str(prelock_lease_mode) == "defer-until-after-planning"
                and str(strategy) == "adaptive-op-strict"
            )
            admitted = False
            if admit_before_begin:
                interaction_gate.acquire()
                admitted = True
            try:
                if defer_prelocks and delay_s:
                    if interaction_gate is None or admitted:
                        time.sleep(delay_s)
                    else:
                        with interaction_gate:
                            time.sleep(delay_s)
                txn = prepare_task_transaction(
                    manager,
                    task,
                    strategy=strategy,
                    runtime_context={
                        "retry_count": attempt,
                        "agent_interval_s": delay_s,
                        "agent_phase": _agent_phase_for_task(task, attempt),
                        "agent_slots": int(agent_slots),
                        "agent_admission_mode": str(agent_admission_mode),
                    },
                )
                should_yield_prelocks = (
                    str(prelock_lease_mode)
                    in {"yield-during-planning", "yield-refresh-regenerate"}
                    and str(strategy) == "adaptive-op-strict"
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
            result = txn.commit(strategy=strategy)
            results.append(result)
            if result.committed or result.state == TransactionState.REJECTED:
                break
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
        background_workers=max(0, int(background_workers)),
        background_commits=int(background_counts.get("committed", 0)),
        background_aborts=int(background_counts.get("aborted", 0)),
        lease_refresh_regenerations=lease_refresh_regenerations,
        prelock_queue_depth_sum=prelock_queue_depth_sum,
        prelock_queue_depth_observations=prelock_queue_depth_observations,
        prelock_queue_depth_max=prelock_queue_depth_max,
        prelock_handoff_count=prelock_handoff_count,
        prelock_committing_enters=prelock_committing_enters,
        prelock_committing_exits=prelock_committing_exits,
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
        hot = [object_id for object_id in object_ids if object_id.startswith("ycsb:record:0:")]
        return tuple(hot or object_ids[:1])
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry-aware operation ATCC experiments.")
    parser.add_argument("--mode", choices=("matrix", "search"), default="matrix")
    parser.add_argument("--workload", choices=("tpcc", "ycsb"), default="tpcc")
    parser.add_argument("--strategies", default="occ,2pl-pre,adaptive-op-strict")
    parser.add_argument("--policy-variant", default="default")
    parser.add_argument("--search-variants", default="aggressive,balanced,conservative")
    parser.add_argument("--task-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
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
            "the simulated planning delay. 2PL-pre is kept as a blocking baseline."
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
    parser.add_argument("--output", type=Path)

    ycsb = parser.add_argument_group("YCSB options")
    ycsb.add_argument("--records", type=int, default=64)
    ycsb.add_argument("--fields", type=int, default=4)
    ycsb.add_argument("--requests-per-task", type=int, default=4)
    ycsb.add_argument("--candidates", type=int, default=4)
    ycsb.add_argument("--read-weight", type=float, default=0.2)
    ycsb.add_argument("--update-weight", type=float, default=0.8)
    ycsb.add_argument("--zipf-theta", type=float, default=0.99)

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
            "workload_config": workload_config,
            "task_count": args.task_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "workers": args.workers,
            "agent_slots": args.agent_slots,
            "agent_admission_mode": args.agent_admission_mode,
            "planning_delay_s": args.planning_delay_ms / 1000.0,
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
            ),
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
            agent_slots=args.agent_slots,
            agent_admission_mode=args.agent_admission_mode,
            planning_delay_s=args.planning_delay_ms / 1000.0,
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
        )
        report = {
            "mode": "matrix",
            "workload": workload.name,
            "workload_kind": args.workload,
            "workload_config": workload_config,
            "strategies": list(_split_csv(args.strategies)),
            "policy_variant": args.policy_variant,
            "task_count": args.task_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "workers": args.workers,
            "agent_slots": args.agent_slots,
            "agent_admission_mode": args.agent_admission_mode,
            "planning_delay_s": args.planning_delay_ms / 1000.0,
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
            "runs": [run.to_dict() for run in runs],
            "aggregates": aggregate_retry_runs(runs),
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


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
