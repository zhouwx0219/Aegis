"""Deterministic CC strategy matrix evaluation for agent workloads."""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agent.runtime import AgentTransactionManager, TransactionResult, TransactionState
from agent.workloads import (
    AgentTask,
    AgentWorkload,
    prepare_task_transaction,
    register_workload,
)


ManagerFactory = Callable[[], AgentTransactionManager]
Regenerator = Optional[Callable[[Any], None]]


@dataclasses.dataclass(frozen=True)
class StrategyRunSummary:
    workload: str
    workload_manifest: Dict[str, Any]
    strategy: str
    seed: int
    task_count: int
    contention_window: int
    committed: int
    rejected: int
    aborted: int
    action_counts: Dict[str, int]
    selected_cc_counts: Dict[str, int]
    n_merge: int
    n_reselect: int
    n_regen: int
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StrategyAggregateSummary:
    workload: str
    workload_manifest: Dict[str, Any]
    strategy: str
    seeds: Tuple[int, ...]
    runs: int
    task_count_per_run: int
    total_task_count: int
    contention_window: int
    committed_total: int
    rejected_total: int
    aborted_total: int
    committed_mean: float
    rejected_mean: float
    aborted_mean: float
    action_counts_total: Dict[str, int]
    selected_cc_counts_total: Dict[str, int]
    n_merge_total: int
    n_reselect_total: int
    n_regen_total: int
    n_merge_mean: float
    n_reselect_mean: float
    n_regen_mean: float
    elapsed_s_total: float
    elapsed_s_mean: float

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["seeds"] = list(self.seeds)
        return row


def run_strategy_matrix(
    workload: AgentWorkload,
    strategies: Iterable[str],
    *,
    task_count: int,
    seed: int = 0,
    contention_window: int = 1,
    manager_factory: ManagerFactory = AgentTransactionManager,
    regenerator: Regenerator = None,
) -> Sequence[StrategyRunSummary]:
    """Run the same generated tasks against multiple CC strategies.

    contention_window controls how many tasks are begun from the same manager
    snapshot before any of them commits. A value above one creates reproducible
    stale-snapshot conflicts for comparing strict and semantic policies.
    """

    if task_count < 0:
        raise ValueError("task_count must be non-negative")
    if contention_window <= 0:
        raise ValueError("contention_window must be positive")

    task_batch = tuple(workload.generate_tasks(task_count, seed=seed))
    summaries: List[StrategyRunSummary] = []
    for strategy in strategies:
        summaries.append(
            _run_one_strategy(
                workload,
                task_batch,
                str(strategy),
                seed=seed,
                contention_window=contention_window,
                manager_factory=manager_factory,
                regenerator=regenerator,
            )
        )
    return tuple(summaries)


def run_strategy_matrix_repeated(
    workload: AgentWorkload,
    strategies: Iterable[str],
    *,
    task_count: int,
    seeds: Iterable[int],
    contention_window: int = 1,
    manager_factory: ManagerFactory = AgentTransactionManager,
    regenerator: Regenerator = None,
) -> Tuple[Sequence[StrategyRunSummary], Sequence[StrategyAggregateSummary]]:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    strategy_tuple = tuple(str(strategy) for strategy in strategies)
    if not strategy_tuple:
        raise ValueError("at least one strategy is required")

    runs: List[StrategyRunSummary] = []
    for seed in seed_tuple:
        runs.extend(
            run_strategy_matrix(
                workload,
                strategy_tuple,
                task_count=task_count,
                seed=seed,
                contention_window=contention_window,
                manager_factory=manager_factory,
                regenerator=regenerator,
            )
        )
    aggregates = [
        _aggregate_runs(strategy, seed_tuple, [run for run in runs if run.strategy == strategy])
        for strategy in strategy_tuple
    ]
    return tuple(runs), tuple(aggregates)


def _run_one_strategy(
    workload: AgentWorkload,
    tasks: Sequence[AgentTask],
    strategy: str,
    *,
    seed: int,
    contention_window: int,
    manager_factory: ManagerFactory,
    regenerator: Regenerator,
) -> StrategyRunSummary:
    manager = manager_factory()
    register_workload(manager, workload)

    started_at = time.perf_counter()
    results: List[TransactionResult] = []
    for offset in range(0, len(tasks), contention_window):
        window = tasks[offset : offset + contention_window]
        transactions = [prepare_task_transaction(manager, task) for task in window]
        for transaction in transactions:
            results.append(
                transaction.commit(strategy=strategy, regenerator=regenerator)
            )
    elapsed_s = time.perf_counter() - started_at
    return _summarize(
        workload.name,
        workload.manifest().to_dict(),
        strategy,
        seed,
        contention_window,
        results,
        manager.traces(),
        elapsed_s,
    )


def _summarize(
    workload_name: str,
    workload_manifest: Dict[str, Any],
    strategy: str,
    seed: int,
    contention_window: int,
    results: Sequence[TransactionResult],
    traces: Sequence[Dict[str, Any]],
    elapsed_s: float,
) -> StrategyRunSummary:
    action_counts = Counter(result.action for result in results)
    selected_cc_counts: Counter[str] = Counter()
    for trace in traces:
        for event in trace.get("events", ()):
            if event.get("kind") == "validate":
                selected = event.get("detail", {}).get("selected_cc")
                if selected:
                    selected_cc_counts[str(selected)] += 1

    return StrategyRunSummary(
        workload=workload_name,
        workload_manifest=workload_manifest,
        strategy=strategy,
        seed=seed,
        task_count=len(results),
        contention_window=contention_window,
        committed=sum(1 for result in results if result.committed),
        rejected=sum(1 for result in results if result.state == TransactionState.REJECTED),
        aborted=sum(1 for result in results if result.state == TransactionState.ABORTED),
        action_counts=dict(sorted(action_counts.items())),
        selected_cc_counts=dict(sorted(selected_cc_counts.items())),
        n_merge=sum(result.n_merge for result in results),
        n_reselect=sum(result.n_reselect for result in results),
        n_regen=sum(result.n_regen for result in results),
        elapsed_s=elapsed_s,
    )


def _aggregate_runs(
    strategy: str, seeds: Tuple[int, ...], runs: Sequence[StrategyRunSummary]
) -> StrategyAggregateSummary:
    if not runs:
        raise ValueError(f"no runs to aggregate for strategy: {strategy}")
    run_count = len(runs)
    action_counts: Counter[str] = Counter()
    selected_cc_counts: Counter[str] = Counter()
    for run in runs:
        action_counts.update(run.action_counts)
        selected_cc_counts.update(run.selected_cc_counts)

    committed_total = sum(run.committed for run in runs)
    rejected_total = sum(run.rejected for run in runs)
    aborted_total = sum(run.aborted for run in runs)
    n_merge_total = sum(run.n_merge for run in runs)
    n_reselect_total = sum(run.n_reselect for run in runs)
    n_regen_total = sum(run.n_regen for run in runs)
    elapsed_s_total = sum(run.elapsed_s for run in runs)

    return StrategyAggregateSummary(
        workload=runs[0].workload,
        workload_manifest=runs[0].workload_manifest,
        strategy=strategy,
        seeds=seeds,
        runs=run_count,
        task_count_per_run=runs[0].task_count,
        total_task_count=sum(run.task_count for run in runs),
        contention_window=runs[0].contention_window,
        committed_total=committed_total,
        rejected_total=rejected_total,
        aborted_total=aborted_total,
        committed_mean=committed_total / run_count,
        rejected_mean=rejected_total / run_count,
        aborted_mean=aborted_total / run_count,
        action_counts_total=dict(sorted(action_counts.items())),
        selected_cc_counts_total=dict(sorted(selected_cc_counts.items())),
        n_merge_total=n_merge_total,
        n_reselect_total=n_reselect_total,
        n_regen_total=n_regen_total,
        n_merge_mean=n_merge_total / run_count,
        n_reselect_mean=n_reselect_total / run_count,
        n_regen_mean=n_regen_total / run_count,
        elapsed_s_total=elapsed_s_total,
        elapsed_s_mean=elapsed_s_total / run_count,
    )
