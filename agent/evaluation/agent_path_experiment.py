"""Compare traditional K-transaction candidates with ASTRA multi-branch tasks."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

from agent.runtime import (
    AdaptivePolicyTable,
    AgentTransactionManager,
    OperationPolicyTable,
    TransactionResult,
    TransactionState,
)
from agent.workloads import (
    AgentCandidate,
    AgentTask,
    AgentWorkload,
    TPCCConfig,
    YCSBConfig,
    build_agent_workload,
    prepare_task_transaction,
    register_workload,
)


ManagerFactory = Callable[[], AgentTransactionManager]


@dataclasses.dataclass(frozen=True)
class AgentPathRunSummary:
    workload: str
    workload_manifest: Mapping[str, Any]
    path: str
    strategy: str
    seed: int
    task_count: int
    candidates_per_task: float
    contention_window: int
    execution_mode: str
    planning_delay_s: float
    committed_tasks: int
    conflict_aborts: int
    loser_aborts: int
    rejected: int
    physical_transactions: int
    action_counts: Mapping[str, int]
    selected_cc_counts: Mapping[str, int]
    operation_policy_counts: Mapping[str, int]
    operation_rule_counts: Mapping[str, int]
    n_merge: int
    n_reselect: int
    n_regen: int
    prelock_wait_s: float
    elapsed_s: float

    @property
    def commit_rate(self) -> float:
        return self.committed_tasks / self.task_count if self.task_count else 0.0

    @property
    def logical_throughput(self) -> float:
        return self.task_count / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def commit_throughput(self) -> float:
        return self.committed_tasks / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def aborts_per_task(self) -> float:
        return (self.conflict_aborts + self.loser_aborts) / self.task_count if self.task_count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["commit_rate"] = self.commit_rate
        row["logical_throughput"] = self.logical_throughput
        row["commit_throughput"] = self.commit_throughput
        row["aborts_per_task"] = self.aborts_per_task
        row["prelock_wait_per_task_s"] = (
            self.prelock_wait_s / self.task_count if self.task_count else 0.0
        )
        return row


def run_agent_path_matrix(
    workload: AgentWorkload,
    strategies: Iterable[str],
    *,
    paths: Iterable[str] = ("traditional-k", "astra"),
    task_count: int,
    seed: int = 0,
    contention_window: int = 1,
    manager_factory: ManagerFactory = AgentTransactionManager,
    execution_mode: str = "stale-window",
    planning_delay_s: float = 0.0,
) -> Sequence[AgentPathRunSummary]:
    if task_count < 0:
        raise ValueError("task_count must be non-negative")
    if contention_window <= 0:
        raise ValueError("contention_window must be positive")
    if execution_mode not in {"stale-window", "concurrent"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    if planning_delay_s < 0:
        raise ValueError("planning_delay_s must be non-negative")

    tasks = tuple(workload.generate_tasks(task_count, seed=seed))
    summaries: List[AgentPathRunSummary] = []
    for path in paths:
        normalized_path = _normalize_path(path)
        for strategy in strategies:
            summaries.append(
                _run_one_path(
                    workload,
                    tasks,
                    normalized_path,
                    str(strategy),
                    seed=seed,
                    contention_window=contention_window,
                    manager_factory=manager_factory,
                    execution_mode=execution_mode,
                    planning_delay_s=planning_delay_s,
                )
            )
    return tuple(summaries)


def learn_new_order_threshold(
    *,
    thresholds: Iterable[int],
    workload_config: TPCCConfig,
    strategies: Tuple[str, ...] = ("adaptive",),
    train_seeds: Iterable[int] = (0, 1, 2),
    task_count: int = 100,
    contention_window: int = 8,
) -> Dict[str, Any]:
    """Train a simple ATCC-style threshold table for TPC-C NewOrder.

    The current policy family has one tunable knob: the distinct-write-target
    threshold above which NewOrder switches from semantic optimistic rebase to
    agent-level pessimistic validation. The score rewards commits and semantic
    merges, and penalizes conflict aborts and loser aborts.
    """

    threshold_rows = []
    workload = build_agent_workload(
        "tpcc", "semantic", tpcc_config=workload_config
    )
    seed_tuple = tuple(int(seed) for seed in train_seeds)
    for threshold in thresholds:
        manager_factory = lambda threshold=threshold: AgentTransactionManager(
            adaptive_policy=AdaptivePolicyTable.new_order(
                wide_write_threshold=int(threshold)
            )
        )
        runs: List[AgentPathRunSummary] = []
        for seed in seed_tuple:
            runs.extend(
                run_agent_path_matrix(
                    workload,
                    strategies,
                    paths=("astra",),
                    task_count=task_count,
                    seed=seed,
                    contention_window=contention_window,
                    manager_factory=manager_factory,
                )
            )
        score = sum(
            run.committed_tasks * 100
            + run.n_merge
            - run.conflict_aborts * 20
            - run.loser_aborts * 5
            for run in runs
        )
        selected_counts: Counter[str] = Counter()
        for run in runs:
            selected_counts.update(run.selected_cc_counts)
        threshold_rows.append(
            {
                "threshold": int(threshold),
                "score": score,
                "committed_tasks": sum(run.committed_tasks for run in runs),
                "conflict_aborts": sum(run.conflict_aborts for run in runs),
                "n_merge": sum(run.n_merge for run in runs),
                "selected_cc_counts": dict(sorted(selected_counts.items())),
            }
        )
    best = max(threshold_rows, key=lambda row: (row["score"], row["committed_tasks"], row["n_merge"]))
    return {
        "policy_family": "tpcc-new-order-atcc-threshold",
        "best_threshold": best["threshold"],
        "train_seeds": list(seed_tuple),
        "task_count": task_count,
        "contention_window": contention_window,
        "thresholds": threshold_rows,
    }


def learn_new_order_operation_policy(
    *,
    thresholds: Iterable[int],
    workload_config: TPCCConfig,
    train_seeds: Iterable[int] = (0, 1, 2),
    task_count: int = 100,
    contention_window: int = 8,
    lock_cost: float = 0.05,
    hot_counter_miss_cost: float = 2.0,
    strategy: str = "adaptive-op-strict",
    execution_mode: str = "concurrent",
    planning_delay_s: float = 0.001,
) -> Dict[str, Any]:
    """Train an operation-level ATCC table for TPC-C NewOrder.

    The learned family is intentionally inspectable: the district
    `next_order_id` counter can be pessimistic when it appears often enough
    across one agent task's candidate branches, while stock deltas and order
    appends remain optimistic. The score combines measured commit outcomes with
    an operation-cost model: locks have a small cost, and leaving the hot counter
    optimistic carries a predicted conflict-risk cost.
    """

    threshold_values = tuple(int(threshold) for threshold in thresholds)
    if not threshold_values:
        raise ValueError("at least one operation threshold is required")
    if any(threshold <= 0 for threshold in threshold_values):
        raise ValueError("operation thresholds must be positive")
    if task_count < 0:
        raise ValueError("task_count must be non-negative")
    if contention_window <= 0:
        raise ValueError("contention_window must be positive")
    if lock_cost < 0 or hot_counter_miss_cost < 0:
        raise ValueError("operation policy costs must be non-negative")

    workload = build_agent_workload(
        "tpcc", "semantic", tpcc_config=workload_config
    )
    seed_tuple = tuple(int(seed) for seed in train_seeds)
    if not seed_tuple:
        raise ValueError("at least one training seed is required")

    threshold_rows = []
    for threshold in threshold_values:
        manager_factory = lambda threshold=threshold: AgentTransactionManager(
            operation_policy=OperationPolicyTable.tpcc_new_order(
                hot_object_threshold=int(threshold)
            )
        )
        runs: List[AgentPathRunSummary] = []
        for seed in seed_tuple:
            runs.extend(
                run_agent_path_matrix(
                    workload,
                    (strategy,),
                    paths=("astra",),
                    task_count=task_count,
                    seed=seed,
                    contention_window=contention_window,
                    manager_factory=manager_factory,
                    execution_mode=execution_mode,
                    planning_delay_s=planning_delay_s,
                )
            )

        task_total = sum(run.task_count for run in runs)
        committed = sum(run.committed_tasks for run in runs)
        conflict_aborts = sum(run.conflict_aborts for run in runs)
        n_merge = sum(run.n_merge for run in runs)
        operation_policy_counts: Counter[str] = Counter()
        operation_rule_counts: Counter[str] = Counter()
        for run in runs:
            operation_policy_counts.update(run.operation_policy_counts)
            operation_rule_counts.update(run.operation_rule_counts)

        pessimistic_ops = operation_policy_counts.get("pessimistic", 0)
        hot_counter_pessimistic = operation_rule_counts.get(
            "new-order-district-counter-pessimistic", 0
        )
        hot_counter_optimistic = max(0, task_total - hot_counter_pessimistic)
        score = (
            committed * 100.0
            + n_merge * 0.25
            - conflict_aborts * 50.0
            - pessimistic_ops * float(lock_cost)
            - hot_counter_optimistic * float(hot_counter_miss_cost)
        )
        threshold_rows.append(
            {
                "threshold": int(threshold),
                "raw_score": score,
                "score": score,
                "committed_tasks": committed,
                "conflict_aborts": conflict_aborts,
                "n_merge": n_merge,
                "pessimistic_ops": pessimistic_ops,
                "optimistic_ops": operation_policy_counts.get("optimistic", 0),
                "hot_counter_pessimistic": hot_counter_pessimistic,
                "hot_counter_optimistic": hot_counter_optimistic,
                "operation_policy_counts": dict(sorted(operation_policy_counts.items())),
                "operation_rule_counts": dict(sorted(operation_rule_counts.items())),
            }
        )

    equivalent_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in threshold_rows:
        signature = (
            row["pessimistic_ops"],
            row["optimistic_ops"],
            row["hot_counter_pessimistic"],
            row["hot_counter_optimistic"],
            tuple(sorted(row["operation_rule_counts"].items())),
        )
        equivalent_groups.setdefault(signature, []).append(row)
    for group in equivalent_groups.values():
        group_score = sum(row["raw_score"] for row in group) / len(group)
        for row in group:
            row["score"] = group_score
            row["equivalent_thresholds"] = sorted(
                member["threshold"] for member in group
            )

    best = max(
        threshold_rows,
        key=lambda row: (
            row["score"],
            -row["hot_counter_optimistic"],
            -row["pessimistic_ops"],
            row["threshold"],
        ),
    )
    return {
        "policy_family": "tpcc-new-order-operation-atcc-hot-counter-threshold",
        "best_hot_object_threshold": best["threshold"],
        "train_seeds": list(seed_tuple),
        "task_count": task_count,
        "contention_window": contention_window,
        "strategy": strategy,
        "execution_mode": execution_mode,
        "planning_delay_s": planning_delay_s,
        "score_model": {
            "commit_reward": 100.0,
            "merge_reward": 0.25,
            "conflict_abort_penalty": 50.0,
            "pessimistic_lock_cost": float(lock_cost),
            "optimistic_hot_counter_miss_cost": float(hot_counter_miss_cost),
        },
        "thresholds": threshold_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare traditional K-transaction candidates with ASTRA."
    )
    parser.add_argument("--workload", choices=("ycsb", "tpcc"), default="tpcc")
    parser.add_argument("--workload-layer", choices=("semantic", "faithful"), default="semantic")
    parser.add_argument("--paths", default="traditional-k,astra")
    parser.add_argument("--strategies", default="occ,2pl,semantic,adaptive")
    parser.add_argument("--task-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--contention-window", type=int, default=8)
    parser.add_argument(
        "--execution-mode",
        choices=("stale-window", "concurrent"),
        default="stale-window",
    )
    parser.add_argument("--planning-delay-ms", type=float, default=0.0)
    parser.add_argument("--adaptive-policy", choices=("default", "new-order"), default="default")
    parser.add_argument("--wide-write-threshold", type=int, default=8)
    parser.add_argument(
        "--operation-policy",
        choices=(
            "default",
            "new-order",
            "learned-new-order",
            "atcc",
            "tpcc-atcc",
            "ycsb-atcc",
        ),
        default="default",
    )
    parser.add_argument("--operation-hot-threshold", type=int, default=2)
    parser.add_argument("--learn-new-order-threshold", action="store_true")
    parser.add_argument("--thresholds", default="2,4,6,8,10,12,16,24,32,48,64")
    parser.add_argument("--learn-operation-policy", action="store_true")
    parser.add_argument("--operation-thresholds", default="1,2,3,4,5,8,16")
    parser.add_argument("--operation-lock-cost", type=float, default=0.05)
    parser.add_argument("--operation-hot-miss-cost", type=float, default=2.0)
    parser.add_argument(
        "--operation-training-strategy",
        choices=("adaptive-op", "adaptive-op-strict"),
        default="adaptive-op-strict",
    )
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--train-repeats", type=int, default=0)
    parser.add_argument("--train-task-count", type=int, default=0)
    parser.add_argument("--train-contention-window", type=int, default=0)
    parser.add_argument("--output", type=Path)

    ycsb = parser.add_argument_group("YCSB options")
    ycsb.add_argument("--records", type=int, default=100)
    ycsb.add_argument("--fields", type=int, default=10)
    ycsb.add_argument("--requests-per-task", type=int, default=4)
    ycsb.add_argument("--candidates", type=int, default=3)
    ycsb.add_argument("--read-weight", type=float, default=0.5)
    ycsb.add_argument("--update-weight", type=float, default=0.5)
    ycsb.add_argument("--zipf-theta", type=float, default=0.6)

    tpcc = parser.add_argument_group("TPC-C options")
    tpcc.add_argument("--warehouses", type=int, default=1)
    tpcc.add_argument("--districts-per-warehouse", type=int, default=1)
    tpcc.add_argument("--customers-per-district", type=int, default=4)
    tpcc.add_argument("--items", type=int, default=20)
    tpcc.add_argument("--initial-stock", type=int, default=100)
    tpcc.add_argument("--order-lines", type=int, default=4)
    tpcc.add_argument(
        "--transaction-mix",
        default="new_order:1.0",
        help="Comma-separated entries such as new_order:1.0,payment:0.5.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    workload, workload_config = _build_workload(args)
    strategies = tuple(_split_csv(args.strategies))
    paths = tuple(_split_csv(args.paths))
    seeds = tuple(args.seed + offset for offset in range(args.repeats))
    learned_operation_policy = None
    if args.workload == "tpcc" and (
        args.learn_operation_policy or args.operation_policy == "learned-new-order"
    ):
        learned_operation_policy = learn_new_order_operation_policy(
            thresholds=[int(item) for item in _split_csv(args.operation_thresholds)],
            workload_config=_build_tpcc_config(args),
            train_seeds=_training_seeds(args),
            task_count=args.train_task_count or args.task_count,
            contention_window=args.train_contention_window
            or args.contention_window,
            lock_cost=args.operation_lock_cost,
            hot_counter_miss_cost=args.operation_hot_miss_cost,
            strategy=args.operation_training_strategy,
            execution_mode=args.execution_mode,
            planning_delay_s=args.planning_delay_ms / 1000.0,
        )
        if args.operation_policy == "learned-new-order":
            args.operation_hot_threshold = int(
                learned_operation_policy["best_hot_object_threshold"]
            )
    manager_factory = _manager_factory(args)

    runs: List[AgentPathRunSummary] = []
    for seed in seeds:
        runs.extend(
            run_agent_path_matrix(
                workload,
                strategies,
                paths=paths,
                task_count=args.task_count,
                seed=seed,
                contention_window=args.contention_window,
                manager_factory=manager_factory,
                execution_mode=args.execution_mode,
                planning_delay_s=args.planning_delay_ms / 1000.0,
            )
        )

    report: Dict[str, Any] = {
        "workload": workload.name,
        "workload_kind": args.workload,
        "workload_layer": args.workload_layer,
        "workload_config": workload_config,
        "workload_manifest": workload.manifest().to_dict(),
        "paths": list(paths),
        "strategies": list(strategies),
        "task_count": args.task_count,
        "seed": args.seed,
        "seeds": list(seeds),
        "repeats": args.repeats,
        "contention_window": args.contention_window,
        "execution_mode": args.execution_mode,
        "planning_delay_s": args.planning_delay_ms / 1000.0,
        "adaptive_policy": args.adaptive_policy,
        "wide_write_threshold": args.wide_write_threshold,
        "operation_policy": args.operation_policy,
        "operation_hot_threshold": args.operation_hot_threshold,
        "runs": [run.to_dict() for run in runs],
        "aggregates": _aggregate_runs(runs),
    }
    if args.learn_new_order_threshold and args.workload == "tpcc":
        report["learned_policy"] = learn_new_order_threshold(
            thresholds=[int(item) for item in _split_csv(args.thresholds)],
            workload_config=_build_tpcc_config(args),
            train_seeds=seeds,
            task_count=args.task_count,
            contention_window=args.contention_window,
        )
    if learned_operation_policy is not None:
        report["learned_operation_policy"] = learned_operation_policy
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _run_one_path(
    workload: AgentWorkload,
    tasks: Sequence[AgentTask],
    path: str,
    strategy: str,
    *,
    seed: int,
    contention_window: int,
    manager_factory: ManagerFactory,
    execution_mode: str,
    planning_delay_s: float,
) -> AgentPathRunSummary:
    manager = manager_factory()
    register_workload(manager, workload)
    if (
        execution_mode != "concurrent"
        and manager.cc_registry.requires_pre_snapshot_locks(strategy)
    ):
        raise ValueError(
            f"{strategy} requires execution_mode='concurrent' for pre-snapshot locks"
        )
    started_at = time.perf_counter()
    if path == "astra":
        if execution_mode == "concurrent":
            results = _run_astra_concurrent(
                manager,
                tasks,
                strategy,
                contention_window,
                planning_delay_s,
            )
        else:
            results = _run_astra(manager, tasks, strategy, contention_window)
        loser_aborts = 0
        physical_transactions = len(results)
    elif path == "traditional-k":
        if execution_mode == "concurrent":
            results, loser_aborts, physical_transactions = (
                _run_traditional_k_concurrent(
                    manager,
                    tasks,
                    strategy,
                    contention_window,
                    planning_delay_s,
                )
            )
        else:
            results, loser_aborts, physical_transactions = _run_traditional_k(
                manager, tasks, strategy, contention_window
            )
    else:
        raise ValueError(f"unsupported path: {path}")
    elapsed_s = time.perf_counter() - started_at
    return _summarize(
        workload,
        path,
        strategy,
        seed,
        contention_window,
        execution_mode,
        planning_delay_s,
        results,
        loser_aborts,
        physical_transactions,
        manager.traces(),
        elapsed_s,
    )


def _run_astra(
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    contention_window: int,
) -> List[TransactionResult]:
    results: List[TransactionResult] = []
    for offset in range(0, len(tasks), contention_window):
        window = tasks[offset : offset + contention_window]
        transactions = [
            prepare_task_transaction(manager, task, strategy=strategy)
            for task in window
        ]
        for transaction in transactions:
            results.append(transaction.commit(strategy=strategy))
    return results


def _run_astra_concurrent(
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    workers: int,
    planning_delay_s: float,
) -> List[TransactionResult]:
    def run_task(task: AgentTask) -> TransactionResult:
        transaction = prepare_task_transaction(manager, task, strategy=strategy)
        if planning_delay_s:
            time.sleep(planning_delay_s)
        return transaction.commit(strategy=strategy)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_task, tasks))


def _run_traditional_k_concurrent(
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    workers: int,
    planning_delay_s: float,
) -> Tuple[List[TransactionResult], int, int]:
    def run_task(task: AgentTask) -> List[TransactionResult]:
        winner_index = _winner_index(task.candidates)
        task_results = []
        for index, candidate in enumerate(task.candidates):
            single = _single_candidate_task(task, candidate)
            if index == winner_index:
                transaction = prepare_task_transaction(
                    manager, single, strategy=strategy
                )
                if planning_delay_s:
                    time.sleep(planning_delay_s)
                task_results.append(transaction.commit(strategy=strategy))
            else:
                transaction = prepare_task_transaction(manager, single)
                task_results.append(transaction.abort("traditional_k_loser"))
        return task_results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        grouped = list(executor.map(run_task, tasks))
    results = [result for group in grouped for result in group]
    loser_aborts = sum(len(task.candidates) - 1 for task in tasks)
    physical_transactions = sum(len(task.candidates) for task in tasks)
    return results, loser_aborts, physical_transactions


def _run_traditional_k(
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    contention_window: int,
) -> Tuple[List[TransactionResult], int, int]:
    results: List[TransactionResult] = []
    loser_aborts = 0
    physical_transactions = 0
    for offset in range(0, len(tasks), contention_window):
        window = tasks[offset : offset + contention_window]
        prepared = []
        for task in window:
            candidate_txns = []
            for candidate in task.candidates:
                single = _single_candidate_task(task, candidate)
                candidate_txns.append(
                    prepare_task_transaction(manager, single, strategy=strategy)
                )
            physical_transactions += len(candidate_txns)
            winner_index = _winner_index(task.candidates)
            prepared.append((candidate_txns, winner_index))

        for candidate_txns, winner_index in prepared:
            for index, transaction in enumerate(candidate_txns):
                if index == winner_index:
                    results.append(transaction.commit(strategy=strategy))
                else:
                    results.append(transaction.abort("traditional_k_loser"))
                    loser_aborts += 1
    return results, loser_aborts, physical_transactions


def _single_candidate_task(task: AgentTask, candidate: AgentCandidate) -> AgentTask:
    return dataclasses.replace(
        task,
        task_id=f"{task.task_id}:{candidate.candidate_id}",
        candidates=(candidate,),
    )


def _winner_index(candidates: Sequence[AgentCandidate]) -> int:
    return max(range(len(candidates)), key=lambda index: candidates[index].quality)


def _summarize(
    workload: AgentWorkload,
    path: str,
    strategy: str,
    seed: int,
    contention_window: int,
    execution_mode: str,
    planning_delay_s: float,
    results: Sequence[TransactionResult],
    loser_aborts: int,
    physical_transactions: int,
    traces: Sequence[Mapping[str, Any]],
    elapsed_s: float,
) -> AgentPathRunSummary:
    action_counts = Counter(result.action for result in results)
    selected_cc_counts: Counter[str] = Counter()
    operation_policy_counts: Counter[str] = Counter()
    operation_rule_counts: Counter[str] = Counter()
    prelock_wait_s = 0.0
    for trace in traces:
        prelock_wait_s += float(trace.get("prelock_wait_s", 0.0) or 0.0)
        for event in trace.get("events", ()):
            if event.get("kind") == "validate":
                selected = event.get("detail", {}).get("selected_cc")
                if selected:
                    selected_cc_counts[str(selected)] += 1
                for decision in event.get("detail", {}).get("operation_policy_decisions", ()):
                    policy = decision.get("policy")
                    rule = decision.get("rule")
                    if policy:
                        operation_policy_counts[str(policy)] += 1
                    if rule:
                        operation_rule_counts[str(rule)] += 1
    committed_tasks = sum(1 for result in results if result.committed)
    rejected = sum(1 for result in results if result.state == TransactionState.REJECTED)
    conflict_aborts = sum(
        1
        for result in results
        if result.state == TransactionState.ABORTED
        and result.reason != "traditional_k_loser"
    )
    task_count = len(results) - loser_aborts
    return AgentPathRunSummary(
        workload=workload.name,
        workload_manifest=workload.manifest().to_dict(),
        path=path,
        strategy=strategy,
        seed=seed,
        task_count=task_count,
        candidates_per_task=physical_transactions / task_count if task_count else 0.0,
        contention_window=contention_window,
        execution_mode=execution_mode,
        planning_delay_s=planning_delay_s,
        committed_tasks=committed_tasks,
        conflict_aborts=conflict_aborts,
        loser_aborts=loser_aborts,
        rejected=rejected,
        physical_transactions=physical_transactions,
        action_counts=dict(sorted(action_counts.items())),
        selected_cc_counts=dict(sorted(selected_cc_counts.items())),
        operation_policy_counts=dict(sorted(operation_policy_counts.items())),
        operation_rule_counts=dict(sorted(operation_rule_counts.items())),
        n_merge=sum(result.n_merge for result in results),
        n_reselect=sum(result.n_reselect for result in results),
        n_regen=sum(result.n_regen for result in results),
        prelock_wait_s=prelock_wait_s,
        elapsed_s=elapsed_s,
    )


def _aggregate_runs(runs: Sequence[AgentPathRunSummary]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[AgentPathRunSummary]] = {}
    for run in runs:
        grouped.setdefault((run.path, run.strategy), []).append(run)
    rows = []
    for (path, strategy), group in sorted(grouped.items()):
        action_counts: Counter[str] = Counter()
        selected_counts: Counter[str] = Counter()
        operation_policy_counts: Counter[str] = Counter()
        operation_rule_counts: Counter[str] = Counter()
        for run in group:
            action_counts.update(run.action_counts)
            selected_counts.update(run.selected_cc_counts)
            operation_policy_counts.update(run.operation_policy_counts)
            operation_rule_counts.update(run.operation_rule_counts)
        task_count = sum(run.task_count for run in group)
        elapsed = sum(run.elapsed_s for run in group)
        committed = sum(run.committed_tasks for run in group)
        conflict_aborts = sum(run.conflict_aborts for run in group)
        loser_aborts = sum(run.loser_aborts for run in group)
        prelock_wait_s = sum(run.prelock_wait_s for run in group)
        rows.append(
            {
                "path": path,
                "strategy": strategy,
                "runs": len(group),
                "task_count": task_count,
                "execution_mode": group[0].execution_mode,
                "planning_delay_s": group[0].planning_delay_s,
                "committed_tasks": committed,
                "commit_rate": committed / task_count if task_count else 0.0,
                "conflict_aborts": conflict_aborts,
                "loser_aborts": loser_aborts,
                "physical_transactions": sum(run.physical_transactions for run in group),
                "aborts_per_task": (conflict_aborts + loser_aborts) / task_count if task_count else 0.0,
                "n_merge": sum(run.n_merge for run in group),
                "n_reselect": sum(run.n_reselect for run in group),
                "n_regen": sum(run.n_regen for run in group),
                "prelock_wait_s": prelock_wait_s,
                "prelock_wait_per_task_s": prelock_wait_s / task_count
                if task_count
                else 0.0,
                "elapsed_s": elapsed,
                "logical_throughput": task_count / elapsed if elapsed > 0 else 0.0,
                "commit_throughput": committed / elapsed if elapsed > 0 else 0.0,
                "action_counts": dict(sorted(action_counts.items())),
                "selected_cc_counts": dict(sorted(selected_counts.items())),
                "operation_policy_counts": dict(sorted(operation_policy_counts.items())),
                "operation_rule_counts": dict(sorted(operation_rule_counts.items())),
            }
        )
    return rows


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
        return build_agent_workload("ycsb", args.workload_layer, ycsb_config=config), dataclasses.asdict(config)
    config = _build_tpcc_config(args)
    return build_agent_workload("tpcc", args.workload_layer, tpcc_config=config), dataclasses.asdict(config)


def _build_tpcc_config(args: argparse.Namespace) -> TPCCConfig:
    return TPCCConfig(
        warehouses=args.warehouses,
        districts_per_warehouse=args.districts_per_warehouse,
        customers_per_district=args.customers_per_district,
        items=args.items,
        initial_stock=args.initial_stock,
        order_lines=args.order_lines,
        candidates_per_task=args.candidates,
        transaction_mix=_parse_mix(args.transaction_mix),
    )


def _manager_factory(args: argparse.Namespace) -> ManagerFactory:
    if args.operation_policy in {"new-order", "learned-new-order"}:
        operation_policy = OperationPolicyTable.tpcc_new_order(
            hot_object_threshold=args.operation_hot_threshold
        )
    elif args.operation_policy == "tpcc-atcc" or (
        args.operation_policy == "atcc" and args.workload == "tpcc"
    ):
        operation_policy = OperationPolicyTable.tpcc_atcc()
    elif args.operation_policy == "ycsb-atcc" or (
        args.operation_policy == "atcc" and args.workload == "ycsb"
    ):
        operation_policy = OperationPolicyTable.ycsb_atcc()
    else:
        operation_policy = OperationPolicyTable.default()
    if args.adaptive_policy == "new-order":
        return lambda: AgentTransactionManager(
            adaptive_policy=AdaptivePolicyTable.new_order(
                wide_write_threshold=args.wide_write_threshold
            ),
            operation_policy=operation_policy,
        )
    return lambda: AgentTransactionManager(operation_policy=operation_policy)


def _training_seeds(args: argparse.Namespace) -> Tuple[int, ...]:
    seed = args.seed if args.train_seed is None else args.train_seed
    repeats = args.repeats if args.train_repeats <= 0 else args.train_repeats
    return tuple(seed + offset for offset in range(repeats))


def _parse_mix(value: str) -> Tuple[Tuple[str, float], ...]:
    entries = []
    for item in _split_csv(value):
        name, weight = item.split(":", 1)
        entries.append((name.strip(), float(weight)))
    return tuple(entries)


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _normalize_path(path: str) -> str:
    normalized = path.strip().lower().replace("_", "-")
    aliases = {
        "traditional": "traditional-k",
        "k-transactions": "traditional-k",
        "astra": "astra",
        "semantic": "astra",
    }
    return aliases.get(normalized, normalized)


if __name__ == "__main__":
    raise SystemExit(main())
