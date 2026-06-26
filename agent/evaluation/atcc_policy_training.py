"""Offline policy-table training for phase-aware ATCC.

The original ATCC system trains a compact table from workload feedback and then
uses that table for low-latency runtime decisions.  This module mirrors that
workflow for the data-agent runtime: run configurable agent-like TPCC/YCSB
episodes, accumulate the phase-aware Q table plus hot-object telemetry, and
emit a JSON artifact that the retry evaluator can load with --policy-artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TextIO

from agent.evaluation.atcc_schema import ATCC_STATE_SCHEMA, atcc_state_schema
from agent.evaluation.atcc_retry_experiment import (
    RetryRunSummary,
    _build_workload,
    _operation_policy,
    _run_one_retry,
    aggregate_retry_runs,
)
from agent.workloads import AgentWorkload


def train_phase_atcc_policy(
    workload: AgentWorkload,
    *,
    workload_kind: str,
    workload_config: Optional[Dict[str, Any]] = None,
    episodes: int,
    task_count: int,
    seed: int,
    workers: int,
    agent_slots: int,
    planning_delay_s: float,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    agent_admission_mode: str = "planning-only",
    tokens_per_operation: float = 2703.0,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    atcc_lock_wait_cost_per_s: Optional[float] = None,
    atcc_lock_action_cost: Optional[float] = None,
    atcc_lock_queue_depth_cost: Optional[float] = None,
    atcc_lock_handoff_cost: Optional[float] = None,
    atcc_committing_count_cost: Optional[float] = None,
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
) -> Dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if task_count <= 0:
        raise ValueError("task_count must be positive")

    policy = _operation_policy(str(workload_kind), "phase-rl")
    if policy.atcc_module is None:
        raise ValueError("phase-rl policy did not create a phase-aware ATCC module")
    if atcc_lock_wait_cost_per_s is not None:
        policy.atcc_module.lock_wait_cost_per_s = float(atcc_lock_wait_cost_per_s)
    if atcc_lock_action_cost is not None:
        policy.atcc_module.lock_action_cost = float(atcc_lock_action_cost)
    if atcc_lock_queue_depth_cost is not None:
        policy.atcc_module.lock_queue_depth_cost = float(atcc_lock_queue_depth_cost)
    if atcc_lock_handoff_cost is not None:
        policy.atcc_module.lock_handoff_cost = float(atcc_lock_handoff_cost)
    if atcc_committing_count_cost is not None:
        policy.atcc_module.committing_count_cost = float(atcc_committing_count_cost)

    runs: list[RetryRunSummary] = []
    started_at = time.perf_counter()
    for episode in range(int(episodes)):
        run_seed = int(seed) + episode
        tasks = tuple(workload.generate_tasks(task_count, seed=run_seed))
        runs.append(
            _run_one_retry(
                workload,
                tasks,
                "adaptive-op-strict",
                workload_kind=str(workload_kind),
                policy_variant="phase-rl",
                seed=run_seed,
                workers=workers,
                agent_slots=agent_slots,
                agent_admission_mode=agent_admission_mode,
                planning_delay_s=planning_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                tokens_per_operation=tokens_per_operation,
                operation_policy=policy,
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

    table = policy.to_dict()
    module = table.get("atcc_module") or {}
    learner = module.get("learner") or {}
    telemetry = table.get("telemetry") or {}
    training_config = {
        "episodes": int(episodes),
        "task_count": int(task_count),
        "seed": int(seed),
        "workers": int(workers),
        "agent_slots": int(agent_slots),
        "agent_admission_mode": str(agent_admission_mode),
        "planning_delay_s": float(planning_delay_s),
        "latency_distribution": str(latency_distribution),
        "latency_cv": float(latency_cv),
        "latency_max_s": float(latency_max_s),
        "max_attempts": int(max_attempts),
        "tokens_per_operation": float(tokens_per_operation),
        "background_workers": int(background_workers),
        "background_interval_s": float(background_interval_s),
        "background_strategy": str(background_strategy),
        "object_lock_scheduler": str(object_lock_scheduler),
        "object_lock_priority_burst": int(object_lock_priority_burst),
        "prelock_wait_budget_s": float(prelock_wait_budget_s),
        "prelock_wait_budget_mode": str(prelock_wait_budget_mode),
        "prelock_lease_mode": str(prelock_lease_mode),
        "atcc_lock_wait_cost_per_s": (
            None
            if atcc_lock_wait_cost_per_s is None
            else float(atcc_lock_wait_cost_per_s)
        ),
        "atcc_lock_action_cost": (
            None if atcc_lock_action_cost is None else float(atcc_lock_action_cost)
        ),
        "atcc_lock_queue_depth_cost": (
            None
            if atcc_lock_queue_depth_cost is None
            else float(atcc_lock_queue_depth_cost)
        ),
        "atcc_lock_handoff_cost": (
            None
            if atcc_lock_handoff_cost is None
            else float(atcc_lock_handoff_cost)
        ),
        "atcc_committing_count_cost": (
            None
            if atcc_committing_count_cost is None
            else float(atcc_committing_count_cost)
        ),
    }
    return {
        "artifact_type": "phase-aware-atcc-policy-artifact",
        "artifact_version": 2,
        "training_method": "offline-simulation-tabular-q-learning",
        "source_system": "data-agent-runtime",
        "atcc_state_schema": atcc_state_schema(),
        "workload": workload.name,
        "workload_kind": str(workload_kind),
        "workload_config": dict(workload_config or {}),
        "strategy": "adaptive-op-strict",
        "policy_variant": "phase-rl",
        "training_config": training_config,
        "training_elapsed_s": time.perf_counter() - started_at,
        "runs": [run.to_dict() for run in runs],
        "aggregates": aggregate_retry_runs(runs),
        "stats": _artifact_stats(table),
        "operation_policy_table": table,
    }


def _artifact_stats(table: Dict[str, Any]) -> Dict[str, Any]:
    module = table.get("atcc_module") or {}
    learner = module.get("learner") or {}
    visits = dict(learner.get("visits", {}))
    action_visits: Counter[str] = Counter()
    for key, count in visits.items():
        _state, separator, action = str(key).rpartition("|")
        if separator:
            action_visits[action] += int(count)
    q_values = dict(learner.get("q_values", {}))
    telemetry = dict(table.get("telemetry", {}))
    runtime_stats = dict(table.get("atcc_runtime_stats", {}))
    return {
        "atcc_state_count": len(q_values),
        "atcc_state_schema_version": int(ATCC_STATE_SCHEMA["version"]),
        "atcc_state_has_object_class": any(
            "class=" in str(state_key) for state_key in q_values
        ),
        "atcc_update_count": int(learner.get("updates", 0)),
        "atcc_action_visits": dict(sorted(action_visits.items())),
        "atcc_runtime_observation_count": int(
            runtime_stats.get("observations", 0)
        ),
        "atcc_runtime_abort_rate": float(
            runtime_stats.get("ewma_abort_rate", 0.0)
        ),
        "atcc_runtime_lock_wait_s": float(
            runtime_stats.get("ewma_lock_wait_s", 0.0)
        ),
        "atcc_runtime_latency_s": float(
            runtime_stats.get("ewma_latency_s", 0.0)
        ),
        "atcc_runtime_lock_queue_depth": float(
            runtime_stats.get("ewma_lock_queue_depth", 0.0)
        ),
        "atcc_runtime_lock_handoff_count": float(
            runtime_stats.get("ewma_lock_handoff_count", 0.0)
        ),
        "atcc_runtime_committing_count": float(
            runtime_stats.get("ewma_committing_count", 0.0)
        ),
        "telemetry_key_count": len(telemetry),
        "telemetry_observation_count": sum(
            int(value.get("observations", 0))
            for value in telemetry.values()
            if isinstance(value, dict)
        ),
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a phase-aware ATCC policy table.")
    parser.add_argument("--workload", choices=("tpcc", "ycsb"), default="tpcc")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--agent-slots", type=int, default=0)
    parser.add_argument(
        "--agent-admission-mode",
        choices=("planning-only", "before-begin"),
        default="planning-only",
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
    parser.add_argument("--tokens-per-operation", type=float, default=2703.0)
    parser.add_argument("--background-workers", type=int, default=0)
    parser.add_argument("--background-interval-ms", type=float, default=0.0)
    parser.add_argument("--background-strategy", default="occ")
    parser.add_argument(
        "--object-lock-scheduler",
        choices=("race", "priority", "bounded-priority"),
        default="race",
    )
    parser.add_argument("--object-lock-priority-burst", type=int, default=2)
    parser.add_argument("--prelock-wait-budget-ms", type=float, default=0.0)
    parser.add_argument(
        "--prelock-wait-budget-mode",
        choices=("transaction", "object"),
        default="transaction",
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
    )
    parser.add_argument(
        "--atcc-lock-wait-cost-per-s",
        type=float,
        help="Override the phase-aware ATCC reward penalty for one second of lock wait.",
    )
    parser.add_argument(
        "--atcc-lock-action-cost",
        type=float,
        help="Override the fixed phase-aware ATCC reward penalty for taking a lock action.",
    )
    parser.add_argument(
        "--atcc-lock-queue-depth-cost",
        type=float,
        help="Override the phase-aware ATCC reward penalty for each queued lock waiter.",
    )
    parser.add_argument(
        "--atcc-lock-handoff-cost",
        type=float,
        help="Override the phase-aware ATCC reward penalty for each lock handoff signal.",
    )
    parser.add_argument(
        "--atcc-committing-count-cost",
        type=float,
        help="Override the phase-aware ATCC reward penalty for each committing-pressure signal.",
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
    artifact = train_phase_atcc_policy(
        workload,
        workload_kind=args.workload,
        workload_config=workload_config,
        episodes=args.episodes,
        task_count=args.task_count,
        seed=args.seed,
        workers=args.workers,
        agent_slots=args.agent_slots,
        agent_admission_mode=args.agent_admission_mode,
        planning_delay_s=args.planning_delay_ms / 1000.0,
        latency_distribution=args.latency_distribution,
        latency_cv=args.latency_cv,
        latency_max_s=args.latency_max_ms / 1000.0,
        max_attempts=args.max_attempts,
        tokens_per_operation=args.tokens_per_operation,
        background_workers=args.background_workers,
        background_interval_s=args.background_interval_ms / 1000.0,
        background_strategy=args.background_strategy,
        object_lock_scheduler=args.object_lock_scheduler,
        object_lock_priority_burst=args.object_lock_priority_burst,
        prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
        prelock_wait_budget_mode=args.prelock_wait_budget_mode,
        prelock_lease_mode=args.prelock_lease_mode,
        atcc_lock_wait_cost_per_s=args.atcc_lock_wait_cost_per_s,
        atcc_lock_action_cost=args.atcc_lock_action_cost,
        atcc_lock_queue_depth_cost=args.atcc_lock_queue_depth_cost,
        atcc_lock_handoff_cost=args.atcc_lock_handoff_cost,
        atcc_committing_count_cost=args.atcc_committing_count_cost,
    )
    text = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
