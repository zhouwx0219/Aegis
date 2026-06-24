"""Reward-cost search for phase-aware ATCC policy artifacts.

This utility keeps ATCC cost tuning reproducible: each candidate reward-cost
setting is trained, evaluated against the same OCC/2PL/ATCC matrix, and written
as explicit JSON artifacts before any larger confirmation run is considered.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

from agent.evaluation.atcc_policy_training import train_phase_atcc_policy
from agent.evaluation.atcc_retry_experiment import (
    _build_workload,
    aggregate_retry_runs,
    run_retry_matrix,
)
from agent.workloads import AgentWorkload


EVAL_STRATEGIES = ("occ", "2pl-pre", "adaptive-op-strict")


def run_reward_cost_search(
    workload: AgentWorkload,
    *,
    workload_kind: str,
    workload_config: Optional[Mapping[str, Any]] = None,
    output_dir: Path = Path("results/atcc_reward_cost_search"),
    lock_wait_costs: Iterable[float] = (70.0, 100.0, 150.0),
    lock_action_costs: Iterable[float] = (0.02,),
    train_episodes: int,
    train_task_count: int,
    eval_task_count: int,
    eval_repeats: int,
    seed: int,
    workers: int,
    agent_slots: int,
    planning_delay_s: float,
    latency_distribution: str,
    latency_cv: float,
    latency_max_s: float,
    max_attempts: int,
    tokens_per_operation: float = 2703.0,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    write_files: bool = True,
) -> Dict[str, Any]:
    lock_wait_values = tuple(float(value) for value in lock_wait_costs)
    lock_action_values = tuple(float(value) for value in lock_action_costs)
    if not lock_wait_values:
        raise ValueError("lock_wait_costs must not be empty")
    if not lock_action_values:
        raise ValueError("lock_action_costs must not be empty")
    if train_episodes <= 0 or train_task_count <= 0 or eval_task_count <= 0:
        raise ValueError("task counts and train_episodes must be positive")
    if eval_repeats <= 0:
        raise ValueError("eval_repeats must be positive")

    output_dir = Path(output_dir)
    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for wait_cost in lock_wait_values:
        for action_cost in lock_action_values:
            label = _cost_label(wait_cost, action_cost)
            artifact = train_phase_atcc_policy(
                workload,
                workload_kind=workload_kind,
                workload_config=dict(workload_config or {}),
                episodes=train_episodes,
                task_count=train_task_count,
                seed=seed,
                workers=workers,
                agent_slots=agent_slots,
                planning_delay_s=planning_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                tokens_per_operation=tokens_per_operation,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
                atcc_lock_wait_cost_per_s=wait_cost,
                atcc_lock_action_cost=action_cost,
            )
            artifact_path = output_dir / f"phase_atcc_policy_{label}.json"
            if write_files:
                _write_json(artifact_path, artifact)

            runs = run_retry_matrix(
                workload,
                EVAL_STRATEGIES,
                workload_kind=workload_kind,
                policy_variant="phase-rl",
                task_count=eval_task_count,
                seed=seed + 1000,
                repeats=eval_repeats,
                workers=workers,
                agent_slots=agent_slots,
                planning_delay_s=planning_delay_s,
                latency_distribution=latency_distribution,
                latency_cv=latency_cv,
                latency_max_s=latency_max_s,
                max_attempts=max_attempts,
                tokens_per_operation=tokens_per_operation,
                policy_artifact=artifact,
                policy_epsilon=0.0,
                background_workers=background_workers,
                background_interval_s=background_interval_s,
                background_strategy=background_strategy,
            )
            aggregates = aggregate_retry_runs(runs)
            eval_report = {
                "artifact_type": "phase-aware-atcc-reward-cost-evaluation",
                "lock_wait_cost_per_s": wait_cost,
                "lock_action_cost": action_cost,
                "policy_artifact": str(artifact_path),
                "runs": [run.to_dict() for run in runs],
                "aggregates": aggregates,
            }
            eval_path = output_dir / f"phase_atcc_eval_{label}.json"
            if write_files:
                _write_json(eval_path, eval_report)
            candidates.append(
                _candidate_summary(
                    wait_cost=wait_cost,
                    action_cost=action_cost,
                    artifact_path=artifact_path,
                    eval_path=eval_path,
                    aggregates=aggregates,
                )
            )

    ranked = sorted(
        candidates,
        key=lambda row: (
            row["atcc_commit_rate"],
            row["atcc_throughput"],
            -row["atcc_p99_latency_s"],
            -row["atcc_prelock_wait_per_task_s"],
            -row["atcc_pessimistic_decisions"],
        ),
        reverse=True,
    )
    report = {
        "artifact_type": "phase-aware-atcc-reward-cost-search",
        "source_system": "data-agent-runtime",
        "workload": workload.name,
        "workload_kind": workload_kind,
        "workload_config": dict(workload_config or {}),
        "config": {
            "train_episodes": int(train_episodes),
            "train_task_count": int(train_task_count),
            "eval_task_count": int(eval_task_count),
            "eval_repeats": int(eval_repeats),
            "seed": int(seed),
            "workers": int(workers),
            "agent_slots": int(agent_slots),
            "planning_delay_s": float(planning_delay_s),
            "latency_distribution": str(latency_distribution),
            "latency_cv": float(latency_cv),
            "latency_max_s": float(latency_max_s),
            "max_attempts": int(max_attempts),
            "tokens_per_operation": float(tokens_per_operation),
            "background_workers": int(background_workers),
            "background_interval_s": float(background_interval_s),
            "background_strategy": str(background_strategy),
            "lock_wait_costs": list(lock_wait_values),
            "lock_action_costs": list(lock_action_values),
        },
        "ranking_note": (
            "Ranking is for screening only; promising candidates still require "
            "larger multi-seed confirmation before becoming headline results."
        ),
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": ranked[0] if ranked else None,
    }
    if write_files:
        _write_json(output_dir / "atcc_reward_cost_search.json", report)
    return report


def _candidate_summary(
    *,
    wait_cost: float,
    action_cost: float,
    artifact_path: Path,
    eval_path: Path,
    aggregates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_strategy = {str(row.get("strategy")): dict(row) for row in aggregates}
    atcc = by_strategy.get("adaptive-op-strict", {})
    occ = by_strategy.get("occ", {})
    two_pl = by_strategy.get("2pl-pre", {})
    atcc_policies = dict(atcc.get("operation_policy_counts", {}) or {})
    two_pl_policies = dict(two_pl.get("operation_policy_counts", {}) or {})
    return {
        "lock_wait_cost_per_s": float(wait_cost),
        "lock_action_cost": float(action_cost),
        "policy_artifact": str(artifact_path),
        "evaluation": str(eval_path),
        "atcc_commit_rate": float(atcc.get("commit_rate", 0.0)),
        "atcc_throughput": float(atcc.get("committed_throughput", 0.0)),
        "atcc_attempts_per_task": float(atcc.get("attempts_per_task", 0.0)),
        "atcc_p99_latency_s": float(atcc.get("agent_latency_p99_s", 0.0)),
        "atcc_wasted_tokens_per_task": float(
            atcc.get("estimated_wasted_tokens_per_task", 0.0)
        ),
        "atcc_prelock_wait_per_task_s": float(
            atcc.get("prelock_wait_per_task_s", 0.0)
        ),
        "atcc_pessimistic_decisions": int(atcc_policies.get("pessimistic", 0)),
        "occ_commit_rate": float(occ.get("commit_rate", 0.0)),
        "occ_throughput": float(occ.get("committed_throughput", 0.0)),
        "occ_wasted_tokens_per_task": float(
            occ.get("estimated_wasted_tokens_per_task", 0.0)
        ),
        "two_pl_throughput": float(two_pl.get("committed_throughput", 0.0)),
        "two_pl_p99_latency_s": float(two_pl.get("agent_latency_p99_s", 0.0)),
        "two_pl_pessimistic_decisions": int(
            two_pl_policies.get("pessimistic", 0)
        ),
        "atcc_vs_occ_throughput_x": _ratio(
            atcc.get("committed_throughput", 0.0),
            occ.get("committed_throughput", 0.0),
        ),
        "atcc_vs_occ_waste_reduction_pct": _reduction_pct(
            atcc.get("estimated_wasted_tokens_per_task", 0.0),
            occ.get("estimated_wasted_tokens_per_task", 0.0),
        ),
        "atcc_vs_2pl_throughput_x": _ratio(
            atcc.get("committed_throughput", 0.0),
            two_pl.get("committed_throughput", 0.0),
        ),
        "atcc_vs_2pl_pessimistic_decision_delta": int(
            atcc_policies.get("pessimistic", 0)
        )
        - int(two_pl_policies.get("pessimistic", 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search phase-aware ATCC reward lock costs."
    )
    parser.add_argument("--workload", choices=("tpcc", "ycsb"), default="tpcc")
    parser.add_argument("--output-dir", type=Path, default=Path("results/atcc_reward_cost_search"))
    parser.add_argument("--lock-wait-costs", default="70,100,150")
    parser.add_argument("--lock-action-costs", default="0.02")
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--train-task-count", type=int, default=100)
    parser.add_argument("--eval-task-count", type=int, default=100)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--agent-slots", type=int, default=4)
    parser.add_argument("--planning-delay-ms", type=float, default=50.0)
    parser.add_argument(
        "--latency-distribution",
        choices=("fixed", "lognormal", "pareto"),
        default="lognormal",
    )
    parser.add_argument("--latency-cv", type=float, default=0.8)
    parser.add_argument("--latency-max-ms", type=float, default=200.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--tokens-per-operation", type=float, default=2703.0)
    parser.add_argument("--background-workers", type=int, default=8)
    parser.add_argument("--background-interval-ms", type=float, default=1.0)
    parser.add_argument("--background-strategy", default="occ")
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
    report = run_reward_cost_search(
        workload,
        workload_kind=args.workload,
        workload_config=workload_config,
        output_dir=args.output_dir,
        lock_wait_costs=_parse_float_csv(args.lock_wait_costs),
        lock_action_costs=_parse_float_csv(args.lock_action_costs),
        train_episodes=args.train_episodes,
        train_task_count=args.train_task_count,
        eval_task_count=args.eval_task_count,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
        workers=args.workers,
        agent_slots=args.agent_slots,
        planning_delay_s=args.planning_delay_ms / 1000.0,
        latency_distribution=args.latency_distribution,
        latency_cv=args.latency_cv,
        latency_max_s=args.latency_max_ms / 1000.0,
        max_attempts=args.max_attempts,
        tokens_per_operation=args.tokens_per_operation,
        background_workers=args.background_workers,
        background_interval_s=args.background_interval_ms / 1000.0,
        background_strategy=args.background_strategy,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _cost_label(wait_cost: float, action_cost: float) -> str:
    return (
        "wait"
        + _number_label(wait_cost)
        + "_action"
        + _number_label(action_cost)
    )


def _number_label(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _parse_float_csv(text: str) -> Tuple[float, ...]:
    values = tuple(
        float(part.strip())
        for part in str(text).split(",")
        if part.strip()
    )
    if not values:
        raise ValueError("expected at least one numeric CSV value")
    return values


def _ratio(numerator: Any, denominator: Any) -> float:
    denom = float(denominator or 0.0)
    if denom <= 0.0:
        return 0.0
    return float(numerator or 0.0) / denom


def _reduction_pct(value: Any, baseline: Any) -> float:
    base = float(baseline or 0.0)
    if base <= 0.0:
        return 0.0
    return (1.0 - float(value or 0.0) / base) * 100.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
