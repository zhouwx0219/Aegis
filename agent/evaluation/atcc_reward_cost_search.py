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


EVAL_STRATEGIES = ("occ", "2pl", "adaptive-op-strict")

RANKING_WEIGHTS = {
    "commit_rate": 4.0,
    "agent_throughput_vs_2pl": 2.0,
    "total_throughput_vs_2pl": 1.0,
    "background_throughput_vs_2pl": 1.5,
    "tail_latency_vs_2pl": 1.5,
    "waste_reduction_vs_occ": 1.0,
    "prelock_wait_penalty": -0.5,
    "excess_pessimistic_decisions_penalty": -0.25,
}


def run_reward_cost_search(
    workload: AgentWorkload,
    *,
    workload_kind: str,
    workload_config: Optional[Mapping[str, Any]] = None,
    output_dir: Path = Path("results/atcc_reward_cost_search"),
    lock_wait_costs: Iterable[float] = (70.0, 100.0, 150.0),
    lock_action_costs: Iterable[float] = (0.02,),
    lock_queue_depth_costs: Iterable[float] = (0.05,),
    lock_handoff_costs: Iterable[float] = (0.03,),
    committing_count_costs: Iterable[float] = (0.005,),
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
    agent_admission_mode: str = "planning-only",
    tokens_per_operation: float = 2703.0,
    background_workers: int = 0,
    background_interval_s: float = 0.0,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    object_lock_schedulers: Optional[Iterable[str]] = None,
    object_lock_priority_bursts: Optional[Iterable[int]] = None,
    prelock_wait_budget_s_values: Optional[Iterable[float]] = None,
    prelock_wait_budget_modes: Optional[Iterable[str]] = None,
    prelock_lease_modes: Optional[Iterable[str]] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    lock_wait_values = tuple(float(value) for value in lock_wait_costs)
    lock_action_values = tuple(float(value) for value in lock_action_costs)
    queue_depth_values = tuple(float(value) for value in lock_queue_depth_costs)
    handoff_values = tuple(float(value) for value in lock_handoff_costs)
    committing_values = tuple(float(value) for value in committing_count_costs)
    scheduler_values = _string_values(
        object_lock_schedulers,
        fallback=object_lock_scheduler,
        name="object_lock_schedulers",
    )
    priority_burst_values = _int_values(
        object_lock_priority_bursts,
        fallback=object_lock_priority_burst,
        name="object_lock_priority_bursts",
    )
    wait_budget_values = _float_values(
        prelock_wait_budget_s_values,
        fallback=prelock_wait_budget_s,
        name="prelock_wait_budget_s_values",
    )
    wait_budget_mode_values = _string_values(
        prelock_wait_budget_modes,
        fallback=prelock_wait_budget_mode,
        name="prelock_wait_budget_modes",
    )
    lease_mode_values = _string_values(
        prelock_lease_modes,
        fallback=prelock_lease_mode,
        name="prelock_lease_modes",
    )
    if not lock_wait_values:
        raise ValueError("lock_wait_costs must not be empty")
    if not lock_action_values:
        raise ValueError("lock_action_costs must not be empty")
    if not queue_depth_values:
        raise ValueError("lock_queue_depth_costs must not be empty")
    if not handoff_values:
        raise ValueError("lock_handoff_costs must not be empty")
    if not committing_values:
        raise ValueError("committing_count_costs must not be empty")
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
            for queue_depth_cost in queue_depth_values:
                for handoff_cost in handoff_values:
                    for committing_cost in committing_values:
                        for scheduler in scheduler_values:
                            for priority_burst in priority_burst_values:
                                for wait_budget_s in wait_budget_values:
                                    for wait_budget_mode in wait_budget_mode_values:
                                        for lease_mode in lease_mode_values:
                                            label = _cost_label(
                                                wait_cost,
                                                action_cost,
                                                queue_depth_cost,
                                                handoff_cost,
                                                committing_cost,
                                                scheduler=scheduler,
                                                priority_burst=priority_burst,
                                                wait_budget_s=wait_budget_s,
                                                wait_budget_mode=wait_budget_mode,
                                                lease_mode=lease_mode,
                                            )
                                            artifact = train_phase_atcc_policy(
                                                workload,
                                                workload_kind=workload_kind,
                                                workload_config=dict(workload_config or {}),
                                                episodes=train_episodes,
                                                task_count=train_task_count,
                                                seed=seed,
                                                workers=workers,
                                                agent_slots=agent_slots,
                                                agent_admission_mode=agent_admission_mode,
                                                planning_delay_s=planning_delay_s,
                                                latency_distribution=latency_distribution,
                                                latency_cv=latency_cv,
                                                latency_max_s=latency_max_s,
                                                max_attempts=max_attempts,
                                                tokens_per_operation=tokens_per_operation,
                                                background_workers=background_workers,
                                                background_interval_s=background_interval_s,
                                                background_strategy=background_strategy,
                                                object_lock_scheduler=scheduler,
                                                object_lock_priority_burst=priority_burst,
                                                prelock_wait_budget_s=wait_budget_s,
                                                prelock_wait_budget_mode=wait_budget_mode,
                                                prelock_lease_mode=lease_mode,
                                                atcc_lock_wait_cost_per_s=wait_cost,
                                                atcc_lock_action_cost=action_cost,
                                                atcc_lock_queue_depth_cost=queue_depth_cost,
                                                atcc_lock_handoff_cost=handoff_cost,
                                                atcc_committing_count_cost=committing_cost,
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
                                                agent_admission_mode=agent_admission_mode,
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
                                                object_lock_scheduler=scheduler,
                                                object_lock_priority_burst=priority_burst,
                                                prelock_wait_budget_s=wait_budget_s,
                                                prelock_wait_budget_mode=wait_budget_mode,
                                                prelock_lease_mode=lease_mode,
                                            )
                                            aggregates = aggregate_retry_runs(runs)
                                            eval_report = {
                                                "artifact_type": "phase-aware-atcc-reward-cost-evaluation",
                                                "lock_wait_cost_per_s": wait_cost,
                                                "lock_action_cost": action_cost,
                                                "lock_queue_depth_cost": queue_depth_cost,
                                                "lock_handoff_cost": handoff_cost,
                                                "committing_count_cost": committing_cost,
                                                "agent_admission_mode": str(agent_admission_mode),
                                                "object_lock_scheduler": str(scheduler),
                                                "object_lock_priority_burst": int(priority_burst),
                                                "prelock_wait_budget_s": float(wait_budget_s),
                                                "prelock_wait_budget_mode": str(wait_budget_mode),
                                                "prelock_lease_mode": str(lease_mode),
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
                                                    queue_depth_cost=queue_depth_cost,
                                                    handoff_cost=handoff_cost,
                                                    committing_cost=committing_cost,
                                                    object_lock_scheduler=scheduler,
                                                    object_lock_priority_burst=priority_burst,
                                                    prelock_wait_budget_s=wait_budget_s,
                                                    prelock_wait_budget_mode=wait_budget_mode,
                                                    prelock_lease_mode=lease_mode,
                                                    artifact_path=artifact_path,
                                                    eval_path=eval_path,
                                                    aggregates=aggregates,
                                                )
                                            )

    candidates = _score_candidates(candidates)
    ranked = _rank_candidates(candidates)
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
            "object_lock_scheduler": str(scheduler_values[0]),
            "object_lock_schedulers": list(scheduler_values),
            "object_lock_priority_burst": int(priority_burst_values[0]),
            "object_lock_priority_bursts": list(priority_burst_values),
            "prelock_wait_budget_s": float(wait_budget_values[0]),
            "prelock_wait_budget_s_values": list(wait_budget_values),
            "prelock_wait_budget_mode": str(wait_budget_mode_values[0]),
            "prelock_wait_budget_modes": list(wait_budget_mode_values),
            "prelock_lease_mode": str(lease_mode_values[0]),
            "prelock_lease_modes": list(lease_mode_values),
            "lock_wait_costs": list(lock_wait_values),
            "lock_action_costs": list(lock_action_values),
            "lock_queue_depth_costs": list(queue_depth_values),
            "lock_handoff_costs": list(handoff_values),
            "committing_count_costs": list(committing_values),
        },
        "ranking_note": (
            "Ranking uses a capped multi-objective score so OCC-collapse "
            "throughput wins do not hide tail latency, wasted-token, or "
            "background-starvation regressions. It is for screening only; "
            "promising candidates still require larger multi-seed confirmation "
            "before becoming headline results."
        ),
        "ranking_metric": "multi_objective_score",
        "ranking_weights": dict(RANKING_WEIGHTS),
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
    queue_depth_cost: float,
    handoff_cost: float,
    committing_cost: float,
    object_lock_scheduler: str,
    object_lock_priority_burst: int,
    prelock_wait_budget_s: float,
    prelock_wait_budget_mode: str,
    prelock_lease_mode: str,
    artifact_path: Path,
    eval_path: Path,
    aggregates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_strategy = {str(row.get("strategy")): dict(row) for row in aggregates}
    atcc = by_strategy.get("adaptive-op-strict", {})
    occ = by_strategy.get("occ", {})
    two_pl = by_strategy.get("2pl", {})
    atcc_policies = dict(atcc.get("operation_policy_counts", {}) or {})
    two_pl_policies = dict(two_pl.get("operation_policy_counts", {}) or {})
    atcc_throughput = float(atcc.get("committed_throughput", 0.0))
    atcc_background_throughput = float(atcc.get("background_throughput", 0.0))
    occ_throughput = float(occ.get("committed_throughput", 0.0))
    occ_background_throughput = float(occ.get("background_throughput", 0.0))
    two_pl_throughput = float(two_pl.get("committed_throughput", 0.0))
    two_pl_background_throughput = float(
        two_pl.get("background_throughput", 0.0)
    )
    atcc_total_throughput = atcc_throughput + atcc_background_throughput
    occ_total_throughput = occ_throughput + occ_background_throughput
    two_pl_total_throughput = two_pl_throughput + two_pl_background_throughput
    return {
        "lock_wait_cost_per_s": float(wait_cost),
        "lock_action_cost": float(action_cost),
        "lock_queue_depth_cost": float(queue_depth_cost),
        "lock_handoff_cost": float(handoff_cost),
        "committing_count_cost": float(committing_cost),
        "object_lock_scheduler": str(object_lock_scheduler),
        "object_lock_priority_burst": int(object_lock_priority_burst),
        "prelock_wait_budget_s": float(prelock_wait_budget_s),
        "prelock_wait_budget_mode": str(prelock_wait_budget_mode),
        "prelock_lease_mode": str(prelock_lease_mode),
        "prelock_lease_semantics": _lease_mode_semantics(prelock_lease_mode),
        "atcc_long_transaction_window_comparable": (
            _is_long_transaction_window_comparable(prelock_lease_mode)
        ),
        "policy_artifact": str(artifact_path),
        "evaluation": str(eval_path),
        "atcc_commit_rate": float(atcc.get("commit_rate", 0.0)),
        "atcc_throughput": atcc_throughput,
        "atcc_total_throughput": atcc_total_throughput,
        "atcc_attempts_per_task": float(atcc.get("attempts_per_task", 0.0)),
        "atcc_p99_latency_s": float(atcc.get("agent_latency_p99_s", 0.0)),
        "atcc_wasted_tokens_per_task": float(
            atcc.get("estimated_wasted_tokens_per_task", 0.0)
        ),
        "atcc_lease_refresh_regenerations": int(
            atcc.get("lease_refresh_regenerations", 0)
        ),
        "atcc_lease_refresh_regenerations_per_task": float(
            atcc.get("lease_refresh_regenerations_per_task", 0.0)
        ),
        "atcc_estimated_refresh_tokens": float(
            atcc.get("estimated_refresh_tokens", 0.0)
        ),
        "atcc_estimated_refresh_tokens_per_task": float(
            atcc.get("estimated_refresh_tokens_per_task", 0.0)
        ),
        "atcc_prelock_wait_per_task_s": float(
            atcc.get("prelock_wait_per_task_s", 0.0)
        ),
        "atcc_background_throughput": atcc_background_throughput,
        "atcc_pessimistic_decisions": int(atcc_policies.get("pessimistic", 0)),
        "occ_commit_rate": float(occ.get("commit_rate", 0.0)),
        "occ_throughput": occ_throughput,
        "occ_total_throughput": occ_total_throughput,
        "occ_background_throughput": occ_background_throughput,
        "occ_wasted_tokens_per_task": float(
            occ.get("estimated_wasted_tokens_per_task", 0.0)
        ),
        "two_pl_throughput": two_pl_throughput,
        "two_pl_total_throughput": two_pl_total_throughput,
        "two_pl_background_throughput": two_pl_background_throughput,
        "two_pl_p99_latency_s": float(two_pl.get("agent_latency_p99_s", 0.0)),
        "two_pl_pessimistic_decisions": int(
            two_pl_policies.get("pessimistic", 0)
        ),
        "atcc_vs_occ_throughput_x": _ratio(
            atcc_throughput,
            occ_throughput,
        ),
        "atcc_vs_occ_waste_reduction_pct": _reduction_pct(
            atcc.get("estimated_wasted_tokens_per_task", 0.0),
            occ.get("estimated_wasted_tokens_per_task", 0.0),
        ),
        "atcc_vs_2pl_throughput_x": _ratio(
            atcc_throughput,
            two_pl_throughput,
        ),
        "atcc_vs_2pl_total_throughput_x": _ratio(
            atcc_total_throughput,
            two_pl_total_throughput,
        ),
        "atcc_vs_2pl_background_throughput_x": _ratio(
            atcc_background_throughput,
            two_pl_background_throughput,
        ),
        "atcc_vs_2pl_tail_latency_x": _ratio(
            two_pl.get("agent_latency_p99_s", 0.0),
            atcc.get("agent_latency_p99_s", 0.0),
        ),
        "atcc_vs_2pl_pessimistic_decision_delta": int(
            atcc_policies.get("pessimistic", 0)
        )
        - int(two_pl_policies.get("pessimistic", 0)),
    }


def _rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> Sequence[Dict[str, Any]]:
    scored = _score_candidates(candidates)
    return sorted(
        scored,
        key=lambda row: (
            row["multi_objective_score"],
            row["atcc_commit_rate"],
            row["atcc_throughput"],
            -row["atcc_p99_latency_s"],
        ),
        reverse=True,
    )


def _score_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> Sequence[Dict[str, Any]]:
    return [_score_candidate(dict(row)) for row in candidates]


def _score_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    components = _ranking_score_components(row)
    row["ranking_score_components"] = components
    row["multi_objective_score"] = sum(
        RANKING_WEIGHTS[name] * value
        for name, value in components.items()
    )
    return row


def _lease_mode_semantics(lease_mode: str) -> str:
    mode = str(lease_mode or "hold").strip().lower()
    if mode == "hold":
        return "pre-planning-snapshot-held-locks"
    if mode == "yield-during-planning":
        return "pre-planning-snapshot-yielded-locks"
    if mode == "yield-refresh-regenerate":
        return "refresh-regenerate-after-planning"
    if mode == "defer-until-after-planning":
        return "post-planning-snapshot"
    return "unknown"


def _is_long_transaction_window_comparable(lease_mode: str) -> bool:
    return _lease_mode_semantics(lease_mode) in {
        "pre-planning-snapshot-held-locks",
        "pre-planning-snapshot-yielded-locks",
    }


def _ranking_score_components(row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "commit_rate": _clamp(float(row.get("atcc_commit_rate", 0.0)), 0.0, 1.0),
        "agent_throughput_vs_2pl": _capped_ratio(
            row.get("atcc_vs_2pl_throughput_x", 0.0)
        ),
        "total_throughput_vs_2pl": _capped_ratio(
            row.get("atcc_vs_2pl_total_throughput_x", 0.0)
        ),
        "background_throughput_vs_2pl": _capped_ratio(
            row.get("atcc_vs_2pl_background_throughput_x", 0.0)
        ),
        "tail_latency_vs_2pl": _capped_ratio(
            row.get("atcc_vs_2pl_tail_latency_x")
            or _ratio(
                row.get("two_pl_p99_latency_s", 0.0),
                row.get("atcc_p99_latency_s", 0.0),
            )
        ),
        "waste_reduction_vs_occ": _clamp(
            1.0
            + float(row.get("atcc_vs_occ_waste_reduction_pct", 0.0)) / 100.0,
            0.0,
            2.0,
        ),
        "prelock_wait_penalty": _clamp(
            float(row.get("atcc_prelock_wait_per_task_s", 0.0)) / 0.1,
            0.0,
            2.0,
        ),
        "excess_pessimistic_decisions_penalty": _clamp(
            max(0.0, float(row.get("atcc_vs_2pl_pessimistic_decision_delta", 0.0)))
            / 100.0,
            0.0,
            2.0,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search phase-aware ATCC reward lock costs."
    )
    parser.add_argument("--workload", choices=("tpcc", "ycsb"), default="tpcc")
    parser.add_argument("--output-dir", type=Path, default=Path("results/atcc_reward_cost_search"))
    parser.add_argument("--lock-wait-costs", default="70,100,150")
    parser.add_argument("--lock-action-costs", default="0.02")
    parser.add_argument("--lock-queue-depth-costs", default="0.05")
    parser.add_argument("--lock-handoff-costs", default="0.03")
    parser.add_argument("--committing-count-costs", default="0.005")
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--train-task-count", type=int, default=100)
    parser.add_argument("--eval-task-count", type=int, default=100)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--agent-slots", type=int, default=4)
    parser.add_argument(
        "--agent-admission-mode",
        choices=("planning-only", "before-begin"),
        default="planning-only",
    )
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
        lock_queue_depth_costs=_parse_float_csv(args.lock_queue_depth_costs),
        lock_handoff_costs=_parse_float_csv(args.lock_handoff_costs),
        committing_count_costs=_parse_float_csv(args.committing_count_costs),
        train_episodes=args.train_episodes,
        train_task_count=args.train_task_count,
        eval_task_count=args.eval_task_count,
        eval_repeats=args.eval_repeats,
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
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _cost_label(
    wait_cost: float,
    action_cost: float,
    queue_depth_cost: float,
    handoff_cost: float,
    committing_cost: float,
    *,
    scheduler: str = "",
    priority_burst: Optional[int] = None,
    wait_budget_s: Optional[float] = None,
    wait_budget_mode: str = "",
    lease_mode: str = "",
) -> str:
    label = (
        "wait"
        + _number_label(wait_cost)
        + "_action"
        + _number_label(action_cost)
        + "_queue"
        + _number_label(queue_depth_cost)
        + "_handoff"
        + _number_label(handoff_cost)
        + "_committing"
        + _number_label(committing_cost)
    )
    if scheduler:
        label += "_sched" + _text_label(scheduler)
    if priority_burst is not None:
        label += "_burst" + _number_label(float(priority_burst))
    if wait_budget_s is not None:
        label += "_waitbudget" + _number_label(float(wait_budget_s) * 1000.0) + "ms"
    if wait_budget_mode:
        label += "_waitmode" + _text_label(wait_budget_mode)
    if lease_mode:
        label += "_lease" + _text_label(lease_mode)
    return label


def _number_label(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _text_label(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _string_values(
    values: Optional[Iterable[str]],
    *,
    fallback: str,
    name: str,
) -> Tuple[str, ...]:
    if values is None:
        result = (str(fallback),)
    else:
        result = tuple(str(value).strip() for value in values if str(value).strip())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _int_values(
    values: Optional[Iterable[int]],
    *,
    fallback: int,
    name: str,
) -> Tuple[int, ...]:
    if values is None:
        result = (int(fallback),)
    else:
        result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _float_values(
    values: Optional[Iterable[float]],
    *,
    fallback: float,
    name: str,
) -> Tuple[float, ...]:
    if values is None:
        result = (float(fallback),)
    else:
        result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


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


def _capped_ratio(value: Any) -> float:
    return _clamp(float(value or 0.0), 0.0, 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
