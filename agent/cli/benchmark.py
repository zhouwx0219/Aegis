"""Run a small, delivery-oriented YCSB/TPC-C benchmark.

This module is intentionally a thin wrapper over the research retry runner.  It
keeps the public command understandable while preserving the same runtime path
used by the experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO

from agent.evaluation.atcc_retry_experiment import (
    aggregate_retry_runs,
    run_retry_matrix,
)
from agent.workloads import TPCCConfig, YCSBConfig, build_agent_workload


QUICK_STRATEGIES = (
    "occ",
    "adaptive-op-strict",
    "transaction-atcc-strict",
)
FULL_STRATEGIES = (
    "occ",
    "2pl-nowait",
    "2pl-wait-die",
    "mvcc-full",
    "silo-full",
    "tictoc-full",
    "adaptive-op-strict",
    "transaction-atcc-strict",
    "adaptive-hybrid",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("ycsb", "tpcc", "all"), default="all")
    parser.add_argument("--profile", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--strategies", choices=("quick", "full"), default="quick")
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=920104)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--agent-slots", type=int, default=1)
    parser.add_argument("--planning-delay-ms", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser


def run_delivery_benchmark(
    *,
    workload: str,
    profile: str,
    strategies: str,
    task_count: int,
    seed: int,
    repeats: int,
    workers: int,
    agent_slots: int,
    planning_delay_ms: float,
    max_attempts: int,
) -> Dict[str, Any]:
    workload_names = ("ycsb", "tpcc") if workload == "all" else (workload,)
    strategy_names = QUICK_STRATEGIES if strategies == "quick" else FULL_STRATEGIES
    runs = []
    workload_reports = []
    for workload_name in workload_names:
        agent_workload, workload_config = _profile_workload(workload_name, profile)
        workload_runs = run_retry_matrix(
            agent_workload,
            strategy_names,
            workload_kind=workload_name,
            policy_variant=(
                "ycsb-strict-tuned" if workload_name == "ycsb" else "default"
            ),
            task_count=task_count,
            seed=seed,
            repeats=repeats,
            workers=workers,
            agent_slots=agent_slots,
            agent_admission_mode="before-begin",
            planning_delay_s=planning_delay_ms / 1000.0,
            latency_distribution="fixed",
            latency_cv=0.0,
            latency_max_s=0.0,
            max_attempts=max_attempts,
            background_workers=0,
            object_lock_scheduler="bounded-priority",
            object_lock_priority_burst=2,
            prelock_wait_budget_s=0.070,
            prelock_wait_budget_mode="object",
            prelock_lease_mode=(
                "yield-refresh-regenerate"
                if workload_name == "ycsb"
                else "hold"
            ),
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )
        runs.extend(workload_runs)
        workload_reports.append(
            {
                "workload": agent_workload.name,
                "workload_kind": workload_name,
                "profile": profile,
                "workload_config": workload_config,
                "runs": [run.to_dict() for run in workload_runs],
                "aggregates": aggregate_retry_runs(workload_runs),
            }
        )
    return {
        "mode": "delivery-benchmark",
        "profile": profile,
        "strategy_set": strategies,
        "strategies": list(strategy_names),
        "task_count": task_count,
        "seed": seed,
        "repeats": repeats,
        "workers": workers,
        "agent_slots": agent_slots,
        "planning_delay_s": planning_delay_ms / 1000.0,
        "max_attempts": max_attempts,
        "workloads": workload_reports,
        "aggregates": aggregate_retry_runs(runs),
    }


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
) -> int:
    args = build_parser().parse_args(argv)
    report = run_delivery_benchmark(
        workload=args.workload,
        profile=args.profile,
        strategies=args.strategies,
        task_count=args.task_count,
        seed=args.seed,
        repeats=args.repeats,
        workers=args.workers,
        agent_slots=args.agent_slots,
        planning_delay_ms=args.planning_delay_ms,
        max_attempts=args.max_attempts,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    out.write(payload + "\n")
    return 0


def _profile_workload(workload: str, profile: str):
    if workload == "ycsb":
        config = _ycsb_config(profile)
        return (
            build_agent_workload("ycsb", "semantic", ycsb_config=config),
            dataclasses.asdict(config),
        )
    config = _tpcc_config(profile)
    return (
        build_agent_workload("tpcc", "semantic", tpcc_config=config),
        dataclasses.asdict(config),
    )


def _ycsb_config(profile: str) -> YCSBConfig:
    profiles = {
        "low": YCSBConfig(
            record_count=128,
            field_count=10,
            requests_per_task=4,
            candidates_per_task=2,
            read_weight=0.95,
            update_weight=0.05,
            zipf_theta=0.0,
            hotspot_fraction=0.0,
            hotspot_access_probability=0.0,
        ),
        "medium": YCSBConfig(
            record_count=96,
            field_count=10,
            requests_per_task=6,
            candidates_per_task=2,
            read_weight=0.90,
            update_weight=0.10,
            zipf_theta=0.7,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.50,
        ),
        "high": YCSBConfig(
            record_count=64,
            field_count=10,
            requests_per_task=8,
            candidates_per_task=2,
            read_weight=0.50,
            update_weight=0.50,
            zipf_theta=0.99,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.75,
        ),
    }
    return profiles[profile]


def _tpcc_config(profile: str) -> TPCCConfig:
    profiles = {
        "low": TPCCConfig(
            warehouses=4,
            districts_per_warehouse=4,
            customers_per_district=40,
            items=160,
            order_lines=4,
            candidates_per_task=2,
            transaction_mix=(("new_order", 1.0),),
        ),
        "medium": TPCCConfig(
            warehouses=2,
            districts_per_warehouse=3,
            customers_per_district=40,
            items=120,
            order_lines=6,
            candidates_per_task=2,
            transaction_mix=(("new_order", 1.0),),
        ),
        "high": TPCCConfig(
            warehouses=1,
            districts_per_warehouse=2,
            customers_per_district=40,
            items=100,
            order_lines=8,
            candidates_per_task=2,
            transaction_mix=(("new_order", 1.0),),
        ),
    }
    return profiles[profile]


if __name__ == "__main__":
    raise SystemExit(main())
