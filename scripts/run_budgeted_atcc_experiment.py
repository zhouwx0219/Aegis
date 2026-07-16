#!/usr/bin/env python3
"""Run a budget-limited ATCC paper-style experiment matrix.

This runner is intentionally separate from the normal matrix CLI because the
requested five-hour budget conflicts with the paper-length warmup/measurement
settings. It records the actual budgeted settings in every output row.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

from agent.benchmarks import MixedBenchmarkConfig, run_mixed_benchmark
from agent.cli.train_atcc import train_policy_matrix


CCS = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,dynamic-atcc"
SEEDS = (920104, 920105, 920106)
CLIENT_COUNTS = (8, 16, 24, 32, 40)
AGENT_RATIOS = (1.0, 0.8)

VARIANTS = (
    {
        "workload_variant": "tpcc_low_w100",
        "workload": "tpcc",
        "level": "low",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": 100,
    },
    {
        "workload_variant": "tpcc_medium",
        "workload": "tpcc",
        "level": "medium",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": None,
    },
    {
        "workload_variant": "tpcc_high_w1",
        "workload": "tpcc",
        "level": "high",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": 1,
    },
    {
        "workload_variant": "ycsb_low",
        "workload": "ycsb",
        "level": "low",
        "ycsb_zipf_theta": 0.0,
        "tpcc_warehouses": None,
    },
    {
        "workload_variant": "ycsb_medium_z07",
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": 0.7,
        "tpcc_warehouses": None,
    },
    {
        "workload_variant": "ycsb_medium_z08",
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": 0.8,
        "tpcc_warehouses": None,
    },
    {
        "workload_variant": "ycsb_high_z099",
        "workload": "ycsb",
        "level": "high",
        "ycsb_zipf_theta": 0.99,
        "tpcc_warehouses": None,
    },
)

CSV_FIELDS = [
    "run_id",
    "source_system",
    "system",
    "cc",
    "workload",
    "workload_variant",
    "level",
    "ycsb_zipf_theta",
    "tpcc_warehouses",
    "client_mix",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "seed",
    "repeat",
    "warmup_s",
    "duration_s",
    "budget_limited",
    "policy",
    "policy_mode",
    "status",
    "agent_task_tps",
    "agent_tps",
    "total_tps",
    "bottom_txn_attempt_tps",
    "bottom_txn_commit_tps",
    "underlying_txn_attempt_tps",
    "underlying_txn_commit_tps",
    "native_throughput",
    "background_tps",
    "agent_commit_rate",
    "agent_task_completion_rate",
    "agent_abort_rate",
    "agent_attempt_abort_rate",
    "agent_avg_retry_count",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_p9999_latency_ms",
    "agent_avg_tokens",
    "agent_total_tokens",
    "wasted_reasoning_ms",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_retries",
    "read_conflicts",
    "write_conflicts",
    "version_conflict_count",
    "guarded_conflict_checks",
    "conflict_pressure_count",
    "raw_action_counts",
    "run_seconds",
    "coverage_note",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--measure-seconds", type=float, default=2.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--training-episodes", type=int, default=3)
    parser.add_argument("--training-duration", type=float, default=2.0)
    parser.add_argument("--budget-seconds", type=float, default=18000.0)
    parser.add_argument("--reserve-seconds", type=float, default=600.0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--policy", type=Path, default=None)
    args = parser.parse_args()

    started_at = time.time()
    run_id = args.run_id or time.strftime("budgeted_atcc_%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = (args.policy or (output_dir / f"{run_id}_policy.json")).resolve()
    internal_csv = output_dir / f"{run_id}_internal.csv"
    raw_jsonl = output_dir / f"{run_id}_internal_raw.jsonl"
    manifest_path = output_dir / f"{run_id}_manifest.json"

    manifest = {
        "run_id": run_id,
        "budget_limited": True,
        "budget_seconds": float(args.budget_seconds),
        "warmup_seconds": float(args.warmup_seconds),
        "measure_seconds": float(args.measure_seconds),
        "training_episodes": int(args.training_episodes),
        "training_duration": float(args.training_duration),
        "client_counts": list(CLIENT_COUNTS),
        "agent_ratios": list(AGENT_RATIOS),
        "seeds": list(SEEDS),
        "cc": CCS,
        "variants": list(VARIANTS),
        "policy": str(policy_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": [
            "Five-hour budget conflicts with paper-length warmup/measurement.",
            "Rows record actual warmup_s and duration_s used.",
            "TPC-C low uses 100 warehouses; TPC-C high uses 1 warehouse.",
        ],
    }
    write_json(manifest_path, manifest)

    if not args.skip_training or not policy_path.exists():
        train_report = train_policy_matrix(
            benchmark="mixed",
            workloads=("ycsb", "tpcc"),
            levels=("low", "medium", "high"),
            workload_profile="paper",
            episodes=int(args.training_episodes),
            tasks=100,
            workers=8,
            duration_s=float(args.training_duration),
            clients=40,
            agent_ratio=0.8,
            agents=32,
            background=8,
            background_mode="procedure",
            retries=0,
            retry_until_commit=True,
            max_attempts_per_task=5,
            agent_retry_backoff_min_ms=1,
            agent_retry_backoff_max_ms=5,
            background_retry_backoff_min_ms=1,
            background_retry_backoff_max_ms=3,
            tokens_per_operation=2703,
            seed=920104,
            reasoning_profile="agentic",
            reasoning_scale=2.0,
            actions="auto",
        )
        write_json(output_dir / f"{run_id}_train_report.json", train_report)
        write_json(policy_path, train_report["policy"])

    rows_written = 0
    with internal_csv.open("w", newline="", encoding="utf-8") as csv_handle, raw_jsonl.open(
        "w", encoding="utf-8"
    ) as raw_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for config in iter_matrix_configs():
            elapsed = time.time() - started_at
            if elapsed + float(args.reserve_seconds) >= float(args.budget_seconds):
                manifest["stopped_reason"] = "budget_reserve_reached"
                break
            try:
                if float(args.warmup_seconds) > 0:
                    run_one(
                        config,
                        duration_s=float(args.warmup_seconds),
                        policy_path=policy_path,
                    )
                measured_started = time.time()
                report = run_one(
                    config,
                    duration_s=float(args.measure_seconds),
                    policy_path=policy_path,
                )
                run_seconds = time.time() - measured_started
                raw_handle.write(json.dumps({"config": config, "report": report}, sort_keys=True) + "\n")
                for result in report["cc_results"]:
                    writer.writerow(
                        result_row(
                            run_id=run_id,
                            config=config,
                            result=result,
                            policy_path=policy_path,
                            duration_s=float(args.measure_seconds),
                            warmup_s=float(args.warmup_seconds),
                            run_seconds=run_seconds,
                            status="ok",
                            error="",
                        )
                    )
                    rows_written += 1
                csv_handle.flush()
                raw_handle.flush()
            except Exception as exc:  # Keep partial matrix recoverable.
                writer.writerow(
                    error_row(
                        run_id=run_id,
                        config=config,
                        policy_path=policy_path,
                        duration_s=float(args.measure_seconds),
                        warmup_s=float(args.warmup_seconds),
                        error=str(exc),
                    )
                )
                rows_written += 1
                csv_handle.flush()
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["elapsed_seconds"] = time.time() - started_at
    manifest["internal_rows"] = rows_written
    write_json(manifest_path, manifest)
    print(internal_csv)
    print(f"rows={rows_written}")
    print(f"manifest={manifest_path}")
    return 0


def iter_matrix_configs() -> Iterable[dict[str, Any]]:
    repeat = 0
    for variant in VARIANTS:
        for clients in CLIENT_COUNTS:
            for agent_ratio in AGENT_RATIOS:
                for seed in SEEDS:
                    yield {
                        **variant,
                        "clients": int(clients),
                        "agent_ratio": float(agent_ratio),
                        "client_mix": "all_agent" if float(agent_ratio) == 1.0 else "agent80_backend20",
                        "seed": int(seed),
                        "repeat": repeat,
                    }
                    repeat += 1


def run_one(config: dict[str, Any], *, duration_s: float, policy_path: Path) -> dict[str, Any]:
    return run_mixed_benchmark(
        MixedBenchmarkConfig(
            workload=config["workload"],
            level=config["level"],
            workload_profile="paper",
            ycsb_zipf_theta=config["ycsb_zipf_theta"],
            tpcc_warehouses=config["tpcc_warehouses"],
            cc=CCS,
            duration_s=float(duration_s),
            clients=int(config["clients"]),
            agent_ratio=float(config["agent_ratio"]),
            reasoning_profile="agentic",
            reasoning_scale=2.0,
            seed=int(config["seed"]),
            retries=0,
            retry_until_commit=True,
            max_attempts_per_task=5,
            agent_retry_backoff_min_ms=1,
            agent_retry_backoff_max_ms=5,
            background_retry_backoff_min_ms=1,
            background_retry_backoff_max_ms=3,
            tokens_per_operation=2703,
            background_mode="procedure",
            policy=policy_path,
            policy_mode="eval",
        )
    )


def result_row(
    *,
    run_id: str,
    config: dict[str, Any],
    result: dict[str, Any],
    policy_path: Path,
    duration_s: float,
    warmup_s: float,
    run_seconds: float,
    status: str,
    error: str,
) -> dict[str, Any]:
    row = common_row(run_id, config, policy_path, duration_s, warmup_s, status, error)
    row.update(
        {
            "cc": result.get("cc", ""),
            "agent_task_tps": result.get("agent_task_tps", ""),
            "agent_tps": result.get("agent_tps", ""),
            "total_tps": result.get("total_tps", ""),
            "bottom_txn_attempt_tps": result.get("bottom_txn_attempt_tps", ""),
            "bottom_txn_commit_tps": result.get("bottom_txn_commit_tps", ""),
            "underlying_txn_attempt_tps": result.get("underlying_txn_attempt_tps", ""),
            "underlying_txn_commit_tps": result.get("underlying_txn_commit_tps", ""),
            "native_throughput": result.get("native_throughput", ""),
            "background_tps": result.get("background_tps", ""),
            "agent_commit_rate": result.get("agent_commit_rate", ""),
            "agent_task_completion_rate": result.get("agent_task_completion_rate", ""),
            "agent_abort_rate": result.get("agent_abort_rate", ""),
            "agent_attempt_abort_rate": result.get("agent_attempt_abort_rate", ""),
            "agent_avg_retry_count": result.get("agent_avg_retry_count", ""),
            "agent_p50_latency_ms": result.get("agent_p50_latency_ms", ""),
            "agent_p95_latency_ms": result.get("agent_p95_latency_ms", ""),
            "agent_p99_latency_ms": result.get("agent_p99_latency_ms", ""),
            "agent_p9999_latency_ms": result.get("agent_p9999_latency_ms", ""),
            "agent_avg_tokens": result.get("agent_avg_tokens", ""),
            "agent_total_tokens": result.get("agent_total_tokens", ""),
            "wasted_reasoning_ms": result.get("wasted_reasoning_ms", ""),
            "agent_attempts": result.get("agent_attempts", ""),
            "agent_commits": result.get("agent_commits", ""),
            "agent_aborts": result.get("agent_aborts", ""),
            "agent_completed_tasks": result.get("agent_completed_tasks", ""),
            "agent_failed_tasks": result.get("agent_failed_tasks", ""),
            "background_attempts": result.get("background_attempts", ""),
            "background_commits": result.get("background_commits", ""),
            "background_aborts": result.get("background_aborts", ""),
            "background_retries": result.get("background_retries", ""),
            "read_conflicts": result.get("read_conflicts", ""),
            "write_conflicts": result.get("write_conflicts", ""),
            "version_conflict_count": result.get("version_conflict_count", ""),
            "guarded_conflict_checks": result.get("guarded_conflict_checks", ""),
            "conflict_pressure_count": result.get("conflict_pressure_count", ""),
            "raw_action_counts": json.dumps(result.get("action_counts", {}), sort_keys=True),
            "run_seconds": f"{run_seconds:.3f}",
        }
    )
    return row


def error_row(
    *,
    run_id: str,
    config: dict[str, Any],
    policy_path: Path,
    duration_s: float,
    warmup_s: float,
    error: str,
) -> dict[str, Any]:
    row = common_row(run_id, config, policy_path, duration_s, warmup_s, "error", error)
    row["cc"] = ""
    return row


def common_row(
    run_id: str,
    config: dict[str, Any],
    policy_path: Path,
    duration_s: float,
    warmup_s: float,
    status: str,
    error: str,
) -> dict[str, Any]:
    clients = int(config["clients"])
    agent_ratio = float(config["agent_ratio"])
    agent_workers = max(1, int(round(clients * agent_ratio)))
    background_workers = max(0, clients - agent_workers)
    return {
        "run_id": run_id,
        "source_system": "cast-das",
        "system": "cast-das",
        "cc": "",
        "workload": config["workload"],
        "workload_variant": config["workload_variant"],
        "level": config["level"],
        "ycsb_zipf_theta": "" if config["ycsb_zipf_theta"] is None else config["ycsb_zipf_theta"],
        "tpcc_warehouses": "" if config["tpcc_warehouses"] is None else config["tpcc_warehouses"],
        "client_mix": config["client_mix"],
        "clients": clients,
        "agent_ratio": agent_ratio,
        "agent_workers": agent_workers,
        "background_workers": background_workers,
        "seed": config["seed"],
        "repeat": config["repeat"],
        "warmup_s": warmup_s,
        "duration_s": duration_s,
        "budget_limited": True,
        "policy": str(policy_path),
        "policy_mode": "eval",
        "status": status,
        "coverage_note": "budgeted-paper-5h; requested paper warmup/measurement shortened to fit total budget",
        "error": error,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
