#!/usr/bin/env python3
"""Run a fixed-trace CAST-DAS/DBx1000 unified CC comparison matrix.

This runner exists for the Polaris/Bamboo comparison path: CAST-DAS generates a
single concrete transaction trace, and both the Python runtime and patched
DBx1000-family systems replay that exact trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmarks import MixedBenchmarkConfig, run_mixed_benchmark  # noqa: E402
from agent.cli.train_atcc import make_policy, resolve_trainable_actions, training_episode_row  # noqa: E402

CLIENT_COUNTS = (8, 16, 24, 32, 40)
AGENT_RATIOS = (1.0, 0.8)
SEEDS = (920104, 920105, 920106)
TRAINING_CLIENTS = (24, 40)
TRAINING_SEEDS = (810104, 810105, 810106)
INTERNAL_CCS = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,dynamic-atcc"
EXTERNAL_SYSTEMS = "bamboo,polaris"

VARIANTS: dict[str, dict[str, str]] = {
    "tpcc_low_w100": {
        "workload": "tpcc",
        "level": "low",
        "ycsb_zipf_theta": "",
        "tpcc_warehouses": "100",
    },
    "tpcc_medium": {
        "workload": "tpcc",
        "level": "medium",
        "ycsb_zipf_theta": "",
        "tpcc_warehouses": "",
    },
    "tpcc_high_w1": {
        "workload": "tpcc",
        "level": "high",
        "ycsb_zipf_theta": "",
        "tpcc_warehouses": "1",
    },
    "ycsb_low": {
        "workload": "ycsb",
        "level": "low",
        "ycsb_zipf_theta": "0.0",
        "tpcc_warehouses": "",
    },
    "ycsb_medium_z07": {
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": "0.7",
        "tpcc_warehouses": "",
    },
    "ycsb_medium_z08": {
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": "0.8",
        "tpcc_warehouses": "",
    },
    "ycsb_high_z099": {
        "workload": "ycsb",
        "level": "high",
        "ycsb_zipf_theta": "0.99",
        "tpcc_warehouses": "",
    },
}

RAW_PREFIX_FIELDS = [
    "run_id",
    "experiment_mode",
    "measurement_note",
    "transactions_per_worker",
    "warmup_transactions_per_worker",
    "measure_transactions_per_worker",
    "warmup_seconds",
    "measure_seconds",
    "client_mix",
    "cc_label",
    "cc_family",
    "policy",
    "policy_mode",
    "ycsb_zipf_theta",
    "tpcc_warehouses",
    "trace_csv",
]

SUMMARY_FIELDS = [
    "run_id",
    "experiment_mode",
    "measurement_note",
    "workload",
    "workload_variant",
    "level",
    "ycsb_zipf_theta",
    "tpcc_warehouses",
    "client_mix",
    "clients",
    "agent_ratio",
    "cc_label",
    "cc_family",
    "source_system",
    "system",
    "cc",
    "n_repeats",
    "n_ok",
    "n_error",
    "total_tps_mean",
    "total_tps_std",
    "bottom_txn_attempt_tps_mean",
    "bottom_txn_commit_tps_mean",
    "underlying_txn_attempt_tps_mean",
    "underlying_txn_commit_tps_mean",
    "native_throughput_mean",
    "agent_task_tps_mean",
    "agent_tps_mean",
    "background_tps_mean",
    "agent_task_completion_rate_mean",
    "agent_commit_rate_mean",
    "agent_attempt_abort_rate_mean",
    "agent_logical_attempts_mean",
    "agent_admission_deferrals_mean",
    "agent_admission_deferral_rate_mean",
    "agent_avg_retry_count_mean",
    "agent_p50_latency_ms_mean",
    "agent_p95_latency_ms_mean",
    "agent_p99_latency_ms_mean",
    "agent_p999_latency_ms_mean",
    "agent_p9999_latency_ms_mean",
    "agent_time_to_success_p50_ms_mean",
    "agent_time_to_success_p95_ms_mean",
    "agent_time_to_success_p99_ms_mean",
    "agent_time_to_success_p999_ms_mean",
    "agent_time_to_success_p9999_ms_mean",
    "background_commit_rate_mean",
    "wasted_reasoning_ms_mean",
    "read_conflicts_mean",
    "write_conflicts_mean",
    "conflict_abort_count_mean",
    "reservation_admission_abort_count_mean",
    "lock_timeout_abort_count_mean",
    "full_commit_lock_timeout_abort_count_mean",
    "hot_commit_lock_timeout_abort_count_mean",
    "begin_lock_timeout_abort_count_mean",
    "version_validation_abort_count_mean",
    "agent_avg_tokens_mean",
    "agent_total_tokens_mean",
    "version_conflict_count_mean",
    "guarded_conflict_checks_mean",
    "conflict_pressure_count_mean",
    "agent_reservation_wait_ms_total_mean",
    "agent_reservation_wait_ms_mean_mean",
    "background_reservation_wait_ms_total_mean",
    "background_reservation_wait_ms_mean_mean",
    "reservation_guard_wait_ms_total_mean",
    "admission_yield_ms_total_mean",
    "elapsed_s_mean",
    "atcc_total_tps_speedup",
    "atcc_agent_tps_speedup",
    "atcc_abort_rate_delta",
    "status",
    "errors",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--external-root", type=Path, default=Path("/home/chenht/castdas_external_cc"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--clients", default=",".join(str(value) for value in CLIENT_COUNTS))
    parser.add_argument("--agent-ratios", default=",".join(str(value) for value in AGENT_RATIOS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument("--transactions-per-worker", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--measure-seconds", type=float, default=0.0)
    parser.add_argument("--no-cycle-trace", action="store_false", dest="cycle_trace")
    parser.add_argument("--internal-cc", default=INTERNAL_CCS)
    parser.add_argument("--external-systems", default=EXTERNAL_SYSTEMS)
    parser.add_argument("--internal-runner", choices=("fair", "simple"), default="fair")
    parser.add_argument("--reasoning-profile", default="agentic")
    parser.add_argument("--reasoning-scale", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--external-timeout", type=float, default=120.0)
    parser.add_argument("--budget-seconds", type=float, default=18000.0)
    parser.add_argument("--reserve-seconds", type=float, default=300.0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--training-episodes", type=int, default=3)
    parser.add_argument("--training-duration", type=float, default=2.0)
    parser.add_argument("--training-clients", default=",".join(str(value) for value in TRAINING_CLIENTS))
    parser.add_argument("--training-seeds", default=",".join(str(value) for value in TRAINING_SEEDS))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.set_defaults(cycle_trace=True)
    args = parser.parse_args()

    if args.transactions_per_worker <= 0:
        raise SystemExit("--transactions-per-worker must be positive")
    if args.warmup_seconds < 0 or args.measure_seconds < 0:
        raise SystemExit("--warmup-seconds/--measure-seconds must be non-negative")

    started_at = time.time()
    run_id = args.run_id or time.strftime("unified_trace_%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    traces_dir = output_dir / "traces"
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, traces_dir, runs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    policy_path = (args.policy or output_dir / f"{run_id}.policy.json").resolve()
    train_report_path = output_dir / f"{run_id}.train_report.json"
    raw_csv = output_dir / f"{run_id}.unified_raw.csv"
    summary_csv = output_dir / f"{run_id}.unified_summary.csv"
    manifest_path = output_dir / f"{run_id}.manifest.json"
    progress_csv = output_dir / f"{run_id}.progress.csv"

    variants = checked_variants(split_csv(args.variants))
    clients = [int(value) for value in split_csv(args.clients)]
    agent_ratios = [float(value) for value in split_csv(args.agent_ratios)]
    seeds = [int(value) for value in split_csv(args.seeds)]
    training_clients = [int(value) for value in split_csv(args.training_clients)]
    training_seeds = [int(value) for value in split_csv(args.training_seeds)]

    manifest = {
        "run_id": run_id,
        "experiment_mode": experiment_mode(args.internal_runner),
        "measurement_note": measurement_note(
            0 if float(args.measure_seconds) > 0 else args.transactions_per_worker,
            args.internal_runner,
            warmup_seconds=float(args.warmup_seconds),
            measure_seconds=float(args.measure_seconds),
        ),
        "output_dir": str(output_dir),
        "policy": str(policy_path),
        "variants": variants,
        "clients": clients,
        "agent_ratios": agent_ratios,
        "seeds": seeds,
        "transactions_per_worker": int(args.transactions_per_worker),
        "warmup_seconds": float(args.warmup_seconds),
        "measure_seconds": float(args.measure_seconds),
        "cycle_trace": bool(args.cycle_trace),
        "internal_cc": split_csv(args.internal_cc),
        "atcc_policy_control": "policy_action_and_priority",
        "performance_guards_enabled": False,
        "atcc_runtime_fast_paths_enabled": False,
        "sparse_state_risk_prior": False,
        "safety_guards_enabled": True,
        "external_systems": split_csv(args.external_systems),
        "training_clients": training_clients,
        "training_seeds": training_seeds,
        "budget_seconds": float(args.budget_seconds),
        "reserve_seconds": float(args.reserve_seconds),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
    }
    write_json(manifest_path, manifest)

    if not args.skip_training and not policy_path.exists():
        run_training(
            policy_path=policy_path,
            train_report_path=train_report_path,
            variants=variants,
            agent_ratios=agent_ratios,
            clients=training_clients,
            seeds=training_seeds,
            episodes=args.training_episodes,
            duration_s=args.training_duration,
            reasoning_scale=args.reasoning_scale,
            logs_dir=logs_dir,
        )
    elif not policy_path.exists():
        raise SystemExit(f"missing policy with --skip-training: {policy_path}")

    matrix = list(iter_matrix(variants, clients, agent_ratios, seeds))
    write_progress_header(progress_csv)
    stopped_reason = ""
    for index, config in enumerate(matrix, start=1):
        elapsed = time.time() - started_at
        if elapsed + float(args.reserve_seconds) >= float(args.budget_seconds):
            stopped_reason = "budget_reserve_reached"
            break
        trace_id = trace_id_for(config)
        trace_csv = traces_dir / f"{trace_id}.csv"
        warmup_trace_csv = traces_dir / f"{trace_id}.warmup.csv"
        internal_csv = runs_dir / f"{trace_id}.internal.csv"
        external_csv = runs_dir / f"{trace_id}.external.csv"
        measure_tpw = trace_transactions_per_worker(
            config["variant"],
            seconds=float(args.measure_seconds),
            fallback=int(args.transactions_per_worker),
        )
        warmup_tpw = (
            trace_transactions_per_worker(
                config["variant"],
                seconds=float(args.warmup_seconds),
                fallback=int(args.transactions_per_worker),
            )
            if float(args.warmup_seconds) > 0
            else 0
        )
        append_progress(progress_csv, run_id, index, len(matrix), trace_id, "start", elapsed, "")
        try:
            if warmup_tpw > 0 and not (args.resume and warmup_trace_csv.exists()):
                run_checked(
                    [
                        sys.executable,
                        str(THIS_DIR / "generate_castdas_trace.py"),
                        "--output",
                        str(warmup_trace_csv),
                        "--trace-id",
                        f"{trace_id}.warmup",
                        "--variant",
                        config["variant"],
                        "--clients",
                        str(config["clients"]),
                        "--agent-ratio",
                        str(config["agent_ratio"]),
                        "--seed",
                        str(int(config["seed"]) + 1_000_000),
                        "--repeat",
                        str(config["repeat"]),
                        "--transactions-per-worker",
                        str(warmup_tpw),
                        "--reasoning-profile",
                        args.reasoning_profile,
                        "--reasoning-scale",
                        str(args.reasoning_scale),
                    ],
                    logs_dir / f"{trace_id}.generate_warmup.log",
                )
            if not (args.resume and trace_csv.exists()):
                run_checked(
                    [
                        sys.executable,
                        str(THIS_DIR / "generate_castdas_trace.py"),
                        "--output",
                        str(trace_csv),
                        "--trace-id",
                        trace_id,
                        "--variant",
                        config["variant"],
                        "--clients",
                        str(config["clients"]),
                        "--agent-ratio",
                        str(config["agent_ratio"]),
                        "--seed",
                        str(config["seed"]),
                        "--repeat",
                        str(config["repeat"]),
                        "--transactions-per-worker",
                        str(measure_tpw),
                        "--reasoning-profile",
                        args.reasoning_profile,
                        "--reasoning-scale",
                        str(args.reasoning_scale),
                    ],
                    logs_dir / f"{trace_id}.generate.log",
                )
            if not (args.resume and internal_csv.exists()):
                internal_cmd = [
                    sys.executable,
                    str(internal_runner_path(args.internal_runner)),
                    "--trace",
                    str(trace_csv),
                    "--output",
                    str(internal_csv),
                    "--cc",
                    args.internal_cc,
                    "--policy",
                    str(policy_path),
                    "--max-attempts",
                    str(args.max_attempts),
                ]
                if str(args.internal_runner).strip().lower() == "fair":
                    if warmup_tpw > 0:
                        internal_cmd.extend(
                            [
                                "--warmup-trace",
                                str(warmup_trace_csv),
                                "--warmup-seconds",
                                str(float(args.warmup_seconds)),
                            ]
                        )
                    if float(args.measure_seconds) > 0:
                        internal_cmd.extend(["--measure-seconds", str(float(args.measure_seconds))])
                    if not bool(args.cycle_trace):
                        internal_cmd.append("--no-cycle-trace")
                run_checked(
                    internal_cmd,
                    logs_dir / f"{trace_id}.internal.log",
                )
            if split_csv(args.external_systems) and not (args.resume and external_csv.exists()):
                run_checked(
                    [
                        sys.executable,
                        str(THIS_DIR / "run_dbx1000_trace.py"),
                        "--root",
                        str(args.external_root),
                        "--trace",
                        str(trace_csv),
                        "--systems",
                        args.external_systems,
                        "--output",
                        str(external_csv),
                        "--run-timeout",
                        str(args.external_timeout),
                    ],
                    logs_dir / f"{trace_id}.external.log",
                )
            append_progress(progress_csv, run_id, index, len(matrix), trace_id, "ok", time.time() - started_at, "")
        except subprocess.CalledProcessError as exc:
            append_progress(
                progress_csv,
                run_id,
                index,
                len(matrix),
                trace_id,
                "error",
                time.time() - started_at,
                f"exit={exc.returncode}",
            )

    raw_rows = build_raw_rows(
        run_id=run_id,
        output_dir=output_dir,
        policy_path=policy_path,
        transactions_per_worker=args.transactions_per_worker,
        warmup_seconds=float(args.warmup_seconds),
        measure_seconds=float(args.measure_seconds),
        internal_runner=args.internal_runner,
    )
    write_csv(raw_csv, raw_rows)
    summary_rows = summarize(raw_rows, run_id=run_id)
    write_csv(summary_csv, summary_rows, fieldnames=SUMMARY_FIELDS)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["elapsed_s"] = time.time() - started_at
    manifest["stopped_reason"] = stopped_reason or "completed"
    manifest["raw_rows"] = len(raw_rows)
    manifest["summary_rows"] = len(summary_rows)
    write_json(manifest_path, manifest)

    print(summary_csv)
    print(f"raw={raw_csv}")
    print(f"rows={len(raw_rows)} summary_rows={len(summary_rows)} stopped_reason={manifest['stopped_reason']}")
    return 0


def run_training(
    *,
    policy_path: Path,
    train_report_path: Path,
    variants: list[str],
    agent_ratios: list[float],
    clients: list[int],
    seeds: list[int],
    episodes: int,
    duration_s: float,
    reasoning_scale: float,
    logs_dir: Path,
) -> None:
    log_path = logs_dir / "train_atcc.log"
    trainable_actions = resolve_trainable_actions("mixed", "auto")
    policy = make_policy(
        abort_threshold=0.20,
        min_visits=3,
        protect_cost_threshold_ms=10.0,
        low_conflict_occ_guard=False,
        low_conflict_safe_abort_rate=0.50,
        sparse_state_risk_prior=False,
        commit_value=100.0,
        abort_penalty=80.0,
        reasoning_weight=1.0,
        lock_wait_weight=0.5,
        latency_weight=0.1,
        lock_hold_weight=0.20,
        background_abort_weight=30.0,
        background_tps_loss_weight=25.0,
        trainable_actions=trainable_actions,
        exploration_coefficient=1.5,
    )
    rows: list[dict[str, Any]] = []
    log_lines = [
        "training_scope=paper_variant_matrix",
        f"variants={','.join(variants)}",
        f"clients={','.join(str(value) for value in clients)}",
        f"agent_ratios={','.join(str(value) for value in agent_ratios)}",
        f"seeds={','.join(str(value) for value in seeds)}",
        f"episodes={int(episodes)} duration_s={float(duration_s)}",
    ]
    started = time.perf_counter()
    run_index = 0
    for variant_name in variants:
        variant = VARIANTS[variant_name]
        for agent_ratio in agent_ratios:
            for client_count in clients:
                for episode in range(int(episodes)):
                    seed = int(seeds[episode % len(seeds)])
                    report = run_mixed_benchmark(
                        MixedBenchmarkConfig(
                            workload=variant["workload"],
                            level=variant["level"],
                            workload_profile="paper",
                            ycsb_zipf_theta=optional_float(variant.get("ycsb_zipf_theta", "")),
                            tpcc_warehouses=optional_int(variant.get("tpcc_warehouses", "")),
                            cc="dynamic-atcc",
                            duration_s=float(duration_s),
                            clients=int(client_count),
                            agent_ratio=float(agent_ratio),
                            reasoning_profile="agentic",
                            reasoning_scale=float(reasoning_scale),
                            seed=seed,
                            retries=0,
                            retry_until_commit=True,
                            max_attempts_per_task=5,
                            agent_retry_backoff_min_ms=1,
                            agent_retry_backoff_max_ms=5,
                            background_retry_backoff_min_ms=1,
                            background_retry_backoff_max_ms=3,
                            tokens_per_operation=2703,
                            background_mode="procedure",
                            policy=policy,
                            policy_mode="train",
                            atcc_pure_policy=True,
                        )
                    )
                    row = training_episode_row(run_index, report["cc_results"][0], policy_states=len(policy.rows))
                    row.update(
                        {
                            "run_index": run_index,
                            "workload_variant": variant_name,
                            "workload": variant["workload"],
                            "level": variant["level"],
                            "ycsb_zipf_theta": variant.get("ycsb_zipf_theta", ""),
                            "tpcc_warehouses": variant.get("tpcc_warehouses", ""),
                            "client_mix": client_mix(agent_ratio),
                            "clients": int(client_count),
                            "agent_ratio": float(agent_ratio),
                            "seed": seed,
                            "matrix_round": int(episode),
                        }
                    )
                    rows.append(row)
                    log_lines.append(
                        " ".join(
                            (
                                f"run={run_index}",
                                f"variant={variant_name}",
                                f"clients={client_count}",
                                f"agent_ratio={agent_ratio:g}",
                                f"seed={seed}",
                                f"policy_states={len(policy.rows)}",
                            )
                        )
                    )
                    run_index += 1
    elapsed_s = time.perf_counter() - started
    policy.set_mode("eval")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "mode": "train-atcc",
        "training_scope": "paper_variant_matrix",
        "variants": variants,
        "agent_ratios": agent_ratios,
        "clients": clients,
        "seeds": seeds,
        "episodes": int(episodes),
        "runs": int(run_index),
        "duration_s": float(duration_s),
        "elapsed_s": elapsed_s,
        "reasoning_profile": "agentic",
        "reasoning_scale": float(reasoning_scale),
        "tokens_per_operation": 2703,
        "max_attempts_per_task": 5,
        "background_mode": "procedure",
        "atcc_policy_control": "policy_action_and_priority",
        "performance_guards_enabled": False,
        "atcc_runtime_fast_paths_enabled": False,
        "sparse_state_risk_prior": False,
        "safety_guards_enabled": True,
        "policy_states": len(policy.rows),
        "actions": list(trainable_actions),
        "policy": policy.to_dict(),
        "episodes_detail": rows,
    }
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    train_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checked(cmd: list[str], log_path: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def build_raw_rows(
    *,
    run_id: str,
    output_dir: Path,
    policy_path: Path,
    transactions_per_worker: int,
    warmup_seconds: float,
    measure_seconds: float,
    internal_runner: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted((output_dir / "runs").glob("*.csv")):
        for row in read_csv(csv_path):
            variant_name = str(row.get("workload_variant", ""))
            variant = VARIANTS.get(variant_name, {})
            trace_path = (output_dir / "traces" / f"{row.get('trace_id', '')}.csv").resolve()
            trace_meta = read_json_if_exists(trace_path.with_suffix(".meta.json"))
            warmup_meta = read_json_if_exists(
                (output_dir / "traces" / f"{row.get('trace_id', '')}.warmup.meta.json").resolve()
            )
            row = dict(row)
            cc_label = label_for(row)
            row.update(
                {
                    "run_id": run_id,
                    "experiment_mode": experiment_mode(internal_runner),
                    "measurement_note": measurement_note(
                        int(trace_meta.get("transactions_per_worker", transactions_per_worker) or transactions_per_worker),
                        internal_runner,
                        warmup_seconds=warmup_seconds,
                        measure_seconds=measure_seconds,
                    ),
                    "transactions_per_worker": int(
                        trace_meta.get("transactions_per_worker", transactions_per_worker) or transactions_per_worker
                    ),
                    "warmup_transactions_per_worker": warmup_meta.get("transactions_per_worker", ""),
                    "measure_transactions_per_worker": trace_meta.get("transactions_per_worker", ""),
                    "warmup_seconds": float(warmup_seconds),
                    "measure_seconds": float(measure_seconds),
                    "client_mix": client_mix(row.get("agent_ratio", "")),
                    "cc_label": cc_label,
                    "cc_family": "atcc" if cc_label == "ATCC" else "traditional",
                    "policy": str(policy_path) if cc_label == "ATCC" else "",
                    "policy_mode": "eval" if cc_label == "ATCC" else "",
                    "ycsb_zipf_theta": variant.get("ycsb_zipf_theta", ""),
                    "tpcc_warehouses": variant.get("tpcc_warehouses", ""),
                    "trace_csv": str(trace_path),
                }
            )
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    group_fields = [
        "workload",
        "workload_variant",
        "level",
        "ycsb_zipf_theta",
        "tpcc_warehouses",
        "client_mix",
        "clients",
        "agent_ratio",
        "cc_label",
        "cc_family",
        "source_system",
        "system",
        "cc",
    ]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        first = group[0]
        ok_rows = [row for row in group if str(row.get("status", "")).lower() == "ok"]
        summary = {
            "run_id": run_id,
            "experiment_mode": first.get("experiment_mode", experiment_mode("fair")),
            "measurement_note": first.get("measurement_note", ""),
            "n_repeats": len(group),
            "n_ok": len(ok_rows),
            "n_error": len(group) - len(ok_rows),
            "status": "ok" if ok_rows and len(ok_rows) == len(group) else ("partial" if ok_rows else "error"),
            "errors": " | ".join(short_error(row.get("error", "")) for row in group if row.get("error")),
        }
        summary.update({field: first.get(field, "") for field in group_fields})
        for metric in (
            "total_tps",
            "bottom_txn_attempt_tps",
            "bottom_txn_commit_tps",
            "underlying_txn_attempt_tps",
            "underlying_txn_commit_tps",
            "agent_task_tps",
            "agent_tps",
            "background_tps",
            "agent_task_completion_rate",
            "agent_commit_rate",
            "agent_attempt_abort_rate",
            "agent_logical_attempts",
            "agent_admission_deferrals",
            "agent_admission_deferral_rate",
            "agent_avg_retry_count",
            "agent_p50_latency_ms",
            "agent_p95_latency_ms",
            "agent_p99_latency_ms",
            "agent_p999_latency_ms",
            "agent_p9999_latency_ms",
            "agent_time_to_success_p50_ms",
            "agent_time_to_success_p95_ms",
            "agent_time_to_success_p99_ms",
            "agent_time_to_success_p999_ms",
            "agent_time_to_success_p9999_ms",
            "background_commit_rate",
            "wasted_reasoning_ms",
            "read_conflicts",
            "write_conflicts",
            "version_conflict_count",
            "guarded_conflict_checks",
            "conflict_pressure_count",
            "conflict_abort_count",
            "reservation_admission_abort_count",
            "lock_timeout_abort_count",
            "full_commit_lock_timeout_abort_count",
            "hot_commit_lock_timeout_abort_count",
            "begin_lock_timeout_abort_count",
            "version_validation_abort_count",
            "agent_reservation_wait_ms_total",
            "agent_reservation_wait_ms_mean",
            "background_reservation_wait_ms_total",
            "background_reservation_wait_ms_mean",
            "reservation_guard_wait_ms_total",
            "admission_yield_ms_total",
            "agent_avg_tokens",
            "agent_total_tokens",
            "elapsed_s",
            "native_throughput",
        ):
            values = [parse_float(row.get(metric)) for row in ok_rows]
            values = [value for value in values if value is not None]
            mean_key = f"{metric}_mean" if metric != "elapsed_s" else "elapsed_s_mean"
            if metric == "total_tps":
                summary["total_tps_mean"] = mean(values)
                summary["total_tps_std"] = stdev(values)
            elif metric == "elapsed_s":
                summary["elapsed_s_mean"] = mean(values)
            elif metric in {
                "wasted_reasoning_ms",
                "read_conflicts",
                "write_conflicts",
                "version_conflict_count",
                "guarded_conflict_checks",
                "conflict_pressure_count",
                "conflict_abort_count",
                "reservation_admission_abort_count",
                "lock_timeout_abort_count",
                "full_commit_lock_timeout_abort_count",
                "hot_commit_lock_timeout_abort_count",
                "begin_lock_timeout_abort_count",
                "version_validation_abort_count",
                "agent_avg_tokens",
                "agent_total_tokens",
            }:
                summary[f"{metric}_mean"] = mean(values)
            else:
                summary[mean_key] = mean(values)
        summaries.append(summary)

    atcc_by_config = {
        (
            row.get("workload_variant", ""),
            row.get("client_mix", ""),
            str(row.get("clients", "")),
            str(row.get("agent_ratio", "")),
        ): row
        for row in summaries
        if row.get("cc_label") == "ATCC"
    }
    for row in summaries:
        key = (
            row.get("workload_variant", ""),
            row.get("client_mix", ""),
            str(row.get("clients", "")),
            str(row.get("agent_ratio", "")),
        )
        atcc = atcc_by_config.get(key)
        if not atcc:
            row["atcc_total_tps_speedup"] = ""
            row["atcc_agent_tps_speedup"] = ""
            row["atcc_abort_rate_delta"] = ""
            continue
        row["atcc_total_tps_speedup"] = ratio(atcc.get("total_tps_mean"), row.get("total_tps_mean"))
        row["atcc_agent_tps_speedup"] = ratio(atcc.get("agent_tps_mean"), row.get("agent_tps_mean"))
        atcc_abort = parse_float(atcc.get("agent_attempt_abort_rate_mean"))
        this_abort = parse_float(row.get("agent_attempt_abort_rate_mean"))
        row["atcc_abort_rate_delta"] = (
            fmt(atcc_abort - this_abort) if atcc_abort is not None and this_abort is not None else ""
        )
    return [{field: row.get(field, "") for field in SUMMARY_FIELDS} for row in summaries]


def iter_matrix(
    variants: list[str],
    clients: list[int],
    agent_ratios: list[float],
    seeds: list[int],
) -> Iterable[dict[str, Any]]:
    for variant in variants:
        for client_count in clients:
            for agent_ratio in agent_ratios:
                for repeat, seed in enumerate(seeds):
                    yield {
                        "variant": variant,
                        "clients": int(client_count),
                        "agent_ratio": float(agent_ratio),
                        "repeat": int(repeat),
                        "seed": int(seed),
                    }


def trace_id_for(config: dict[str, Any]) -> str:
    ratio = f"{float(config['agent_ratio']):g}".replace(".", "p")
    return f"{config['variant']}_c{config['clients']}_a{ratio}_r{config['repeat']}_s{config['seed']}"


def checked_variants(values: list[str]) -> list[str]:
    unknown = [value for value in values if value not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {','.join(unknown)}")
    return values


def trace_transactions_per_worker(variant_name: str, *, seconds: float, fallback: int) -> int:
    if float(seconds) <= 0:
        return int(fallback)
    level = str(VARIANTS.get(variant_name, {}).get("level", "low")).strip().lower()
    per_second = {
        "low": 512,
        "medium": 128,
        "high": 48,
    }.get(level, 128)
    return max(int(fallback), int(math.ceil(float(seconds) * per_second)))


def optional_float(value: object) -> float | None:
    parsed = parse_float(value)
    return parsed


def optional_int(value: object) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def label_for(row: dict[str, Any]) -> str:
    system = str(row.get("system", "")).lower()
    cc = str(row.get("cc", "")).lower()
    if cc == "dynamic-atcc":
        return "ATCC"
    if cc == "occ":
        return "OCC"
    if cc == "2pl-nowait":
        return "2PL-nowait"
    if cc == "2pl-wait-die":
        return "2PL-wait-die"
    if cc == "mvcc":
        return "MVCC"
    if cc == "silo":
        return "Silo"
    if cc == "tictoc":
        return "TicToc"
    if system == "bamboo" or cc == "bamboo":
        return "Bamboo"
    if system == "polaris" or cc in {"silo_prio", "polaris"}:
        return "Polaris"
    return str(row.get("cc", ""))


def client_mix(agent_ratio: object) -> str:
    value = parse_float(agent_ratio)
    if value is None:
        return ""
    if abs(value - 1.0) < 1e-9:
        return "all_agent"
    if abs(value - 0.8) < 1e-9:
        return "agent80_backend20"
    return f"agent_ratio_{value:g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = list(RAW_PREFIX_FIELDS)
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_progress_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "index", "total", "trace_id", "stage", "elapsed_s", "message"],
        )
        writer.writeheader()


def append_progress(
    path: Path,
    run_id: str,
    index: int,
    total: int,
    trace_id: str,
    stage: str,
    elapsed_s: float,
    message: str,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "index", "total", "trace_id", "stage", "elapsed_s", "message"],
        )
        writer.writerow(
            {
                "run_id": run_id,
                "index": index,
                "total": total,
                "trace_id": trace_id,
                "stage": stage,
                "elapsed_s": f"{elapsed_s:.3f}",
                "message": message,
            }
        )


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"raw_stdout": text}
    return json.loads(text[start : end + 1])


def split_csv(value: str) -> list[str]:
    if str(value).strip().lower() in {"", "none", "null", "-"}:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def experiment_mode(internal_runner: str) -> str:
    if str(internal_runner).strip().lower() == "fair":
        return "fixed_trace_agent_fair"
    return "fixed_trace_replay"


def internal_runner_path(internal_runner: str) -> Path:
    if str(internal_runner).strip().lower() == "fair":
        return THIS_DIR / "run_castdas_trace_fair.py"
    return THIS_DIR / "run_castdas_trace.py"


def measurement_note(
    transactions_per_worker: int,
    internal_runner: str = "fair",
    *,
    warmup_seconds: float = 0.0,
    measure_seconds: float = 0.0,
) -> str:
    if float(measure_seconds) > 0:
        length_note = (
            "measure trace length is computed per variant and recorded in raw rows"
            if int(transactions_per_worker) <= 0
            else f"measure trace has {int(transactions_per_worker)} transactions per worker and cycles if exhausted"
        )
        return (
            "steady-state fixed trace source replay with CAST-DAS paper agent runtime; "
            f"warmup={float(warmup_seconds):g}s measure={float(measure_seconds):g}s; "
            f"{length_note}; "
            "ATCC preplan/reservation/deferred paths enabled; no outcome oracle"
        )
    if str(internal_runner).strip().lower() == "fair":
        return (
            "fixed transaction-count replay with CAST-DAS paper agent runtime; "
            f"{int(transactions_per_worker)} transactions per worker; "
            "ATCC preplan/reservation/deferred paths enabled"
        )
    return (
        "fixed transaction-count replay under 5h budget; "
        f"{int(transactions_per_worker)} transactions per worker; no paper-length warmup window"
    )


def parse_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def mean(values: list[float]) -> str:
    return fmt(statistics.fmean(values)) if values else ""


def stdev(values: list[float]) -> str:
    return fmt(statistics.stdev(values)) if len(values) > 1 else ""


def ratio(numerator: object, denominator: object) -> str:
    top = parse_float(numerator)
    bottom = parse_float(denominator)
    if top is None or bottom is None or bottom == 0:
        return ""
    return fmt(top / bottom)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def short_error(value: object, limit: int = 240) -> str:
    text = str(value or "").strip().replace("\n", " | ")
    return text[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
