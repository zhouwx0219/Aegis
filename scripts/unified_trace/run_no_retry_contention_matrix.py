#!/usr/bin/env python3
"""Run the four paper-retry scalability matrices used by Figures 7-10."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "unified_trace" / "generate_castdas_trace.py"
RUNNER = ROOT / "scripts" / "unified_trace" / "run_castdas_trace_fair.py"
SYSTEMS = "2pl-wait-die,bamboo,silo,polaris,paper-atcc"
WORKLOAD_VARIANTS = {
    "ycsb_high": "ycsb_high_z099",
    "tpcc_high_w1": "tpcc_high_w1",
    "ycsb_low": "ycsb_low",
    "tpcc_low_w100": "tpcc_low_w100",
    "ycsb_medium": "ycsb_medium_z08",
}
DEFAULT_WORKLOADS = (
    "ycsb_medium",
    "ycsb_high",
    "tpcc_low_w100",
    "tpcc_high_w1",
)
CLIENTS = (8, 16, 24, 32, 40)
PAPER_MAX_RETRIES = 5
PAPER_MAX_ATTEMPTS = PAPER_MAX_RETRIES + 1
CLEAN_FIELDS = (
    "experiment",
    "max_retries",
    "max_attempts",
    "workload",
    "workload_variant",
    "level",
    "warehouses",
    "clients",
    "agent_ratio",
    "seed",
    "repeat",
    "cc",
    "access_set_visibility",
    "status",
    "measurement_window_s",
    "agent_tps",
    "native_throughput",
    "agent_commit_rate",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_attempts",
    "agent_logical_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_avg_retry_count",
    "wasted_reasoning_ms",
    "agent_avg_tokens",
    "agent_total_tokens",
    "agent_wasted_reasoning_tokens",
    "agent_wasted_tokens_per_commit",
    "agent_wasted_token_ratio",
    "read_conflicts",
    "write_conflicts",
    "version_validation_abort_count",
    "lock_timeout_abort_count",
    "lock_preempted_abort_count",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-policy", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=0.25)
    parser.add_argument("--measure-seconds", type=float, default=1.5)
    parser.add_argument("--transactions-per-worker", type=int, default=64)
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument(
        "--max-retries",
        type=int,
        default=PAPER_MAX_RETRIES,
        help="Retries after the initial attempt (paper default: 5).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    max_attempts = int(args.max_retries) + 1

    workload_labels = tuple(
        value.strip() for value in str(args.workloads).split(",") if value.strip()
    )
    unknown = [value for value in workload_labels if value not in WORKLOAD_VARIANTS]
    if not workload_labels or unknown:
        raise SystemExit(f"unknown or empty workloads: {','.join(unknown)}")

    output_dir = args.output_dir.resolve()
    traces = output_dir / "traces"
    runs = output_dir / "runs"
    traces.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    for label in workload_labels:
        variant = WORKLOAD_VARIANTS[label]
        group_rows: list[dict[str, str]] = []
        total = len(CLIENTS) * args.repeats
        completed = 0
        for clients in CLIENTS:
            for repeat in range(args.repeats):
                seed = 992_000 + clients + repeat
                stem = f"{label}_c{clients}_r{repeat}_s{seed}"
                trace = traces / f"{stem}.csv"
                result = runs / f"{stem}.csv"
                if not trace.exists():
                    subprocess.run(
                        [
                            sys.executable,
                            str(GENERATOR),
                            "--output",
                            str(trace),
                            "--variant",
                            variant,
                            "--clients",
                            str(clients),
                            "--agent-ratio",
                            "1.0",
                            "--seed",
                            str(seed),
                            "--repeat",
                            str(repeat),
                            "--transactions-per-worker",
                            str(args.transactions_per_worker),
                            "--reasoning-profile",
                            "agentic",
                            "--reasoning-scale",
                            "1.0",
                        ],
                        cwd=ROOT,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                if args.force or not valid_result(result, max_attempts=max_attempts):
                    command = build_run_command(
                        trace=trace,
                        result=result,
                        paper_policy=args.paper_policy.resolve(),
                        max_attempts=max_attempts,
                        warmup_seconds=args.warmup_seconds,
                        measure_seconds=args.measure_seconds,
                    )
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                with result.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        row["experiment"] = label
                        row["max_retries"] = str(args.max_retries)
                        row["max_attempts"] = str(max_attempts)
                        row["warehouses"] = (
                            "1" if label == "tpcc_high_w1" else
                            "100" if label == "tpcc_low_w100" else ""
                        )
                        group_rows.append(row)
                completed += 1
                print(f"[{label}] {completed}/{total} {stem}", flush=True)
        write_csv(output_dir / f"{label}_raw.csv", group_rows)
        print(f"COMPLETE {label}", flush=True)

    all_rows: list[dict[str, str]] = []
    for label in workload_labels:
        with (output_dir / f"{label}_raw.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            all_rows.extend(csv.DictReader(handle))
    artifact_prefix = (
        "zero_retry"
        if args.max_retries == 0
        else "five_retry"
        if args.max_retries == PAPER_MAX_RETRIES
        else f"retry_{args.max_retries}"
    )
    write_csv(output_dir / f"{artifact_prefix}_four_workloads_raw.csv", all_rows)
    write_csv(
        output_dir / f"{artifact_prefix}_four_workloads_clean.csv",
        [{field: row.get(field, "") for field in CLEAN_FIELDS} for row in all_rows],
    )
    return 0


def build_run_command(
    *,
    trace: Path,
    result: Path,
    paper_policy: Path,
    max_attempts: int,
    warmup_seconds: float,
    measure_seconds: float,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--trace",
        str(trace),
        "--warmup-trace",
        str(trace),
        "--output",
        str(result),
        "--cc",
        SYSTEMS,
        "--paper-policy",
        str(paper_policy),
        "--policy-mode",
        "eval",
        "--max-attempts",
        str(max_attempts),
        "--warmup-seconds",
        str(warmup_seconds),
        "--measure-seconds",
        str(measure_seconds),
        "--paper-switching",
        "dynamic",
        "--paper-priority",
        "enabled",
        "--paper-delayed-write-apply",
        "enabled",
        "--paper-performance-guards",
        "enabled",
        "--tpcc-replay-capacity",
        "1",
        "--ycsb-replay-capacity",
        "1",
        "--disable-atcc-retry-cache",
    ]
    if max_attempts > 1:
        command.append("--allow-retries")
    return command


def valid_result(path: Path, *, max_attempts: int) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = len(SYSTEMS.split(","))
    return len(rows) == expected and all(
        row.get("status") == "ok"
        and int(float(row.get("max_attempts", 0) or 0)) == int(max_attempts)
        and int(float(row.get("retry_budget", -1) or -1))
        == max(0, int(max_attempts) - 1)
        and int(float(row.get("tpcc_replay_capacity", 0) or 0)) == 1
        and int(float(row.get("ycsb_replay_capacity", 0) or 0)) == 1
        for row in rows
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
