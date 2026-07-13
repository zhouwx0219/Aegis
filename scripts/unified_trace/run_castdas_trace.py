#!/usr/bin/env python3
"""Replay a fixed CAST-DAS trace inside the Python runtime."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmarks.mixed import MixedBenchmarkConfig, registry_for
from agent.runtime import AgentTransactionManager


CCS = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,dynamic-atcc"

FIELDS = [
    "trace_id",
    "source_system",
    "system",
    "cc",
    "workload",
    "workload_variant",
    "level",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "seed",
    "repeat",
    "status",
    "elapsed_s",
    "total_tps",
    "agent_task_tps",
    "agent_tps",
    "background_tps",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "agent_task_completion_rate",
    "agent_commit_rate",
    "agent_attempt_abort_rate",
    "agent_avg_retry_count",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_p9999_latency_ms",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_commit_rate",
    "total_reasoning_delay_ms",
    "wasted_reasoning_ms",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", default=CCS)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    rows = read_trace(args.trace)
    output_rows = []
    for cc in split_csv(args.cc):
        started = time.perf_counter()
        try:
            output_rows.append(run_trace(rows, cc=cc, policy=args.policy, max_attempts=args.max_attempts))
        except Exception as exc:  # keep matrix runners moving and make failures explicit
            sample = rows[0] if rows else {}
            output_rows.append(error_row(sample, cc, exc, time.perf_counter() - started))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(args.output)
    print(f"rows={len(output_rows)}")
    return 0


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty trace: {path}")
    for row in rows:
        row["_ops"] = json.loads(row["ops_json"])
        row["_context"] = json.loads(row.get("context_json") or "{}")
    return rows


def run_trace(
    rows: list[dict[str, Any]],
    *,
    cc: str,
    policy: Path | None,
    max_attempts: int,
) -> dict[str, Any]:
    sample = rows[0]
    agent_workers = int(float(sample["agent_workers"]))
    background_workers = int(float(sample["background_workers"]))
    config = MixedBenchmarkConfig(
        workload=sample["workload"],
        level=sample["level"],
        workload_profile="paper",
        cc=cc,
        clients=int(float(sample["clients"])),
        agent_ratio=float(sample["agent_ratio"]),
        policy=policy,
        policy_mode="eval" if policy else "online",
        background_mode="procedure",
    ).normalized()
    manager = AgentTransactionManager(cc_registry=registry_for(config))
    for key in sorted(trace_keys(rows)):
        manager.register_object(obj_id(key), "0", kind="row")

    by_worker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_worker[int(float(row["worker_id"]))].append(row)
    for worker_rows in by_worker.values():
        worker_rows.sort(key=lambda row: int(float(row["sequence"])))

    lock = threading.Lock()
    counters = Counter()
    agent_latencies: list[float] = []
    agent_retry_counts: list[int] = []
    total_reasoning_ms = 0
    wasted_reasoning_ms = 0
    barrier = threading.Barrier(len(by_worker) + 1)
    threads = [
        threading.Thread(
            target=worker_main,
            args=(manager, cc, worker_rows, max_attempts, barrier, lock, counters, agent_latencies, agent_retry_counts),
        )
        for _worker, worker_rows in sorted(by_worker.items())
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    elapsed_s = max(0.001, time.perf_counter() - started)

    for row in rows:
        if row["client_type"] == "agent":
            total_reasoning_ms += int(float(row.get("total_reasoning_delay_ms") or 0))
    wasted_reasoning_ms = int(counters["agent_aborts"]) * 0
    agent_attempts = int(counters["agent_attempts"])
    agent_commits = int(counters["agent_commits"])
    agent_aborts = int(counters["agent_aborts"])
    agent_completed = int(counters["agent_completed_tasks"])
    agent_failed = int(counters["agent_failed_tasks"])
    background_attempts = int(counters["background_attempts"])
    background_commits = int(counters["background_commits"])
    background_aborts = int(counters["background_aborts"])
    return {
        **base_row(sample, cc),
        "status": "ok",
        "elapsed_s": elapsed_s,
        "total_tps": (agent_commits + background_commits) / elapsed_s,
        "agent_task_tps": agent_completed / elapsed_s,
        "agent_tps": agent_commits / elapsed_s,
        "background_tps": background_commits / elapsed_s,
        "agent_attempts": agent_attempts,
        "agent_commits": agent_commits,
        "agent_aborts": agent_aborts,
        "agent_completed_tasks": agent_completed,
        "agent_failed_tasks": agent_failed,
        "agent_task_completion_rate": agent_completed / (agent_completed + agent_failed) if agent_completed + agent_failed else 0.0,
        "agent_commit_rate": agent_commits / agent_attempts if agent_attempts else 0.0,
        "agent_attempt_abort_rate": agent_aborts / agent_attempts if agent_attempts else 0.0,
        "agent_avg_retry_count": average(agent_retry_counts),
        "agent_p50_latency_ms": percentile(agent_latencies, 50),
        "agent_p95_latency_ms": percentile(agent_latencies, 95),
        "agent_p99_latency_ms": percentile(agent_latencies, 99),
        "agent_p9999_latency_ms": percentile(agent_latencies, 99.99),
        "background_attempts": background_attempts,
        "background_commits": background_commits,
        "background_aborts": background_aborts,
        "background_commit_rate": background_commits / background_attempts if background_attempts else 0.0,
        "total_reasoning_delay_ms": total_reasoning_ms,
        "wasted_reasoning_ms": wasted_reasoning_ms,
        "error": "",
    }


def worker_main(
    manager: AgentTransactionManager,
    cc: str,
    rows: list[dict[str, Any]],
    max_attempts: int,
    barrier: threading.Barrier,
    lock: threading.Lock,
    counters: Counter,
    agent_latencies: list[float],
    agent_retry_counts: list[int],
) -> None:
    barrier.wait()
    for row in rows:
        is_agent = row["client_type"] == "agent"
        started = time.perf_counter()
        committed = False
        attempts_done = 0
        for attempt in range(max(1, max_attempts if is_agent else 1)):
            attempts_done += 1
            if is_agent:
                sleep_ms(int(float(row.get("retry_delay_ms") or 0)) if attempt > 0 else 0)
                sleep_ms(int(float(row.get("explore_delay_ms") or 0)) + int(float(row.get("refine_delay_ms") or 0)))
            result = execute_attempt(manager, cc, row, attempt)
            if is_agent:
                with lock:
                    counters["agent_attempts"] += 1
                    if result.committed:
                        counters["agent_commits"] += 1
                    else:
                        counters["agent_aborts"] += 1
            else:
                with lock:
                    counters["background_attempts"] += 1
                    if result.committed:
                        counters["background_commits"] += 1
                    else:
                        counters["background_aborts"] += 1
            if result.committed:
                committed = True
                break
        if is_agent:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with lock:
                if committed:
                    counters["agent_completed_tasks"] += 1
                    agent_latencies.append(elapsed_ms)
                    agent_retry_counts.append(max(0, attempts_done - 1))
                else:
                    counters["agent_failed_tasks"] += 1


def execute_attempt(manager: AgentTransactionManager, cc: str, row: dict[str, Any], attempt: int):
    keys = [int(op["key"]) for op in row["_ops"]]
    metadata = {
        "workload": row["workload"],
        "task_type": row["task_type"],
        "retry_count": attempt,
        "context": {**row["_context"], "retry_count": attempt, "client_type": row["client_type"]},
        "agentic": {
            "phase_count": 3 if row["client_type"] == "agent" else 0,
            "reasoning_delay_ms": int(float(row.get("total_reasoning_delay_ms") or 0)) if row["client_type"] == "agent" else 0,
            "retry_delay_ms": int(float(row.get("retry_delay_ms") or 0)) if row["client_type"] == "agent" else 0,
            "background_workers": int(float(row.get("background_workers") or 0)),
            "target_selection_seed": stable_selection_seed(row),
        },
    }
    txn = manager.begin(
        f"{row['trace_id']}:{row['worker_id']}:{row['sequence']}:{attempt}",
        metadata,
        snapshot_object_ids=tuple(obj_id(key) for key in keys),
    )
    for op in row["_ops"]:
        oid = obj_id(int(op["key"]))
        if op["kind"] == "read":
            txn.read(oid)
        else:
            txn.write(oid, op.get("value") or f"v:{row['worker_id']}:{row['sequence']}:{attempt}")
    if row["client_type"] == "agent":
        sleep_ms(int(float(row.get("commit_delay_ms") or 0)))
    return txn.commit(cc)


def trace_keys(rows: list[dict[str, Any]]) -> set[int]:
    keys = set()
    for row in rows:
        for op in row["_ops"]:
            keys.add(int(op["key"]))
    return keys


def obj_id(key: int) -> str:
    return f"trace:key:{int(key)}"


def sleep_ms(value: int) -> None:
    if value > 0:
        time.sleep(value / 1000.0)


def stable_selection_seed(row: dict[str, Any]) -> int:
    text = f"{row['trace_id']}:{row['worker_id']}:{row['sequence']}"
    value = 0
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def base_row(row: dict[str, Any], cc: str) -> dict[str, Any]:
    return {
        "trace_id": row.get("trace_id", ""),
        "source_system": "cast-das-trace",
        "system": "cast-das",
        "cc": cc,
        "workload": row.get("workload", ""),
        "workload_variant": row.get("workload_variant", ""),
        "level": row.get("level", ""),
        "clients": row.get("clients", ""),
        "agent_ratio": row.get("agent_ratio", ""),
        "agent_workers": row.get("agent_workers", ""),
        "background_workers": row.get("background_workers", ""),
        "seed": row.get("seed", ""),
        "repeat": row.get("repeat", ""),
    }


def error_row(sample: dict[str, Any], cc: str, exc: Exception, elapsed_s: float) -> dict[str, Any]:
    return {
        **base_row(sample, cc),
        "status": "error",
        "elapsed_s": elapsed_s,
        "error": repr(exc),
    }


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def average(values: list[int]) -> float:
    return statistics.fmean(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
