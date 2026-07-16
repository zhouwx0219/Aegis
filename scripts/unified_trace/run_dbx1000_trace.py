#!/usr/bin/env python3
"""Replay CAST-DAS fixed traces on patched DBx1000-family systems."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_RE = re.compile(r"\[summary\]\s+(?P<body>.*)")
LATENCY_RE = re.compile(r"\[all_debug1\s+thd=(?P<thd>\d+)\]\s+(?P<body>.*)")

SYSTEMS = {
    "bamboo": {
        "source_dir": "Bamboo-Public",
        "work_dir": "Bamboo-Public_trace",
        "python": "python",
        "alg": "BAMBOO",
        "compat": True,
        "compiler_candidates": [],
    },
    "polaris": {
        "source_dir": "polaris",
        "work_dir": "polaris_trace",
        "python": "python3",
        "alg": "SILO_PRIO",
        "compat": False,
        "compiler_candidates": [
            "/home/chenht/miniconda3-castdas/bin/x86_64-conda-linux-gnu-g++",
            "/usr/bin/g++-13",
            "/usr/bin/g++-12",
            "/usr/bin/g++-11",
            "/usr/bin/g++-10",
        ],
    },
}

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
    "agent_p999_latency_ms",
    "agent_p9999_latency_ms",
    "agent_time_to_success_p50_ms",
    "agent_time_to_success_p95_ms",
    "agent_time_to_success_p99_ms",
    "agent_time_to_success_p999_ms",
    "agent_time_to_success_p9999_ms",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_commit_rate",
    "total_reasoning_delay_ms",
    "wasted_reasoning_ms",
    "read_conflicts",
    "write_conflicts",
    "conflict_abort_count",
    "agent_avg_tokens",
    "agent_total_tokens",
    "native_throughput",
    "txn_cnt",
    "abort_cnt",
    "user_abort_cnt",
    "run_seconds",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--systems", default="bamboo,polaris")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-timeout", type=float, default=120.0)
    parser.add_argument("--patch-agentic", type=Path, default=ROOT / "scripts" / "external_cc" / "patch_dbx1000_agentic.py")
    parser.add_argument("--patch-trace", type=Path, default=Path(__file__).with_name("patch_dbx1000_trace_replay.py"))
    args = parser.parse_args()

    trace_rows = read_trace(args.trace)
    output_rows = []
    for system in split_csv(args.systems):
        meta = SYSTEMS[system]
        repo = prepare_repo(args.root.resolve(), system, meta, args.patch_agentic, args.patch_trace)
        output_rows.append(run_one(repo, system, meta, trace_rows, args.trace, args.run_timeout))
        write_rows(args.output, output_rows)
    return 0


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty trace: {path}")
    return rows


def prepare_repo(root: Path, system: str, meta: dict, patch_agentic: Path, patch_trace: Path) -> Path:
    src = root / meta["source_dir"]
    dst = root / meta["work_dir"]
    if not src.exists():
        raise SystemExit(f"missing external repo for {system}: {src}")
    if dst.exists():
        shutil.rmtree(str(dst))
    ignore = shutil.ignore_patterns(".git", "outputs", "log", "*.o", "*.d", "rundb", "temp.out")
    shutil.copytree(str(src), str(dst), ignore=ignore)
    (dst / "outputs").mkdir(exist_ok=True)
    (dst / "log").mkdir(exist_ok=True)
    cmd = [sys.executable, str(patch_agentic), str(dst)]
    if meta.get("compat"):
        cmd.append("--compat")
    compiler = first_existing(meta.get("compiler_candidates", []))
    if compiler:
        cmd.extend(["--compiler", compiler])
    subprocess.check_call(cmd)
    subprocess.check_call([sys.executable, str(patch_trace), str(dst)])
    return dst


def run_one(
    repo: Path,
    system: str,
    meta: dict,
    trace_rows: list[dict[str, str]],
    trace_path: Path,
    run_timeout: float,
) -> dict[str, str]:
    sample = trace_rows[0]
    tsv_path = repo / "outputs" / "castdas_trace.tsv"
    trace_stats = write_dbx1000_trace(tsv_path, trace_rows)
    overrides = common_overrides(sample, trace_stats, meta["alg"])
    cmd = [meta["python"], "test.py", "experiments/default.json"]
    cmd.extend(f"{key}={value}" for key, value in sorted(overrides.items()))
    started = time.time()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        env=run_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    try:
        out, _ = proc.communicate(timeout=float(run_timeout) if run_timeout else None)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_tree(proc)
        out, _ = proc.communicate()
    elapsed = time.time() - started
    parsed = parse_summary(out)
    latency_by_thread = parse_latency_distribution(out)
    log_dir = repo / "outputs" / "castdas_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{system}_{meta['alg']}_{sample['trace_id']}.log").write_text(out)
    row = base_row(sample, system, meta["alg"])
    txn_cnt = number(parsed.get("txn_cnt"))
    abort_cnt = number(parsed.get("abort_cnt"))
    agent_commits = number(parsed.get("castdas_agent_txn_cnt"))
    agent_aborts = number(parsed.get("castdas_agent_abort_cnt"))
    bg_commits = number(parsed.get("castdas_background_txn_cnt"))
    bg_aborts = number(parsed.get("castdas_background_abort_cnt"))
    native_tps = number(parsed.get("throughput"))
    user_abort_cnt = number(parsed.get("user_abort_cnt"))
    conflict_abort_count = (
        max(0.0, abort_cnt - (user_abort_cnt or 0.0))
        if abort_cnt is not None
        else None
    )
    total_tps = (
        native_tps
        if native_tps is not None
        else (txn_cnt / elapsed if txn_cnt is not None and elapsed > 0 else None)
    )
    agent_attempts = (
        (agent_commits or 0) + (agent_aborts or 0)
        if agent_commits is not None or agent_aborts is not None
        else None
    )
    expected_agent_tasks = float(trace_stats.get("agent_rows", 0) or 0)
    agent_failed_tasks = (
        max(0.0, expected_agent_tasks - agent_commits)
        if agent_commits is not None and expected_agent_tasks > 0
        else None
    )
    agent_completion_rate = (
        min(1.0, agent_commits / expected_agent_tasks)
        if agent_commits is not None and expected_agent_tasks > 0
        else None
    )
    agent_retry_count = (
        agent_aborts / agent_commits
        if agent_aborts is not None and agent_commits is not None and agent_commits > 0
        else None
    )
    avg_tokens = (
        (1.0 + agent_retry_count) * trace_stats["avg_agent_ops"] * trace_stats["tokens_per_operation"]
        if agent_retry_count is not None and trace_stats["avg_agent_ops"] > 0
        else None
    )
    agent_latencies_ms = latency_samples_for_threads(
        latency_by_thread,
        int(float(sample.get("agent_workers") or 0)),
    )
    avg_latency_ms = number(parsed.get("latency"))
    if avg_latency_ms is not None:
        avg_latency_ms *= 1000.0

    def commit_share_tps(commits: float | None) -> float | None:
        if total_tps is None or commits is None or txn_cnt is None or txn_cnt <= 0:
            return None
        return total_tps * (commits / txn_cnt)

    row.update(
        {
            "status": "ok" if parsed and not timed_out else "error",
            "elapsed_s": f"{elapsed:.6f}",
            "total_tps": fmt(total_tps),
            "agent_task_tps": fmt(commit_share_tps(agent_commits)),
            "agent_tps": fmt(commit_share_tps(agent_commits)),
            "background_tps": fmt(commit_share_tps(bg_commits)),
            "agent_attempts": fmt(agent_attempts),
            "agent_commits": fmt(agent_commits),
            "agent_aborts": fmt(agent_aborts),
            "agent_completed_tasks": fmt(agent_commits),
            "agent_failed_tasks": fmt(agent_failed_tasks),
            "agent_task_completion_rate": fmt(agent_completion_rate),
            "agent_commit_rate": fmt(agent_commits / (agent_commits + agent_aborts) if agent_commits is not None and agent_aborts is not None and agent_commits + agent_aborts else None),
            "agent_attempt_abort_rate": fmt(agent_aborts / (agent_commits + agent_aborts) if agent_commits is not None and agent_aborts is not None and agent_commits + agent_aborts else None),
            "agent_avg_retry_count": fmt(agent_retry_count),
            "agent_p50_latency_ms": fmt(percentile(agent_latencies_ms, 50) if agent_latencies_ms else avg_latency_ms),
            "agent_p95_latency_ms": fmt(percentile(agent_latencies_ms, 95) if agent_latencies_ms else None),
            "agent_p99_latency_ms": fmt(percentile(agent_latencies_ms, 99) if agent_latencies_ms else None),
            "agent_p999_latency_ms": fmt(percentile(agent_latencies_ms, 99.9) if agent_latencies_ms else None),
            "agent_p9999_latency_ms": fmt(percentile(agent_latencies_ms, 99.99) if agent_latencies_ms else None),
            "agent_time_to_success_p50_ms": fmt(percentile(agent_latencies_ms, 50) if agent_latencies_ms else avg_latency_ms),
            "agent_time_to_success_p95_ms": fmt(percentile(agent_latencies_ms, 95) if agent_latencies_ms else None),
            "agent_time_to_success_p99_ms": fmt(percentile(agent_latencies_ms, 99) if agent_latencies_ms else None),
            "agent_time_to_success_p999_ms": fmt(percentile(agent_latencies_ms, 99.9) if agent_latencies_ms else None),
            "agent_time_to_success_p9999_ms": fmt(percentile(agent_latencies_ms, 99.99) if agent_latencies_ms else None),
            "background_attempts": fmt((bg_commits or 0) + (bg_aborts or 0) if bg_commits is not None or bg_aborts is not None else None),
            "background_commits": fmt(bg_commits),
            "background_aborts": fmt(bg_aborts),
            "background_commit_rate": fmt(bg_commits / (bg_commits + bg_aborts) if bg_commits is not None and bg_aborts is not None and bg_commits + bg_aborts else None),
            "total_reasoning_delay_ms": trace_stats["total_agent_delay_ms"],
            "wasted_reasoning_ms": fmt(agent_aborts * trace_stats["avg_agent_delay_ms"] if agent_aborts is not None else None),
            "read_conflicts": "",
            "write_conflicts": "",
            "conflict_abort_count": fmt(conflict_abort_count),
            "agent_avg_tokens": fmt(avg_tokens),
            "agent_total_tokens": fmt(avg_tokens * agent_commits if avg_tokens is not None and agent_commits is not None else None),
            "native_throughput": parsed.get("throughput", ""),
            "txn_cnt": parsed.get("txn_cnt", ""),
            "abort_cnt": parsed.get("abort_cnt", ""),
            "user_abort_cnt": parsed.get("user_abort_cnt", ""),
            "run_seconds": f"{elapsed:.6f}",
            "error": "" if parsed and not timed_out else ("timeout; " + tail(out) if timed_out else tail(out)),
        }
    )
    return row


def write_dbx1000_trace(path: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    max_key = 0
    max_ops = 0
    total_agent_delay = 0
    total_agent_ops = 0
    agent_rows = 0
    background_rows = 0
    per_worker = {}
    workload = str(rows[0].get("workload", "")).strip().lower() if rows else ""
    configured_warehouses = int(float(rows[0].get("tpcc_warehouses") or 0)) if rows else 0
    tpcc_num_wh = max(0, configured_warehouses)
    tpcc_payment_count = 0
    tpcc_new_order_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (int(float(r["worker_id"])), int(float(r["sequence"])))):
            ops = json.loads(row["ops_json"])
            tokens = []
            for op in ops:
                key = int(op["key"])
                max_key = max(max_key, key)
                tokens.append(("W" if op["kind"] == "write" else "R") + f":{key}")
            max_ops = max(max_ops, len(tokens))
            pre = (
                int(float(row.get("explore_delay_ms") or 0))
                + int(float(row.get("refine_delay_ms") or 0))
                + sum(
                    int(float(op.get("delay_ms") or 0))
                    for op in ops
                    if str(op.get("phase", "")).strip().lower()
                    in {"explore", "refine"}
                )
            )
            retry = int(float(row.get("retry_delay_ms") or 0))
            commit = int(float(row.get("commit_delay_ms") or 0)) + sum(
                int(float(op.get("delay_ms") or 0))
                for op in ops
                if str(op.get("phase", "")).strip().lower() == "commit"
            )
            if row["client_type"] == "agent":
                total_agent_delay += pre + commit
                total_agent_ops += len(ops)
                agent_rows += 1
            else:
                background_rows += 1
            worker = int(float(row["worker_id"]))
            per_worker[worker] = per_worker.get(worker, 0) + 1
            native_payload = ""
            if workload == "tpcc":
                native_payload = tpcc_native_payload(row, ops)
                context = parse_json_object(row.get("context_json", ""))
                tpcc_num_wh = max(tpcc_num_wh, int(context.get("warehouse", 0) or 0) + 1)
                task_type = str(row.get("task_type", "")).strip().lower()
                if task_type == "payment":
                    tpcc_payment_count += 1
                elif task_type == "new_order":
                    tpcc_new_order_count += 1
            handle.write(
                "\t".join(
                    [
                        str(worker),
                        row["client_type"],
                        str(int(float(row["sequence"]))),
                        str(pre),
                        str(retry),
                        str(commit),
                        ",".join(tokens),
                        native_payload,
                    ]
                )
                + "\n"
            )
    return {
        "max_key": max_key,
        "table_size": max_key + 1,
        "max_ops": max_ops,
        "transactions_per_worker": max(per_worker.values()) if per_worker else 0,
        "total_agent_delay_ms": total_agent_delay,
        "avg_agent_delay_ms": (total_agent_delay / agent_rows) if agent_rows else 0,
        "avg_agent_ops": (total_agent_ops / agent_rows) if agent_rows else 0,
        "tokens_per_operation": 2703,
        "agent_rows": agent_rows,
        "background_rows": background_rows,
        "tpcc_num_wh": max(1, tpcc_num_wh),
        "tpcc_payment_count": tpcc_payment_count,
        "tpcc_new_order_count": tpcc_new_order_count,
    }


def tpcc_native_payload(row: dict[str, str], ops: list[dict[str, object]]) -> str:
    context = parse_json_object(row.get("context_json", ""))
    task_type = str(row.get("task_type", "")).strip().lower()
    warehouse = int(context.get("warehouse", 0) or 0) + 1
    district = int(context.get("district", 0) or 0) + 1
    customer = int(context.get("customer", 0) or 0) + 1
    if task_type == "payment":
        amount = 1
        for op in ops:
            value = str(op.get("value", ""))
            if value.startswith("payment-ytd:"):
                amount = parse_trailing_int(value, default=amount)
                break
        return ";".join(
            [
                "task=payment",
                f"w={warehouse}",
                f"d={district}",
                f"c={customer}",
                f"dw={warehouse}",
                f"cw={warehouse}",
                f"cd={district}",
                f"amount={max(1, amount)}",
            ]
        )

    items: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for op in ops:
        object_id = str(op.get("object_id", ""))
        parts = object_id.split(":")
        if len(parts) != 5 or parts[0] != "tpcc" or parts[1] != "stock":
            continue
        try:
            supply_w = int(parts[2]) + 1
            item_id = int(parts[3]) + 1
        except ValueError:
            continue
        key = (supply_w, item_id)
        if key in seen:
            continue
        seen.add(key)
        quantity = parse_trailing_int(str(op.get("value", "")), default=1)
        items.append((item_id, supply_w, max(1, quantity)))
    if not items:
        items.append((1, warehouse, 1))
    encoded_items = "|".join(f"{item}:{supply}:{quantity}" for item, supply, quantity in items)
    return ";".join(
        [
            "task=new_order",
            f"w={warehouse}",
            f"d={district}",
            f"c={customer}",
            f"items={encoded_items}",
        ]
    )


def parse_json_object(text: str) -> dict[str, object]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_trailing_int(text: str, *, default: int) -> int:
    match = re.search(r"(\d+)$", str(text))
    if not match:
        return int(default)
    return int(match.group(1))


def common_overrides(sample: dict[str, str], stats: dict[str, int], alg: str) -> dict[str, object]:
    max_ops = max(1, int(stats["max_ops"]))
    table_size = max(1, int(stats["table_size"]))
    workload = str(sample.get("workload", "")).strip().lower()
    overrides: dict[str, object] = {
        "THREAD_CNT": int(float(sample["clients"])),
        "CC_ALG": alg,
        "CASTDAS_AGENTIC": "true",
        "CASTDAS_AGENT_RATIO": sample["agent_ratio"],
        "CASTDAS_REASONING_SCALE": 0,
        "CASTDAS_TRACE_REPLAY": "true",
        "CASTDAS_TRACE_PATH": '"outputs/castdas_trace.tsv"',
        "CASTDAS_TRACE_MAX_OPS": max_ops,
        "TERMINATE_BY_COUNT": "true",
        "MAX_TXN_PER_PART": int(stats["transactions_per_worker"]),
        "MAX_RUNTIME": 3600,
        "WARMUP": 0,
        "ABORT_BUFFER_ENABLE": "true",
        "ABORT_BUFFER_SIZE": 8,
        "ABORT_PENALTY": 1000,
        "MAX_ROW_PER_TXN": max_ops,
        "MAX_WRITE_SET": max_ops,
        "REQ_PER_QUERY": max_ops,
        "SYNTH_TABLE_SIZE": table_size,
        "INIT_PARALLELISM": 1,
        "PART_CNT": 1,
        "VIRTUAL_PART_CNT": 1,
        "PART_PER_TXN": 1,
        "PERC_MULTI_PART": 0,
        "UNSET_NUMA": "true",
        "OUTPUT_TO_FILE": "false",
        "PRT_LAT_DISTR": "true",
        "NDEBUG": "true",
    }
    if workload == "tpcc":
        payment_count = int(stats.get("tpcc_payment_count", 0) or 0)
        new_order_count = int(stats.get("tpcc_new_order_count", 0) or 0)
        total_tpcc = max(1, payment_count + new_order_count)
        max_rows = max(64, max_ops + 16)
        overrides.update(
            {
                "WORKLOAD": "TPCC",
                "TPCC_SMALL": "true",
                "TPCC_USER_ABORT": "false",
                "NUM_WH": max(1, int(stats.get("tpcc_num_wh", 1) or 1)),
                "PERC_PAYMENT": payment_count / total_tpcc,
                "PERC_DELIVERY": 0,
                "PERC_ORDERSTATUS": 0,
                "PERC_STOCKLEVEL": 0,
                "MAX_ROW_PER_TXN": max_rows,
                "MAX_WRITE_SET": max_rows,
                "MAX_LOCK_CNT": max(20 * int(float(sample["clients"])), max_rows * int(float(sample["clients"])) * 2),
            }
        )
    else:
        overrides.update(
            {
                "WORKLOAD": "YCSB",
                "READ_PERC": 0.5,
                "WRITE_PERC": 1,
                "SYNTHETIC_YCSB": "false",
            }
        )
    return overrides


def base_row(sample: dict[str, str], system: str, cc: str) -> dict[str, str]:
    return {
        "trace_id": sample.get("trace_id", ""),
        "source_system": "external-dbx1000-trace",
        "system": system,
        "cc": cc,
        "workload": sample.get("workload", ""),
        "workload_variant": sample.get("workload_variant", ""),
        "level": sample.get("level", ""),
        "clients": sample.get("clients", ""),
        "agent_ratio": sample.get("agent_ratio", ""),
        "agent_workers": sample.get("agent_workers", ""),
        "background_workers": sample.get("background_workers", ""),
        "seed": sample.get("seed", ""),
        "repeat": sample.get("repeat", ""),
    }


def parse_summary(output: str) -> dict[str, str]:
    result = {}
    for line in output.splitlines():
        match = SUMMARY_RE.search(line.strip())
        if not match:
            continue
        for token in match.group("body").split(","):
            token = token.strip()
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def parse_latency_distribution(output: str) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for line in output.splitlines():
        match = LATENCY_RE.search(line.strip())
        if not match:
            continue
        thread_id = int(match.group("thd"))
        values = []
        for token in match.group("body").split(","):
            token = token.strip()
            if not token:
                continue
            parsed = number(token)
            if parsed is not None:
                values.append(parsed / 1_000_000.0)
        result[thread_id] = values
    return result


def latency_samples_for_threads(latency_by_thread: dict[int, list[float]], agent_workers: int) -> list[float]:
    samples: list[float] = []
    for thread_id in range(max(0, int(agent_workers))):
        samples.extend(latency_by_thread.get(thread_id, ()))
    return samples


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def run_env() -> dict[str, str]:
    env = dict(os.environ)
    conda_lib = "/home/chenht/miniconda3-castdas/lib"
    if Path(conda_lib).exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = conda_lib if not existing else conda_lib + ":" + existing
    return env


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, 9)
        else:
            proc.kill()
    except ProcessLookupError:
        return


def first_existing(paths: list[str]) -> str:
    for path in paths:
        if Path(path).exists():
            return str(path)
    return ""


def number(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fmt(value: object) -> str:
    parsed = number(value)
    if parsed is None:
        return ""
    return f"{parsed:.10g}"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tail(text: str, limit: int = 800) -> str:
    clean = " | ".join(line.strip() for line in text.splitlines()[-20:])
    return clean[-limit:]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
