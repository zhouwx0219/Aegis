#!/usr/bin/env python3
"""Combine budgeted CAST-DAS and external DBx1000 experiment CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
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
    "status",
    "total_tps",
    "agent_tps",
    "agent_task_tps",
    "background_tps",
    "speedup_vs_occ_total_tps",
    "speedup_vs_occ_agent_task_tps",
    "baseline_cc",
    "baseline_total_tps",
    "baseline_agent_task_tps",
    "total_txn_cnt",
    "total_abort_cnt",
    "txn_abort_rate",
    "commit_rate",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_commit_rate",
    "agent_abort_rate",
    "agent_attempt_abort_rate",
    "agent_avg_retry_count",
    "agent_task_completion_rate",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_p9999_latency_ms",
    "agent_avg_tokens",
    "agent_total_tokens",
    "wasted_reasoning_ms",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_abort_rate",
    "read_conflicts",
    "write_conflicts",
    "raw_action_counts",
    "agent_delay_ms_total",
    "run_seconds",
    "policy",
    "policy_mode",
    "coverage_note",
    "error",
]

VARIANT_META = {
    "ycsb_low": {"ycsb_zipf_theta": "0.0", "tpcc_warehouses": ""},
    "ycsb_medium_z07": {"ycsb_zipf_theta": "0.7", "tpcc_warehouses": ""},
    "ycsb_medium_z08": {"ycsb_zipf_theta": "0.8", "tpcc_warehouses": ""},
    "ycsb_high_z099": {"ycsb_zipf_theta": "0.99", "tpcc_warehouses": ""},
    "tpcc_low_w100": {"ycsb_zipf_theta": "", "tpcc_warehouses": "100"},
    "tpcc_medium": {"ycsb_zipf_theta": "", "tpcc_warehouses": "2"},
    "tpcc_high_w1": {"ycsb_zipf_theta": "", "tpcc_warehouses": "1"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    rows = []
    internal = run_dir / f"{args.run_id}_internal.csv"
    if internal.exists():
        rows.extend(read_internal(internal))
    external_rows = []
    for external in external_files(run_dir, args.run_id):
        external_rows.extend(read_external(external, args.run_id))
    rows.extend(dedupe_external_rows(external_rows))

    fill_internal_occ_baselines(rows)
    rows.sort(key=sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    print(args.output)
    print(f"rows={len(rows)}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_internal(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(path):
        normalized = {field: row.get(field, "") for field in FIELDS}
        normalized["source_system"] = "cast-das"
        normalized["system"] = "cast-das"
        normalized["total_txn_cnt"] = row.get("agent_commits", "")
        normalized["total_abort_cnt"] = row.get("agent_aborts", "")
        normalized["baseline_cc"] = "occ"
        rows.append(normalized)
    return rows


def read_external(path: Path, run_id: str) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(path):
        ratio = number(row.get("agent_ratio"))
        clients = number(row.get("clients")) or 0.0
        agent_workers = int(round(clients * ratio)) if ratio is not None else 0
        background_workers = int(clients - agent_workers)
        duration = number(row.get("duration_s"))
        agent_commits = number(row.get("agent_txn_cnt"))
        background_commits = number(row.get("background_txn_cnt"))
        txn_cnt = number(row.get("txn_cnt"))
        abort_cnt = number(row.get("abort_cnt"))
        variant = row.get("workload_variant", "") or f"{row.get('workload', '')}_{row.get('level', '')}"
        meta = VARIANT_META.get(variant, {})
        client_mix = "all_agent" if ratio == 1.0 else "agent80_backend20" if ratio == 0.8 else f"agent_ratio_{row.get('agent_ratio', '')}"
        normalized = {
            "run_id": run_id,
            "source_system": "external-dbx1000",
            "system": row.get("system", ""),
            "cc": row.get("cc", ""),
            "workload": row.get("workload", ""),
            "workload_variant": variant,
            "level": row.get("level", ""),
            "ycsb_zipf_theta": meta.get("ycsb_zipf_theta", ""),
            "tpcc_warehouses": meta.get("tpcc_warehouses", ""),
            "client_mix": client_mix,
            "clients": row.get("clients", ""),
            "agent_ratio": row.get("agent_ratio", ""),
            "agent_workers": str(agent_workers),
            "background_workers": str(background_workers),
            "seed": "",
            "repeat": row.get("repeat", ""),
            "warmup_s": row.get("warmup_s", "0"),
            "duration_s": row.get("duration_s", ""),
            "budget_limited": "True",
            "status": row.get("status", ""),
            "total_tps": row.get("throughput", ""),
            "agent_tps": fmt(agent_commits / duration if agent_commits is not None and duration else None),
            "agent_task_tps": fmt(agent_commits / duration if agent_commits is not None and duration else None),
            "background_tps": fmt(background_commits / duration if background_commits is not None and duration else None),
            "total_txn_cnt": row.get("txn_cnt", ""),
            "total_abort_cnt": row.get("abort_cnt", ""),
            "txn_abort_rate": fmt(abort_cnt / txn_cnt if abort_cnt is not None and txn_cnt else None),
            "commit_rate": fmt(txn_cnt / (txn_cnt + abort_cnt) if txn_cnt is not None and abort_cnt is not None and txn_cnt + abort_cnt else None),
            "agent_commits": row.get("agent_txn_cnt", ""),
            "agent_aborts": row.get("agent_abort_cnt", ""),
            "background_commits": row.get("background_txn_cnt", ""),
            "background_aborts": row.get("background_abort_cnt", ""),
            "agent_delay_ms_total": row.get("agent_delay_ms", ""),
            "run_seconds": row.get("run_seconds", ""),
            "policy_mode": "",
            "coverage_note": "external DBx1000 budgeted run; no cross-engine speedup computed",
            "error": row.get("error", ""),
        }
        rows.append(normalized)
    return rows


def external_files(run_dir: Path, run_id: str) -> list[Path]:
    patterns = (f"{run_id}_external_*.csv", f"{run_id}.external_*.csv")
    files = []
    seen = set()
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def dedupe_external_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("source_system", ""),
            row.get("system", ""),
            row.get("cc", ""),
            row.get("workload_variant", ""),
            row.get("clients", ""),
            row.get("agent_ratio", ""),
            row.get("repeat", ""),
        )
        current = deduped.get(key)
        if current is None or row_rank(row) >= row_rank(current):
            deduped[key] = row
    return list(deduped.values())


def row_rank(row: dict[str, str]) -> tuple[int, int]:
    status_rank = 1 if row.get("status") == "ok" else 0
    metric_rank = 1 if number(row.get("total_tps")) is not None else 0
    return status_rank, metric_rank


def fill_internal_occ_baselines(rows: list[dict[str, str]]) -> None:
    baselines = {}
    for row in rows:
        if row.get("source_system") != "cast-das" or row.get("cc") != "occ":
            continue
        key = baseline_key(row)
        baselines[key] = {
            "total_tps": number(row.get("total_tps")),
            "agent_task_tps": number(row.get("agent_task_tps")),
        }
    for row in rows:
        if row.get("source_system") != "cast-das":
            continue
        baseline = baselines.get(baseline_key(row))
        if not baseline:
            continue
        row["baseline_cc"] = "occ"
        row["baseline_total_tps"] = fmt(baseline["total_tps"])
        row["baseline_agent_task_tps"] = fmt(baseline["agent_task_tps"])
        total = number(row.get("total_tps"))
        agent = number(row.get("agent_task_tps"))
        row["speedup_vs_occ_total_tps"] = fmt(total / baseline["total_tps"] if total is not None and baseline["total_tps"] else None)
        row["speedup_vs_occ_agent_task_tps"] = fmt(agent / baseline["agent_task_tps"] if agent is not None and baseline["agent_task_tps"] else None)


def baseline_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("workload_variant", ""),
        row.get("client_mix", ""),
        row.get("clients", ""),
        row.get("seed", ""),
    )


def sort_key(row: dict[str, str]) -> tuple[object, ...]:
    source_order = {"cast-das": 0, "external-dbx1000": 1}
    mix_order = {"all_agent": 0, "agent80_backend20": 1}
    return (
        source_order.get(row.get("source_system", ""), 99),
        row.get("system", ""),
        row.get("workload_variant", ""),
        int(number(row.get("clients")) or 0),
        mix_order.get(row.get("client_mix", ""), 99),
        int(number(row.get("repeat")) or 0),
        int(number(row.get("seed")) or 0),
        row.get("cc", ""),
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
