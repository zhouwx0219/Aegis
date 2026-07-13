#!/usr/bin/env python3
"""Build a normalized CC metrics comparison CSV from experiment outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FIELDS = [
    "run_id",
    "source_file",
    "source_system",
    "system",
    "cc",
    "workload",
    "level",
    "client_mix",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "duration_s",
    "runs",
    "seed",
    "status",
    "coverage_note",
    "total_tps",
    "bottom_txn_attempt_tps",
    "bottom_txn_commit_tps",
    "underlying_txn_attempt_tps",
    "underlying_txn_commit_tps",
    "native_throughput",
    "agent_tps",
    "agent_task_tps",
    "background_tps",
    "speedup_vs_occ_total_tps",
    "speedup_vs_occ_agent_tps",
    "speedup_vs_silo_total_tps",
    "baseline_cc",
    "baseline_total_tps",
    "total_txn_cnt",
    "total_abort_cnt",
    "total_attempt_cnt",
    "txn_abort_rate",
    "commit_rate",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_commit_rate",
    "agent_abort_rate_per_commit",
    "agent_attempt_abort_rate",
    "agent_avg_retry_count",
    "agent_task_completion_rate",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_retries",
    "background_abort_rate",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_p9999_latency_ms",
    "agent_avg_tokens",
    "agent_total_tokens",
    "wasted_reasoning_ms",
    "agent_delay_ms_total",
    "read_conflicts",
    "write_conflicts",
    "version_conflict_count",
    "guarded_conflict_checks",
    "conflict_pressure_count",
    "raw_action_counts",
    "reservation_waiter_count_mean",
    "reservation_unique_target_count_mean",
    "reservation_front_queue_wait_ms_mean",
    "reservation_blocked_target_checks_mean",
    "reservation_owner_blocked_checks_mean",
    "reservation_writer_blocked_checks_mean",
    "reserve_hot_rw_k_attempts_mean",
    "reserve_hot_rw_k_target_size_mean",
    "reserve_read_write_set_attempts_mean",
    "reserve_read_write_set_target_size_mean",
    "reserve_read_write_set_hot_coverage_ratio_mean",
    "atcc_pure_policy",
    "atcc_agent_guardrail",
    "atcc_hot_rw_k",
    "workload_profile",
    "reasoning_profile",
    "reasoning_scale",
    "background_mode",
    "retry_until_commit",
    "max_attempts_per_task",
    "run_seconds",
    "error",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return value
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
        return "" if value in ("", None) else str(value)
    return f"{parsed:.10g}"


def safe_div(num: object, den: object) -> float | None:
    parsed_num = number(num)
    parsed_den = number(den)
    if parsed_num is None or parsed_den in (None, 0.0):
        return None
    return parsed_num / parsed_den


def add_castdas(rows: Iterable[dict[str, str]], source_file: Path, run_id: str) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        agent_attempts = first(row, "raw_agent_attempts")
        agent_commits = first(row, "raw_agent_commits")
        agent_aborts = first(row, "raw_agent_aborts")
        background_attempts = first(row, "raw_background_attempts")
        background_commits = first(row, "raw_background_commits")
        background_aborts = first(row, "raw_background_aborts")

        total_attempts = sum_existing(agent_attempts, background_attempts)
        total_aborts = sum_existing(agent_aborts, background_aborts)
        total_commits = sum_existing(agent_commits, background_commits)

        normalized.append(
            {
                "run_id": run_id,
                "source_file": source_file.name,
                "source_system": "cast-das",
                "system": "cast-das",
                "cc": row.get("cc", ""),
                "workload": row.get("workload", ""),
                "level": row.get("level", ""),
                "client_mix": row.get("client_mix", ""),
                "clients": row.get("clients", ""),
                "agent_ratio": first(row, "agent_ratio_config"),
                "agent_workers": row.get("agent_workers", ""),
                "background_workers": row.get("background_workers", ""),
                "duration_s": row.get("duration_s", ""),
                "runs": row.get("runs", ""),
                "seed": row.get("seed", ""),
                "status": "ok",
                "coverage_note": "complete_castdas_fast_matrix_duration_0.1s",
                "total_tps": first(row, "total_tps_mean", "raw_total_tps"),
                "bottom_txn_attempt_tps": first(row, "bottom_txn_attempt_tps_mean", "raw_bottom_txn_attempt_tps"),
                "bottom_txn_commit_tps": first(row, "bottom_txn_commit_tps_mean", "raw_bottom_txn_commit_tps"),
                "underlying_txn_attempt_tps": first(row, "underlying_txn_attempt_tps_mean", "raw_underlying_txn_attempt_tps"),
                "underlying_txn_commit_tps": first(row, "underlying_txn_commit_tps_mean", "raw_underlying_txn_commit_tps"),
                "native_throughput": first(row, "native_throughput_mean", "raw_native_throughput"),
                "agent_tps": first(row, "agent_tps_mean", "raw_agent_tps"),
                "agent_task_tps": first(row, "agent_task_tps_mean", "raw_agent_task_tps"),
                "background_tps": first(row, "background_tps_mean", "raw_background_tps"),
                "speedup_vs_occ_total_tps": first(row, "total_tps_speedup_vs_occ"),
                "speedup_vs_occ_agent_tps": first(row, "agent_tps_speedup_vs_occ"),
                "speedup_vs_silo_total_tps": "",
                "baseline_cc": "occ",
                "baseline_total_tps": "",
                "total_txn_cnt": fmt(total_commits),
                "total_abort_cnt": fmt(total_aborts),
                "total_attempt_cnt": fmt(total_attempts),
                "txn_abort_rate": fmt(safe_div(total_aborts, total_commits)),
                "commit_rate": fmt(safe_div(total_commits, total_attempts)),
                "agent_attempts": agent_attempts,
                "agent_commits": agent_commits,
                "agent_aborts": agent_aborts,
                "agent_commit_rate": first(row, "agent_commit_rate_mean", "raw_agent_commit_rate"),
                "agent_abort_rate_per_commit": first(row, "agent_abort_rate_mean", "raw_agent_abort_rate"),
                "agent_attempt_abort_rate": first(row, "agent_attempt_abort_rate_mean", "raw_agent_attempt_abort_rate"),
                "agent_avg_retry_count": first(row, "agent_avg_retry_count_mean", "raw_agent_avg_retry_count"),
                "agent_task_completion_rate": first(row, "agent_task_completion_rate_mean", "raw_agent_task_completion_rate"),
                "agent_completed_tasks": first(row, "raw_agent_completed_tasks"),
                "agent_failed_tasks": first(row, "raw_agent_failed_tasks"),
                "background_attempts": background_attempts,
                "background_commits": background_commits,
                "background_aborts": background_aborts,
                "background_retries": first(row, "raw_background_retries"),
                "background_abort_rate": fmt(safe_div(background_aborts, background_attempts)),
                "agent_p50_latency_ms": first(row, "agent_p50_latency_ms_mean", "raw_agent_p50_latency_ms"),
                "agent_p95_latency_ms": first(row, "agent_p95_latency_ms_mean", "raw_agent_p95_latency_ms"),
                "agent_p99_latency_ms": first(row, "agent_p99_latency_ms_mean", "raw_agent_p99_latency_ms"),
                "agent_p9999_latency_ms": first(row, "agent_p9999_latency_ms_mean", "raw_agent_p9999_latency_ms"),
                "agent_avg_tokens": first(row, "agent_avg_tokens_mean", "raw_agent_avg_tokens"),
                "agent_total_tokens": first(row, "agent_total_tokens_mean", "raw_agent_total_tokens"),
                "wasted_reasoning_ms": first(row, "wasted_reasoning_ms_mean", "raw_wasted_reasoning_ms"),
                "agent_delay_ms_total": "",
                "read_conflicts": first(row, "raw_read_conflicts"),
                "write_conflicts": first(row, "raw_write_conflicts"),
                "version_conflict_count": first(row, "version_conflict_count_mean", "raw_version_conflict_count"),
                "guarded_conflict_checks": first(row, "guarded_conflict_checks_mean", "raw_guarded_conflict_checks"),
                "conflict_pressure_count": first(row, "conflict_pressure_count_mean", "raw_conflict_pressure_count"),
                "raw_action_counts": row.get("raw_action_counts", ""),
                "reservation_waiter_count_mean": row.get("reservation_waiter_count_mean", ""),
                "reservation_unique_target_count_mean": row.get("reservation_unique_target_count_mean", ""),
                "reservation_front_queue_wait_ms_mean": row.get("reservation_front_queue_wait_ms_mean", ""),
                "reservation_blocked_target_checks_mean": row.get("reservation_blocked_target_checks_mean", ""),
                "reservation_owner_blocked_checks_mean": row.get("reservation_owner_blocked_checks_mean", ""),
                "reservation_writer_blocked_checks_mean": row.get("reservation_writer_blocked_checks_mean", ""),
                "reserve_hot_rw_k_attempts_mean": row.get("reserve_hot_rw_k_attempts_mean", ""),
                "reserve_hot_rw_k_target_size_mean": row.get("reserve_hot_rw_k_target_size_mean_mean", ""),
                "reserve_read_write_set_attempts_mean": row.get("reserve_read_write_set_attempts_mean", ""),
                "reserve_read_write_set_target_size_mean": row.get("reserve_read_write_set_target_size_mean_mean", ""),
                "reserve_read_write_set_hot_coverage_ratio_mean": row.get("reserve_read_write_set_hot_coverage_ratio_mean_mean", ""),
                "atcc_pure_policy": row.get("atcc_pure_policy", ""),
                "atcc_agent_guardrail": row.get("atcc_agent_guardrail", ""),
                "atcc_hot_rw_k": row.get("atcc_hot_rw_k", ""),
                "workload_profile": row.get("workload_profile", ""),
                "reasoning_profile": row.get("reasoning_profile", ""),
                "reasoning_scale": row.get("reasoning_scale", ""),
                "background_mode": row.get("background_mode", ""),
                "retry_until_commit": row.get("retry_until_commit", ""),
                "max_attempts_per_task": row.get("max_attempts_per_task", ""),
                "run_seconds": "",
                "error": "",
            }
        )
    return normalized


def sum_existing(*values: object) -> float | None:
    parsed = [number(value) for value in values]
    present = [value for value in parsed if value is not None]
    if not present:
        return None
    return sum(present)


def add_external(
    rows: Iterable[dict[str, str]],
    source_file: Path,
    run_id: str,
    coverage_note: str,
) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        clients = number(row.get("clients")) or 0.0
        ratio = number(row.get("agent_ratio"))
        agent_workers = int(round(clients * ratio)) if ratio is not None else 0
        background_workers = int(clients - agent_workers)
        duration = number(row.get("duration_s")) or 0.0

        txn_cnt = number(row.get("txn_cnt"))
        abort_cnt = number(row.get("abort_cnt"))
        attempts = sum_existing(txn_cnt, abort_cnt)
        agent_commits = number(row.get("agent_txn_cnt"))
        agent_aborts = number(row.get("agent_abort_cnt"))
        agent_attempts = sum_existing(agent_commits, agent_aborts)
        background_commits = number(row.get("background_txn_cnt"))
        background_aborts = number(row.get("background_abort_cnt"))
        background_attempts = sum_existing(background_commits, background_aborts)
        client_mix = (
            "all_agent"
            if ratio == 1.0
            else "agent80_backend20"
            if ratio == 0.8
            else f"agent_ratio_{row.get('agent_ratio', '')}"
        )

        normalized.append(
            {
                "run_id": run_id,
                "source_file": source_file.name,
                "source_system": "external-dbx1000",
                "system": row.get("system", ""),
                "cc": row.get("cc", ""),
                "workload": row.get("workload", ""),
                "level": row.get("level", ""),
                "client_mix": client_mix,
                "clients": row.get("clients", ""),
                "agent_ratio": row.get("agent_ratio", ""),
                "agent_workers": str(agent_workers),
                "background_workers": str(background_workers),
                "duration_s": row.get("duration_s", ""),
                "runs": "1",
                "seed": "",
                "status": row.get("status", ""),
                "coverage_note": coverage_note,
                "total_tps": row.get("throughput", ""),
                "bottom_txn_attempt_tps": fmt(attempts / duration if attempts is not None and duration else None),
                "bottom_txn_commit_tps": row.get("throughput", ""),
                "underlying_txn_attempt_tps": fmt(attempts / duration if attempts is not None and duration else None),
                "underlying_txn_commit_tps": row.get("throughput", ""),
                "native_throughput": row.get("throughput", ""),
                "agent_tps": fmt(agent_commits / duration if agent_commits is not None and duration else None),
                "agent_task_tps": fmt(agent_commits / duration if agent_commits is not None and duration else None),
                "background_tps": fmt(background_commits / duration if background_commits is not None and duration else None),
                "speedup_vs_occ_total_tps": "",
                "speedup_vs_occ_agent_tps": "",
                "speedup_vs_silo_total_tps": "",
                "baseline_cc": "",
                "baseline_total_tps": "",
                "total_txn_cnt": row.get("txn_cnt", ""),
                "total_abort_cnt": row.get("abort_cnt", ""),
                "total_attempt_cnt": fmt(attempts),
                "txn_abort_rate": fmt(safe_div(abort_cnt, txn_cnt)),
                "commit_rate": fmt(safe_div(txn_cnt, attempts)),
                "agent_attempts": fmt(agent_attempts),
                "agent_commits": row.get("agent_txn_cnt", ""),
                "agent_aborts": row.get("agent_abort_cnt", ""),
                "agent_commit_rate": fmt(safe_div(agent_commits, agent_attempts)),
                "agent_abort_rate_per_commit": fmt(safe_div(agent_aborts, agent_commits)),
                "agent_attempt_abort_rate": fmt(safe_div(agent_aborts, agent_attempts)),
                "agent_avg_retry_count": "",
                "agent_task_completion_rate": "",
                "agent_completed_tasks": "",
                "agent_failed_tasks": "",
                "background_attempts": fmt(background_attempts),
                "background_commits": row.get("background_txn_cnt", ""),
                "background_aborts": row.get("background_abort_cnt", ""),
                "background_retries": "",
                "background_abort_rate": fmt(safe_div(background_aborts, background_attempts)),
                "agent_p50_latency_ms": "",
                "agent_p95_latency_ms": "",
                "agent_p99_latency_ms": "",
                "agent_p9999_latency_ms": "",
                "agent_avg_tokens": "",
                "agent_total_tokens": "",
                "wasted_reasoning_ms": "",
                "agent_delay_ms_total": row.get("agent_delay_ms", ""),
                "read_conflicts": "",
                "write_conflicts": "",
                "version_conflict_count": "",
                "guarded_conflict_checks": "",
                "conflict_pressure_count": "",
                "raw_action_counts": "",
                "reservation_waiter_count_mean": "",
                "reservation_unique_target_count_mean": "",
                "reservation_front_queue_wait_ms_mean": "",
                "reservation_blocked_target_checks_mean": "",
                "reservation_owner_blocked_checks_mean": "",
                "reservation_writer_blocked_checks_mean": "",
                "reserve_hot_rw_k_attempts_mean": "",
                "reserve_hot_rw_k_target_size_mean": "",
                "reserve_read_write_set_attempts_mean": "",
                "reserve_read_write_set_target_size_mean": "",
                "reserve_read_write_set_hot_coverage_ratio_mean": "",
                "atcc_pure_policy": "",
                "atcc_agent_guardrail": "",
                "atcc_hot_rw_k": "",
                "workload_profile": "paper-mapped-dbx1000" if "paper" in run_id else "compressed-dbx1000",
                "reasoning_profile": "agentic-delay-sim",
                "reasoning_scale": "1.0",
                "background_mode": "native-dbx1000-thread",
                "retry_until_commit": "",
                "max_attempts_per_task": "",
                "run_seconds": row.get("run_seconds", ""),
                "error": row.get("error", ""),
            }
        )
    return normalized


def fill_baselines(rows: list[dict[str, str]]) -> None:
    occ_baseline = {}
    silo_baseline = {}
    for row in rows:
        total = number(row.get("total_tps"))
        if total is None:
            continue
        if row["source_system"] == "cast-das" and row["cc"] == "occ":
            key = (row["run_id"], row["client_mix"], row["workload"], row["level"], row["clients"])
            occ_baseline[key] = total
        if row["source_system"] == "external-dbx1000" and row["cc"].upper() == "SILO":
            key = (
                row["run_id"],
                row["system"],
                row["client_mix"],
                row["workload"],
                row["level"],
                row["clients"],
            )
            silo_baseline[key] = total

    for row in rows:
        total = number(row.get("total_tps"))
        if row["source_system"] == "cast-das":
            key = (row["run_id"], row["client_mix"], row["workload"], row["level"], row["clients"])
            base = occ_baseline.get(key)
            if base is not None:
                row["baseline_cc"] = "occ"
                row["baseline_total_tps"] = fmt(base)
                if not row.get("speedup_vs_occ_total_tps"):
                    row["speedup_vs_occ_total_tps"] = fmt(total / base if total is not None and base else None)
        elif row["source_system"] == "external-dbx1000":
            key = (
                row["run_id"],
                row["system"],
                row["client_mix"],
                row["workload"],
                row["level"],
                row["clients"],
            )
            base = silo_baseline.get(key)
            if base is not None:
                row["baseline_cc"] = "SILO"
                row["baseline_total_tps"] = fmt(base)
                row["speedup_vs_silo_total_tps"] = fmt(total / base if total is not None and base else None)


def sort_key(row: dict[str, str]) -> tuple[object, ...]:
    level_order = {"low": 0, "medium": 1, "high": 2}
    mix_order = {"all_agent": 0, "agent80_backend20": 1}
    return (
        row["run_id"],
        row["source_system"],
        row["system"],
        row["workload"],
        level_order.get(row["level"], 99),
        int(number(row["clients"]) or 0),
        mix_order.get(row["client_mix"], 99),
        row["cc"],
    )


def main() -> int:
    root = Path("results")
    output = root / "cc_metrics_comparison_20260710.csv"
    rows: list[dict[str, str]] = []

    castdas_src = root / "atcc_clients_fast_20260710_150641_summary.csv"
    if castdas_src.exists():
        rows.extend(add_castdas(read_csv(castdas_src), castdas_src, "castdas_atcc_fast_20260710_150641"))

    external_paper = root / "external_cc_paper_clients_target_20260710_163643.csv"
    if external_paper.exists():
        rows.extend(
            add_external(
                read_csv(external_paper),
                external_paper,
                "external_paper_clients_target_20260710_163643",
                "partial_external_paper_client_matrix_half_hour_cap; ycsb complete; tpcc low complete; tpcc medium partial; tpcc high not reached",
            )
        )

    external_compressed = root / "external_cc_target_plus_silo_20260710_160837.csv"
    if external_compressed.exists():
        rows.extend(
            add_external(
                read_csv(external_compressed),
                external_compressed,
                "external_compressed_target_plus_silo_20260710_160837",
                "compressed_external_matrix_clients_4_8_duration_1s_all_ok_includes_silo_baselines",
            )
        )

    fill_baselines(rows)
    rows.sort(key=sort_key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    counts = defaultdict(int)
    errors = defaultdict(int)
    for row in rows:
        counts[row["run_id"]] += 1
        if row.get("status") != "ok":
            errors[row["run_id"]] += 1
    print(output)
    print(f"rows={len(rows)} fields={len(FIELDS)}")
    for run_id in sorted(counts):
        print(f"{run_id}: rows={counts[run_id]} errors={errors[run_id]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
