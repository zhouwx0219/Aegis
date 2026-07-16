#!/usr/bin/env python3
"""Replace corrected ATCC rows and rebuild the final paper-main CC summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


KEY_FIELDS = ("workload_variant", "client_mix", "clients", "cc_label")
RAW_KEY_FIELDS = ("workload_variant", "client_mix", "clients", "seed", "cc_label")
TRADITIONAL_LABELS = {
    "OCC", "2PL-nowait", "2PL-wait-die", "MVCC", "Silo", "TicToc", "Bamboo", "Polaris"
}
SUM_FIELDS = {
    "bottom_txn_attempts": "bottom_txn_attempts",
    "bottom_txn_commits": "bottom_txn_commits",
    "agent_attempts": "agent_attempts",
    "agent_commits": "agent_commits",
    "agent_aborts": "agent_aborts",
    "agent_completed_tasks": "agent_completed_tasks",
    "agent_failed_tasks": "agent_failed_tasks",
    "background_attempts": "background_attempts",
    "background_commits": "background_commits",
    "background_aborts": "background_aborts",
    "background_retries": "background_retries",
    "total_reasoning_delay_ms": "total_reasoning_delay_ms",
    "wasted_reasoning_ms": "wasted_reasoning_ms",
    "read_conflicts": "read_conflicts",
    "write_conflicts": "write_conflicts",
    "version_conflict_count": "version_conflict_count",
    "guarded_conflict_checks": "guarded_conflict_checks",
    "conflict_pressure_count": "conflict_pressure_count",
    "agent_reservation_wait_ms_total": "agent_reservation_wait_ms_total",
    "background_reservation_wait_ms_total": "background_reservation_wait_ms_total",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str], fields: tuple[str, ...] = KEY_FIELDS) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def number(value: object) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def ratio(numerator: object, denominator: object) -> str:
    num, den = number(numerator), number(denominator)
    return fmt(num / den) if num is not None and den not in (None, 0.0) else ""


def aggregate_replacement_raw(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for group_key, group in grouped.items():
        values: dict[str, str] = {
            "raw_row_count": str(len(group)),
            "seeds": ",".join(sorted({row["seed"] for row in group}, key=int)),
        }
        for source, target in SUM_FIELDS.items():
            parsed = [number(row.get(source)) for row in group]
            present = [value for value in parsed if value is not None]
            values[f"{target}_total"] = fmt(sum(present)) if present else ""
            values[f"{target}_mean"] = fmt(statistics.fmean(present)) if present else ""
        actions: Counter[str] = Counter()
        for row in group:
            try:
                actions.update({str(k): int(v) for k, v in json.loads(row.get("raw_action_counts", "{}") or "{}").items()})
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        values["raw_action_counts_sum_json"] = json.dumps(dict(sorted(actions.items())), sort_keys=True)
        output[group_key] = values
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--replacement-raw", type=Path, required=True)
    parser.add_argument("--replacement-summary", type=Path, required=True)
    parser.add_argument("--output-raw", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    base_raw = read_csv(args.base_raw)
    replacement_raw = read_csv(args.replacement_raw)
    replacement_summary_rows = read_csv(args.replacement_summary)
    replacement_summary = {key(row): row for row in replacement_summary_rows}
    replacement_raw_keys = {key(row, RAW_KEY_FIELDS) for row in replacement_raw}
    merged_raw = [row for row in base_raw if key(row, RAW_KEY_FIELDS) not in replacement_raw_keys]
    merged_raw.extend(replacement_raw)

    expected_replacement_raw = len(replacement_summary) * 3
    if (
        len(base_raw) != 1620
        or len(replacement_raw) != expected_replacement_raw
        or len(merged_raw) != 1620
    ):
        raise SystemExit(
            f"unexpected raw coverage: base={len(base_raw)} replacement={len(replacement_raw)} merged={len(merged_raw)}"
        )
    merged_raw_key_counts = Counter(key(row, RAW_KEY_FIELDS) for row in merged_raw)
    duplicates = [item for item, count in merged_raw_key_counts.items() if count != 1]
    if duplicates:
        raise SystemExit(f"raw key uniqueness failure: {duplicates[:3]}")

    base_summary = read_csv(args.base_summary)
    replacement_aggregates = aggregate_replacement_raw(replacement_raw)
    replaced = 0
    for row in base_summary:
        row_key = key(row)
        replacement = replacement_summary.get(row_key)
        if replacement is None:
            continue
        for field, value in replacement.items():
            if field in row:
                row[field] = value
        row.update(replacement_aggregates[row_key])
        row["source_summary_csv"] = str(args.replacement_summary)
        row["source_raw_csv"] = str(args.replacement_raw)
        replaced += 1
    if replaced != len(replacement_summary) or len(base_summary) != 540:
        raise SystemExit(f"unexpected summary coverage: rows={len(base_summary)} replaced={replaced}")
    for row in base_summary:
        row["run_id"] = "paper_main_atcc_trained_final"

    by_config: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in base_summary:
        by_config[(row["workload_variant"], row["client_mix"], row["clients"])].append(row)
    for config_rows in by_config.values():
        if len(config_rows) != 9:
            raise SystemExit(f"expected 9 CC rows, got {len(config_rows)} for {key(config_rows[0])[:3]}")
        atcc = next(row for row in config_rows if row["cc_label"] == "ATCC")
        traditional = [row for row in config_rows if row["cc_label"] in TRADITIONAL_LABELS]
        best_agent = max(traditional, key=lambda row: number(row.get("agent_task_tps_mean")) or float("-inf"))
        best_total = max(traditional, key=lambda row: number(row.get("total_tps_mean")) or float("-inf"))
        for row in config_rows:
            row["best_agent_baseline"] = best_agent["cc_label"]
            row["best_total_baseline"] = best_total["cc_label"]
            row["best_agent_task_tps"] = best_agent.get("agent_task_tps_mean", "")
            row["best_total_tps"] = best_total.get("total_tps_mean", "")
            row["atcc_agent_task_tps_vs_best_agent_speedup"] = ratio(
                atcc.get("agent_task_tps_mean"), best_agent.get("agent_task_tps_mean")
            )
            row["atcc_total_tps_vs_best_total_speedup"] = ratio(
                atcc.get("total_tps_mean"), best_total.get("total_tps_mean")
            )
            row["atcc_p99_latency_ms"] = atcc.get("agent_p99_latency_ms_mean", "")
            row["best_agent_baseline_p99_latency_ms"] = best_agent.get("agent_p99_latency_ms_mean", "")
            row["p99_latency_ratio_atcc_over_best_agent"] = ratio(
                atcc.get("agent_p99_latency_ms_mean"), best_agent.get("agent_p99_latency_ms_mean")
            )
            row["atcc_background_tps"] = atcc.get("background_tps_mean", "")
            row["best_agent_baseline_background_tps"] = best_agent.get("background_tps_mean", "")
            row["background_tps_ratio_atcc_over_best_agent"] = ratio(
                atcc.get("background_tps_mean"), best_agent.get("background_tps_mean")
            )

    raw_fields = list(merged_raw[0])
    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    with args.output_raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in raw_fields} for row in merged_raw)

    summary_fields = [field for field in base_summary[0] if "p9999" not in field]
    with args.output_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in summary_fields} for row in base_summary)

    print(f"raw={args.output_raw} rows={len(merged_raw)}")
    print(f"summary={args.output_summary} rows={len(base_summary)} fields={len(summary_fields)} replaced={replaced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
