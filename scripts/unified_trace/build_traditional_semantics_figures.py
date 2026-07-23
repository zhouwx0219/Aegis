#!/usr/bin/env python3
"""Aggregate a coherent five-system Figure 1-6 experiment matrix."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SYSTEM_LABELS = {
    "2pl-wait-die": "2PL",
    "bamboo": "Bamboo",
    "silo": "Silo",
    "polaris": "Polaris",
    "paper-atcc": "Aegis",
}
SYSTEM_ORDER = {name: index for index, name in enumerate(SYSTEM_LABELS.values())}
GROUPS = {
    "ycsb_scalability": ("Figure 1", "figure_ycsb_scalability_raw.csv"),
    "tpcc_scalability": ("Figure 2", "figure_tpcc_scalability_raw.csv"),
    "contention_sensitivity": ("Figure 4", "figure_contention_sensitivity_raw.csv"),
    "shape_sensitivity": ("Figure 5", "figure_shape_sensitivity_raw.csv"),
    "ratio_control_sensitivity": ("Figure 6", "figure_ratio_control_sensitivity_raw.csv"),
}
OUTPUT_FIELDS = (
    "figure",
    "experiment",
    "parameter",
    "parameter_value",
    "workload",
    "system",
    "n_seeds",
    "agent_tps_mean",
    "agent_tps_std",
    "commit_rate_mean",
    "commit_rate_std",
    "p50_latency_ms_mean",
    "p50_latency_ms_std",
    "p95_latency_ms_mean",
    "p95_latency_ms_std",
    "p99_latency_ms_mean",
    "p99_latency_ms_std",
    "wasted_reasoning_ms_per_commit_mean",
    "wasted_reasoning_ms_per_commit_std",
    "wasted_tokens_per_commit_mean",
    "wasted_tokens_per_commit_std",
)
METRICS = (
    "agent_tps",
    "commit_rate",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "wasted_reasoning_ms_per_commit",
    "wasted_tokens_per_commit",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    normalized: list[dict[str, Any]] = []
    normalized.extend(load_regular_matrix(args.matrix_dir))
    normalized.extend(
        load_credit(
            args.matrix_dir / "credit_review" / "credit_review_figure3_raw.csv",
            expected_cc=SYSTEM_LABELS,
        )
    )
    summary = aggregate(normalized)
    validate_summary(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master = args.output_dir / "aegis_figures_1_6_no_retry_semantics_fixed.csv"
    write_csv(master, summary)
    for figure_number in range(1, 7):
        figure = f"Figure {figure_number}"
        rows = [row for row in summary if row["figure"] == figure]
        write_csv(args.output_dir / f"figure{figure_number}_plot_data.csv", rows)
    print(master)
    return 0


def load_regular_matrix(directory: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment, (figure, filename) in GROUPS.items():
        for row in read_csv(directory / filename):
            cc = str(row.get("cc", ""))
            if cc not in SYSTEM_LABELS:
                continue
            output.append(normalize_regular(row, figure))
    return output


def normalize_regular(row: dict[str, str], figure: str) -> dict[str, Any]:
    require_ok(row)
    commits = as_float(row, "agent_commits")
    wasted_reasoning = as_float(row, "wasted_reasoning_ms")
    return {
        "figure": figure,
        "experiment": row["experiment"],
        "parameter": row["parameter"],
        "parameter_value": row["parameter_value"],
        "workload": row["workload"],
        "cc": row["cc"],
        "system": SYSTEM_LABELS[row["cc"]],
        "seed": int(float(row["seed"])),
        "agent_tps": as_float(row, "agent_tps"),
        "commit_rate": as_float(row, "agent_commit_rate"),
        "p50_latency_ms": as_float(row, "agent_p50_latency_ms"),
        "p95_latency_ms": as_float(row, "agent_p95_latency_ms"),
        "p99_latency_ms": as_float(row, "agent_p99_latency_ms"),
        "wasted_reasoning_ms_per_commit": wasted_reasoning / commits if commits else 0.0,
        "wasted_tokens_per_commit": as_float(row, "agent_wasted_tokens_per_commit"),
    }


def load_credit(path: Path, *, expected_cc: Iterable[str]) -> list[dict[str, Any]]:
    expected = frozenset(expected_cc)
    output: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row.get("cc") not in expected:
            continue
        require_ok(row)
        output.append(
            {
                "figure": "Figure 3",
                "experiment": "agentic_native_credit_review",
                "parameter": "workers",
                "parameter_value": row["clients"],
                "workload": "CreditReview",
                "cc": row["cc"],
                "system": SYSTEM_LABELS[row["cc"]],
                "seed": int(float(row["seed"])),
                **{metric: as_float(row, metric) for metric in METRICS},
            }
        )
    return output


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = ("figure", "experiment", "parameter", "parameter_value", "workload", "system")
    for row in rows:
        grouped[tuple(str(row[field]) for field in key_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        summary: dict[str, Any] = dict(zip(key_fields, key))
        summary["n_seeds"] = len({int(row["seed"]) for row in group})
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    output.sort(key=sort_key)
    return output


def validate_summary(rows: list[dict[str, Any]]) -> None:
    expected_counts = {
        "Figure 1": 25,
        "Figure 2": 25,
        "Figure 3": 25,
        "Figure 4": 50,
        "Figure 5": 100,
        "Figure 6": 37,
    }
    actual = defaultdict(int)
    for row in rows:
        actual[str(row["figure"])] += 1
        if int(row["n_seeds"]) != 3:
            raise RuntimeError(f"expected 3 seeds: {row}")
    if dict(actual) != expected_counts:
        raise RuntimeError(f"unexpected figure row counts: {dict(actual)}")


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    figure = int(str(row["figure"]).split()[-1])
    try:
        value: Any = float(row["parameter_value"])
    except ValueError:
        value = str(row["parameter_value"])
    return (
        figure,
        str(row["parameter"]),
        str(row["workload"]),
        value,
        SYSTEM_ORDER[str(row["system"])],
    )


def require_ok(row: dict[str, str]) -> None:
    if row.get("status") != "ok":
        raise RuntimeError(f"non-ok experiment row: {row.get('trace_id') or row.get('run_id')}")


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise RuntimeError(f"missing {field} in {row.get('trace_id') or row.get('run_id')}")
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in OUTPUT_FIELDS})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
