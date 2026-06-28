"""Manifest-driven ATCC retry experiment runner.

This module keeps profile-scaled experiments reproducible by declaring each
profile's workload, cost, runtime, and policy-artifact settings in one JSON
manifest, then delegating execution to ``atcc_retry_experiment``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TextIO

from agent.evaluation.atcc_retry_experiment import main as retry_main


def run_manifest_suite(
    manifest_path: Path,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    output_dir = Path(output_dir or manifest.get("output_dir") or manifest_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = dict(manifest.get("defaults", {}) or {})
    profile_reports = []
    run_rows = []
    for profile in manifest.get("profiles", ()) or ():
        if not isinstance(profile, Mapping):
            raise ValueError("profile entries must be JSON objects")
        profile_name = str(profile.get("name", "")).strip()
        if not profile_name:
            raise ValueError("profile entry missing name")
        seeds = tuple(int(seed) for seed in profile.get("seeds", ()) or ())
        if not seeds:
            raise ValueError(f"profile {profile_name} missing seeds")
        profile_run_reports = []
        for seed in seeds:
            run_path = output_dir / f"{profile_name}-seed{seed}-r{int(defaults.get('repeats', 1))}.json"
            args = _retry_args(defaults, profile, seed=seed, output=run_path)
            retry_main(args)
            data = json.loads(run_path.read_text(encoding="utf-8"))
            row = _summary_row(profile_name, run_path.name, data)
            run_rows.append(row)
            profile_run_reports.append(
                {
                    "seed": seed,
                    "output": str(run_path),
                    "summary": row,
                }
            )
        profile_reports.append(
            {
                "name": profile_name,
                "workload": str(profile.get("workload", defaults.get("workload", ""))),
                "runs": profile_run_reports,
            }
        )

    stats = _profile_stats(run_rows)
    _write_summary_csv(output_dir / "manifest-combined-summary.csv", run_rows)
    _write_json(output_dir / "manifest-combined-stats.json", stats)
    report = {
        "artifact_type": "atcc-retry-manifest-suite",
        "artifact_version": 1,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "profiles": profile_reports,
        "stats": stats,
    }
    _write_json(output_dir / "manifest-suite.json", report)
    return report


def _retry_args(
    defaults: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    seed: int,
    output: Path,
) -> list[str]:
    merged = dict(defaults)
    merged.update(dict(profile.get("runtime", {}) or {}))
    workload = str(profile.get("workload", merged.get("workload", "ycsb")))
    args = [
        "--workload",
        workload,
        "--profile-name",
        str(profile["name"]),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    option_map = {
        "strategies": "--strategies",
        "strategy_order": "--strategy-order",
        "interleave_blocks": "--interleave-blocks",
        "task_count": "--task-count",
        "repeats": "--repeats",
        "workers": "--workers",
        "agent_slots": "--agent-slots",
        "planning_delay_ms": "--planning-delay-ms",
        "abort_retry_delay_ms": "--abort-retry-delay-ms",
        "latency_distribution": "--latency-distribution",
        "latency_cv": "--latency-cv",
        "latency_max_ms": "--latency-max-ms",
        "max_attempts": "--max-attempts",
        "tokens_per_operation": "--tokens-per-operation",
        "background_workers": "--background-workers",
        "background_interval_ms": "--background-interval-ms",
        "object_lock_scheduler": "--object-lock-scheduler",
        "object_lock_priority_burst": "--object-lock-priority-burst",
        "prelock_wait_budget_ms": "--prelock-wait-budget-ms",
        "prelock_wait_budget_mode": "--prelock-wait-budget-mode",
        "prelock_lease_mode": "--prelock-lease-mode",
        "agent_execution_mode": "--agent-execution-mode",
        "snapshot_timing": "--snapshot-timing",
        "policy_artifact": "--policy-artifact",
        "policy_variant": "--policy-variant",
    }
    for key, option in option_map.items():
        if key in merged and merged[key] not in (None, ""):
            args.extend([option, str(merged[key])])
    if bool(merged.get("hybrid_fast_through", False)):
        args.append("--hybrid-fast-through")
    if bool(merged.get("hybrid_selected_fast_through", False)):
        args.append("--hybrid-selected-fast-through")

    workload_config = dict(profile.get("workload_config", {}) or {})
    ycsb_map = {
        "record_count": "--records",
        "field_count": "--fields",
        "requests_per_task": "--requests-per-task",
        "candidates_per_task": "--candidates",
        "read_weight": "--read-weight",
        "update_weight": "--update-weight",
        "zipf_theta": "--zipf-theta",
        "hotspot_fraction": "--hotspot-fraction",
        "hotspot_access_probability": "--hotspot-access-probability",
    }
    tpcc_map = {
        "warehouses": "--warehouses",
        "districts_per_warehouse": "--districts-per-warehouse",
        "customers_per_district": "--customers-per-district",
        "items": "--items",
        "initial_stock": "--initial-stock",
        "order_lines": "--order-lines",
        "transaction_mix": "--transaction-mix",
        "candidates_per_task": "--candidates",
    }
    config_map = ycsb_map if workload == "ycsb" else tpcc_map
    for key, option in config_map.items():
        if key in workload_config:
            args.extend([option, str(workload_config[key])])
    return args


def _summary_row(profile: str, filename: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    aggregates = {
        str(row.get("strategy", "")): row
        for row in data.get("aggregates", ())
        if isinstance(row, Mapping)
    }
    hybrid = aggregates.get("adaptive-hybrid")
    if hybrid is None:
        hybrid = next(iter(aggregates.values()))
    baselines = [
        row
        for name, row in aggregates.items()
        if name != "adaptive-hybrid"
        and float(row.get("committed_throughput", 0.0) or 0.0) > 0.0
    ]
    best = max(
        baselines or [hybrid],
        key=lambda row: float(row.get("committed_throughput", 0.0) or 0.0),
    )
    hybrid_throughput = float(hybrid.get("committed_throughput", 0.0) or 0.0)
    best_throughput = float(best.get("committed_throughput", 0.0) or 0.0)
    pairs = data.get("selected_baseline_pairs", {}).get("pairs", ())
    pair_ratio = ""
    if pairs:
        pair_ratio = float(pairs[0].get("hybrid_vs_selected_baseline", 0.0) or 0.0)
    return {
        "profile": profile,
        "file": filename,
        "hybrid_throughput": hybrid_throughput,
        "best_baseline_strategy": str(best.get("strategy", "")),
        "best_baseline_throughput": best_throughput,
        "vs_best_baseline": (
            hybrid_throughput / best_throughput if best_throughput else 0.0
        ),
        "hybrid_commit_rate": float(hybrid.get("commit_rate", 0.0) or 0.0),
        "hybrid_attempts_per_task": float(
            hybrid.get("attempts_per_task", 0.0) or 0.0
        ),
        "selected_pair_ratio": pair_ratio,
    }


def _profile_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for profile in sorted({str(row["profile"]) for row in rows}):
        profile_rows = [row for row in rows if row["profile"] == profile]
        ratios = [float(row["vs_best_baseline"]) for row in profile_rows]
        pairs = [
            float(row["selected_pair_ratio"])
            for row in profile_rows
            if row.get("selected_pair_ratio") != ""
        ]
        stats[profile] = {
            "runs": len(ratios),
            "mean": statistics.mean(ratios),
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "paired_mean_avg": statistics.mean(pairs) if pairs else None,
        }
    return stats


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ATCC retry experiments from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_manifest_suite(args.manifest, output_dir=args.output_dir)
    (stdout or sys.stdout).write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
