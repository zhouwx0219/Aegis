#!/usr/bin/env python3
"""Verify archived zero-retry Figure 7-13 artifacts and recompute claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_entry(repo_root: Path, label: str, entry: Mapping[str, Any]) -> Path:
    path = repo_root / str(entry["path"])
    if not path.is_file():
        raise VerificationError(f"{label}: missing {path}")
    expected = str(entry["sha256"]).lower()
    actual = sha256_file(path)
    if actual != expected:
        raise VerificationError(
            f"{label}: SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VerificationError(f"{path}: empty CSV")
    if any(str(row.get("status", "ok")).lower() != "ok" for row in rows):
        raise VerificationError(f"{path}: contains a failed row")
    return rows


def fmean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def rows_for(rows: Iterable[dict[str, str]], **values: object) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in values.items())
    ]


def system_metrics(rows: list[dict[str, str]], system_field: str = "cc") -> dict[str, dict[str, float]]:
    if not rows:
        raise VerificationError("metric selection produced no rows")
    commit_field = "agent_commit_rate" if "agent_commit_rate" in rows[0] else "commit_rate"
    p99_field = (
        "agent_p99_latency_ms"
        if "agent_p99_latency_ms" in rows[0]
        else "p99_latency_ms"
    )
    systems = sorted({row[system_field] for row in rows})
    return {
        system: {
            "tps": fmean(rows_for(rows, **{system_field: system}), "agent_tps"),
            "commit_rate": fmean(
                rows_for(rows, **{system_field: system}), commit_field
            ),
            "p99_ms": fmean(
                rows_for(rows, **{system_field: system}), p99_field
            ),
        }
        for system in systems
    }


def speedup(metrics: Mapping[str, Mapping[str, float]]) -> tuple[float, str]:
    aegis = metrics["paper-atcc"]["tps"]
    best = max(
        (name for name in metrics if name != "paper-atcc"),
        key=lambda name: metrics[name]["tps"],
    )
    return aegis / metrics[best]["tps"], best


def point_metrics(
    rows: list[dict[str, str]],
    *,
    parameter: str,
    value: str,
    workload: str = "YCSB",
) -> dict[str, Any]:
    selected = rows_for(
        rows,
        workload=workload,
        parameter=parameter,
        parameter_value=value,
    )
    metrics = system_metrics(selected)
    ratio, best = speedup(metrics)
    return {
        "aegis_tps": metrics["paper-atcc"]["tps"],
        "best_baseline": best,
        "best_baseline_tps": metrics[best]["tps"],
        "speedup": ratio,
        "commit_rate": metrics["paper-atcc"]["commit_rate"],
        "p99_ms": metrics["paper-atcc"]["p99_ms"],
    }


def verify_zero_retry_rows(rows: Iterable[Mapping[str, str]], label: str) -> None:
    for row in rows:
        if "max_attempts" in row and int(float(row["max_attempts"])) != 1:
            raise VerificationError(f"{label}: max_attempts is not 1")
        if "retry_budget" in row and int(float(row["retry_budget"])) != 0:
            raise VerificationError(f"{label}: retry_budget is not 0")
        if "agent_avg_retry_count" in row and float(row["agent_avg_retry_count"]) != 0.0:
            raise VerificationError(f"{label}: nonzero retry count")


def verify_matrix(manifest_path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "cast-das-aegis-zero-retry-paper-matrix-manifest":
        raise VerificationError("unexpected manifest type")
    runtime = manifest["runtime"]
    if int(runtime["max_attempts"]) != 1 or int(runtime["retry_budget"]) != 0:
        raise VerificationError("manifest is not zero-retry")
    if bool(runtime["allow_retries"]):
        raise VerificationError("manifest enables retries")

    verify_entry(repo_root, "paper", manifest["paper"])
    for section in ("code", "policies", "artifacts"):
        for label, entry in manifest[section].items():
            verify_entry(repo_root, f"{section}.{label}", entry)

    artifact = manifest["artifacts"]
    credit = read_csv(repo_root / artifact["credit_raw"]["path"])
    scalability = read_csv(repo_root / artifact["scalability"]["path"])
    ablation = read_csv(repo_root / artifact["ablation"]["path"])
    contention = read_csv(repo_root / artifact["figure13_contention"]["path"])
    write_ratio = read_csv(repo_root / artifact["figure13_write_ratio"]["path"])
    shape = read_csv(repo_root / artifact["figure13_shape"]["path"])
    for label, rows in (
        ("scalability", scalability),
        ("ablation", ablation),
        ("figure13_contention", contention),
        ("figure13_write_ratio", write_ratio),
        ("figure13_shape", shape),
    ):
        verify_zero_retry_rows(rows, label)
    credit_run = json.loads(
        (repo_root / artifact["credit_run_manifest"]["path"]).read_text(encoding="utf-8")
    )
    if int(credit_run["max_attempts"]) != 1:
        raise VerificationError("credit run is not zero-retry")

    credit_metrics = system_metrics(credit)
    credit_speedup, credit_best = speedup(credit_metrics)
    scalability_points = {}
    for experiment in ("ycsb_medium", "ycsb_high", "tpcc_low_w100", "tpcc_high_w1"):
        for clients in (8, 24, 40):
            selected = rows_for(scalability, experiment=experiment, clients=clients)
            metrics = system_metrics(selected)
            ratio, best = speedup(metrics)
            scalability_points[f"{experiment}.c{clients}"] = {
                "aegis_tps": metrics["paper-atcc"]["tps"],
                "aegis_commit_rate": metrics["paper-atcc"]["commit_rate"],
                "aegis_p99_ms": metrics["paper-atcc"]["p99_ms"],
                "best_baseline": best,
                "speedup": ratio,
            }

    ycsb_ablation = rows_for(ablation, workload="YCSB")
    variants = sorted({row["system"] for row in ycsb_ablation})
    ablation_metrics = {
        variant: {
            "tps": fmean(rows_for(ycsb_ablation, system=variant), "agent_tps"),
            "p99_ms": fmean(
                rows_for(ycsb_ablation, system=variant), "agent_p99_latency_ms"
            ),
        }
        for variant in variants
    }

    figure13 = {
        "zipf_0": point_metrics(contention, parameter="zipf_theta", value="0.0"),
        "zipf_1.2": point_metrics(contention, parameter="zipf_theta", value="1.2"),
        "write_0.1": point_metrics(write_ratio, parameter="write_ratio", value="0.1"),
        "write_0.9": point_metrics(write_ratio, parameter="write_ratio", value="0.9"),
        "reasoning_0.25": point_metrics(shape, parameter="reasoning_scale", value="0.25"),
        "reasoning_4": point_metrics(shape, parameter="reasoning_scale", value="4.0"),
        "length_4": point_metrics(shape, parameter="transaction_length", value="4"),
        "length_24": point_metrics(shape, parameter="transaction_length", value="24"),
    }

    mechanism_checks = {
        "credit_absolute_paper_level": (
            90.0 <= credit_metrics["paper-atcc"]["tps"] <= 115.0
            and credit_metrics["paper-atcc"]["commit_rate"] >= 0.98
            and credit_metrics["paper-atcc"]["p99_ms"] <= 750.0
        ),
        "ycsb_medium_no_material_regression": (
            scalability_points["ycsb_medium.c40"]["speedup"] >= 0.95
        ),
        "tpcc_low_no_material_regression": (
            scalability_points["tpcc_low_w100.c40"]["speedup"] >= 0.95
        ),
        "tpcc_high_40_commit_rate": (
            scalability_points["tpcc_high_w1.c40"]["aegis_commit_rate"] >= 0.85
        ),
        "tpcc_high_multifold_gain": (
            scalability_points["tpcc_high_w1.c40"]["speedup"] >= 2.5
        ),
        "dynamic_dwa_improves_dynamic": (
            ablation_metrics["Dynamic + DWA"]["tps"]
            > ablation_metrics["Dynamic"]["tps"]
            and ablation_metrics["Dynamic + DWA"]["p99_ms"]
            < ablation_metrics["Dynamic"]["p99_ms"]
        ),
        "figure13_uniform_no_regression": figure13["zipf_0"]["speedup"] >= 0.98,
        "figure13_long_reasoning_gain": figure13["reasoning_4"]["speedup"] >= 2.0,
        "figure13_length_gain_increases": (
            figure13["length_24"]["speedup"] > figure13["length_4"]["speedup"]
        ),
    }
    paper_claim_checks = {
        "ycsb_high_8_speedup": scalability_points["ycsb_high.c8"]["speedup"] >= 2.7,
        "ycsb_high_24_speedup": scalability_points["ycsb_high.c24"]["speedup"] >= 3.5,
        "ycsb_high_40_speedup": scalability_points["ycsb_high.c40"]["speedup"] >= 3.2,
        "credit_relative_speedup": credit_speedup >= 1.15,
        "ablation_full_variant_best": (
            ablation_metrics["Dynamic + DWA + Priority"]["tps"]
            == max(value["tps"] for value in ablation_metrics.values())
            and ablation_metrics["Dynamic + DWA + Priority"]["p99_ms"]
            == min(value["p99_ms"] for value in ablation_metrics.values())
        ),
        "figure13_write_90_speedup": figure13["write_0.9"]["speedup"] >= 6.5,
        "figure13_length_24_speedup": figure13["length_24"]["speedup"] >= 12.0,
    }
    return {
        "manifest": str(manifest_path.resolve()),
        "zero_retry_invariants": True,
        "credit": {
            "aegis": credit_metrics["paper-atcc"],
            "best_baseline": credit_best,
            "speedup": credit_speedup,
        },
        "scalability": scalability_points,
        "ablation_ycsb": ablation_metrics,
        "figure13": figure13,
        "mechanism_checks": mechanism_checks,
        "mechanism_checks_pass": all(mechanism_checks.values()),
        "paper_claim_checks": paper_claim_checks,
        "full_paper_claims_pass": all(paper_claim_checks.values()),
        "latency_caveat": (
            "Zero-retry P99 contains successful first-attempt tasks only; failed tasks are "
            "excluded and cross-system comparisons have survivor bias."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/reproduction/zero_retry_paper_acceptance_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--require-full-paper-claims", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_matrix(args.manifest, repo_root=args.repo_root.resolve())
    except (KeyError, TypeError, ValueError, OSError, VerificationError) as exc:
        print(f"Zero-retry paper matrix verification: FAIL: {exc}")
        return 1
    print("Zero-retry paper matrix verification: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_full_paper_claims and not report["full_paper_claims_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
