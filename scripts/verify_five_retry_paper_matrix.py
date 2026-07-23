#!/usr/bin/env python3
"""Verify the archived five-retry small-scale Aegis paper matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
AEGIS = "paper-atcc"
BASELINES = ("2pl-wait-die", "bamboo", "silo", "polaris")


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_paths(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from iter_paths(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from iter_paths(nested)


def archive_paths(manifest: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    paper = manifest.get("paper", {})
    if isinstance(paper, Mapping):
        values.append(str(paper["path"]))
    for section in ("code", "policies", "artifacts"):
        values.extend(iter_paths(manifest.get(section, {})))
    return sorted(set(values))


def archive_sha256(repo_root: Path, paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(set(str(value).replace("\\", "/") for value in paths)):
        path = repo_root / relative
        if not path.is_file():
            raise VerificationError(f"missing archived file: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def read_csv(repo_root: Path, relative: str) -> list[dict[str, str]]:
    path = repo_root / str(relative)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VerificationError(f"{path}: empty CSV")
    return rows


def verify_five_retry_rows(rows: Iterable[Mapping[str, str]], label: str) -> None:
    for index, row in enumerate(rows, 2):
        if str(row.get("status", "")).strip().lower() != "ok":
            raise VerificationError(f"{label}:{index}: unsuccessful row")
        if int(float(row.get("max_attempts", 0) or 0)) != 6:
            raise VerificationError(f"{label}:{index}: max_attempts is not 6")
        if int(float(row.get("retry_budget", -1) or -1)) != 5:
            raise VerificationError(f"{label}:{index}: retry_budget is not 5")


def fmean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def selected(
    rows: Iterable[dict[str, str]],
    *,
    cc: str | None = None,
    clients: int | None = None,
    parameter: str | None = None,
    value: str | None = None,
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if cc is not None and row.get("cc") != cc:
            continue
        if clients is not None and int(float(row.get("clients", -1))) != clients:
            continue
        if parameter is not None and row.get("parameter") != parameter:
            continue
        if value is not None and row.get("parameter_value") != value:
            continue
        output.append(row)
    return output


def system_metrics(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    if not materialized:
        raise VerificationError("metric selection produced no rows")
    systems = sorted({row["cc"] for row in materialized})
    metrics: dict[str, dict[str, float]] = {}
    for cc in systems:
        values = selected(materialized, cc=cc)
        commit_field = (
            "agent_commit_rate"
            if "agent_commit_rate" in values[0]
            else "commit_rate"
        )
        p99_field = (
            "agent_p99_latency_ms"
            if "agent_p99_latency_ms" in values[0]
            else "p99_latency_ms"
        )
        metrics[cc] = {
            "tps": fmean(values, "agent_tps"),
            "commit_rate": fmean(values, commit_field),
            "p99_ms": fmean(values, p99_field),
        }
    return metrics


def combine_point(
    baseline_rows: Iterable[dict[str, str]],
    aegis_rows: Iterable[dict[str, str]],
) -> dict[str, dict[str, float]]:
    baselines = [row for row in baseline_rows if row.get("cc") in BASELINES]
    aegis = [row for row in aegis_rows if row.get("cc") == AEGIS]
    metrics = system_metrics([*baselines, *aegis])
    missing = [cc for cc in (*BASELINES, AEGIS) if cc not in metrics]
    if missing:
        raise VerificationError("point is missing systems: " + ", ".join(missing))
    return metrics


def point_report(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    best = max(BASELINES, key=lambda cc: metrics[cc]["tps"])
    aegis_tps = float(metrics[AEGIS]["tps"])
    return {
        "aegis": dict(metrics[AEGIS]),
        "best_baseline": best,
        "best_baseline_tps": float(metrics[best]["tps"]),
        "speedup_vs_best": aegis_tps / float(metrics[best]["tps"]),
        "speedup_vs_2pl": aegis_tps / float(metrics["2pl-wait-die"]["tps"]),
    }


def load_point(
    repo_root: Path,
    baseline_path: str,
    aegis_paths: str | Iterable[str],
    *,
    clients: int | None = None,
    parameter: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    baseline = selected(
        read_csv(repo_root, baseline_path),
        clients=clients,
        parameter=parameter,
        value=value,
    )
    paths = [aegis_paths] if isinstance(aegis_paths, str) else list(aegis_paths)
    aegis: list[dict[str, str]] = []
    for path in paths:
        aegis.extend(
            selected(
                read_csv(repo_root, path),
                cc=AEGIS,
                clients=clients,
                parameter=parameter,
                value=value,
            )
        )
    return point_report(combine_point(baseline, aegis))


def verify_matrix(manifest_path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "cast-das-aegis-five-retry-paper-matrix-manifest":
        raise VerificationError("unexpected manifest type")
    runtime = manifest["runtime"]
    if (
        int(runtime["max_attempts"]) != 6
        or int(runtime["retry_budget"]) != 5
        or not bool(runtime["allow_retries"])
    ):
        raise VerificationError("manifest does not use five-retry paper semantics")

    paper_path = repo_root / str(manifest["paper"]["path"])
    if sha256_file(paper_path) != str(manifest["paper"]["sha256"]).lower():
        raise VerificationError("paper SHA-256 mismatch")
    actual_archive_hash = archive_sha256(repo_root, archive_paths(manifest))
    if actual_archive_hash != str(manifest["archive_sha256"]).lower():
        raise VerificationError(
            "archive SHA-256 mismatch: "
            f"expected {manifest['archive_sha256']}, got {actual_archive_hash}"
        )

    csv_paths = sorted(
        {
            path
            for path in iter_paths(manifest["artifacts"])
            if str(path).lower().endswith(".csv")
        }
    )
    for path in csv_paths:
        verify_five_retry_rows(read_csv(repo_root, path), path)

    artifacts = manifest["artifacts"]
    medium = artifacts["ycsb_medium"]
    ycsb_medium: dict[str, Any] = {}
    for clients in (8, 16, 32):
        ycsb_medium[str(clients)] = load_point(
            repo_root,
            medium["baseline_short"],
            medium["aegis_short"][str(clients)],
            clients=clients,
        )
    for clients in (24, 40):
        ycsb_medium[str(clients)] = load_point(
            repo_root,
            medium["baseline_steady"][str(clients)],
            medium["aegis_steady"][str(clients)],
            clients=clients,
        )

    high = artifacts["ycsb_high"]
    ycsb_high = {
        str(clients): load_point(
            repo_root,
            high["baseline"][str(clients)],
            high["aegis"][str(clients)],
            clients=clients,
        )
        for clients in (8, 16, 24, 32, 40)
    }

    tpcc_low_artifacts = artifacts["tpcc_low"]
    tpcc_low = {
        str(clients): load_point(
            repo_root,
            tpcc_low_artifacts["baseline"],
            tpcc_low_artifacts["aegis"][str(clients)],
            clients=clients,
        )
        for clients in (8, 16, 24, 32, 40)
    }
    tpcc_low_steady_metrics = system_metrics(
        [
            *read_csv(repo_root, tpcc_low_artifacts["steady_baseline"]),
            *read_csv(repo_root, tpcc_low_artifacts["steady_aegis"]),
        ]
    )
    tpcc_low["40_steady"] = {
        "aegis": tpcc_low_steady_metrics[AEGIS],
        "silo": tpcc_low_steady_metrics["silo"],
        "speedup_vs_silo": (
            tpcc_low_steady_metrics[AEGIS]["tps"]
            / tpcc_low_steady_metrics["silo"]["tps"]
        ),
    }

    tpcc_high_artifacts = artifacts["tpcc_high"]
    tpcc_high = {
        str(clients): load_point(
            repo_root,
            tpcc_high_artifacts["baseline"],
            tpcc_high_artifacts["aegis"][str(clients)],
            clients=clients,
        )
        for clients in (8, 16, 24, 32, 40)
    }

    credit_metrics = system_metrics(read_csv(repo_root, artifacts["credit"]))
    credit = point_report(credit_metrics)

    ablation_rows = read_csv(repo_root, artifacts["ablation"])
    variants: dict[str, dict[str, float]] = {}
    for name in sorted({row["system"] for row in ablation_rows}):
        rows = [row for row in ablation_rows if row["system"] == name]
        variants[name] = {
            "tps": fmean(rows, "agent_tps"),
            "p99_ms": fmean(rows, "agent_p99_latency_ms"),
        }
    full_name = "Dynamic + DWA + Priority"

    figure = artifacts["figure13"]
    figure13 = {
        "zipf_0": load_point(
            repo_root, figure["zipf_0"], figure["zipf_0"],
            parameter="zipf_theta", value="0.0",
        ),
        "zipf_1.2": load_point(
            repo_root, figure["zipf_1.2"], figure["zipf_1.2"],
            parameter="zipf_theta", value="1.2",
        ),
        "write_0.1": load_point(
            repo_root, figure["write_0.1"], figure["write_0.1"],
            parameter="write_ratio", value="0.1",
        ),
        "write_0.9": load_point(repo_root, figure["write_0.9"], figure["write_0.9"]),
        "reasoning_0.25": load_point(
            repo_root, figure["shape_low"], figure["shape_low"],
            parameter="reasoning_scale", value="0.25",
        ),
        "reasoning_4": load_point(
            repo_root, figure["reasoning_4"], figure["reasoning_4"]
        ),
        "length_4": load_point(
            repo_root, figure["shape_low"], figure["shape_low"],
            parameter="transaction_length", value="4",
        ),
        "length_24": load_point(
            repo_root, figure["length_24_baseline"], figure["length_24_aegis"]
        ),
        "length_24_steady": load_point(
            repo_root,
            figure["length_24_steady_baseline"],
            figure["length_24_steady_aegis"],
        ),
    }

    tolerance = float(manifest["acceptance"]["paper_ratio_relative_tolerance"])
    checks = {
        "figure7_medium_comparable": min(
            point["speedup_vs_best"] for point in ycsb_medium.values()
        ) >= 0.90,
        "figure8_high_8": (
            ycsb_high["8"]["speedup_vs_best"] >= 2.93 * (1.0 - tolerance)
        ),
        "figure8_high_24": (
            ycsb_high["24"]["speedup_vs_best"] >= 3.94 * (1.0 - tolerance)
        ),
        "figure8_high_40": (
            ycsb_high["40"]["speedup_vs_best"] >= 3.52 * (1.0 - tolerance)
        ),
        "figure9_low_tpcc_comparable": min(
            point["speedup_vs_best"]
            for label, point in tpcc_low.items()
            if label != "40_steady"
        ) >= 0.89,
        "figure9_low_tpcc_commit": (
            tpcc_low["40_steady"]["aegis"]["commit_rate"] >= 0.95
        ),
        "figure10_high_tpcc_8_tps": (
            tpcc_high["8"]["aegis"]["tps"] >= 21.6 * (1.0 - tolerance)
        ),
        "figure10_high_tpcc_24_tps": (
            tpcc_high["24"]["aegis"]["tps"] >= 50.2 * (1.0 - tolerance)
        ),
        "figure10_high_tpcc_24_speedup": (
            tpcc_high["24"]["speedup_vs_best"] >= 3.59 * (1.0 - tolerance)
        ),
        "figure10_high_tpcc_40_tps": (
            tpcc_high["40"]["aegis"]["tps"] >= 53.1 * (1.0 - tolerance)
        ),
        "figure10_high_tpcc_40_commit": (
            tpcc_high["40"]["aegis"]["commit_rate"] >= 0.877
        ),
        "figure11_credit_tps": credit["aegis"]["tps"] >= 103.5 * (1.0 - tolerance),
        "figure11_credit_speedup": credit["speedup_vs_best"] >= 1.24,
        "figure11_credit_commit": credit["aegis"]["commit_rate"] >= 0.98,
        "figure11_credit_p99": credit["aegis"]["p99_ms"] <= 640.0 * 1.10,
        "figure12_full_highest_tps": variants[full_name]["tps"]
        == max(value["tps"] for value in variants.values()),
        "figure12_full_lowest_p99": variants[full_name]["p99_ms"]
        == min(value["p99_ms"] for value in variants.values()),
        "figure13_uniform_no_regression": figure13["zipf_0"]["speedup_vs_best"] >= 0.95,
        "figure13_contention_gain_increases": (
            figure13["zipf_1.2"]["speedup_vs_best"]
            > figure13["zipf_0"]["speedup_vs_best"]
        ),
        "figure13_write_gain_increases": (
            figure13["write_0.9"]["speedup_vs_best"]
            > figure13["write_0.1"]["speedup_vs_best"]
        ),
        "figure13_write_90_peak": (
            figure13["write_0.9"]["speedup_vs_2pl"]
            >= 7.36 * (1.0 - tolerance)
        ),
        "figure13_reasoning_gain_increases": (
            figure13["reasoning_4"]["speedup_vs_best"]
            > figure13["reasoning_0.25"]["speedup_vs_best"]
        ),
        "figure13_reasoning_4_tps": (
            figure13["reasoning_4"]["aegis"]["tps"] >= 43.1 * (1.0 - tolerance)
        ),
        "figure13_reasoning_4_competitor": (
            figure13["reasoning_4"]["best_baseline_tps"] <= 10.8 * 1.05
        ),
        "figure13_length_gain_increases": (
            figure13["length_24"]["speedup_vs_best"]
            > figure13["length_4"]["speedup_vs_best"]
        ),
        "figure13_length_24_peak": (
            figure13["length_24"]["speedup_vs_2pl"]
            >= 13.43 * (1.0 - tolerance)
        ),
    }
    return {
        "manifest": str(manifest_path.resolve()),
        "archive_sha256": actual_archive_hash,
        "runtime": dict(runtime),
        "ycsb_medium": ycsb_medium,
        "ycsb_high": ycsb_high,
        "tpcc_low": tpcc_low,
        "tpcc_high": tpcc_high,
        "credit": credit,
        "ablation": variants,
        "figure13": figure13,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "comparison_note": (
            "Figure 13 reports both speedup_vs_best and speedup_vs_2pl. "
            "The paper peak checks use the latter; no best-baseline ratio is hidden."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/reproduction/five_retry_paper_acceptance_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_matrix(args.manifest, repo_root=args.repo_root.resolve())
    except (KeyError, TypeError, ValueError, OSError, VerificationError) as exc:
        print(f"Five-retry paper matrix verification: FAIL: {exc}")
        return 1
    print("Five-retry paper matrix verification: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all and not report["all_checks_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
