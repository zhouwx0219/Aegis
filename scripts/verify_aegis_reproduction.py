#!/usr/bin/env python3
"""Verify the archived small-scale Aegis reproduction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    """Raised when an archived reproduction invariant does not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(repo_root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def verify_hashed_artifact(
    repo_root: Path,
    label: str,
    entry: Mapping[str, Any],
) -> Path:
    path = resolve_artifact(repo_root, str(entry.get("path", "")))
    if not path.is_file():
        raise VerificationError(f"{label}: missing artifact: {path}")
    expected = str(entry.get("sha256", "")).strip().lower()
    actual = sha256_file(path)
    if not expected:
        raise VerificationError(f"{label}: manifest has no SHA-256")
    if actual != expected:
        raise VerificationError(
            f"{label}: SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def numeric_seed_entries(files: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    entries = []
    for value, entry in files.items():
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, Mapping):
            raise VerificationError(f"seed {seed}: invalid file entry")
        entries.append((seed, entry))
    return sorted(entries)


def read_result_rows(path: Path, *, expected_seed: int) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VerificationError(f"{path}: result CSV has no rows")
    by_cc: dict[str, dict[str, str]] = {}
    for row in rows:
        cc = str(row.get("cc", "")).strip()
        if not cc:
            continue
        if str(row.get("status", "")).strip().lower() != "ok":
            raise VerificationError(f"{path}: {cc} row is not successful")
        row_seed = int(row.get("seed", expected_seed) or expected_seed)
        if row_seed != expected_seed:
            raise VerificationError(
                f"{path}: expected seed {expected_seed}, found {row_seed}"
            )
        if cc in by_cc:
            raise VerificationError(f"{path}: duplicate {cc} row")
        by_cc[cc] = row
    return by_cc


def mean_tps_by_cc(seed_rows: Mapping[int, Mapping[str, Mapping[str, str]]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for rows in seed_rows.values():
        for cc, row in rows.items():
            values.setdefault(cc, []).append(float(row["agent_tps"]))
    seed_count = len(seed_rows)
    incomplete = sorted(cc for cc, samples in values.items() if len(samples) != seed_count)
    if incomplete:
        raise VerificationError(
            "systems missing from one or more seeds: " + ", ".join(incomplete)
        )
    return {cc: statistics.fmean(samples) for cc, samples in values.items()}


def close_enough(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)


def verify_metric(label: str, actual: float, expected: Any) -> None:
    expected_value = float(expected)
    if not close_enough(actual, expected_value):
        raise VerificationError(
            f"{label}: manifest says {expected_value:.12g}, recomputed {actual:.12g}"
        )


def load_path_rows(
    repo_root: Path,
    label: str,
    payload: Mapping[str, Any],
) -> dict[int, dict[str, dict[str, str]]]:
    seed_rows = {}
    for seed, entry in numeric_seed_entries(dict(payload.get("files", {}))):
        path = verify_hashed_artifact(repo_root, f"{label}.{seed}", entry)
        seed_rows[seed] = read_result_rows(path, expected_seed=seed)
    if not seed_rows:
        raise VerificationError(f"{label}: no per-seed result files")
    return seed_rows


def verify_speedup_path(
    repo_root: Path,
    label: str,
    payload: Mapping[str, Any],
) -> tuple[float, dict[int, dict[str, dict[str, str]]]]:
    seed_rows = load_path_rows(repo_root, label, payload)
    means = mean_tps_by_cc(seed_rows)
    if "paper-atcc" not in means:
        raise VerificationError(f"{label}: paper-atcc row is missing")
    baselines = {cc: value for cc, value in means.items() if cc != "paper-atcc"}
    if not baselines:
        raise VerificationError(f"{label}: no baseline rows")
    best_cc = max(baselines, key=baselines.__getitem__)
    speedup = means["paper-atcc"] / baselines[best_cc]
    verify_metric(f"{label}.aegis_tps_mean", means["paper-atcc"], payload["aegis_tps_mean"])
    verify_metric(
        f"{label}.best_mean_baseline_tps",
        baselines[best_cc],
        payload["best_mean_baseline_tps"],
    )
    if best_cc != str(payload.get("best_mean_baseline", "")):
        raise VerificationError(
            f"{label}.best_mean_baseline: manifest says "
            f"{payload.get('best_mean_baseline')}, recomputed {best_cc}"
        )
    verify_metric(
        f"{label}.speedup_vs_best_mean_baseline",
        speedup,
        payload["speedup_vs_best_mean_baseline"],
    )
    return speedup, seed_rows


def iter_hash_entries(manifest: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    paper = manifest.get("paper", {})
    if isinstance(paper, Mapping):
        yield "paper", paper
    training = manifest.get("training", {})
    if isinstance(training, Mapping):
        for name in ("policy", "report"):
            entry = training.get(name, {})
            if isinstance(entry, Mapping):
                yield f"training.{name}", entry
    for path_name in ("pure_policy_path",):
        payload = manifest.get(path_name, {})
        if not isinstance(payload, Mapping):
            continue
        files = payload.get("files", {})
        if not isinstance(files, Mapping):
            continue
        for seed, entry in numeric_seed_entries(files):
            yield f"{path_name}.{seed}", entry
    zero = manifest.get("zero_retry_path", {})
    if isinstance(zero, Mapping):
        files = zero.get("files", {})
        if isinstance(files, Mapping):
            for name in ("raw", "summary"):
                entry = files.get(name, {})
                if isinstance(entry, Mapping) and entry.get("path"):
                    yield f"zero_retry_path.{name}", entry


def verify_reproduction(
    manifest_path: Path,
    *,
    repo_root: Path = ROOT,
    min_guarded_speedup: float = 2.8,
    min_zero_retry_speedup: float = 2.0,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "cast-das-aegis-small-scale-reproduction-manifest":
        raise VerificationError("unexpected reproduction manifest type")

    for label, entry in iter_hash_entries(manifest):
        verify_hashed_artifact(repo_root, label, entry)

    training = dict(manifest.get("training", {}))
    evaluation = dict(manifest.get("evaluation", {}))
    training_seeds = {int(value) for value in training.get("seeds", [])}
    evaluation_seeds = {int(value) for value in evaluation.get("seeds", [])}
    overlap = sorted(training_seeds & evaluation_seeds)
    if overlap:
        raise VerificationError(f"training/evaluation seed overlap: {overlap}")
    if sorted(int(value) for value in evaluation.get("training_seed_overlap", [])) != overlap:
        raise VerificationError("manifest training_seed_overlap is inconsistent")

    policy_path = resolve_artifact(repo_root, training["policy"]["path"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    medoids_per_group = int(policy.get("medoids_per_group", 0))
    if medoids_per_group <= 0:
        raise VerificationError("compiled policy does not record medoids_per_group")

    guarded = dict(manifest.get("guarded_path", {}))
    guarded_speedup, _guarded_rows = verify_speedup_path(
        repo_root, "guarded_path", guarded
    )
    if guarded_speedup < float(min_guarded_speedup):
        raise VerificationError(
            f"guarded speedup {guarded_speedup:.4f}x is below "
            f"{float(min_guarded_speedup):.4f}x"
        )

    zero = dict(manifest.get("zero_retry_path", {}))
    if int(zero.get("max_attempts", 0)) != 1:
        raise VerificationError("zero-retry max_attempts must be 1")
    if int(zero.get("retry_budget", -1)) != 0:
        raise VerificationError("zero-retry retry_budget must be 0")
    if bool(zero.get("allow_retries", True)):
        raise VerificationError("zero-retry allow_retries must be false")
    zero_speedup, zero_rows = verify_speedup_path(repo_root, "zero_retry_path", zero)
    if zero_speedup < float(min_zero_retry_speedup):
        raise VerificationError(
            f"zero-retry speedup {zero_speedup:.4f}x is below "
            f"{float(min_zero_retry_speedup):.4f}x"
        )
    retry_counts = [
        float(row["agent_avg_retry_count"])
        for rows in zero_rows.values()
        for row in rows.values()
    ]
    if any(value != 0.0 for value in retry_counts):
        raise VerificationError("zero-retry CSV contains a nonzero average retry count")

    return {
        "manifest": str(manifest_path),
        "training_seeds": sorted(training_seeds),
        "evaluation_seeds": sorted(evaluation_seeds),
        "policy_medoids_per_group": medoids_per_group,
        "guarded_speedup": guarded_speedup,
        "pure_policy_speedup": float(
            manifest.get("pure_policy_path", {}).get(
                "speedup_vs_best_mean_baseline", 0.0
            )
        ),
        "zero_retry_speedup": zero_speedup,
        "zero_retry_all_average_retries": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/reproduction/op24_w90_experiment_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--min-guarded-speedup", type=float, default=2.8)
    parser.add_argument("--min-zero-retry-speedup", type=float, default=2.0)
    args = parser.parse_args()

    try:
        report = verify_reproduction(
            args.manifest,
            repo_root=args.repo_root.resolve(),
            min_guarded_speedup=args.min_guarded_speedup,
            min_zero_retry_speedup=args.min_zero_retry_speedup,
        )
    except (KeyError, TypeError, ValueError, OSError, VerificationError) as exc:
        print(f"Aegis reproduction verification: FAIL: {exc}")
        return 1

    print("Aegis reproduction verification: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
