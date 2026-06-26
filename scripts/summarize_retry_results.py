#!/usr/bin/env python3
"""Summarize ATCC retry-experiment JSON files into one CSV.

This helper intentionally imports only Python stdlib so it can run on Windows
or WSL without loading the Linux `cast_core` extension.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_json_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*.json")):
        if path.name.startswith("."):
            continue
        yield path


def _profile_name(path: Path) -> str:
    return path.stem


def _row(profile: str, aggregate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "profile": profile,
        "strategy": aggregate.get("strategy", ""),
        "policy_variant": aggregate.get("policy_variant", ""),
        "throughput": aggregate.get("committed_throughput", 0.0),
        "commit_rate": aggregate.get("commit_rate", 0.0),
        "committed_tasks": aggregate.get("committed_tasks", 0),
        "failed_tasks": aggregate.get("final_failed_tasks", 0),
        "attempts_per_task": aggregate.get("attempts_per_task", 0.0),
        "wasted_tokens_per_task": aggregate.get(
            "estimated_wasted_tokens_per_task", 0.0
        ),
        "p50_latency_s": aggregate.get("agent_latency_p50_s", 0.0),
        "p95_latency_s": aggregate.get("agent_latency_p95_s", 0.0),
        "p99_latency_s": aggregate.get("agent_latency_p99_s", 0.0),
        "conflict_aborts": aggregate.get("conflict_aborts", 0),
        "background_commits": aggregate.get("background_commits", 0),
        "background_aborts": aggregate.get("background_aborts", 0),
        "operation_policy_counts": json.dumps(
            aggregate.get("operation_policy_counts", {}), ensure_ascii=False
        ),
        "operation_rule_counts": json.dumps(
            aggregate.get("operation_rule_counts", {}), ensure_ascii=False
        ),
    }


def summarize(input_dir: Path, output: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _iter_json_files(input_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        aggregates = data.get("aggregates", [])
        if not isinstance(aggregates, list):
            continue
        for aggregate in aggregates:
            if isinstance(aggregate, dict):
                rows.append(_row(_profile_name(path), aggregate))
    if not rows:
        raise SystemExit(f"no retry-experiment aggregates found under {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.input_dir / "summary.csv")
    rows = summarize(args.input_dir, output)
    print(f"wrote {output} ({len(rows)} rows)")
    for row in rows:
        if row["strategy"] == "adaptive-op-strict":
            print(
                f"{row['profile']}: ATCC throughput={float(row['throughput']):.2f}, "
                f"commit={float(row['commit_rate']):.2%}, "
                f"attempts/task={float(row['attempts_per_task']):.2f}, "
                f"p99={float(row['p99_latency_s']):.3f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
