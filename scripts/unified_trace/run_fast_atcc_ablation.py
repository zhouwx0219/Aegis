#!/usr/bin/env python3
"""Run one-seed orthogonal ATCC ablations on an existing fixed trace."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "unified_trace" / "run_castdas_trace_fair.py"


def variants(
    experiment: str, *, switching_delayed_write: bool = True
) -> tuple[dict[str, str | bool], ...]:
    if experiment == "switching-priority":
        return (
            {"label": "Static", "switching": "static", "priority": "disabled", "delay": switching_delayed_write},
            {"label": "Static + Priority", "switching": "static", "priority": "enabled", "delay": switching_delayed_write},
            {"label": "Dynamic", "switching": "dynamic", "priority": "disabled", "delay": switching_delayed_write},
            {"label": "Dynamic + Priority", "switching": "dynamic", "priority": "enabled", "delay": switching_delayed_write},
        )
    if experiment == "dwa-priority":
        return (
            {"label": "Static", "switching": "static", "priority": "disabled", "delay": False},
            {"label": "Static + DWA", "switching": "static", "priority": "disabled", "delay": True},
            {"label": "Static + DWA + Priority", "switching": "static", "priority": "enabled", "delay": True},
            {"label": "Dynamic", "switching": "dynamic", "priority": "disabled", "delay": False},
            {"label": "Dynamic + DWA", "switching": "dynamic", "priority": "disabled", "delay": True},
            {"label": "Dynamic + DWA + Priority", "switching": "dynamic", "priority": "enabled", "delay": True},
        )
    return (
        {"label": "Static", "switching": "static", "priority": "disabled", "delay": False},
        {"label": "Static + Delayed Write Apply", "switching": "static", "priority": "disabled", "delay": True},
        {"label": "Dynamic", "switching": "dynamic", "priority": "disabled", "delay": False},
        {"label": "Dynamic + Delayed Write Apply", "switching": "dynamic", "priority": "disabled", "delay": True},
    )


def slug(value: str) -> str:
    return "_".join(value.lower().replace("+", " ").split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("switching-priority", "delayed-write", "dwa-priority"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warmup-trace", type=Path, default=None)
    parser.add_argument("--paper-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=0.25)
    parser.add_argument("--measure-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--priority-quantum-scale", type=float, default=1.0)
    parser.add_argument("--static-conflict-threshold", type=float, default=0.20)
    parser.add_argument("--static-protection-mask", type=int, default=4)
    parser.add_argument(
        "--switching-delayed-write",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    selected_variants = variants(
        args.experiment,
        switching_delayed_write=args.switching_delayed_write == "enabled",
    )
    for variant in selected_variants:
        output = args.output_dir / f"{slug(str(variant['label']))}.csv"
        command = [
            sys.executable,
            str(RUNNER),
            "--trace", str(args.trace),
            "--output", str(output),
            "--cc", "paper-atcc",
            "--paper-policy", str(args.paper_policy),
            "--policy-mode", "eval",
            "--paper-switching", str(variant["switching"]),
            "--paper-priority", str(variant["priority"]),
            "--priority-quantum-scale", str(args.priority_quantum_scale),
            "--paper-static-conflict-threshold", str(args.static_conflict_threshold),
            "--paper-static-protection-mask", str(args.static_protection_mask),
            "--paper-delayed-write-apply", "enabled" if variant["delay"] else "disabled",
            "--disable-atcc-retry-cache",
            "--max-attempts", str(args.max_attempts),
            "--warmup-seconds", str(args.warmup_seconds),
            "--measure-seconds", str(args.measure_seconds),
        ]
        if args.warmup_trace is not None:
            command.extend(("--warmup-trace", str(args.warmup_trace)))
        if args.max_attempts > 1:
            command.append("--allow-retries")
        if args.experiment in {"delayed-write", "dwa-priority"}:
            # DWA measures how long an immediate WLock covers the Agent's
            # post-write reasoning suffix. Deferred replay moves that suffix
            # before begin(), eliminating the mechanism under test.
            command.append("--disable-paper-deferred-replay")
        if variant["priority"] == "enabled" and args.experiment == "switching-priority":
            command.append("--paper-commit-admission-priority")
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with output.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        row.update(
            experiment=args.experiment,
            variant=variant["label"],
            switching=variant["switching"],
            priority=variant["priority"],
            delayed_write_apply="enabled" if variant["delay"] else "disabled",
        )
        combined.append(row)
        print(
            f"{variant['label']}: TPS={row.get('agent_tps')} "
            f"commit={row.get('agent_commit_rate')} "
            f"P99={row.get('agent_p99_latency_ms')}",
            flush=True,
        )

    summary = args.output_dir / f"{args.experiment}_raw.csv"
    fields = list(combined[0])
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
