#!/usr/bin/env python3
"""Refresh only Aegis rows in an existing bounded experiment group."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.unified_trace.run_aegis_two_hour_experiments import (
    ROOT,
    RUNNER,
    cases_for,
    export_group,
    group_names,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=group_names(), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-policy", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=0.25)
    parser.add_argument("--measure-seconds", type=float, default=1.5)
    parser.add_argument("--parameter", default="")
    parser.add_argument("--parameter-value", default="")
    parser.add_argument(
        "--fresh-results-dir",
        type=Path,
        default=None,
        help="Merge precomputed one-row Aegis CSVs instead of rerunning them.",
    )
    args = parser.parse_args()

    all_cases = cases_for(args.group)
    cases = all_cases
    if args.parameter:
        cases = [case for case in cases if case.parameter == args.parameter]
    if args.parameter_value:
        cases = [case for case in cases if case.value == args.parameter_value]
    case_dir = args.output_dir / "cases" / args.group
    trace_dir = args.output_dir / "traces" / args.group
    for index, case in enumerate(cases, 1):
        if "paper-atcc" not in case.systems.split(","):
            continue
        trace = trace_dir / f"{case.case_id}.csv"
        result = case_dir / f"{case.case_id}.csv"
        external_fresh = args.fresh_results_dir is not None
        fresh = (
            args.fresh_results_dir / f"{case.case_id}.csv"
            if external_fresh
            else case_dir / f"{case.case_id}.aegis.csv"
        )
        if external_fresh:
            if not fresh.exists():
                raise FileNotFoundError(f"missing fresh result: {fresh}")
        else:
            command = [
                sys.executable,
                str(RUNNER),
                "--trace", str(trace),
                "--output", str(fresh),
                "--cc", "paper-atcc",
                "--paper-policy", str(args.paper_policy),
                "--policy-mode", "eval",
                "--max-attempts", "1",
                "--warmup-seconds", str(args.warmup_seconds),
                "--measure-seconds", str(args.measure_seconds),
                "--disable-atcc-retry-cache",
                "--paper-switching", case.switching,
                "--paper-priority", case.priority,
                "--priority-quantum-scale", str(case.priority_scale),
            ]
            if args.warmup_seconds > 0:
                command.extend(("--warmup-trace", str(trace)))
            if case.execution_workers > 0:
                command.extend(("--execution-workers", str(case.execution_workers)))
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with result.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0]) if rows else []
        with fresh.open(newline="", encoding="utf-8-sig") as handle:
            new_row = next(csv.DictReader(handle))
        rows = [new_row if row.get("cc") == "paper-atcc" else row for row in rows]
        with result.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if not external_fresh:
            fresh.unlink()
        print(f"[{args.group}] {index}/{len(cases)} {case.case_id}", flush=True)

    export_group(
        args.group,
        all_cases,
        case_dir,
        args.output_dir / f"figure_{args.group}_raw.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
