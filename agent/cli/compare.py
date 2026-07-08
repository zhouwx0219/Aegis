"""Run a concurrent benchmark comparing CC strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TextIO

from agent.benchmarks import BenchmarkConfig, run_cc_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", "-w", choices=("ycsb", "tpcc"), default="ycsb")
    parser.add_argument("--level", "-l", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--workload-profile", choices=("small", "paper"), default="small")
    parser.add_argument("--cc", default="all", help="Comma-separated CC names or all.")
    parser.add_argument("--tasks", "-n", type=int, default=10)
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--retries", "-r", type=int, default=0)
    parser.add_argument(
        "--reasoning-profile",
        choices=("none", "light", "agentic", "heavy"),
        default="agentic",
    )
    parser.add_argument("--reasoning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=920104)
    parser.add_argument("--policy", type=Path, help="ATCC policy artifact for dynamic variants.")
    parser.add_argument(
        "--policy-mode",
        choices=("train", "eval", "online"),
        help="ATCC policy update mode. Defaults to eval when --policy is set, otherwise online.",
    )
    parser.add_argument("--output", "-o", type=Path)
    return parser


def run_compare(
    *,
    workload: str,
    level: str,
    cc: str,
    tasks: int,
    seed: int,
    workload_profile: str = "small",
    workers: int = 8,
    retries: int = 0,
    reasoning_profile: str = "agentic",
    reasoning_scale: float = 1.0,
    policy: Optional[Path] = None,
    policy_mode: Optional[str] = None,
) -> Dict[str, Any]:
    mode = policy_mode or ("eval" if policy else "online")
    return run_cc_benchmark(
        BenchmarkConfig(
            workload=workload,
            level=level,
            workload_profile=workload_profile,
            cc=cc,
            tasks=tasks,
            workers=workers,
            retries=retries,
            reasoning_profile=reasoning_profile,
            reasoning_scale=reasoning_scale,
            seed=seed,
            policy=policy,
            policy_mode=mode,
        )
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
) -> int:
    args = build_parser().parse_args(argv)
    report = run_compare(
        workload=args.workload,
        level=args.level,
        workload_profile=args.workload_profile,
        cc=args.cc,
        tasks=args.tasks,
        workers=args.workers,
        retries=args.retries,
        reasoning_profile=args.reasoning_profile,
        reasoning_scale=args.reasoning_scale,
        seed=args.seed,
        policy=args.policy,
        policy_mode=args.policy_mode,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    out.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
