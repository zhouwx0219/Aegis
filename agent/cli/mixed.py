"""Run mixed agent/background starvation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, TextIO

from agent.benchmarks import MixedBenchmarkConfig, run_mixed_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", "-w", choices=("ycsb", "tpcc"), default="tpcc")
    parser.add_argument("--level", "-l", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--workload-profile", choices=("small", "paper"), default="small")
    parser.add_argument("--cc", default="occ,dynamic-atcc")
    parser.add_argument("--duration", "-d", type=float, default=3.0)
    parser.add_argument("--clients", "-c", type=int, default=0, help="Total clients. When set, derives agents/background from --agent-ratio.")
    parser.add_argument("--agent-ratio", type=float, default=0.80)
    parser.add_argument("--agents", "-a", type=int, default=2)
    parser.add_argument("--background", "-b", type=int, default=8)
    parser.add_argument(
        "--reasoning-profile",
        choices=("none", "light", "agentic", "heavy"),
        default="agentic",
    )
    parser.add_argument("--reasoning-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=920104)
    parser.add_argument("--retries", "-r", type=int, default=0)
    parser.add_argument("--retry-until-commit", action="store_true")
    parser.add_argument("--max-attempts-per-task", type=int, default=100)
    parser.add_argument("--agent-retry-backoff-ms", default="500,5000")
    parser.add_argument("--background-retry-backoff-ms", default="10,30")
    parser.add_argument("--tokens-per-operation", type=int, default=2703)
    parser.add_argument("--background-wait", action="store_true")
    parser.add_argument("--background-mode", choices=("hotspot", "procedure"), default="hotspot")
    parser.add_argument("--reservation-ttl-s", type=float, default=5.0)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--policy-mode",
        choices=("train", "eval", "online"),
        help="ATCC policy update mode. Defaults to eval when --policy is set, otherwise online.",
    )
    parser.add_argument("--output", "-o", type=Path)
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
) -> int:
    args = build_parser().parse_args(argv)
    policy_mode = args.policy_mode or ("eval" if args.policy else "online")
    agent_backoff = parse_range(args.agent_retry_backoff_ms, field="agent-retry-backoff-ms")
    background_backoff = parse_range(args.background_retry_backoff_ms, field="background-retry-backoff-ms")
    report = run_mixed_benchmark(
        MixedBenchmarkConfig(
            workload=args.workload,
            level=args.level,
            workload_profile=args.workload_profile,
            cc=args.cc,
            duration_s=args.duration,
            clients=args.clients,
            agent_ratio=args.agent_ratio,
            agent_workers=args.agents,
            background_workers=args.background,
            reasoning_profile=args.reasoning_profile,
            reasoning_scale=args.reasoning_scale,
            seed=args.seed,
            retries=args.retries,
            retry_until_commit=args.retry_until_commit,
            max_attempts_per_task=args.max_attempts_per_task,
            agent_retry_backoff_min_ms=agent_backoff[0],
            agent_retry_backoff_max_ms=agent_backoff[1],
            background_retry_backoff_min_ms=background_backoff[0],
            background_retry_backoff_max_ms=background_backoff[1],
            tokens_per_operation=args.tokens_per_operation,
            background_wait=args.background_wait,
            background_mode=args.background_mode,
            reservation_ttl_s=args.reservation_ttl_s,
            policy=args.policy,
            policy_mode=policy_mode,
        )
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


def parse_range(value: str, *, field: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{field} must be min,max")
    low, high = (int(parts[0]), int(parts[1]))
    if low < 0 or high < 0 or low > high:
        raise ValueError(f"{field} must be non-negative min<=max")
    return low, high


if __name__ == "__main__":
    raise SystemExit(main())
