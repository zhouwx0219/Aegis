"""Profile-suite runner for migrated phase-aware ATCC experiments.

This module turns the manual ATCC migration experiments into one reproducible
workflow: train a phase-aware policy table, evaluate OCC/2PL/ATCC on fixed
agent-like YCSB/TPC-C profiles, and emit JSON plus a compact Markdown table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

from agent.evaluation.atcc_policy_training import train_phase_atcc_policy
from agent.evaluation.atcc_retry_experiment import (
    aggregate_retry_runs,
    run_retry_matrix,
)
from agent.evaluation.atcc_schema import (
    atcc_artifact_schema_status,
    atcc_state_schema,
)
from agent.workloads import AgentWorkload, TPCCConfig, YCSBConfig, build_agent_workload


@dataclasses.dataclass(frozen=True)
class ATCCProfileSpec:
    name: str
    workload_kind: str
    description: str
    config: Mapping[str, Any]


PROFILE_SPECS: Tuple[ATCCProfileSpec, ...] = (
    ATCCProfileSpec(
        name="ycsb-low",
        workload_kind="ycsb",
        description="Read-heavy uniform YCSB profile.",
        config={
            "record_count": 128,
            "field_count": 4,
            "requests_per_task": 10,
            "candidates_per_task": 4,
            "read_weight": 0.95,
            "update_weight": 0.05,
            "zipf_theta": 0.0,
            "hotspot_fraction": 0.0,
            "hotspot_access_probability": 0.0,
        },
    ),
    ATCCProfileSpec(
        name="ycsb-medium",
        workload_kind="ycsb",
        description="Read-heavy YCSB profile with moderate Zipf skew.",
        config={
            "record_count": 128,
            "field_count": 4,
            "requests_per_task": 10,
            "candidates_per_task": 4,
            "read_weight": 0.90,
            "update_weight": 0.10,
            "zipf_theta": 0.7,
            "hotspot_fraction": 0.10,
            "hotspot_access_probability": 0.50,
        },
    ),
    ATCCProfileSpec(
        name="ycsb-high",
        workload_kind="ycsb",
        description="Mixed read/update YCSB profile with high Zipf skew.",
        config={
            "record_count": 128,
            "field_count": 4,
            "requests_per_task": 10,
            "candidates_per_task": 4,
            "read_weight": 0.50,
            "update_weight": 0.50,
            "zipf_theta": 0.99,
            "hotspot_fraction": 0.10,
            "hotspot_access_probability": 0.75,
        },
    ),
    ATCCProfileSpec(
        name="tpcc-low",
        workload_kind="tpcc",
        description="TPC-C NewOrder spread across warehouses and districts.",
        config={
            "warehouses": 8,
            "districts_per_warehouse": 2,
            "customers_per_district": 16,
            "items": 64,
            "initial_stock": 1000,
            "order_lines": 8,
            "candidates_per_task": 4,
            "transaction_mix": (("new_order", 1.0),),
        },
    ),
    ATCCProfileSpec(
        name="tpcc-medium",
        workload_kind="tpcc",
        description="TPC-C NewOrder with moderate district-counter contention.",
        config={
            "warehouses": 2,
            "districts_per_warehouse": 2,
            "customers_per_district": 32,
            "items": 128,
            "initial_stock": 1000,
            "order_lines": 8,
            "candidates_per_task": 4,
            "transaction_mix": (("new_order", 1.0),),
        },
    ),
    ATCCProfileSpec(
        name="tpcc-high",
        workload_kind="tpcc",
        description="TPC-C NewOrder concentrated on a small hotspot.",
        config={
            "warehouses": 1,
            "districts_per_warehouse": 2,
            "customers_per_district": 32,
            "items": 128,
            "initial_stock": 1000,
            "order_lines": 8,
            "candidates_per_task": 4,
            "transaction_mix": (("new_order", 1.0),),
        },
    ),
)

PROFILE_BY_NAME = {profile.name: profile for profile in PROFILE_SPECS}
DEFAULT_TRAINING_PROFILE = {
    "ycsb": "ycsb-high",
    "tpcc": "tpcc-high",
}
CORE_EVAL_STRATEGIES = ("occ", "2pl", "adaptive-op-strict")
FULL_EVAL_STRATEGIES = (
    "occ",
    "2pl-nowait",
    "2pl-wait-die",
    "mvcc-full",
    "silo-full",
    "tictoc-full",
    "adaptive-op-strict",
)


def run_profile_suite(
    *,
    profiles: Iterable[str] = (),
    output_dir: Path = Path("results/phase_atcc_profiles"),
    strategy_set: str = "core",
    train_per_profile: bool = False,
    train_episodes: int = 3,
    train_task_count: int = 500,
    eval_task_count: int = 500,
    eval_repeats: int = 3,
    seed: int = 0,
    workers: int = 16,
    agent_slots: int = 4,
    agent_admission_mode: str = "planning-only",
    planning_delay_s: float = 0.010,
    abort_retry_delay_s: float = 0.0,
    latency_distribution: str = "lognormal",
    latency_cv: float = 0.8,
    latency_max_s: float = 0.080,
    max_attempts: int = 6,
    tokens_per_operation: float = 2703.0,
    background_workers: int = 4,
    background_interval_s: float = 0.002,
    background_strategy: str = "occ",
    object_lock_scheduler: str = "race",
    object_lock_priority_burst: int = 2,
    prelock_wait_budget_s: float = 0.0,
    prelock_wait_budget_mode: str = "transaction",
    prelock_lease_mode: str = "hold",
    agent_execution_mode: str = "legacy",
    snapshot_timing: str = "before-planning",
    profile_eval_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    selected = _select_profiles(profiles)
    if train_episodes <= 0:
        raise ValueError("train_episodes must be positive")
    if train_task_count <= 0 or eval_task_count <= 0:
        raise ValueError("task counts must be positive")
    if eval_repeats <= 0:
        raise ValueError("eval_repeats must be positive")
    eval_strategies = _eval_strategies(strategy_set)

    output_dir = Path(output_dir)
    artifacts: Dict[str, Dict[str, Any]] = {}
    artifact_paths: Dict[str, Path] = {}
    training_specs = _training_specs(selected, train_per_profile=train_per_profile)
    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)

    for train_spec in training_specs:
        workload = _build_workload(train_spec)
        artifact = train_phase_atcc_policy(
            workload,
            workload_kind=train_spec.workload_kind,
            workload_config=dict(dataclasses.asdict(workload.config)),
            episodes=train_episodes,
            task_count=train_task_count,
            seed=seed,
            workers=workers,
            agent_slots=agent_slots,
            agent_admission_mode=agent_admission_mode,
            planning_delay_s=planning_delay_s,
            latency_distribution=latency_distribution,
            latency_cv=latency_cv,
            latency_max_s=latency_max_s,
            max_attempts=max_attempts,
            tokens_per_operation=tokens_per_operation,
            background_workers=background_workers,
            background_interval_s=background_interval_s,
            background_strategy=background_strategy,
            object_lock_scheduler=object_lock_scheduler,
            object_lock_priority_burst=object_lock_priority_burst,
            prelock_wait_budget_s=prelock_wait_budget_s,
            prelock_wait_budget_mode=prelock_wait_budget_mode,
            prelock_lease_mode=prelock_lease_mode,
            agent_execution_mode=agent_execution_mode,
            snapshot_timing=snapshot_timing,
        )
        key = _artifact_key(train_spec, train_per_profile=train_per_profile)
        artifacts[key] = artifact
        artifact_path = output_dir / f"phase_atcc_{train_spec.name}_policy.json"
        artifact_paths[key] = artifact_path
        if write_files:
            _write_json(artifact_path, artifact)

    profile_reports = []
    eval_overrides = {
        str(name): dict(value)
        for name, value in dict(profile_eval_overrides or {}).items()
    }
    for spec in selected:
        profile_override = dict(eval_overrides.get(spec.name, {}) or {})
        workload_config_override = dict(
            profile_override.get("workload_config", {}) or {}
        )
        eval_spec = _with_config_override(spec, workload_config_override)
        workload = _build_workload(eval_spec)
        key = _artifact_key(spec, train_per_profile=train_per_profile)
        artifact = artifacts[key]
        eval_planning_delay_s = float(
            profile_override.get("planning_delay_s", planning_delay_s)
        )
        eval_abort_retry_delay_s = float(
            profile_override.get("abort_retry_delay_s", abort_retry_delay_s)
        )
        runs = run_retry_matrix(
            workload,
            eval_strategies,
            workload_kind=eval_spec.workload_kind,
            policy_variant="phase-rl",
            task_count=eval_task_count,
            seed=seed,
            repeats=eval_repeats,
            workers=workers,
            agent_slots=agent_slots,
            agent_admission_mode=agent_admission_mode,
            planning_delay_s=eval_planning_delay_s,
            abort_retry_delay_s=eval_abort_retry_delay_s,
            latency_distribution=latency_distribution,
            latency_cv=latency_cv,
            latency_max_s=latency_max_s,
            max_attempts=max_attempts,
            tokens_per_operation=tokens_per_operation,
            policy_artifact=artifact,
            policy_epsilon=0.0,
            background_workers=background_workers,
            background_interval_s=background_interval_s,
            background_strategy=background_strategy,
            object_lock_scheduler=object_lock_scheduler,
            object_lock_priority_burst=object_lock_priority_burst,
            prelock_wait_budget_s=prelock_wait_budget_s,
            prelock_wait_budget_mode=prelock_wait_budget_mode,
            prelock_lease_mode=prelock_lease_mode,
            agent_execution_mode=agent_execution_mode,
            snapshot_timing=snapshot_timing,
        )
        aggregates = aggregate_retry_runs(runs)
        eval_report = {
            "artifact_type": "phase-aware-atcc-profile-evaluation",
            "atcc_state_schema": atcc_state_schema(),
            "profile": spec.name,
            "description": spec.description,
            "workload": workload.name,
            "workload_kind": eval_spec.workload_kind,
            "workload_config": dict(dataclasses.asdict(workload.config)),
            "strategies": list(eval_strategies),
            "policy_variant": "phase-rl",
            "policy_artifact": str(artifact_paths[key]),
            "policy_artifact_schema": atcc_artifact_schema_status(artifact),
            "task_count": eval_task_count,
            "seed": seed,
            "repeats": eval_repeats,
            "workers": workers,
            "agent_slots": agent_slots,
            "agent_admission_mode": agent_admission_mode,
            "planning_delay_s": eval_planning_delay_s,
            "abort_retry_delay_s": eval_abort_retry_delay_s,
            "latency_distribution": latency_distribution,
            "latency_cv": latency_cv,
            "latency_max_s": latency_max_s,
            "max_attempts": max_attempts,
            "tokens_per_operation": tokens_per_operation,
            "background_workers": background_workers,
            "background_interval_s": background_interval_s,
            "background_strategy": background_strategy,
            "object_lock_scheduler": object_lock_scheduler,
            "object_lock_priority_burst": object_lock_priority_burst,
            "prelock_wait_budget_s": prelock_wait_budget_s,
            "prelock_wait_budget_mode": prelock_wait_budget_mode,
            "prelock_lease_mode": prelock_lease_mode,
            "agent_execution_mode": agent_execution_mode,
            "snapshot_timing": snapshot_timing,
            "runs": [run.to_dict() for run in runs],
            "aggregates": aggregates,
            "comparisons": _comparisons(aggregates),
        }
        eval_path = output_dir / f"phase_atcc_{spec.name}_eval.json"
        if write_files:
            _write_json(eval_path, eval_report)
        profile_reports.append(
            {
                "profile": spec.name,
                "description": spec.description,
                "workload_kind": eval_spec.workload_kind,
                "workload_config": eval_report["workload_config"],
                "eval_overrides": profile_override,
                "policy_artifact": str(artifact_paths[key]),
                "policy_artifact_schema": eval_report["policy_artifact_schema"],
                "evaluation": str(eval_path),
                "aggregates": aggregates,
                "comparisons": eval_report["comparisons"],
            }
        )

    report = {
        "artifact_type": "phase-aware-atcc-profile-suite",
        "artifact_version": 2,
        "source_system": "data-agent-runtime",
        "training_method": "offline-simulation-tabular-q-learning",
        "atcc_state_schema": atcc_state_schema(),
        "agent_admission_mode": agent_admission_mode,
        "object_lock_scheduler": object_lock_scheduler,
        "object_lock_priority_burst": object_lock_priority_burst,
        "prelock_wait_budget_s": prelock_wait_budget_s,
        "prelock_wait_budget_mode": prelock_wait_budget_mode,
        "prelock_lease_mode": prelock_lease_mode,
        "agent_execution_mode": agent_execution_mode,
        "snapshot_timing": snapshot_timing,
        "profiles": profile_reports,
        "config": {
            "strategy_set": str(strategy_set),
            "train_per_profile": train_per_profile,
            "train_episodes": train_episodes,
            "train_task_count": train_task_count,
            "eval_task_count": eval_task_count,
            "eval_repeats": eval_repeats,
            "seed": seed,
            "workers": workers,
            "agent_slots": agent_slots,
            "agent_admission_mode": agent_admission_mode,
            "planning_delay_s": planning_delay_s,
            "abort_retry_delay_s": abort_retry_delay_s,
            "latency_distribution": latency_distribution,
            "latency_cv": latency_cv,
            "latency_max_s": latency_max_s,
            "max_attempts": max_attempts,
            "tokens_per_operation": tokens_per_operation,
            "background_workers": background_workers,
            "background_interval_s": background_interval_s,
            "background_strategy": background_strategy,
            "object_lock_scheduler": object_lock_scheduler,
            "object_lock_priority_burst": object_lock_priority_burst,
            "prelock_wait_budget_s": prelock_wait_budget_s,
            "prelock_wait_budget_mode": prelock_wait_budget_mode,
            "prelock_lease_mode": prelock_lease_mode,
            "agent_execution_mode": agent_execution_mode,
            "snapshot_timing": snapshot_timing,
            "strategies": list(eval_strategies),
            "profile_eval_overrides": eval_overrides,
        },
    }
    report["markdown"] = render_markdown_report(report)
    if write_files:
        _write_json(output_dir / "phase_atcc_profile_suite.json", report)
        (output_dir / "phase_atcc_profile_suite.md").write_text(
            report["markdown"] + "\n",
            encoding="utf-8",
        )
    return report


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-aware ATCC Profile Suite",
        "",
        "State schema: {name} v{version}".format(
            name=str(report.get("atcc_state_schema", {}).get("name", "")),
            version=int(report.get("atcc_state_schema", {}).get("version", 0) or 0),
        ),
        "",
        "## Artifact Schema",
        "",
        "| Profile | Policy artifact | Schema version | Compatible |",
        "|---|---|---:|---:|",
    ]
    for profile in report.get("profiles", ()):
        schema = dict(profile.get("policy_artifact_schema", {}) or {})
        lines.append(
            "| {profile} | {artifact} | {version} | {compatible} |".format(
                profile=str(profile.get("profile", "")),
                artifact=str(profile.get("policy_artifact", "")),
                version=int(schema.get("state_schema_version", 0) or 0),
                compatible="yes" if bool(schema.get("compatible", False)) else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Profile | Strategy | Commit rate | Throughput | Attempts/task | P99 latency | Wasted tokens/task | Pessimistic decisions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in report.get("profiles", ()):
        name = str(profile.get("profile", ""))
        for row in profile.get("aggregates", ()):
            lines.append(
                "| {profile} | {strategy} | {commit_rate:.3f} | {throughput:.2f}/s | "
                "{attempts:.2f} | {p99:.4f}s | {wasted:.1f} | {pessimistic} |".format(
                    profile=name,
                    strategy=_strategy_label(row),
                    commit_rate=float(row.get("commit_rate", 0.0)),
                    throughput=float(row.get("committed_throughput", 0.0)),
                    attempts=float(row.get("attempts_per_task", 0.0)),
                    p99=float(row.get("agent_latency_p99_s", 0.0)),
                    wasted=float(row.get("estimated_wasted_tokens_per_task", 0.0)),
                    pessimistic=int(
                        dict(row.get("operation_policy_counts", {})).get(
                            "pessimistic", 0
                        )
                    ),
                )
            )
    lines.extend(["", "## Relative Comparisons", ""])
    for profile in report.get("profiles", ()):
        comparisons = profile.get("comparisons", {})
        lines.append(f"### {profile.get('profile', '')}")
        for baseline in ("occ", "2pl"):
            delta = comparisons.get(f"adaptive-op-strict_vs_{baseline}")
            if not delta:
                continue
            lines.append(
                "- ATCC vs {baseline}: throughput {throughput:+.1f}%, "
                "P99 {p99:+.1f}%, wasted tokens {wasted:+.1f}%, "
                "attempts {attempts:+.1f}%".format(
                    baseline=_strategy_label({"strategy": baseline}),
                    throughput=float(delta.get("throughput_pct", 0.0)),
                    p99=float(delta.get("p99_latency_pct", 0.0)),
                    wasted=float(delta.get("wasted_tokens_pct", 0.0)),
                    attempts=float(delta.get("attempts_pct", 0.0)),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed phase-aware ATCC YCSB/TPC-C profile experiments."
    )
    parser.add_argument(
        "--profiles",
        default="all",
        help=(
            "Comma-separated profile names. Use all, ycsb, tpcc, or explicit "
            f"names: {','.join(PROFILE_BY_NAME)}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase_atcc_profiles"))
    parser.add_argument(
        "--strategy-set",
        choices=("core", "full"),
        default="core",
        help=(
            "'core' compares OCC/2PL/ATCC; 'full' also includes "
            "2PL nowait/wait-die, MVCC, SILO, and TicToc."
        ),
    )
    parser.add_argument("--train-per-profile", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--train-task-count", type=int, default=500)
    parser.add_argument("--eval-task-count", type=int, default=500)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--agent-slots", type=int, default=4)
    parser.add_argument(
        "--agent-admission-mode",
        choices=("planning-only", "before-begin"),
        default="planning-only",
    )
    parser.add_argument("--planning-delay-ms", type=float, default=10.0)
    parser.add_argument("--abort-retry-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--profile-eval-overrides",
        default="",
        help=(
            "JSON object keyed by profile name. Supports workload_config plus "
            "planning_delay_ms/planning_delay_s and "
            "abort_retry_delay_ms/abort_retry_delay_s."
        ),
    )
    parser.add_argument(
        "--profile-eval-overrides-file",
        default="",
        help=(
            "Path to a JSON object keyed by profile name. File overrides are "
            "loaded before --profile-eval-overrides, so inline JSON can "
            "override file defaults."
        ),
    )
    parser.add_argument(
        "--latency-distribution",
        choices=("fixed", "lognormal", "pareto"),
        default="lognormal",
    )
    parser.add_argument("--latency-cv", type=float, default=0.8)
    parser.add_argument("--latency-max-ms", type=float, default=80.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--tokens-per-operation", type=float, default=2703.0)
    parser.add_argument("--background-workers", type=int, default=4)
    parser.add_argument("--background-interval-ms", type=float, default=2.0)
    parser.add_argument("--background-strategy", default="occ")
    parser.add_argument(
        "--object-lock-scheduler",
        choices=("race", "priority", "bounded-priority"),
        default="race",
    )
    parser.add_argument("--object-lock-priority-burst", type=int, default=2)
    parser.add_argument("--prelock-wait-budget-ms", type=float, default=0.0)
    parser.add_argument(
        "--prelock-wait-budget-mode",
        choices=("transaction", "object"),
        default="transaction",
    )
    parser.add_argument(
        "--prelock-lease-mode",
        choices=(
            "hold",
            "yield-during-planning",
            "yield-refresh-regenerate",
            "defer-until-after-planning",
        ),
        default="hold",
    )
    parser.add_argument(
        "--agent-execution-mode",
        choices=("legacy", "staged", "staged-local"),
        default="legacy",
    )
    parser.add_argument(
        "--snapshot-timing",
        choices=("before-planning", "after-planning"),
        default="before-planning",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_profile_suite(
        profiles=_split_csv(args.profiles),
        output_dir=args.output_dir,
        strategy_set=args.strategy_set,
        train_per_profile=args.train_per_profile,
        train_episodes=args.train_episodes,
        train_task_count=args.train_task_count,
        eval_task_count=args.eval_task_count,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
        workers=args.workers,
        agent_slots=args.agent_slots,
        agent_admission_mode=args.agent_admission_mode,
        planning_delay_s=args.planning_delay_ms / 1000.0,
        abort_retry_delay_s=args.abort_retry_delay_ms / 1000.0,
        latency_distribution=args.latency_distribution,
        latency_cv=args.latency_cv,
        latency_max_s=args.latency_max_ms / 1000.0,
        max_attempts=args.max_attempts,
        tokens_per_operation=args.tokens_per_operation,
        background_workers=args.background_workers,
        background_interval_s=args.background_interval_ms / 1000.0,
        background_strategy=args.background_strategy,
        object_lock_scheduler=args.object_lock_scheduler,
        object_lock_priority_burst=args.object_lock_priority_burst,
        prelock_wait_budget_s=args.prelock_wait_budget_ms / 1000.0,
        prelock_wait_budget_mode=args.prelock_wait_budget_mode,
        prelock_lease_mode=args.prelock_lease_mode,
        agent_execution_mode=args.agent_execution_mode,
        snapshot_timing=args.snapshot_timing,
        profile_eval_overrides=_merge_profile_eval_overrides(
            _load_profile_eval_overrides_file(args.profile_eval_overrides_file),
            _parse_profile_eval_overrides(args.profile_eval_overrides),
        ),
    )
    (stdout or sys.stdout).write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def _select_profiles(profiles: Iterable[str]) -> Tuple[ATCCProfileSpec, ...]:
    names = tuple(str(name).strip().lower() for name in profiles if str(name).strip())
    if not names or names == ("all",):
        return PROFILE_SPECS
    expanded = []
    for name in names:
        if name == "all":
            expanded.extend(profile.name for profile in PROFILE_SPECS)
        elif name in {"ycsb", "tpcc"}:
            expanded.extend(
                profile.name
                for profile in PROFILE_SPECS
                if profile.workload_kind == name
            )
        elif name in PROFILE_BY_NAME:
            expanded.append(name)
        else:
            raise ValueError(f"unknown ATCC profile: {name}")
    seen = set()
    selected = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            selected.append(PROFILE_BY_NAME[name])
    return tuple(selected)


def _training_specs(
    selected: Sequence[ATCCProfileSpec],
    *,
    train_per_profile: bool,
) -> Tuple[ATCCProfileSpec, ...]:
    if train_per_profile:
        return tuple(selected)
    names = {
        DEFAULT_TRAINING_PROFILE[profile.workload_kind]
        for profile in selected
    }
    return tuple(PROFILE_BY_NAME[name] for name in sorted(names))


def _artifact_key(spec: ATCCProfileSpec, *, train_per_profile: bool) -> str:
    if train_per_profile:
        return spec.name
    return spec.workload_kind


def _build_workload(spec: ATCCProfileSpec) -> AgentWorkload:
    if spec.workload_kind == "ycsb":
        return build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(**dict(spec.config)),
        )
    if spec.workload_kind == "tpcc":
        return build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(**dict(spec.config)),
        )
    raise ValueError(f"unsupported workload kind: {spec.workload_kind}")


def _with_config_override(
    spec: ATCCProfileSpec,
    override: Mapping[str, Any],
) -> ATCCProfileSpec:
    if not override:
        return spec
    config = dict(spec.config)
    config.update(dict(override))
    return dataclasses.replace(spec, config=config)


def _eval_strategies(strategy_set: str) -> Tuple[str, ...]:
    normalized = str(strategy_set or "core").strip().lower()
    if normalized == "core":
        return CORE_EVAL_STRATEGIES
    if normalized == "full":
        return FULL_EVAL_STRATEGIES
    raise ValueError(f"unsupported strategy set: {strategy_set}")


def _comparisons(aggregates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_strategy = {str(row.get("strategy", "")): row for row in aggregates}
    atcc = by_strategy.get("adaptive-op-strict")
    if atcc is None:
        return {}
    comparisons = {}
    for baseline_name in ("occ", "2pl"):
        baseline = by_strategy.get(baseline_name)
        if baseline is None:
            continue
        comparisons[f"adaptive-op-strict_vs_{baseline_name}"] = {
            "throughput_pct": _pct_delta(
                atcc.get("committed_throughput", 0.0),
                baseline.get("committed_throughput", 0.0),
            ),
            "p99_latency_pct": _pct_delta(
                atcc.get("agent_latency_p99_s", 0.0),
                baseline.get("agent_latency_p99_s", 0.0),
            ),
            "wasted_tokens_pct": _pct_delta(
                atcc.get("estimated_wasted_tokens_per_task", 0.0),
                baseline.get("estimated_wasted_tokens_per_task", 0.0),
            ),
            "attempts_pct": _pct_delta(
                atcc.get("attempts_per_task", 0.0),
                baseline.get("attempts_per_task", 0.0),
            ),
            "pessimistic_decision_delta": (
                int(
                    dict(atcc.get("operation_policy_counts", {})).get(
                        "pessimistic", 0
                    )
                )
                - int(
                    dict(baseline.get("operation_policy_counts", {})).get(
                        "pessimistic", 0
                    )
                )
            ),
        }
    return comparisons


def _strategy_label(row: Mapping[str, Any]) -> str:
    strategy = str(row.get("strategy", row))
    if strategy == "adaptive-op-strict":
        return "ATCC"
    if strategy == "2pl":
        return "2PL"
    if strategy == "2pl-pre":
        return "2PL-pre-oracle"
    return strategy.upper()


def _pct_delta(value: Any, baseline: Any) -> float:
    base = float(baseline or 0.0)
    if base == 0.0:
        return 0.0
    return (float(value or 0.0) - base) / base * 100.0


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_profile_eval_overrides(raw: str) -> Dict[str, Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("--profile-eval-overrides must be a JSON object")
    parsed: Dict[str, Dict[str, Any]] = {}
    for profile, row in data.items():
        if not isinstance(row, Mapping):
            raise ValueError("profile override rows must be JSON objects")
        normalized = dict(row)
        if "planning_delay_ms" in normalized:
            normalized["planning_delay_s"] = (
                float(normalized.pop("planning_delay_ms")) / 1000.0
            )
        if "abort_retry_delay_ms" in normalized:
            normalized["abort_retry_delay_s"] = (
                float(normalized.pop("abort_retry_delay_ms")) / 1000.0
            )
        parsed[str(profile)] = normalized
    return parsed


def _load_profile_eval_overrides_file(path: str) -> Dict[str, Dict[str, Any]]:
    text = str(path or "").strip()
    if not text:
        return {}
    return _parse_profile_eval_overrides(Path(text).read_text(encoding="utf-8"))


def _merge_profile_eval_overrides(
    *sources: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        for profile, row in dict(source or {}).items():
            target = dict(merged.get(str(profile), {}) or {})
            for key, value in dict(row or {}).items():
                if key == "workload_config":
                    workload_config = dict(target.get("workload_config", {}) or {})
                    workload_config.update(dict(value or {}))
                    target[key] = workload_config
                else:
                    target[key] = value
            merged[str(profile)] = target
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
