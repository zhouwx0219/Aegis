#!/usr/bin/env python3
"""Replay a fixed CAST-DAS trace with the paper agent runtime semantics.

Unlike ``run_castdas_trace.py``, this runner preserves the mixed benchmark's
agent execution path: reasoning happens inside the transaction/ATCC runtime
window, ATCC decisions are made after observed execution phases, reservations/deferred begin
paths are honored, and wasted reasoning is counted on aborts.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmarks.mixed import (  # noqa: E402
    ATCCAdmissionConflict,
    MixedBenchmarkConfig,
    MixedCounters,
    average,
    admission_failure_reason,
    attempt_failure_reason,
    background_write_guard,
    operation_write_targets,
    percentile,
    registry_for,
    run_agent_attempt,
    observe_atcc_admission_conflict,
    task_targets,
    tpcc_replay_gate_pressure,
)
from agent.benchmarks.phases import PlannedPhase, PlannedTask, sleep_for_reasoning  # noqa: E402
from agent.cc import LockConflict  # noqa: E402
from agent.cc.atcc.ppo import DiscretePPOPolicy, EpsilonGreedyPolicy  # noqa: E402
from agent.runtime import AgentTransactionManager  # noqa: E402
from agent.runtime import CompiledPhasePolicy  # noqa: E402
from agent.runtime.paper_policy import StaticThresholdPhasePolicy  # noqa: E402
from agent.runtime.priority import PriorityConfig  # noqa: E402
from agent.workloads import AgentOperation, AgentTask, apply_operation  # noqa: E402


CCS = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,paper-atcc"

FIELDS = [
    "trace_id",
    "source_system",
    "system",
    "cc",
    "access_set_visibility",
    "workload",
    "workload_variant",
    "level",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "seed",
    "repeat",
    "paper_switching",
    "paper_priority",
    "paper_performance_guards",
    "paper_delayed_write_apply",
    "paper_policy_mode",
    "paper_policy_path",
    "atcc_retry_cache_enabled",
    "paper_deferred_replay_enabled",
    "tpcc_replay_capacity",
    "ycsb_replay_capacity",
    "max_attempts",
    "retry_budget",
    "status",
    "elapsed_s",
    "measurement_window_s",
    "agent_drain_s",
    "bottom_txn_attempts",
    "bottom_txn_commits",
    "bottom_txn_attempt_tps",
    "bottom_txn_commit_tps",
    "underlying_txn_attempt_tps",
    "underlying_txn_commit_tps",
    "native_throughput",
    "total_tps",
    "drain_total_tps",
    "steady_agent_commits",
    "steady_background_commits",
    "agent_task_tps",
    "agent_drain_task_tps",
    "agent_tps",
    "background_tps",
    "agent_attempts",
    "agent_logical_attempts",
    "agent_admission_deferrals",
    "agent_admission_deferral_rate",
    "agent_commits",
    "agent_aborts",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "agent_task_completion_rate",
    "agent_commit_rate",
    "agent_attempt_abort_rate",
    "agent_avg_retry_count",
    "agent_p50_latency_ms",
    "agent_p95_latency_ms",
    "agent_p99_latency_ms",
    "agent_p999_latency_ms",
    "agent_p9999_latency_ms",
    "agent_time_to_success_p50_ms",
    "agent_time_to_success_p95_ms",
    "agent_time_to_success_p99_ms",
    "agent_time_to_success_p999_ms",
    "agent_time_to_success_p9999_ms",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_commit_rate",
    "background_retries",
    "agent_reservation_wait_ms_total",
    "agent_reservation_wait_ms_mean",
    "agent_overload_admission_wait_ms_total",
    "agent_overload_admission_wait_ms_mean",
    "agent_overload_admission_events",
    "agent_tpcc_replay_gate_wait_ms_total",
    "agent_tpcc_replay_gate_wait_ms_mean",
    "agent_tpcc_replay_gate_wait_events",
    "background_reservation_wait_ms_total",
    "background_reservation_wait_ms_mean",
    "background_overload_admission_wait_ms_total",
    "background_overload_admission_wait_ms_mean",
    "background_overload_admission_events",
    "background_begin_ms_mean",
    "background_apply_ms_mean",
    "background_commit_wall_ms_mean",
    "background_row_ms_mean",
    "reservation_guard_wait_ms_total",
    "total_reasoning_delay_ms",
    "agent_initial_reasoning_invocations",
    "agent_retry_reasoning_invocations",
    "agent_cached_retry_replays",
    "agent_initial_reasoning_tokens",
    "agent_retry_reasoning_tokens",
    "agent_retry_cache_saved_tokens",
    "agent_counterfactual_no_cache_tokens",
    "agent_avg_tokens_without_retry_cache",
    "agent_retry_cache_savings_ratio",
    "wasted_reasoning_ms",
    "read_conflicts",
    "write_conflicts",
    "version_conflict_count",
    "reservation_admission_abort_count",
    "lock_timeout_abort_count",
    "lock_preempted_abort_count",
    "full_commit_lock_timeout_abort_count",
    "hot_commit_lock_timeout_abort_count",
    "begin_lock_timeout_abort_count",
    "version_validation_abort_count",
    "paper_read_lock_acquires",
    "paper_write_lock_acquires",
    "paper_lock_wait_events",
    "paper_lock_wait_ms",
    "paper_agent_lock_wait_events",
    "paper_agent_lock_wait_ms",
    "paper_background_lock_wait_events",
    "paper_background_lock_wait_ms",
    "paper_wounds",
    "paper_wounds_agent_to_agent",
    "paper_wounds_agent_to_background",
    "paper_wounds_background_to_agent",
    "paper_wounds_background_to_background",
    "paper_wound_events",
    "paper_lock_timeouts",
    "paper_priority_reorders",
    "paper_live_contexts",
    "paper_live_contexts_by_status",
    "paper_live_context_ids",
    "paper_background_fast_publishes",
    "paper_background_fast_publish_failures",
    "paper_background_publisher_queue_events",
    "paper_background_publisher_queue_wait_ms",
    "paper_background_publisher_queue_timeouts",
    "paper_background_pre_admission_yields",
    "paper_background_pre_admission_objects",
    "paper_background_native_batch_attempts",
    "paper_background_native_batch_commits",
    "paper_background_native_batch_read_only_commits",
    "paper_background_native_batch_validation_failures",
    "paper_background_native_batch_admission_fallbacks",
    "paper_background_native_batch_pin_fallbacks",
    "paper_background_native_batch_unsupported_fallbacks",
    "paper_commit_admission_conflicts",
    "paper_commit_admission_conflict_objects",
    "paper_agent_blind_write_rebases",
    "paper_tpcc_exact_risk_wlocks",
    "paper_tpcc_family_risk_wlocks",
    "paper_tpcc_exact_guard_checks",
    "paper_tpcc_exact_guard_insufficient_evidence",
    "paper_tpcc_exact_guard_max_exact_changes",
    "paper_tpcc_exact_guard_max_family_changes",
    "paper_tpcc_exact_guard_max_total_changes",
    "paper_occ_native_fast_publishes",
    "paper_occ_native_fast_publish_failures",
    "paper_background_publish_fallbacks",
    "paper_background_publish_fallback_active_reader",
    "paper_background_publish_fallback_active_writer",
    "paper_background_publish_fallback_version_mismatch",
    "paper_background_publish_fallback_commit_latch",
    "paper_background_publish_fallback_missing_private_version",
    "paper_background_publish_fallback_multi_object_atomicity",
    "paper_background_publish_fallback_unsupported_operation",
    "paper_version_private_prepares",
    "paper_version_private_discards",
    "paper_version_atomic_publishes",
    "paper_version_published_objects",
    "paper_version_gc_versions",
    "paper_version_history_versions",
    "paper_version_pinned_transactions",
    "paper_version_private_transactions",
    "paper_version_commit_table_entries",
    "paper_version_native_publish_attempts",
    "paper_version_native_publishes",
    "paper_version_native_publish_pin_fallbacks",
    "paper_version_native_publish_disjoint_pin_bypasses",
    "paper_version_read_only_bypasses",
    "paper_version_background_version_change_events",
    "paper_version_background_changed_objects",
    "paper_version_version_risk_read_locks",
    "paper_version_object_boundary_acquires",
    "paper_version_object_boundary_waits",
    "paper_version_pinned_read_guard_acquires",
    "paper_version_pinned_read_guard_conflicts",
    "paper_retry_validation_conflicts",
    "paper_retry_mask_escalations",
    "paper_retry_full_observed_escalations",
    "paper_retry_inherited_attempts",
    "paper_retry_tracked_tasks",
    "paper_retry_validation_conflicts_first_attempt",
    "paper_retry_validation_conflicts_retry_attempt",
    "paper_retry_conflict_hot_read",
    "paper_retry_conflict_cold_read",
    "paper_retry_conflict_hot_write",
    "paper_retry_conflict_cold_write",
    "paper_retry_conflict_read_before_write",
    "paper_retry_conflict_blind_write",
    "paper_retry_conflict_object_warehouse",
    "paper_retry_conflict_object_district",
    "paper_retry_conflict_object_stock",
    "paper_retry_conflict_object_customer",
    "paper_retry_conflict_object_other",
    "paper_retry_conflict_after_tpcc_exact_guard",
    "paper_retry_conflict_objects",
    "paper_lock_acquires_by_phase",
    "paper_hotness_observed_objects",
    "paper_hotness_total_accesses",
    "paper_hotness_hot_objects",
    "paper_hotness_validation_failures",
    "paper_hotness_lock_wait_events",
    "paper_hotness_lock_wait_ms",
    "paper_hotness_wounds",
    "guarded_conflict_checks",
    "conflict_pressure_count",
    "conflict_abort_count",
    "raw_action_counts",
    "admission_yield_ms_total",
    "raw_admission_yield_counts",
    "agent_avg_tokens",
    "agent_total_tokens",
    "agent_committed_reasoning_tokens",
    "agent_wasted_reasoning_tokens",
    "agent_tokens_per_committed_txn",
    "agent_wasted_tokens_per_commit",
    "agent_wasted_token_ratio",
    "error",
]

COMMIT_TIMING_PHASES = (
    "interceptor",
    "hotness",
    "policy",
    "lock",
    "validate",
    "install",
    "publish",
    "gc",
)
FIELDS[-1:-1] = [
    "paper_commit_timing_transactions",
    "paper_commit_timing_agent_transactions",
    "paper_commit_timing_background_transactions",
    "paper_commit_timing_samples",
    "paper_commit_timing_agent_samples",
    "paper_commit_timing_background_samples",
] + [
    f"paper_commit_timing_{role}{phase}_ms_mean"
    for role in ("", "agent_", "background_")
    for phase in COMMIT_TIMING_PHASES
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warmup-trace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", default=CCS)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--paper-policy", type=Path, default=None)
    parser.add_argument("--trajectory-output", type=Path, default=None)
    parser.add_argument("--paper-exploration-seed", type=int, default=None)
    parser.add_argument("--paper-exploration-stay-probability", type=float, default=0.5)
    parser.add_argument("--paper-exploration-epsilon", type=float, default=0.2)
    parser.add_argument("--policy-mode", choices=("eval", "train", "online"), default="eval")
    parser.add_argument("--paper-switching", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--paper-static-conflict-threshold", type=float, default=0.20)
    parser.add_argument("--paper-static-protection-mask", type=int, default=4)
    parser.add_argument("--paper-priority", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--paper-commit-admission-priority", action="store_true")
    parser.add_argument(
        "--paper-performance-guards",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--paper-delayed-write-apply",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument("--priority-quantum-scale", type=float, default=1.0)
    parser.add_argument("--disable-atcc-retry-cache", action="store_true")
    parser.add_argument("--disable-paper-deferred-replay", action="store_true")
    parser.add_argument("--tpcc-replay-capacity", type=int, default=1)
    parser.add_argument("--ycsb-replay-capacity", type=int, default=1)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum logical transaction attempts, including the first attempt.",
    )
    parser.add_argument(
        "--allow-retries",
        action="store_true",
        help="Explicitly opt into comparative retry experiments.",
    )
    parser.add_argument("--tokens-per-operation", type=int, default=2703)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--measure-seconds", type=float, default=0.0)
    parser.add_argument("--execution-workers", type=int, default=0)
    parser.add_argument("--no-cycle-trace", action="store_false", dest="cycle_trace")
    parser.set_defaults(cycle_trace=True)
    args = parser.parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    if args.tpcc_replay_capacity < 1:
        raise SystemExit("--tpcc-replay-capacity must be positive")
    if args.ycsb_replay_capacity < 1:
        raise SystemExit("--ycsb-replay-capacity must be positive")
    if args.max_attempts != 1 and not args.allow_retries:
        raise SystemExit(
            "paper experiments disable retries by default; pass --allow-retries explicitly"
        )

    rows = read_trace(args.trace)
    warmup_rows = read_trace(args.warmup_trace) if args.warmup_trace else []
    output_rows = []
    for cc in split_csv(args.cc):
        started = time.perf_counter()
        try:
            output_rows.append(
                run_trace(
                    rows,
                    warmup_rows=warmup_rows,
                    cc=cc,
                    policy=args.policy,
                    paper_policy=args.paper_policy,
                    trajectory_output=args.trajectory_output,
                    paper_exploration_seed=args.paper_exploration_seed,
                    paper_exploration_stay_probability=args.paper_exploration_stay_probability,
                    paper_exploration_epsilon=args.paper_exploration_epsilon,
                    policy_mode=args.policy_mode,
                    paper_switching=args.paper_switching,
                    paper_static_conflict_threshold=args.paper_static_conflict_threshold,
                    paper_static_protection_mask=args.paper_static_protection_mask,
                    paper_priority=args.paper_priority,
                    paper_commit_admission_priority=args.paper_commit_admission_priority,
                    paper_performance_guards=args.paper_performance_guards,
                    paper_delayed_write_apply=args.paper_delayed_write_apply,
                    priority_quantum_scale=args.priority_quantum_scale,
                    atcc_retry_cache_enabled=not args.disable_atcc_retry_cache,
                    paper_deferred_replay_enabled=not args.disable_paper_deferred_replay,
                    tpcc_replay_capacity=args.tpcc_replay_capacity,
                    ycsb_replay_capacity=args.ycsb_replay_capacity,
                    max_attempts=args.max_attempts,
                    tokens_per_operation=args.tokens_per_operation,
                    warmup_seconds=args.warmup_seconds,
                    measure_seconds=args.measure_seconds,
                    execution_workers=args.execution_workers,
                    cycle_trace=bool(args.cycle_trace),
                )
            )
        except Exception as exc:
            sample = rows[0] if rows else {}
            output_rows.append(error_row(sample, cc, exc, time.perf_counter() - started))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(args.output)
    print(f"rows={len(output_rows)}")
    return 0


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty trace: {path}")
    for row in rows:
        row["_ops"] = json.loads(row["ops_json"])
        row["_context"] = json.loads(row.get("context_json") or "{}")
        row["_task"] = task_from_row(row)
        row["_planned0"] = planned_from_row(row, attempt=0)
    return rows


def paper_performance_guards_enabled(
    cc: str,
    *,
    setting: str,
) -> bool:
    return str(cc).strip().lower() == "paper-atcc" and str(
        setting
    ).strip().lower() == "enabled"


def run_trace(
    rows: list[dict[str, Any]],
    *,
    warmup_rows: list[dict[str, Any]] | None = None,
    cc: str,
    policy: Path | None,
    paper_policy: Path | None = None,
    trajectory_output: Path | None = None,
    paper_exploration_seed: int | None = None,
    paper_exploration_stay_probability: float = 0.5,
    paper_exploration_epsilon: float = 0.2,
    policy_mode: str = "eval",
    paper_switching: str = "dynamic",
    paper_static_conflict_threshold: float = 0.20,
    paper_static_protection_mask: int = 4,
    paper_priority: str = "enabled",
    paper_commit_admission_priority: bool = False,
    paper_performance_guards: str = "disabled",
    paper_delayed_write_apply: str = "disabled",
    priority_quantum_scale: float = 1.0,
    atcc_retry_cache_enabled: bool = True,
    paper_deferred_replay_enabled: bool = True,
    tpcc_replay_capacity: int = 1,
    ycsb_replay_capacity: int = 1,
    max_attempts: int,
    tokens_per_operation: int,
    warmup_seconds: float = 0.0,
    measure_seconds: float = 0.0,
    execution_workers: int = 0,
    cycle_trace: bool = True,
) -> dict[str, Any]:
    sample = rows[0]
    config = MixedBenchmarkConfig(
        workload=sample["workload"],
        level=sample["level"],
        workload_profile="paper",
        cc=cc,
        clients=int(float(sample["clients"])),
        agent_ratio=float(sample["agent_ratio"]),
        agent_workers=int(float(sample["agent_workers"])),
        background_workers=int(float(sample["background_workers"])),
        policy=policy,
        policy_mode=str(policy_mode) if policy else "online",
        paper_policy=paper_policy,
        atcc_pure_policy=True,
        background_mode="procedure",
        retry_until_commit=True,
        max_attempts_per_task=max_attempts,
        agent_retry_backoff_min_ms=1,
        agent_retry_backoff_max_ms=5,
        background_retry_backoff_min_ms=10,
        background_retry_backoff_max_ms=30,
        tokens_per_operation=tokens_per_operation,
        atcc_retry_cache_enabled=bool(atcc_retry_cache_enabled),
        paper_deferred_replay_enabled=bool(paper_deferred_replay_enabled),
    ).normalized()
    quantum_scale = float(priority_quantum_scale)
    if quantum_scale <= 0.0:
        raise ValueError("priority quantum scale must be positive")
    if paper_exploration_seed is not None:
        if cc != "paper-atcc":
            raise ValueError("paper exploration is only valid for paper-atcc")
        if paper_policy is not None:
            epsilon = float(paper_exploration_epsilon)
            if not 0.0 <= epsilon <= 1.0:
                raise ValueError("paper exploration epsilon must be in [0, 1]")
            compiled_policy = EpsilonGreedyPolicy(
                CompiledPhasePolicy.load(paper_policy),
                seed=paper_exploration_seed,
                epsilon=epsilon,
            )
        else:
            stay_probability = float(paper_exploration_stay_probability)
            if not 0.0 <= stay_probability <= 1.0:
                raise ValueError("paper exploration stay probability must be in [0, 1]")
            compiled_policy = DiscretePPOPolicy(
                seed=paper_exploration_seed,
                stay_probability=stay_probability,
            )
    else:
        compiled_policy = CompiledPhasePolicy.load(paper_policy) if paper_policy is not None else None
    if str(paper_switching).strip().lower() == "static":
        compiled_policy = StaticThresholdPhasePolicy(
            conflict_abort_threshold=paper_static_conflict_threshold,
            protection_mask=paper_static_protection_mask,
        )
    manager = AgentTransactionManager(
        cc_registry=registry_for(config),
        record_traces=False,
        paper_policy=compiled_policy,
        collect_trajectories=trajectory_output is not None or paper_exploration_seed is not None,
        low_conflict_occ_guard=cc == "paper-atcc",
        performance_guards_enabled=paper_performance_guards_enabled(
            cc,
            setting=paper_performance_guards,
        ),
        commit_admission_priority_enabled=bool(paper_commit_admission_priority),
        delayed_write_apply_enabled=(
            str(paper_delayed_write_apply).strip().lower() == "enabled"
        ),
        priority_config=PriorityConfig(
            sql_quantum_ms=10.0 * quantum_scale,
            interval_quantum_ms=10.0 * quantum_scale,
            blocked_quantum_ms=100.0 * quantum_scale,
        ),
        priority_enabled=str(paper_priority).strip().lower() == "enabled",
    )
    manager.paper_static_initial_mask = (
        int(paper_static_protection_mask) & 0xF
        if str(paper_switching).strip().lower() == "static"
        else 0
    )
    manager.paper_force_runtime_path = not bool(paper_deferred_replay_enabled)
    manager.tpcc_replay_capacity = max(1, int(tpcc_replay_capacity))
    manager.ycsb_replay_capacity = max(1, int(ycsb_replay_capacity))
    all_rows = list(rows)
    if warmup_rows:
        all_rows.extend(warmup_rows)
    for object_id in sorted(trace_object_ids(all_rows)):
        manager.register_object(object_id, "0", kind="row")
    if warmup_rows:
        run_rows(
            manager,
            cc,
            config,
            warmup_rows,
            max_attempts,
            duration_s=float(warmup_seconds),
            cycle_trace=cycle_trace,
            execution_workers=execution_workers,
        )
        manager.trajectory_collector.clear()
        manager.reset_measurement_diagnostics()
    counters, elapsed_s = run_rows(
        manager,
        cc,
        config,
        rows,
        max_attempts,
        duration_s=float(measure_seconds),
        cycle_trace=cycle_trace,
        execution_workers=execution_workers,
    )
    result = result_row(
        sample,
        cc,
        counters,
        elapsed_s,
        tokens_per_operation,
        rows,
        manager,
        runtime_config={
            "paper_switching": str(paper_switching).strip().lower(),
            "paper_priority": str(paper_priority).strip().lower(),
            "paper_performance_guards": str(
                paper_performance_guards
            ).strip().lower(),
            "paper_delayed_write_apply": str(
                paper_delayed_write_apply
            ).strip().lower(),
            "paper_policy_mode": str(policy_mode).strip().lower(),
            "paper_policy_path": str(paper_policy.resolve()) if paper_policy else "",
            "atcc_retry_cache_enabled": bool(atcc_retry_cache_enabled),
            "paper_deferred_replay_enabled": bool(paper_deferred_replay_enabled),
            "tpcc_replay_capacity": int(tpcc_replay_capacity),
            "ycsb_replay_capacity": int(ycsb_replay_capacity),
            "max_attempts": int(max_attempts),
            "retry_budget": max(0, int(max_attempts) - 1),
        },
    )
    if trajectory_output is not None and cc == "paper-atcc":
        transitions = [
            {
                **dataclasses.asdict(transition),
                "state": dataclasses.asdict(transition.state),
                "next_state": dataclasses.asdict(transition.next_state),
            }
            for transition in manager.trajectory_collector.snapshot()
        ]
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        trajectory_output.write_text(
            json.dumps({"transitions": transitions}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if policy is not None and str(policy_mode).strip().lower() == "train":
        trained_policy = getattr(manager.cc_registry.resolve(cc), "policy", None)
        if trained_policy is not None and hasattr(trained_policy, "save_json"):
            trained_policy.save_json(policy)
    return result


def run_rows(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    rows: list[dict[str, Any]],
    max_attempts: int,
    *,
    duration_s: float = 0.0,
    cycle_trace: bool = True,
    execution_workers: int = 0,
) -> tuple[MixedCounters, float]:
    by_worker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        original_worker = int(float(row["worker_id"]))
        slot = (
            original_worker % int(execution_workers)
            if int(execution_workers) > 0
            else original_worker
        )
        by_worker[slot].append(row)
    for worker_rows in by_worker.values():
        worker_rows.sort(
            key=lambda row: (
                int(float(row["sequence"])),
                int(float(row["worker_id"])),
            )
        )

    worker_kinds = {
        worker_id: {str(row["client_type"]) for row in worker_rows}
        for worker_id, worker_rows in by_worker.items()
    }
    mixed_workers = [
        worker_id for worker_id, kinds in worker_kinds.items() if len(kinds) != 1
    ]
    if mixed_workers:
        raise ValueError(
            f"fixed trace workers must have one client type: {mixed_workers}"
        )
    agent_worker_count = sum(kinds == {"agent"} for kinds in worker_kinds.values())
    background_worker_count = len(worker_kinds) - agent_worker_count
    fixed_count_coordination = (
        FixedCountRunCoordinator(agent_worker_count)
        if float(duration_s) <= 0.0
        and bool(cycle_trace)
        and agent_worker_count > 0
        and background_worker_count > 0
        else None
    )
    agent_admission_cap = paper_agent_admission_cap(
        manager,
        cc,
        config,
        agent_worker_count=agent_worker_count,
    )
    if 0 < agent_admission_cap < agent_worker_count:
        if (
            str(config.workload).strip().lower() == "tpcc"
            and str(config.level).strip().lower() == "high"
            and int(config.background_workers) == 0
        ):
            agent_admission = TPCCReplayPressureAdmission(
                manager,
                full_limit=agent_worker_count,
                pressure_limit=agent_admission_cap,
            )
        else:
            agent_admission = ObservedPressureAdmission(
                manager,
                full_limit=agent_worker_count,
                pressure_limit=agent_admission_cap,
            )
    else:
        agent_admission = None
    background_admission_cap = paper_background_admission_cap(
        manager,
        cc,
        config,
        background_worker_count=background_worker_count,
    )
    background_admission = (
        threading.BoundedSemaphore(background_admission_cap)
        if 0 < background_admission_cap < background_worker_count
        else None
    )

    lock = threading.Lock()
    counters = MixedCounters()
    timed_window = (
        TimedRunWindow(float(duration_s)) if float(duration_s) > 0.0 else None
    )
    barrier = threading.Barrier(
        len(by_worker) + 1,
        action=(timed_window.start if timed_window is not None else None),
    )
    thread_errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=worker_main_guarded,
            args=(
                manager,
                cc,
                config,
                worker_rows,
                max_attempts,
                barrier,
                lock,
                counters,
                float(duration_s),
                bool(cycle_trace),
                fixed_count_coordination,
                timed_window,
                agent_admission,
                background_admission,
                thread_errors,
                lock,
            ),
        )
        for _worker, worker_rows in sorted(by_worker.items())
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    barrier.wait()
    if timed_window is not None:
        started = timed_window.started_at
    for thread in threads:
        thread.join()
    if thread_errors:
        raise RuntimeError(f"worker failed: {thread_errors[0]}") from thread_errors[0]
    elapsed_s = max(0.001, time.perf_counter() - started)
    counters.measurement_window_s = (
        float(timed_window.duration_s) if timed_window is not None else elapsed_s
    )
    counters.agent_drain_s = max(
        0.0, elapsed_s - counters.measurement_window_s
    )
    return counters, elapsed_s


def worker_main_guarded(*args: Any) -> None:
    thread_errors = args[-2]
    error_lock = args[-1]
    try:
        worker_main(*args[:-2])
    except BaseException as exc:
        with error_lock:
            thread_errors.append(exc)


def worker_main(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    rows: list[dict[str, Any]],
    max_attempts: int,
    barrier: threading.Barrier,
    lock: threading.Lock,
    counters: MixedCounters,
    duration_s: float,
    cycle_trace: bool,
    fixed_count_coordination: "FixedCountRunCoordinator | None" = None,
    timed_window: "TimedRunWindow | None" = None,
    agent_admission: "threading.BoundedSemaphore | None" = None,
    background_admission: "threading.BoundedSemaphore | None" = None,
) -> None:
    rng = random.Random(int(float(rows[0]["seed"])) + int(float(rows[0]["worker_id"])))
    barrier.wait()
    client_type = str(rows[0]["client_type"])
    try:
        if fixed_count_coordination is not None and client_type != "agent":
            selected_rows = continuous_background_rows(rows, fixed_count_coordination)
        else:
            selected_rows = timed_rows(
                rows,
                duration_s=duration_s,
                cycle_trace=cycle_trace,
                deadline=(timed_window.deadline if timed_window is not None else None),
            )
        measurement_deadline = (
            timed_window.deadline if timed_window is not None else float("inf")
        )
        background_batch_size = paper_background_batch_size(manager, cc, config)
        if client_type != "agent" and background_batch_size > 1:
            run_paper_background_batches(
                manager,
                cc,
                config,
                selected_rows,
                rng,
                lock,
                counters,
                measurement_deadline,
                batch_size=background_batch_size,
            )
            return
        for logical_instance, row in enumerate(selected_rows):
            if row["client_type"] == "agent":
                logical_row = dict(row)
                logical_row["_logical_instance"] = logical_instance
                admission_started = time.perf_counter()
                if agent_admission is not None:
                    agent_admission.acquire()
                admission_wait_s = time.perf_counter() - admission_started
                try:
                    if agent_admission is None:
                        run_agent_row(
                            manager, cc, config, logical_row, max_attempts,
                            rng, lock, counters, measurement_deadline,
                        )
                    else:
                        run_agent_row(
                            manager,
                            cc,
                            config,
                            logical_row,
                            max_attempts,
                            rng,
                            lock,
                            counters,
                            measurement_deadline,
                            overload_admission_wait_s=admission_wait_s,
                        )
                finally:
                    if agent_admission is not None:
                        agent_admission.release()
                client_think_ms = max(
                    0,
                    int(float(dict(row.get("_context", {}) or {}).get("client_think_ms", 0) or 0)),
                )
                if client_think_ms:
                    time.sleep(client_think_ms / 1000.0)
            else:
                admission_started = time.perf_counter()
                if background_admission is not None:
                    background_admission.acquire()
                admission_wait_s = time.perf_counter() - admission_started
                try:
                    if background_admission is None:
                        run_background_row(
                            manager, cc, config, row, rng, lock, counters,
                            measurement_deadline,
                        )
                    else:
                        run_background_row(
                            manager,
                            cc,
                            config,
                            row,
                            rng,
                            lock,
                            counters,
                            measurement_deadline,
                            overload_admission_wait_s=admission_wait_s,
                        )
                finally:
                    if background_admission is not None:
                        background_admission.release()
    finally:
        if fixed_count_coordination is not None and client_type == "agent":
            fixed_count_coordination.agent_worker_done()


def paper_background_batch_size(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
) -> int:
    """Group only disjoint short background procedures in paper ATCC."""

    registry = getattr(manager, "cc_registry", None)
    resolve = getattr(registry, "resolve", None)
    if not callable(resolve) or getattr(resolve(cc), "name", "") != "paper-atcc":
        return 1
    if int(config.background_workers) <= 0:
        return 1
    if str(config.workload).strip().lower() != "ycsb":
        return 1
    level = str(config.level).strip().lower()
    clients = max(1, int(config.clients) or int(config.agent_workers) + int(config.background_workers))
    background = max(1, int(config.background_workers))
    pressure_window = max(2, min(64, (clients * background) // 8))
    if level == "low":
        return pressure_window
    if level == "medium":
        return max(2, pressure_window // 2)
    return 1


def run_paper_background_batches(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    rows: Iterable[dict[str, Any]],
    rng: random.Random,
    lock: threading.Lock,
    counters: MixedCounters,
    measurement_deadline: float,
    *,
    batch_size: int,
) -> None:
    """Publish disjoint stored procedures as one serializable native batch.

    Every row remains one logical background transaction in the reported
    counters. Rows are grouped only when their complete background procedure
    object sets are disjoint, so the atomic native publish is equivalent to a
    serial order of the original transactions. Agent policy inputs remain
    online-observed; this path consumes no Agent future target.
    """

    pending: list[
        tuple[
            dict[str, Any],
            tuple[tuple[str, int], ...],
            tuple[tuple[str, str], ...],
        ]
    ] = []
    pending_objects: set[str] = set()
    batch_sequence = 0

    def flush() -> None:
        nonlocal batch_sequence, pending, pending_objects
        if not pending:
            return
        batch_sequence += 1
        checks = tuple(item for _row, read_rows, _writes in pending for item in read_rows)
        writes = tuple(item for _row, _reads, write_rows in pending for item in write_rows)
        started = time.perf_counter()
        handled, committed = manager.try_native_background_batch(
            f"paper-bg-batch-{threading.get_ident()}-{batch_sequence}-{rng.randrange(10_000_000)}",
            checks,
            writes,
            sample_metrics=True,
            background_workers=config.background_workers,
            allow_reader_bypass=False,
        )
        elapsed = time.perf_counter() - started
        if handled and committed:
            count = len(pending)
            committed_at = time.perf_counter()
            with lock:
                counters.background_attempts += count
                counters.background_commits += count
                if committed_at <= float(measurement_deadline):
                    counters.steady_background_commits += count
                counters.background_commit_s += elapsed
                counters.background_row_s += elapsed
        else:
            for row, _read_rows, _write_rows in pending:
                run_background_row(
                    manager,
                    cc,
                    config,
                    row,
                    rng,
                    lock,
                    counters,
                    measurement_deadline,
                )
        pending = []
        pending_objects = set()

    for row in rows:
        plan = native_background_rows(manager, row["_task"])
        if plan is None:
            flush()
            run_background_row(
                manager,
                cc,
                config,
                row,
                rng,
                lock,
                counters,
                measurement_deadline,
            )
            continue
        checks, writes = plan
        objects = {str(key) for key, _version in checks}
        objects.update(str(key) for key, _value in writes)
        if pending and (
            len(pending) >= max(2, int(batch_size))
            or bool(objects & pending_objects)
        ):
            flush()
        pending.append((row, checks, writes))
        pending_objects.update(objects)
    flush()


class ObservedPressureAdmission:
    """Shrink Agent concurrency only after online conflict evidence appears."""

    def __init__(
        self,
        manager: AgentTransactionManager,
        *,
        full_limit: int,
        pressure_limit: int,
    ) -> None:
        self._manager = manager
        self._full_limit = max(1, int(full_limit))
        self._pressure_limit = max(1, min(self._full_limit, int(pressure_limit)))
        self._active = 0
        # Begin at the requested worker count.  A cold YCSB trace must pay the
        # same scheduling cost as native OCC; only observed waiters/conflicts
        # are allowed to shrink the window.
        self._limit = self._full_limit
        self._last_adjusted = 0.0
        self._cold_windows = 0
        self._condition = threading.Condition()

    def _adjust_limit(self) -> None:
        now = time.monotonic()
        if now - self._last_adjusted < 0.050:
            return
        self._last_adjusted = now
        metrics = self._manager.paper_runtime_metrics()
        waiter_count = int(metrics.get("waiter_count", 0) or 0)
        conflict_rate = float(metrics.get("conflict_abort_rate", 0.0) or 0.0)
        hot_objects = int(
            self._manager.hotness_tracker.snapshot().get("hot_objects", 0) or 0
        )
        pressured = (
            waiter_count >= 2 or conflict_rate >= 0.12 or hot_objects >= 4
        )
        if pressured:
            self._cold_windows = 0
            shrink = max(1, waiter_count // 2)
            if conflict_rate >= 0.30 or hot_objects >= 4:
                shrink = max(shrink, math.ceil(self._limit * 0.25))
            self._limit = max(self._pressure_limit, self._limit - shrink)
            return
        self._cold_windows += 1
        # Recovery is deliberately slower than pressure response so queues do
        # not repeatedly synchronize at the same hot commit boundary.
        if self._cold_windows >= 3 and self._limit < self._full_limit:
            self._limit += 1
            self._cold_windows = 0

    def acquire(self) -> None:
        with self._condition:
            while True:
                self._adjust_limit()
                if self._active < self._limit:
                    self._active += 1
                    return
                self._condition.wait(timeout=0.005)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("Agent admission released without an owner")
            self._active -= 1
            self._condition.notify_all()


class TPCCReplayPressureAdmission:
    """Adapt the all-Agent TPC-C window from the real replay queue.

    This is intentionally separate from the YCSB/global pressure controller.
    The lower bound keeps enough deferred-reasoning tasks in flight to feed the
    one-warehouse commit replay, while actual replay waiters drive fast shrink
    and an empty queue drives slow recovery.
    """

    def __init__(
        self,
        manager: AgentTransactionManager,
        *,
        full_limit: int,
        pressure_limit: int,
    ) -> None:
        self._manager = manager
        self._full_limit = max(1, int(full_limit))
        self._pressure_limit = max(
            1, min(self._full_limit, int(pressure_limit))
        )
        self._active = 0
        self._limit = self._full_limit
        self._last_adjusted = 0.0
        self._empty_windows = 0
        self._wait_queue: list[tuple[object, float]] = []
        self._max_bypass_wait_s = 1.200
        self._condition = threading.Condition()

    def _adjust_limit(self) -> None:
        now = time.monotonic()
        if now - self._last_adjusted < 0.020:
            return
        self._last_adjusted = now
        pressure = tpcc_replay_gate_pressure(self._manager)
        replay_waiters = int(pressure.get("waiters", 0) or 0)
        replay_active = int(pressure.get("active", 0) or 0)
        runtime = self._manager.paper_runtime_metrics()
        lock_waiters = int(runtime.get("waiter_count", 0) or 0)

        if replay_waiters > 0 or lock_waiters > 0:
            self._empty_windows = 0
            shrink = max(
                1,
                math.ceil(replay_waiters / 2),
                math.ceil(lock_waiters / 2),
            )
            self._limit = max(
                self._pressure_limit,
                self._limit - shrink,
            )
            return

        # A busy gate with no queue is the desired saturation point.  Reopen
        # only after repeated empty observations, one worker at a time.
        if replay_active:
            self._empty_windows = 0
            return
        self._empty_windows += 1
        if self._empty_windows >= 3 and self._limit < self._full_limit:
            self._limit += 1
            self._empty_windows = 0

    def acquire(self) -> None:
        with self._condition:
            token = object()
            self._wait_queue.append((token, time.monotonic()))
            while True:
                self._adjust_limit()
                head_token, head_started = self._wait_queue[0]
                oldest_wait_s = max(0.0, time.monotonic() - head_started)
                bounded_bypass = oldest_wait_s < self._max_bypass_wait_s
                if self._active < self._limit and (
                    head_token is token or bounded_bypass
                ):
                    for index, (queued, _started) in enumerate(self._wait_queue):
                        if queued is token:
                            self._wait_queue.pop(index)
                            break
                    self._active += 1
                    return
                self._condition.wait(timeout=0.005)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("TPC-C admission released without an owner")
            self._active -= 1
            self._adjust_limit()
            self._condition.notify_all()


class FixedCountRunCoordinator:
    """Keep short background workers active while fixed-count agents run."""

    def __init__(self, agent_worker_count: int):
        if int(agent_worker_count) <= 0:
            raise ValueError("agent_worker_count must be positive")
        self._remaining_agent_workers = int(agent_worker_count)
        self._lock = threading.Lock()
        self.stop_background = threading.Event()

    @property
    def remaining_agent_workers(self) -> int:
        with self._lock:
            return self._remaining_agent_workers

    def agent_worker_done(self) -> None:
        with self._lock:
            if self._remaining_agent_workers <= 0:
                raise RuntimeError("agent worker completion reported more than once")
            self._remaining_agent_workers -= 1
            if self._remaining_agent_workers == 0:
                self.stop_background.set()


class TimedRunWindow:
    """One steady-state deadline shared by every replay worker."""

    def __init__(self, duration_s: float):
        self.duration_s = max(0.0, float(duration_s))
        self.started_at = 0.0
        self.deadline = float("inf")

    def start(self) -> float:
        self.started_at = time.perf_counter()
        self.deadline = self.started_at + self.duration_s
        return self.started_at


def continuous_background_rows(
    rows: list[dict[str, Any]],
    coordination: FixedCountRunCoordinator,
) -> Iterable[dict[str, Any]]:
    if not rows:
        return
    index = 0
    while not coordination.stop_background.is_set():
        yield rows[index]
        index = (index + 1) % len(rows)


def timed_rows(
    rows: list[dict[str, Any]],
    *,
    duration_s: float,
    cycle_trace: bool,
    deadline: float | None = None,
) -> Iterable[dict[str, Any]]:
    if not rows:
        return
    if float(duration_s) <= 0:
        for row in rows:
            yield row
        return
    stop_at = (
        float(deadline)
        if deadline is not None
        else time.perf_counter() + float(duration_s)
    )
    index = 0
    row_count = len(rows)
    while time.perf_counter() < stop_at:
        if index >= row_count:
            if not cycle_trace:
                break
            index = 0
        yield rows[index]
        index += 1


def run_agent_row(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    row: dict[str, Any],
    max_attempts: int,
    rng: random.Random,
    lock: threading.Lock,
    counters: MixedCounters,
    measurement_deadline: float = float("inf"),
    *,
    overload_admission_wait_s: float = 0.0,
) -> None:
    task_started_at = time.perf_counter() - max(0.0, float(overload_admission_wait_s))
    final_result: dict[str, Any] = {"committed": False}
    task_reservation_wait_s = 0.0
    attempts_done = 0
    reuse_reasoning = False
    previous_failure_reason = "none"
    prior_retry_cost_ms = 0.0
    committed_at = float("inf")
    transaction_id = (
        f"{row['_task'].task_id}:generation:"
        f"{max(0, int(row.get('_logical_instance', 0) or 0))}"
    )
    for attempt in range(max(1, max_attempts)):
        planned = planned_from_row(row, attempt=attempt)
        admission_deferred = False
        used_cached_retry = bool(reuse_reasoning)
        if used_cached_retry:
            planned = planned_without_reasoning(planned)
            if (
                str(config.workload).strip().lower() == "ycsb"
                and str(config.level).strip().lower() == "high"
            ):
                # Reusing a deterministic plan avoids another LLM call, but a
                # short randomized scheduler yield prevents synchronized hot-
                # key retries from immediately colliding again.
                time.sleep(cached_retry_scheduler_cooldown_s(config, rng))
        try:
            result, action, wait_s, diagnostics = run_agent_attempt(
                manager,
                planned,
                cc,
                ttl_s=config.reservation_ttl_s,
                jitter_ms=0,
                retry_count=attempt,
                background_workers=config.background_workers,
                config=config,
                previous_failure_reason=previous_failure_reason,
                prior_retry_cost_ms=prior_retry_cost_ms,
                transaction_id=transaction_id,
            )
        except LockConflict as exc:
            admission_deferred = isinstance(exc, ATCCAdmissionConflict)
            action, wait_s, diagnostics = observe_atcc_admission_conflict(manager, cc, exc)
            result, action, wait_s, diagnostics = (
                {
                    "committed": False,
                    "wasted_reasoning_ms": 0,
                    "read_conflicts": 0,
                    "write_conflicts": 0,
                    "reservation_conflicts": 1,
                    "failure_reason": admission_failure_reason(exc, action),
                    "error": str(exc),
                },
                action,
                wait_s,
                diagnostics,
            )
            reuse_reasoning = bool(config.atcc_retry_cache_enabled)
        else:
            reuse_reasoning = bool(
                config.atcc_retry_cache_enabled
                and should_reuse_atcc_retry_plan(
                    manager,
                    cc,
                    config,
                    result,
                )
            )
        final_result = result
        task_reservation_wait_s += float(wait_s)
        attempts_done += 1
        with lock:
            counters.agent_logical_attempts += 1
            counters.total_reasoning_ms += int(planned.total_reasoning_delay_ms)
            operation_units = len(row["_task"].operations)
            if not admission_deferred:
                if used_cached_retry:
                    counters.agent_retry_cache_saved_operation_units += operation_units
                    counters.agent_cached_retry_replays += 1
                elif planned.total_reasoning_delay_ms > 0:
                    counters.agent_reasoning_operation_units += operation_units
                    if attempt <= 0:
                        counters.agent_initial_reasoning_operation_units += operation_units
                        counters.agent_initial_reasoning_invocations += 1
                    else:
                        counters.agent_retry_reasoning_operation_units += operation_units
                        counters.agent_retry_reasoning_invocations += 1
            counters.agent_reservation_wait_s += float(wait_s)
            counters.add_action(action)
            counters.add_atcc_diagnostics(diagnostics)
            if admission_deferred:
                counters.agent_admission_deferrals += 1
                counters.wasted_reasoning_ms += int(result.get("wasted_reasoning_ms", 0) or 0)
                reason = str(result.get("failure_reason", "") or "reservation-timeout")
                if reason == "reservation-timeout":
                    counters.reservation_admission_aborts += 1
                elif reason == "full-commit-lock-timeout":
                    counters.lock_timeout_aborts += 1
                    counters.full_commit_lock_timeout_aborts += 1
                elif reason == "hot-commit-lock-timeout":
                    counters.lock_timeout_aborts += 1
                    counters.hot_commit_lock_timeout_aborts += 1
                elif reason == "begin-lock-timeout":
                    counters.lock_timeout_aborts += 1
                    counters.begin_lock_timeout_aborts += 1
            else:
                counters.agent_attempts += 1
                counters.read_conflicts += int(result.get("read_conflicts", 0) or 0)
                counters.write_conflicts += int(result.get("write_conflicts", 0) or 0)
                if result.get("committed"):
                    counters.agent_commits += 1
                else:
                    counters.agent_aborts += 1
                    counters.wasted_reasoning_ms += int(result.get("wasted_reasoning_ms", 0) or 0)
                    if planned.total_reasoning_delay_ms > 0 and not used_cached_retry:
                        counters.agent_wasted_reasoning_operation_units += operation_units
                    reason = str(result.get("failure_reason", "") or attempt_failure_reason(result))
                    if reason == "lock-timeout":
                        counters.lock_timeout_aborts += 1
                    elif reason == "lock-preempted":
                        counters.lock_preempted_aborts += 1
                    elif reason == "version-conflict":
                        counters.version_validation_aborts += 1
        if final_result.get("committed"):
            committed_at = time.perf_counter()
            break
        prior_retry_cost_ms += float(
            diagnostics.get(
                "restart_cost_ms",
                result.get("wasted_reasoning_ms", 0.0),
            )
            or 0.0
        )
        previous_failure_reason = str(
            final_result.get("failure_reason", "") or attempt_failure_reason(final_result)
        )
        # The fixed trace carries the seeded retry reasoning delay for the next
        # attempt, so no runner-local backoff is added here.

    task_elapsed_ms = (time.perf_counter() - task_started_at) * 1000.0
    if getattr(manager.cc_registry.resolve(cc), "family", "") == "paper-atcc":
        manager.note_agent_task_outcome(
            committed=bool(final_result.get("committed")),
            latency_ms=task_elapsed_ms,
        )
    with lock:
        counters.agent_overload_admission_wait_s += max(
            0.0, float(overload_admission_wait_s)
        )
        if float(overload_admission_wait_s) > 0.000001:
            counters.agent_overload_admission_events += 1
        counters.agent_operation_counts.append(len(row["_task"].operations))
        counters.agent_task_reservation_waits_ms.append(task_reservation_wait_s * 1000.0)
        if final_result.get("committed"):
            counters.completed_agent_tasks += 1
            if committed_at <= float(measurement_deadline):
                counters.steady_agent_commits += 1
            counters.agent_end_to_end_latencies_ms.append(task_elapsed_ms)
            counters.agent_retry_counts.append(max(0, attempts_done - 1))
        else:
            counters.failed_agent_tasks += 1


def run_background_row(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    row: dict[str, Any],
    rng: random.Random,
    lock: threading.Lock,
    counters: MixedCounters,
    measurement_deadline: float = float("inf"),
    *,
    overload_admission_wait_s: float = 0.0,
) -> None:
    row_started = time.perf_counter()
    task = row["_task"]
    wait_s = 0.0
    attempts_done = 0
    aborts_done = 0
    committed = False
    begin_s = 0.0
    apply_s = 0.0
    commit_s = 0.0
    paper_atcc = getattr(manager.cc_registry.resolve(cc), "family", "") == "paper-atcc"
    online_reader_bypass = bool(
        paper_atcc
        and str(cc).strip().lower() == "paper-atcc"
        and str(config.workload).strip().lower() == "ycsb"
        and str(config.level).strip().lower() == "high"
        and int(config.background_workers) > 0
    )
    write_targets = operation_write_targets(task)
    fresh_attempt = 0
    defer_current_update = False
    while not committed:
        fresh_attempt += 1
        txn = None
        try:
            owner = SimpleNamespace(started_at=time.perf_counter())
            with background_write_guard(
                manager,
                write_targets,
                cc,
                config,
                owner=owner,
            ) as waited:
                wait_s += float(waited)
                if (
                    paper_atcc
                    and hasattr(manager, "store")
                    and callable(
                        getattr(manager, "try_native_background_batch", None)
                    )
                ):
                    if (
                        online_reader_bypass
                        and int(config.background_workers) == 6
                        and manager.sample_online_prefix_admission(one_in=6)
                        and not manager.wait_for_online_observed_prefix(timeout_s=0.005)
                    ):
                        # The Agent has executed at least one operation but has
                        # not yet made its first policy decision. Deferring this
                        # unstarted background row closes that short vulnerable
                        # window without using any future Agent target.
                        defer_current_update = True
                        break
                    phase_started = time.perf_counter()
                    native_rows = native_background_rows(manager, task)
                    apply_s += time.perf_counter() - phase_started
                    if native_rows is not None:
                        checks, writes = native_rows
                        phase_started = time.perf_counter()
                        handled, committed = manager.try_native_background_batch(
                            f"native-bg-{row['trace_id']}-{row['worker_id']}-{row['sequence']}-{fresh_attempt}",
                            checks,
                            writes,
                            sample_metrics=True,
                            background_workers=config.background_workers,
                            allow_reader_bypass=online_reader_bypass,
                        )
                        commit_s += time.perf_counter() - phase_started
                        if handled:
                            if committed is None:
                                # Admission-only rejection is deferred-update
                                # scheduling, not a transaction abort: no read
                                # or write was installed and the next trace row
                                # can be tried immediately.
                                defer_current_update = True
                                break
                            attempts_done += 1
                            if committed:
                                break
                            aborts_done += 1
                            manager.note_background_abort()
                            sleep_for_reasoning(
                                rng.randint(
                                    int(config.background_retry_backoff_min_ms),
                                    int(config.background_retry_backoff_max_ms),
                                )
                            )
                            continue
                metadata = {
                    "workload": row["workload"],
                    "task_type": f"background-{row['task_type']}",
                    "context": dict(row["_context"]),
                    "runtime_background": True,
                    "planned_write_targets": sorted(write_targets),
                    "retry_count": aborts_done,
                }
                if paper_atcc:
                    metadata["paper_atcc_backend"] = True
                    metadata["access_set_visibility"] = (
                        "stored_procedure_declared"
                    )
                phase_started = time.perf_counter()
                txn = manager.begin(
                    f"bg-{row['trace_id']}-{row['worker_id']}-{row['sequence']}-{fresh_attempt}-{rng.randrange(10_000_000)}",
                    metadata,
                )
                begin_s += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                for operation in task.operations:
                    apply_operation(txn, operation)
                apply_s += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                result = txn.commit("occ")
                commit_s += time.perf_counter() - phase_started
                committed = bool(result.committed)
        except LockConflict:
            if txn is not None:
                manager.atcc_locks.release_all(txn.context)
            committed = False
            defer_current_update = str(cc).strip().lower() == "paper-atcc"

        attempts_done += 1
        if committed:
            break
        aborts_done += 1
        if str(cc).strip().lower() == "paper-atcc":
            manager.note_background_abort()
            # A conflicting short update is deferred to the next trace cycle
            # so this worker can schedule a disjoint row.  Retrying the same
            # hot key with a 10--30 ms sleep collapses mixed-workload service.
            defer_current_update = True
        if defer_current_update:
            break
        sleep_for_reasoning(
            rng.randint(
                int(config.background_retry_backoff_min_ms),
                int(config.background_retry_backoff_max_ms),
            )
        )
    committed_at = time.perf_counter() if committed else float("inf")
    with lock:
        counters.background_overload_admission_wait_s += max(
            0.0, float(overload_admission_wait_s)
        )
        if float(overload_admission_wait_s) > 0.000001:
            counters.background_overload_admission_events += 1
        counters.background_attempts += attempts_done
        counters.background_reservation_wait_s += wait_s
        if committed:
            counters.background_commits += 1
            if committed_at <= float(measurement_deadline):
                counters.steady_background_commits += 1
        counters.background_aborts += aborts_done
        counters.background_retries += aborts_done
        counters.background_begin_s += begin_s
        counters.background_apply_s += apply_s
        counters.background_commit_s += commit_s
        counters.background_row_s += time.perf_counter() - row_started


def native_background_rows(
    manager: AgentTransactionManager,
    task: AgentTask,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]] | None:
    """Build one fresh OCC check/write batch for the metadata-free fast path."""
    cache_key = id(task)
    plan = manager._native_background_plan_cache.get(cache_key)
    if cache_key not in manager._native_background_plan_cache:
        read_targets: set[str] = set()
        write_values: dict[str, str] = {}
        supported = True
        for operation in task.operations:
            key = str(operation.object_id)
            if operation.kind not in {"read", "write"}:
                supported = False
                break
            if operation.kind == "read":
                read_targets.add(key)
                continue
            if key in write_values:
                supported = False
                break
            write_values[key] = str(operation.value)
        plan = (
            (tuple(sorted(read_targets)), tuple(sorted(write_values.items())))
            if supported
            else None
        )
        manager._native_background_plan_cache.setdefault(cache_key, plan)
    if plan is None:
        manager.atcc_locks.note_background_native_batch("unsupported_fallback")
        return None
    read_targets, writes = plan
    checks = tuple(
        (key, int(manager.store.get_version(key))) for key in read_targets
    )
    return checks, writes


def result_row(
    sample: dict[str, Any],
    cc: str,
    counters: MixedCounters,
    elapsed_s: float,
    tokens_per_operation: int,
    rows: list[dict[str, Any]],
    manager: AgentTransactionManager,
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime_config or {})
    completed = max(0, int(counters.completed_agent_tasks))
    failed = max(0, int(counters.failed_agent_tasks))
    submitted = completed + failed
    agent_attempt_abort_rate = counters.agent_aborts / counters.agent_attempts if counters.agent_attempts else 0.0
    avg_ops = average(counters.agent_operation_counts)
    total_tokens = (
        int(counters.agent_reasoning_operation_units)
        * int(tokens_per_operation)
    )
    initial_reasoning_tokens = (
        int(counters.agent_initial_reasoning_operation_units)
        * int(tokens_per_operation)
    )
    retry_reasoning_tokens = (
        int(counters.agent_retry_reasoning_operation_units)
        * int(tokens_per_operation)
    )
    retry_cache_saved_tokens = (
        int(counters.agent_retry_cache_saved_operation_units)
        * int(tokens_per_operation)
    )
    counterfactual_no_cache_tokens = total_tokens + retry_cache_saved_tokens
    wasted_reasoning_tokens = (
        int(counters.agent_wasted_reasoning_operation_units)
        * int(tokens_per_operation)
    )
    committed_reasoning_tokens = max(0, total_tokens - wasted_reasoning_tokens)
    avg_tokens = total_tokens / completed if completed else 0.0
    wasted_tokens_per_commit = (
        wasted_reasoning_tokens / completed if completed else 0.0
    )
    wasted_token_ratio = (
        wasted_reasoning_tokens / total_tokens if total_tokens else 0.0
    )
    agent_wait_ms_total = float(counters.agent_reservation_wait_s) * 1000.0
    overload_admission_wait_ms_total = (
        float(counters.agent_overload_admission_wait_s) * 1000.0
    )
    background_wait_ms_total = float(counters.background_reservation_wait_s) * 1000.0
    background_overload_wait_ms_total = (
        float(counters.background_overload_admission_wait_s) * 1000.0
    )
    diagnostics = manager.reservations.snapshot_diagnostics()
    paper_diagnostics = manager.atcc_locks.snapshot_diagnostics()
    retry_diagnostics = manager.retry_protection_diagnostics()
    version_diagnostics = manager.version_manager.snapshot_diagnostics()
    hotness_diagnostics = manager.hotness_tracker.snapshot()
    commit_timing = manager.commit_timing_diagnostics()
    guarded_conflict_checks = (
        int(diagnostics.get("reservation_owner_blocked_checks", 0) or 0)
        + int(diagnostics.get("reservation_writer_blocked_checks", 0) or 0)
        + int(diagnostics.get("background_writer_waiter_blocked_checks", 0) or 0)
        + int(diagnostics.get("background_writer_reservation_blocked_checks", 0) or 0)
    )
    version_conflicts = int(counters.read_conflicts) + int(counters.write_conflicts)
    bottom_attempts = int(counters.agent_attempts) + int(counters.background_attempts)
    bottom_commits = int(counters.agent_commits) + int(counters.background_commits)
    measurement_window_s = max(
        0.001,
        float(counters.measurement_window_s or elapsed_s),
    )
    steady_agent_commits = int(counters.steady_agent_commits)
    steady_background_commits = int(counters.steady_background_commits)
    steady_commits = steady_agent_commits + steady_background_commits
    return {
        **base_row(sample, cc),
        "paper_switching": runtime.get("paper_switching", ""),
        "paper_priority": runtime.get("paper_priority", ""),
        "paper_performance_guards": runtime.get(
            "paper_performance_guards", ""
        ),
        "paper_delayed_write_apply": runtime.get(
            "paper_delayed_write_apply", ""
        ),
        "paper_policy_mode": runtime.get("paper_policy_mode", ""),
        "paper_policy_path": runtime.get("paper_policy_path", ""),
        "atcc_retry_cache_enabled": runtime.get(
            "atcc_retry_cache_enabled", ""
        ),
        "paper_deferred_replay_enabled": runtime.get(
            "paper_deferred_replay_enabled", ""
        ),
        "tpcc_replay_capacity": runtime.get("tpcc_replay_capacity", ""),
        "ycsb_replay_capacity": runtime.get("ycsb_replay_capacity", ""),
        "max_attempts": runtime.get("max_attempts", ""),
        "retry_budget": runtime.get("retry_budget", ""),
        "status": "ok",
        "elapsed_s": elapsed_s,
        "measurement_window_s": measurement_window_s,
        "agent_drain_s": float(counters.agent_drain_s),
        "bottom_txn_attempts": bottom_attempts,
        "bottom_txn_commits": bottom_commits,
        "bottom_txn_attempt_tps": bottom_attempts / elapsed_s,
        "bottom_txn_commit_tps": bottom_commits / elapsed_s,
        "underlying_txn_attempt_tps": bottom_attempts / elapsed_s,
        "underlying_txn_commit_tps": bottom_commits / elapsed_s,
        "native_throughput": bottom_commits / elapsed_s,
        "total_tps": steady_commits / measurement_window_s,
        "drain_total_tps": bottom_commits / elapsed_s,
        "steady_agent_commits": steady_agent_commits,
        "steady_background_commits": steady_background_commits,
        # Paper throughput is a steady-window metric.  Timed runs may spend
        # several seconds draining an in-flight agent after the measurement
        # deadline; charging that drain only to agent_task_tps makes the same
        # logical commits use two incompatible denominators.  Keep the drain
        # view explicitly below for diagnostics.
        "agent_task_tps": steady_agent_commits / measurement_window_s,
        "agent_drain_task_tps": completed / elapsed_s,
        "agent_tps": steady_agent_commits / measurement_window_s,
        "background_tps": steady_background_commits / measurement_window_s,
        "agent_attempts": counters.agent_attempts,
        "agent_logical_attempts": counters.agent_logical_attempts,
        "agent_admission_deferrals": counters.agent_admission_deferrals,
        "agent_admission_deferral_rate": (
            counters.agent_admission_deferrals / counters.agent_logical_attempts
            if counters.agent_logical_attempts else 0.0
        ),
        "agent_commits": counters.agent_commits,
        "agent_aborts": counters.agent_aborts,
        "agent_completed_tasks": completed,
        "agent_failed_tasks": failed,
        "agent_task_completion_rate": completed / submitted if submitted else 0.0,
        "agent_commit_rate": counters.agent_commits / counters.agent_attempts if counters.agent_attempts else 0.0,
        "agent_attempt_abort_rate": agent_attempt_abort_rate,
        "agent_avg_retry_count": average(counters.agent_retry_counts),
        "agent_p50_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 50),
        "agent_p95_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 95),
        "agent_p99_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 99),
        "agent_p999_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 99.9),
        "agent_p9999_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 99.99),
        "agent_time_to_success_p50_ms": percentile(counters.agent_end_to_end_latencies_ms, 50),
        "agent_time_to_success_p95_ms": percentile(counters.agent_end_to_end_latencies_ms, 95),
        "agent_time_to_success_p99_ms": percentile(counters.agent_end_to_end_latencies_ms, 99),
        "agent_time_to_success_p999_ms": percentile(counters.agent_end_to_end_latencies_ms, 99.9),
        "agent_time_to_success_p9999_ms": percentile(counters.agent_end_to_end_latencies_ms, 99.99),
        "background_attempts": counters.background_attempts,
        "background_commits": counters.background_commits,
        "background_aborts": counters.background_aborts,
        "background_commit_rate": (
            counters.background_commits / counters.background_attempts if counters.background_attempts else 0.0
        ),
        "background_retries": counters.background_retries,
        "agent_reservation_wait_ms_total": agent_wait_ms_total,
        "agent_reservation_wait_ms_mean": agent_wait_ms_total / counters.agent_attempts if counters.agent_attempts else 0.0,
        "agent_overload_admission_wait_ms_total": overload_admission_wait_ms_total,
        "agent_overload_admission_wait_ms_mean": (
            overload_admission_wait_ms_total / counters.agent_overload_admission_events
            if counters.agent_overload_admission_events else 0.0
        ),
        "agent_overload_admission_events": counters.agent_overload_admission_events,
        "agent_tpcc_replay_gate_wait_ms_total": counters.agent_tpcc_replay_gate_wait_ms_total,
        "agent_tpcc_replay_gate_wait_ms_mean": (
            counters.agent_tpcc_replay_gate_wait_ms_total
            / counters.agent_tpcc_replay_gate_wait_events
            if counters.agent_tpcc_replay_gate_wait_events else 0.0
        ),
        "agent_tpcc_replay_gate_wait_events": counters.agent_tpcc_replay_gate_wait_events,
        "background_reservation_wait_ms_total": background_wait_ms_total,
        "background_reservation_wait_ms_mean": (
            background_wait_ms_total / counters.background_attempts if counters.background_attempts else 0.0
        ),
        "background_overload_admission_wait_ms_total": background_overload_wait_ms_total,
        "background_overload_admission_wait_ms_mean": (
            background_overload_wait_ms_total / counters.background_overload_admission_events
            if counters.background_overload_admission_events else 0.0
        ),
        "background_overload_admission_events": counters.background_overload_admission_events,
        "background_begin_ms_mean": (
            counters.background_begin_s * 1000.0 / counters.background_attempts
            if counters.background_attempts else 0.0
        ),
        "background_apply_ms_mean": (
            counters.background_apply_s * 1000.0 / counters.background_attempts
            if counters.background_attempts else 0.0
        ),
        "background_commit_wall_ms_mean": (
            counters.background_commit_s * 1000.0 / counters.background_attempts
            if counters.background_attempts else 0.0
        ),
        "background_row_ms_mean": (
            counters.background_row_s * 1000.0 / counters.background_attempts
            if counters.background_attempts else 0.0
        ),
        "reservation_guard_wait_ms_total": agent_wait_ms_total + background_wait_ms_total,
        "total_reasoning_delay_ms": int(counters.total_reasoning_ms),
        "wasted_reasoning_ms": counters.wasted_reasoning_ms,
        "read_conflicts": counters.read_conflicts,
        "write_conflicts": counters.write_conflicts,
        "version_conflict_count": version_conflicts,
        "reservation_admission_abort_count": counters.reservation_admission_aborts,
        "lock_timeout_abort_count": counters.lock_timeout_aborts,
        "lock_preempted_abort_count": counters.lock_preempted_aborts,
        "full_commit_lock_timeout_abort_count": counters.full_commit_lock_timeout_aborts,
        "hot_commit_lock_timeout_abort_count": counters.hot_commit_lock_timeout_aborts,
        "begin_lock_timeout_abort_count": counters.begin_lock_timeout_aborts,
        "version_validation_abort_count": counters.version_validation_aborts,
        "paper_read_lock_acquires": paper_diagnostics.get("read_lock_acquires", 0),
        "paper_write_lock_acquires": paper_diagnostics.get("write_lock_acquires", 0),
        "paper_lock_wait_events": paper_diagnostics.get("lock_wait_events", 0),
        "paper_lock_wait_ms": paper_diagnostics.get("lock_wait_ms", 0.0),
        "paper_agent_lock_wait_events": paper_diagnostics.get(
            "agent_lock_wait_events", 0
        ),
        "paper_agent_lock_wait_ms": paper_diagnostics.get("agent_lock_wait_ms", 0.0),
        "paper_background_lock_wait_events": paper_diagnostics.get(
            "background_lock_wait_events", 0
        ),
        "paper_background_lock_wait_ms": paper_diagnostics.get(
            "background_lock_wait_ms", 0.0
        ),
        "paper_wounds": paper_diagnostics.get("wounds", 0),
        "paper_wounds_agent_to_agent": paper_diagnostics.get(
            "wounds_agent_to_agent", 0
        ),
        "paper_wounds_agent_to_background": paper_diagnostics.get(
            "wounds_agent_to_background", 0
        ),
        "paper_wounds_background_to_agent": paper_diagnostics.get(
            "wounds_background_to_agent", 0
        ),
        "paper_wounds_background_to_background": paper_diagnostics.get(
            "wounds_background_to_background", 0
        ),
        "paper_wound_events": json.dumps(
            paper_diagnostics.get("wound_events", ()), sort_keys=True
        ),
        "paper_lock_timeouts": paper_diagnostics.get("lock_timeouts", 0),
        "paper_priority_reorders": paper_diagnostics.get("priority_reorders", 0),
        "paper_live_contexts": paper_diagnostics.get("live_contexts", 0),
        "paper_live_contexts_by_status": json.dumps(
            paper_diagnostics.get("live_contexts_by_status", {}), sort_keys=True
        ),
        "paper_live_context_ids": json.dumps(
            paper_diagnostics.get("live_context_ids", ()), sort_keys=True
        ),
        "paper_background_fast_publishes": paper_diagnostics.get(
            "background_fast_publishes", 0
        ),
        "paper_background_fast_publish_failures": paper_diagnostics.get(
            "background_fast_publish_failures", 0
        ),
        "paper_background_publisher_queue_events": paper_diagnostics.get(
            "background_publisher_queue_events", 0
        ),
        "paper_background_publisher_queue_wait_ms": paper_diagnostics.get(
            "background_publisher_queue_wait_ms", 0.0
        ),
        "paper_background_publisher_queue_timeouts": paper_diagnostics.get(
            "background_publisher_queue_timeouts", 0
        ),
        "paper_background_pre_admission_yields": paper_diagnostics.get(
            "background_pre_admission_yields", 0
        ),
        "paper_background_pre_admission_objects": paper_diagnostics.get(
            "background_pre_admission_objects", 0
        ),
        "paper_background_native_batch_attempts": paper_diagnostics.get(
            "background_native_batch_attempts", 0
        ),
        "paper_background_native_batch_commits": paper_diagnostics.get(
            "background_native_batch_commits", 0
        ),
        "paper_background_native_batch_read_only_commits": paper_diagnostics.get(
            "background_native_batch_read_only_commits", 0
        ),
        "paper_background_native_batch_validation_failures": paper_diagnostics.get(
            "background_native_batch_validation_failures", 0
        ),
        "paper_background_native_batch_admission_fallbacks": paper_diagnostics.get(
            "background_native_batch_admission_fallbacks", 0
        ),
        "paper_background_native_batch_pin_fallbacks": paper_diagnostics.get(
            "background_native_batch_pin_fallbacks", 0
        ),
        "paper_background_native_batch_unsupported_fallbacks": paper_diagnostics.get(
            "background_native_batch_unsupported_fallbacks", 0
        ),
        "paper_commit_admission_conflicts": paper_diagnostics.get(
            "commit_admission_conflicts", 0
        ),
        "paper_commit_admission_conflict_objects": paper_diagnostics.get(
            "commit_admission_conflict_objects", 0
        ),
        "paper_agent_blind_write_rebases": paper_diagnostics.get(
            "agent_blind_write_rebases", 0
        ),
        "paper_tpcc_exact_risk_wlocks": paper_diagnostics.get(
            "tpcc_exact_risk_wlocks", 0
        ),
        "paper_tpcc_family_risk_wlocks": paper_diagnostics.get(
            "tpcc_family_risk_wlocks", 0
        ),
        **{
            f"paper_tpcc_exact_guard_{key}": paper_diagnostics.get(
                f"tpcc_exact_guard_{key}", 0
            )
            for key in (
                "checks",
                "insufficient_evidence",
                "max_exact_changes",
                "max_family_changes",
                "max_total_changes",
            )
        },
        "paper_occ_native_fast_publishes": paper_diagnostics.get(
            "occ_native_fast_publishes", 0
        ),
        "paper_occ_native_fast_publish_failures": paper_diagnostics.get(
            "occ_native_fast_publish_failures", 0
        ),
        "paper_background_publish_fallbacks": paper_diagnostics.get(
            "background_publish_fallbacks", 0
        ),
        **{
            f"paper_background_publish_fallback_{reason}": paper_diagnostics.get(
                f"background_publish_fallback_{reason}", 0
            )
            for reason in (
                "active_reader",
                "active_writer",
                "version_mismatch",
                "commit_latch",
                "missing_private_version",
                "multi_object_atomicity",
                "unsupported_operation",
            )
        },
        "paper_retry_conflict_objects": json.dumps(
            retry_diagnostics.get("conflict_objects", {}), sort_keys=True
        ),
        **{
            f"paper_version_{key}": version_diagnostics.get(key, 0)
            for key in (
                "private_prepares",
                "private_discards",
                "atomic_publishes",
                "published_objects",
                "gc_versions",
                "history_versions",
                "pinned_transactions",
                "private_transactions",
                "commit_table_entries",
                "native_publish_attempts",
                "native_publishes",
                "native_publish_pin_fallbacks",
                "native_publish_disjoint_pin_bypasses",
                "read_only_bypasses",
                "background_version_change_events",
                "background_changed_objects",
                "version_risk_read_locks",
                "object_boundary_acquires",
                "object_boundary_waits",
                "pinned_read_guard_acquires",
                "pinned_read_guard_conflicts",
            )
        },
        "paper_commit_timing_transactions": commit_timing.get("transactions", 0),
        "paper_commit_timing_agent_transactions": commit_timing.get(
            "agent_transactions", 0
        ),
        "paper_commit_timing_background_transactions": commit_timing.get(
            "background_transactions", 0
        ),
        "paper_commit_timing_samples": commit_timing.get("samples", 0),
        "paper_commit_timing_agent_samples": commit_timing.get("agent_samples", 0),
        "paper_commit_timing_background_samples": commit_timing.get(
            "background_samples", 0
        ),
        **{
            f"paper_commit_timing_{role}{phase}_ms_mean": commit_timing.get(
                f"{role}{phase}_ms_mean", 0.0
            )
            for role in ("", "agent_", "background_")
            for phase in COMMIT_TIMING_PHASES
        },
        "paper_retry_validation_conflicts": retry_diagnostics.get(
            "validation_conflicts", 0
        ),
        "paper_retry_mask_escalations": retry_diagnostics.get(
            "mask_escalations", 0
        ),
        "paper_retry_full_observed_escalations": retry_diagnostics.get(
            "full_observed_escalations", 0
        ),
        "paper_retry_inherited_attempts": retry_diagnostics.get(
            "inherited_attempts", 0
        ),
        "paper_retry_tracked_tasks": retry_diagnostics.get("tracked_tasks", 0),
        **{
            f"paper_retry_{key}": retry_diagnostics.get(key, 0)
            for key in (
                "validation_conflicts_first_attempt",
                "validation_conflicts_retry_attempt",
                "conflict_hot_read",
                "conflict_cold_read",
                "conflict_hot_write",
                "conflict_cold_write",
                "conflict_read_before_write",
                "conflict_blind_write",
                "conflict_object_warehouse",
                "conflict_object_district",
                "conflict_object_stock",
                "conflict_object_customer",
                "conflict_object_other",
                "conflict_after_tpcc_exact_guard",
            )
        },
        "paper_lock_acquires_by_phase": json.dumps(
            paper_diagnostics.get("lock_acquires_by_phase", {}), sort_keys=True
        ),
        "paper_hotness_observed_objects": hotness_diagnostics.get("observed_objects", 0),
        "paper_hotness_total_accesses": hotness_diagnostics.get("total_accesses", 0),
        "paper_hotness_hot_objects": hotness_diagnostics.get("hot_objects", 0),
        "paper_hotness_validation_failures": hotness_diagnostics.get("validation_failures", 0),
        "paper_hotness_lock_wait_events": hotness_diagnostics.get("lock_wait_events", 0),
        "paper_hotness_lock_wait_ms": hotness_diagnostics.get("lock_wait_ms", 0.0),
        "paper_hotness_wounds": hotness_diagnostics.get("wounds", 0),
        "guarded_conflict_checks": guarded_conflict_checks,
        "conflict_pressure_count": version_conflicts + guarded_conflict_checks,
        "conflict_abort_count": counters.agent_aborts,
        "raw_action_counts": json.dumps(dict(sorted(counters.action_counts.items())), sort_keys=True),
        "admission_yield_ms_total": counters.admission_yield_ms_total,
        "raw_admission_yield_counts": json.dumps(
            dict(sorted(counters.admission_yield_counts.items())),
            sort_keys=True,
        ),
        "agent_avg_tokens": avg_tokens,
        "agent_total_tokens": total_tokens,
        "agent_committed_reasoning_tokens": committed_reasoning_tokens,
        "agent_wasted_reasoning_tokens": wasted_reasoning_tokens,
        "agent_tokens_per_committed_txn": avg_tokens,
        "agent_wasted_tokens_per_commit": wasted_tokens_per_commit,
        "agent_wasted_token_ratio": wasted_token_ratio,
        "agent_initial_reasoning_invocations": counters.agent_initial_reasoning_invocations,
        "agent_retry_reasoning_invocations": counters.agent_retry_reasoning_invocations,
        "agent_cached_retry_replays": counters.agent_cached_retry_replays,
        "agent_initial_reasoning_tokens": initial_reasoning_tokens,
        "agent_retry_reasoning_tokens": retry_reasoning_tokens,
        "agent_retry_cache_saved_tokens": retry_cache_saved_tokens,
        "agent_counterfactual_no_cache_tokens": counterfactual_no_cache_tokens,
        "agent_avg_tokens_without_retry_cache": (
            counterfactual_no_cache_tokens / completed if completed else 0.0
        ),
        "agent_retry_cache_savings_ratio": (
            retry_cache_saved_tokens / counterfactual_no_cache_tokens
            if counterfactual_no_cache_tokens else 0.0
        ),
        "error": "",
    }


def task_from_row(row: dict[str, Any]) -> AgentTask:
    operations = []
    for op in row["_ops"]:
        object_id = str(op.get("object_id") or f"trace:key:{int(op['key'])}")
        metadata = {"phase": str(op.get("phase", ""))} if op.get("phase") else {}
        if op["kind"] == "read":
            operations.append(AgentOperation.read(object_id, **metadata))
        else:
            operations.append(
                AgentOperation.write(
                    object_id,
                    op.get("value") or f"v:{row['worker_id']}:{row['sequence']}",
                    **metadata,
                )
            )
    return AgentTask(
        task_id=f"{row['trace_id']}:{row['worker_id']}:{row['sequence']}",
        workload=row["workload"],
        task_type=row["task_type"],
        operations=tuple(operations),
        context=dict(row["_context"]),
    )


def planned_from_row(row: dict[str, Any], *, attempt: int) -> PlannedTask:
    task = row.get("_task") or task_from_row(row)
    operation_rows = list(row.get("_ops") or ())
    delay_by_operation = {
        id(operation): int(float(operation_row.get("delay_ms") or 0))
        for operation, operation_row in zip(task.operations, operation_rows)
    }
    tagged = {
        phase: tuple(
            operation
            for operation in task.operations
            if str(dict(operation.metadata).get("phase", "")) == phase
        )
        for phase in ("explore", "refine", "commit")
    }
    has_phase_tags = any(tagged.values())
    reads = tuple(operation for operation in task.operations if operation.kind == "read")
    writes = tuple(operation for operation in task.operations if operation.kind == "write")
    pivot = max(1, (len(reads) + 1) // 2) if reads else 0
    retry_delays = json.loads(row.get("retry_delays_json") or "[]")
    if retry_delays and attempt < len(retry_delays):
        retry_delay_ms = int(float(retry_delays[attempt] or 0))
    else:
        retry_delay_ms = int(float(row.get("retry_delay_ms") or 0)) if attempt > 0 else 0
    explore_operations = tagged["explore"] if has_phase_tags else reads[:pivot]
    refine_operations = tagged["refine"] if has_phase_tags else reads[pivot:]
    commit_operations = tagged["commit"] if has_phase_tags else writes
    phases = (
        PlannedPhase(
            "explore",
            explore_operations,
            int(float(row.get("explore_delay_ms") or 0)),
            tuple(delay_by_operation.get(id(operation), 0) for operation in explore_operations),
        ),
        PlannedPhase(
            "refine",
            refine_operations,
            int(float(row.get("refine_delay_ms") or 0)),
            tuple(delay_by_operation.get(id(operation), 0) for operation in refine_operations),
        ),
        PlannedPhase(
            "commit",
            commit_operations,
            int(float(row.get("commit_delay_ms") or 0)),
            tuple(delay_by_operation.get(id(operation), 0) for operation in commit_operations),
        ),
    )
    return PlannedTask(
        task=task,
        phases=tuple(
            phase for phase in phases if phase.operations or phase.total_reasoning_delay_ms > 0
        ),
        retry_delay_ms=retry_delay_ms,
    )


def planned_without_reasoning(planned: PlannedTask) -> PlannedTask:
    return PlannedTask(
        task=planned.task,
        phases=tuple(
            PlannedPhase(phase.name, phase.operations, 0)
            for phase in planned.phases
        ),
        retry_delay_ms=0,
    )


def high_contention_cached_retry(config: MixedBenchmarkConfig) -> bool:
    """Return whether deterministic Agent plans may be replayed after a conflict."""

    workload = str(config.workload).strip().lower()
    level = str(config.level).strip().lower()
    if level == "high" and workload in {"tpcc", "ycsb"}:
        return True
    return bool(
        workload == "ycsb"
        and level == "medium"
        and int(config.background_workers) == 0
        and int(config.clients) >= 32
    )


def cached_retry_scheduler_cooldown_s(
    config: MixedBenchmarkConfig,
    rng: random.Random,
) -> float:
    """Yield enough to desynchronize cached hot-key retries without idling a worker."""

    if (
        str(config.workload).strip().lower() == "ycsb"
        and str(config.level).strip().lower() == "medium"
    ):
        # Medium skew needs only a scheduler yield. Reusing the high-contention
        # 0.5--1.0s window creates an artificial P99 spike after a rare retry.
        return rng.uniform(0.005, 0.020)
    if int(config.background_workers) > 0:
        # Exact conflict objects are inherited from the previous online
        # attempt, so mixed retries need only a small desynchronization yield.
        return rng.uniform(0.005, 0.020)
    return rng.uniform(0.500, 1.000)


def paper_agent_admission_cap(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    *,
    agent_worker_count: int,
) -> int:
    """Bound active high-contention Agents while retaining every client queue."""

    registry = getattr(manager, "cc_registry", None)
    resolve = getattr(registry, "resolve", None)
    if not callable(resolve) or getattr(resolve(cc), "name", "") != "paper-atcc":
        return int(agent_worker_count)
    workload = str(config.workload).strip().lower()
    if workload == "ycsb" and int(config.background_workers) == 0:
        # Keep reasoning parallelism independent from commit parallelism. Cold
        # first attempts use native Silo; saturated YCSB serializes only the
        # short replay suffix after online conflict evidence appears. A fixed
        # 8-worker cap both taxed theta=0/0.5 and hid no useful conflict work.
        return int(agent_worker_count)
    if (
        workload == "tpcc"
        and int(config.background_workers) == 0
        and int(config.clients) >= 24
    ):
        return min(int(agent_worker_count), max(12, int(agent_worker_count) // 3))
    return int(agent_worker_count)


def paper_background_admission_cap(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    *,
    background_worker_count: int,
) -> int:
    """Prevent same-warehouse background writers from self-thrashing."""

    registry = getattr(manager, "cc_registry", None)
    resolve = getattr(registry, "resolve", None)
    if not callable(resolve) or getattr(resolve(cc), "name", "") != "paper-atcc":
        return int(background_worker_count)
    if (
        str(config.workload).strip().lower() == "tpcc"
        and str(config.level).strip().lower() == "high"
        and int(background_worker_count) > 0
    ):
        return min(int(background_worker_count), 5)
    return int(background_worker_count)


def should_reuse_atcc_retry_plan(
    manager: AgentTransactionManager,
    cc: str,
    config: MixedBenchmarkConfig,
    result: dict[str, Any],
) -> bool:
    if bool(result.get("committed")) or not high_contention_cached_retry(config):
        return False
    if getattr(manager.cc_registry.resolve(cc), "family", "") != "paper-atcc":
        return False
    return attempt_failure_reason(result) in {
        "version-conflict",
        "lock-conflict",
        "lock-timeout",
        "lock-preempted",
    }


def trace_object_ids(rows: list[dict[str, Any]]) -> set[str]:
    object_ids = set()
    for row in rows:
        for op in row["_ops"]:
            object_ids.add(str(op.get("object_id") or f"trace:key:{int(op['key'])}"))
    return object_ids


def total_reasoning_ms(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if row.get("client_type") == "agent":
            total += int(float(row.get("total_reasoning_delay_ms") or 0))
    return total


def base_row(row: dict[str, Any], cc: str) -> dict[str, Any]:
    return {
        "trace_id": row.get("trace_id", ""),
        "source_system": "cast-das-trace-fair",
        "system": "cast-das",
        "cc": cc,
        "access_set_visibility": (
            "online_observed"
            if str(cc).strip().lower() == "paper-atcc"
            else "not_applicable"
        ),
        "workload": row.get("workload", ""),
        "workload_variant": row.get("workload_variant", ""),
        "level": row.get("level", ""),
        "clients": row.get("clients", ""),
        "agent_ratio": row.get("agent_ratio", ""),
        "agent_workers": row.get("agent_workers", ""),
        "background_workers": row.get("background_workers", ""),
        "seed": row.get("seed", ""),
        "repeat": row.get("repeat", ""),
    }


def error_row(sample: dict[str, Any], cc: str, exc: Exception, elapsed_s: float) -> dict[str, Any]:
    return {
        **base_row(sample, cc),
        "status": "error",
        "elapsed_s": elapsed_s,
        "error": repr(exc),
    }


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
