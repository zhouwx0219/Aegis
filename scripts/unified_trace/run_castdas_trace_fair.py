#!/usr/bin/env python3
"""Replay a fixed CAST-DAS trace with the paper agent runtime semantics.

Unlike ``run_castdas_trace.py``, this runner preserves the mixed benchmark's
agent execution path: reasoning happens inside the transaction/ATCC runtime
window, ATCC decisions are made before execution, reservations/deferred begin
paths are honored, and wasted reasoning is counted on aborts.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
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
)
from agent.benchmarks.phases import PlannedPhase, PlannedTask, sleep_for_reasoning  # noqa: E402
from agent.cc import LockConflict  # noqa: E402
from agent.cc.atcc.ppo import DiscretePPOPolicy, EpsilonGreedyPolicy  # noqa: E402
from agent.runtime import AgentTransactionManager  # noqa: E402
from agent.runtime import CompiledPhasePolicy  # noqa: E402
from agent.workloads import AgentOperation, AgentTask, apply_operation  # noqa: E402


CCS = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,paper-atcc"

FIELDS = [
    "trace_id",
    "source_system",
    "system",
    "cc",
    "workload",
    "workload_variant",
    "level",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "seed",
    "repeat",
    "status",
    "elapsed_s",
    "bottom_txn_attempts",
    "bottom_txn_commits",
    "bottom_txn_attempt_tps",
    "bottom_txn_commit_tps",
    "underlying_txn_attempt_tps",
    "underlying_txn_commit_tps",
    "native_throughput",
    "total_tps",
    "agent_task_tps",
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
    "background_reservation_wait_ms_total",
    "background_reservation_wait_ms_mean",
    "background_begin_ms_mean",
    "background_apply_ms_mean",
    "background_commit_wall_ms_mean",
    "background_row_ms_mean",
    "reservation_guard_wait_ms_total",
    "total_reasoning_delay_ms",
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
    "paper_lock_timeouts",
    "paper_priority_reorders",
    "paper_background_fast_publishes",
    "paper_background_fast_publish_failures",
    "paper_background_publisher_queue_events",
    "paper_background_publisher_queue_wait_ms",
    "paper_background_publisher_queue_timeouts",
    "paper_background_admission_queue_events",
    "paper_background_admission_queue_wait_ms",
    "paper_background_admission_queue_timeouts",
    "paper_commit_admission_conflicts",
    "paper_commit_admission_conflict_objects",
    "paper_agent_blind_write_rebases",
    "paper_occ_native_fast_publishes",
    "paper_occ_native_fast_publish_failures",
    "paper_background_publish_fallbacks",
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
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--tokens-per-operation", type=int, default=2703)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--measure-seconds", type=float, default=0.0)
    parser.add_argument("--no-cycle-trace", action="store_false", dest="cycle_trace")
    parser.set_defaults(cycle_trace=True)
    args = parser.parse_args()

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
                    max_attempts=args.max_attempts,
                    tokens_per_operation=args.tokens_per_operation,
                    warmup_seconds=args.warmup_seconds,
                    measure_seconds=args.measure_seconds,
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
    max_attempts: int,
    tokens_per_operation: int,
    warmup_seconds: float = 0.0,
    measure_seconds: float = 0.0,
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
        background_retry_backoff_min_ms=1,
        background_retry_backoff_max_ms=3,
        tokens_per_operation=tokens_per_operation,
    ).normalized()
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
    manager = AgentTransactionManager(
        cc_registry=registry_for(config),
        record_traces=False,
        paper_policy=compiled_policy,
        collect_trajectories=trajectory_output is not None or paper_exploration_seed is not None,
        low_conflict_occ_guard=cc == "paper-atcc",
    )
    all_rows = list(rows)
    if warmup_rows:
        all_rows.extend(warmup_rows)
    for object_id in sorted(trace_object_ids(all_rows)):
        manager.register_object(object_id, "0", kind="row")
    if cc == "paper-atcc":
        manager.hotness_tracker.prime_accesses(
            operation.object_id
            for row in all_rows
            for operation in row["_task"].operations
        )
        for row in all_rows:
            manager.hotness_tracker.prime_transaction(
                operation.object_id for operation in row["_task"].operations
            )

    if warmup_rows:
        run_rows(
            manager,
            cc,
            config,
            warmup_rows,
            max_attempts,
            duration_s=float(warmup_seconds),
            cycle_trace=cycle_trace,
        )
        manager.trajectory_collector.clear()
    counters, elapsed_s = run_rows(
        manager,
        cc,
        config,
        rows,
        max_attempts,
        duration_s=float(measure_seconds),
        cycle_trace=cycle_trace,
    )
    result = result_row(sample, cc, counters, elapsed_s, tokens_per_operation, rows, manager)
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
) -> tuple[MixedCounters, float]:
    by_worker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_worker[int(float(row["worker_id"]))].append(row)
    for worker_rows in by_worker.values():
        worker_rows.sort(key=lambda row: int(float(row["sequence"])))

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

    lock = threading.Lock()
    counters = MixedCounters()
    barrier = threading.Barrier(len(by_worker) + 1)
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
    for thread in threads:
        thread.join()
    if thread_errors:
        raise RuntimeError(f"worker failed: {thread_errors[0]}") from thread_errors[0]
    elapsed_s = max(0.001, time.perf_counter() - started)
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
            )
        for row in selected_rows:
            if row["client_type"] == "agent":
                run_agent_row(manager, cc, config, row, max_attempts, rng, lock, counters)
            else:
                run_background_row(manager, cc, config, row, rng, lock, counters)
    finally:
        if fixed_count_coordination is not None and client_type == "agent":
            fixed_count_coordination.agent_worker_done()


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
) -> Iterable[dict[str, Any]]:
    if not rows:
        return
    if float(duration_s) <= 0:
        for row in rows:
            yield row
        return
    deadline = time.perf_counter() + float(duration_s)
    index = 0
    row_count = len(rows)
    while time.perf_counter() < deadline:
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
) -> None:
    task_started_at = time.perf_counter()
    final_result: dict[str, Any] = {"committed": False}
    task_reservation_wait_s = 0.0
    attempts_done = 0
    reuse_reasoning = False
    previous_failure_reason = "none"
    prior_retry_cost_ms = 0.0
    for attempt in range(max(1, max_attempts)):
        planned = planned_from_row(row, attempt=attempt)
        admission_deferred = False
        if reuse_reasoning:
            planned = planned_without_reasoning(planned)
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
            reuse_reasoning = True
        else:
            reuse_reasoning = False
        final_result = result
        task_reservation_wait_s += float(wait_s)
        attempts_done += 1
        with lock:
            counters.agent_logical_attempts += 1
            counters.total_reasoning_ms += int(planned.total_reasoning_delay_ms)
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
                    reason = str(result.get("failure_reason", "") or attempt_failure_reason(result))
                    if reason == "lock-timeout":
                        counters.lock_timeout_aborts += 1
                    elif reason == "lock-preempted":
                        counters.lock_preempted_aborts += 1
                    elif reason == "version-conflict":
                        counters.version_validation_aborts += 1
        if final_result.get("committed"):
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
        counters.agent_operation_counts.append(len(row["_task"].operations))
        counters.agent_task_reservation_waits_ms.append(task_reservation_wait_s * 1000.0)
        if final_result.get("committed"):
            counters.completed_agent_tasks += 1
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
    write_targets = operation_write_targets(task)
    if paper_atcc:
        blocked = manager.atcc_locks.background_pre_admission_block(write_targets)
        if blocked is not None:
            manager.note_background_abort()
            with lock:
                counters.background_attempts += 1
                counters.background_aborts += 1
                counters.background_retries += 1
                counters.background_row_s += time.perf_counter() - row_started
            return
    try:
        owner = SimpleNamespace(started_at=time.perf_counter())
        with background_write_guard(
            manager,
            write_targets,
            cc,
            config,
            owner=owner,
        ) as waited:
            wait_s = float(waited)
            for fresh_attempt in range(1 if paper_atcc else 3):
                metadata = {
                    "workload": row["workload"],
                    "task_type": f"background-{row['task_type']}",
                    "context": dict(row["_context"]),
                    "runtime_background": True,
                    "planned_write_targets": sorted(write_targets),
                }
                if paper_atcc:
                    metadata["paper_atcc_backend"] = True
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
                attempts_done += 1
                committed = bool(result.committed)
                if committed:
                    break
                aborts_done += 1
                if paper_atcc:
                    manager.note_background_abort()
                if str(result.reason) not in {
                    "background-publisher-busy",
                    "background-admission-busy",
                }:
                    break
                if paper_atcc:
                    break
                if not manager.atcc_locks.wait_for_background_admission(
                    write_targets,
                    timeout_s=0.002,
                ):
                    break
    except Exception:
        if "txn" in locals():
            manager.atcc_locks.release_all(txn.context)
        if attempts_done == 0:
            attempts_done = 1
            aborts_done = 1
            if paper_atcc:
                manager.note_background_abort()
        committed = False
    with lock:
        counters.background_attempts += attempts_done
        counters.background_reservation_wait_s += wait_s
        if committed:
            counters.background_commits += 1
        counters.background_aborts += aborts_done
        counters.background_retries += aborts_done
        counters.background_begin_s += begin_s
        counters.background_apply_s += apply_s
        counters.background_commit_s += commit_s
        counters.background_row_s += time.perf_counter() - row_started


def result_row(
    sample: dict[str, Any],
    cc: str,
    counters: MixedCounters,
    elapsed_s: float,
    tokens_per_operation: int,
    rows: list[dict[str, Any]],
    manager: AgentTransactionManager,
) -> dict[str, Any]:
    completed = max(0, int(counters.completed_agent_tasks))
    failed = max(0, int(counters.failed_agent_tasks))
    submitted = completed + failed
    agent_attempt_abort_rate = counters.agent_aborts / counters.agent_attempts if counters.agent_attempts else 0.0
    agent_abort_rate = counters.agent_aborts / completed if completed else 0.0
    avg_ops = average(counters.agent_operation_counts)
    avg_tokens = (1.0 + agent_abort_rate) * avg_ops * int(tokens_per_operation) if avg_ops else 0.0
    agent_wait_ms_total = float(counters.agent_reservation_wait_s) * 1000.0
    background_wait_ms_total = float(counters.background_reservation_wait_s) * 1000.0
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
    return {
        **base_row(sample, cc),
        "status": "ok",
        "elapsed_s": elapsed_s,
        "bottom_txn_attempts": bottom_attempts,
        "bottom_txn_commits": bottom_commits,
        "bottom_txn_attempt_tps": bottom_attempts / elapsed_s,
        "bottom_txn_commit_tps": bottom_commits / elapsed_s,
        "underlying_txn_attempt_tps": bottom_attempts / elapsed_s,
        "underlying_txn_commit_tps": bottom_commits / elapsed_s,
        "native_throughput": bottom_commits / elapsed_s,
        "total_tps": (counters.agent_commits + counters.background_commits) / elapsed_s,
        "agent_task_tps": completed / elapsed_s,
        "agent_tps": counters.agent_commits / elapsed_s,
        "background_tps": counters.background_commits / elapsed_s,
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
        "background_reservation_wait_ms_total": background_wait_ms_total,
        "background_reservation_wait_ms_mean": (
            background_wait_ms_total / counters.background_attempts if counters.background_attempts else 0.0
        ),
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
        "paper_lock_timeouts": paper_diagnostics.get("lock_timeouts", 0),
        "paper_priority_reorders": paper_diagnostics.get("priority_reorders", 0),
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
        "paper_background_admission_queue_events": paper_diagnostics.get(
            "background_admission_queue_events", 0
        ),
        "paper_background_admission_queue_wait_ms": paper_diagnostics.get(
            "background_admission_queue_wait_ms", 0.0
        ),
        "paper_background_admission_queue_timeouts": paper_diagnostics.get(
            "background_admission_queue_timeouts", 0
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
        "agent_total_tokens": avg_tokens * completed,
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
