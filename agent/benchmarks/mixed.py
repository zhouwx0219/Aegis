"""Mixed agent/background starvation benchmark."""

from __future__ import annotations

import contextlib
import dataclasses
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

from agent.benchmarks.phases import PlannedTask, ReasoningProfile, plan_task_phases, sleep_for_reasoning
from agent.cc import ConcurrencyControlRegistry, LockConflict
from agent.cc.atcc.actions import (
    OCC,
    LOCK_BEFORE_COMMIT,
    RESERVE_HOT,
    RESERVE_HOT_RW,
    RESERVE_HOT_RW_K,
    RESERVE_READ_WRITE_SET,
    WRITE_VALIDATE,
    normalize_action,
)
from agent.cc.base import CCPlan, unique_targets
from agent.cc.atcc.features import extract_task_features
from agent.runtime import AgentTransactionManager
from agent.workloads import AgentTask, apply_operation, build_workload, register_workload


@dataclasses.dataclass(frozen=True)
class MixedBenchmarkConfig:
    workload: str = "tpcc"
    level: str = "high"
    workload_profile: str = "small"
    ycsb_zipf_theta: float | None = None
    cc: str = "occ,dynamic-atcc"
    duration_s: float = 3.0
    agent_workers: int = 2
    background_workers: int = 8
    clients: int = 0
    agent_ratio: float = 0.80
    reasoning_profile: str = "agentic"
    reasoning_scale: float = 2.0
    seed: int = 920104
    background_wait: bool = False
    background_mode: str = "hotspot"
    reservation_ttl_s: float = 5.0
    retries: int = 0
    retry_until_commit: bool = False
    max_attempts_per_task: int = 100
    agent_retry_backoff_min_ms: int = 500
    agent_retry_backoff_max_ms: int = 5000
    background_retry_backoff_min_ms: int = 10
    background_retry_backoff_max_ms: int = 30
    tokens_per_operation: int = 2703
    policy: Any = None
    policy_mode: str = "online"
    atcc_hot_rw_k: int = 3
    atcc_bp_background_threshold: int = 6
    atcc_bp_queue_pressure_threshold: int = 2
    atcc_bp_min_windows: int = 3
    atcc_agent_guardrail: bool = False
    atcc_agent_guardrail_queue_threshold: int = 1
    atcc_full_reservation_fallback_ratio: float = 0.0
    atcc_pure_policy: bool = False
    background_admission_cap: int = 0

    def normalized(self) -> "MixedBenchmarkConfig":
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")
        agent_workers = int(self.agent_workers)
        background_workers = int(self.background_workers)
        clients = int(self.clients)
        agent_ratio = float(self.agent_ratio)
        if clients > 0:
            if clients < 2:
                raise ValueError("clients must be at least 2 when set")
            if not 0.0 < agent_ratio <= 1.0:
                raise ValueError("agent ratio must be > 0 and <= 1")
            agent_workers = max(1, int(round(clients * agent_ratio)))
            background_workers = max(0, clients - agent_workers)
        if agent_workers <= 0:
            raise ValueError("agent workers must be positive")
        if background_workers < 0:
            raise ValueError("background workers must be non-negative")
        if self.ycsb_zipf_theta is not None and self.ycsb_zipf_theta < 0:
            raise ValueError("YCSB Zipfian theta must be non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.max_attempts_per_task <= 0:
            raise ValueError("max attempts per task must be positive")
        if self.agent_retry_backoff_min_ms < 0 or self.agent_retry_backoff_max_ms < 0:
            raise ValueError("agent retry backoff must be non-negative")
        if self.agent_retry_backoff_min_ms > self.agent_retry_backoff_max_ms:
            raise ValueError("agent retry backoff min must be <= max")
        if self.background_retry_backoff_min_ms < 0 or self.background_retry_backoff_max_ms < 0:
            raise ValueError("background retry backoff must be non-negative")
        if self.background_retry_backoff_min_ms > self.background_retry_backoff_max_ms:
            raise ValueError("background retry backoff min must be <= max")
        if self.tokens_per_operation <= 0:
            raise ValueError("tokens per operation must be positive")
        if self.atcc_hot_rw_k <= 0:
            raise ValueError("ATCC hot-rw-k target limit must be positive")
        if self.atcc_bp_background_threshold < 0:
            raise ValueError("ATCC BP background threshold must be non-negative")
        if self.atcc_bp_queue_pressure_threshold < 0:
            raise ValueError("ATCC BP queue pressure threshold must be non-negative")
        if self.atcc_bp_min_windows <= 0:
            raise ValueError("ATCC BP min windows must be positive")
        if self.atcc_agent_guardrail_queue_threshold < 0:
            raise ValueError("ATCC agent guardrail queue threshold must be non-negative")
        if not 0.0 <= float(self.atcc_full_reservation_fallback_ratio) <= 1.0:
            raise ValueError("ATCC full reservation fallback ratio must be between 0 and 1")
        if self.background_admission_cap < 0:
            raise ValueError("background admission cap must be non-negative")
        mode = str(self.policy_mode).strip().lower()
        if mode not in {"train", "eval", "online"}:
            raise ValueError(f"unsupported policy mode: {self.policy_mode}")
        background_mode = str(self.background_mode).strip().lower()
        if background_mode not in {"hotspot", "procedure"}:
            raise ValueError(f"unsupported background mode: {self.background_mode}")
        return dataclasses.replace(
            self,
            workload=str(self.workload).strip().lower(),
            level=str(self.level).strip().lower(),
            workload_profile=str(self.workload_profile).strip().lower() or "small",
            ycsb_zipf_theta=self.ycsb_zipf_theta,
            cc=str(self.cc).strip() or "occ",
            agent_workers=agent_workers,
            background_workers=background_workers,
            clients=clients,
            agent_ratio=agent_ratio,
            reasoning_profile=str(self.reasoning_profile).strip().lower() or "agentic",
            background_mode=background_mode,
            policy_mode=mode,
            atcc_hot_rw_k=int(self.atcc_hot_rw_k),
            atcc_bp_background_threshold=int(self.atcc_bp_background_threshold),
            atcc_bp_queue_pressure_threshold=int(self.atcc_bp_queue_pressure_threshold),
            atcc_bp_min_windows=int(self.atcc_bp_min_windows),
            atcc_agent_guardrail=bool(self.atcc_agent_guardrail),
            atcc_agent_guardrail_queue_threshold=int(self.atcc_agent_guardrail_queue_threshold),
            atcc_full_reservation_fallback_ratio=float(self.atcc_full_reservation_fallback_ratio),
            atcc_pure_policy=bool(self.atcc_pure_policy),
            background_admission_cap=int(self.background_admission_cap),
        )


@dataclasses.dataclass
class MixedCounters:
    agent_attempts: int = 0
    agent_commits: int = 0
    agent_aborts: int = 0
    background_attempts: int = 0
    background_commits: int = 0
    background_aborts: int = 0
    agent_reservation_wait_s: float = 0.0
    background_reservation_wait_s: float = 0.0
    wasted_reasoning_ms: int = 0
    read_conflicts: int = 0
    write_conflicts: int = 0
    completed_agent_tasks: int = 0
    failed_agent_tasks: int = 0
    agent_end_to_end_latencies_ms: List[float] = dataclasses.field(default_factory=list)
    agent_retry_counts: List[int] = dataclasses.field(default_factory=list)
    agent_operation_counts: List[int] = dataclasses.field(default_factory=list)
    agent_task_reservation_waits_ms: List[float] = dataclasses.field(default_factory=list)
    background_retries: int = 0
    action_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    reservation_target_sizes: List[int] = dataclasses.field(default_factory=list)
    reserve_read_write_set_target_sizes: List[int] = dataclasses.field(default_factory=list)
    reserve_read_write_set_hot_target_counts: List[int] = dataclasses.field(default_factory=list)
    reserve_read_write_set_hot_coverage_ratios: List[float] = dataclasses.field(default_factory=list)
    reserve_read_write_set_unique_targets: set[str] = dataclasses.field(default_factory=set)
    reserve_read_write_set_unique_hot_targets: set[str] = dataclasses.field(default_factory=set)
    reserve_hot_rw_k_target_sizes: List[int] = dataclasses.field(default_factory=list)
    reserve_hot_rw_k_unique_targets: set[str] = dataclasses.field(default_factory=set)

    def add_action(self, action: str) -> None:
        if action:
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def add_atcc_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        if not diagnostics:
            return
        reservation_target_size = diagnostics.get("reservation_target_size")
        if reservation_target_size is not None:
            self.reservation_target_sizes.append(int(reservation_target_size))
        if diagnostics.get("action") == RESERVE_HOT_RW_K:
            target_size = int(diagnostics.get("target_size", 0) or 0)
            if target_size:
                self.reserve_hot_rw_k_target_sizes.append(target_size)
            self.reserve_hot_rw_k_unique_targets.update(
                str(target) for target in diagnostics.get("targets", ()) if str(target)
            )
        if diagnostics.get("action") != RESERVE_READ_WRITE_SET:
            return
        target_size = int(diagnostics.get("target_size", 0) or 0)
        hot_target_count = int(diagnostics.get("hot_target_count", 0) or 0)
        if target_size:
            self.reserve_read_write_set_target_sizes.append(target_size)
            self.reserve_read_write_set_hot_target_counts.append(hot_target_count)
            self.reserve_read_write_set_hot_coverage_ratios.append(hot_target_count / target_size)
        self.reserve_read_write_set_unique_targets.update(
            str(target) for target in diagnostics.get("targets", ()) if str(target)
        )
        self.reserve_read_write_set_unique_hot_targets.update(
            str(target) for target in diagnostics.get("hot_targets", ()) if str(target)
        )


def run_mixed_benchmark(config: MixedBenchmarkConfig) -> Dict[str, Any]:
    config = config.normalized()
    rows = []
    for strategy in expand_cc(config.cc):
        rows.append(run_mixed_strategy(config, strategy))
    return {
        "mode": "mixed-starvation",
        "workload": config.workload,
        "level": config.level,
        "workload_profile": config.workload_profile,
        "ycsb_zipf_theta": config.ycsb_zipf_theta,
        "duration_s": float(config.duration_s),
        "clients": int(config.clients),
        "agent_ratio": float(config.agent_ratio),
        "agent_workers": int(config.agent_workers),
        "background_workers": int(config.background_workers),
        "reasoning_profile": config.reasoning_profile,
        "reasoning_scale": float(config.reasoning_scale),
        "background_mode": config.background_mode,
        "retry_until_commit": bool(config.retry_until_commit),
        "max_attempts_per_task": int(config.max_attempts_per_task),
        "agent_retry_backoff_ms": [
            int(config.agent_retry_backoff_min_ms),
            int(config.agent_retry_backoff_max_ms),
        ],
        "background_retry_backoff_ms": [
            int(config.background_retry_backoff_min_ms),
            int(config.background_retry_backoff_max_ms),
        ],
        "tokens_per_operation": int(config.tokens_per_operation),
        "policy_mode": config.policy_mode,
        "atcc_hot_rw_k": int(config.atcc_hot_rw_k),
        "atcc_bp_background_threshold": int(config.atcc_bp_background_threshold),
        "atcc_bp_queue_pressure_threshold": int(config.atcc_bp_queue_pressure_threshold),
        "atcc_bp_min_windows": int(config.atcc_bp_min_windows),
        "atcc_agent_guardrail": bool(config.atcc_agent_guardrail),
        "atcc_agent_guardrail_queue_threshold": int(config.atcc_agent_guardrail_queue_threshold),
        "atcc_full_reservation_fallback_ratio": float(config.atcc_full_reservation_fallback_ratio),
        "atcc_pure_policy": bool(config.atcc_pure_policy),
        "background_admission_cap": int(config.background_admission_cap),
        "strategies": expand_cc(config.cc),
        "cc_results": rows,
    }


def run_mixed_strategy(config: MixedBenchmarkConfig, strategy: str) -> Dict[str, Any]:
    manager = AgentTransactionManager(
        cc_registry=registry_for(config),
    )
    workload = build_workload(
        config.workload,
        config.level,
        config.workload_profile,
        ycsb_zipf_theta=config.ycsb_zipf_theta,
    )
    register_workload(manager, workload)
    tasks = list(workload.generate_tasks(256, seed=config.seed))
    background_tasks = list(workload.generate_tasks(512, seed=config.seed + 700_000))
    hot_targets = hot_write_targets(tasks)
    if not hot_targets:
        hot_targets = write_targets(tasks)
    if not hot_targets:
        raise ValueError(f"mixed benchmark needs at least one write target: {config.workload}/{config.level}")
    started_at = time.perf_counter()
    stop_at = started_at + float(config.duration_s)
    background_stop = threading.Event()
    background_admission = (
        threading.BoundedSemaphore(int(config.background_admission_cap))
        if int(config.background_admission_cap) > 0
        else None
    )
    lock = threading.Lock()
    counters = MixedCounters()

    with ThreadPoolExecutor(max_workers=config.agent_workers + config.background_workers) as executor:
        agent_futures = []
        for worker in range(config.agent_workers):
            agent_futures.append(
                executor.submit(
                    agent_worker,
                    manager,
                    tasks,
                    strategy,
                    config,
                    stop_at,
                    counters,
                    lock,
                    worker,
                )
            )
        background_futures = []
        for worker in range(config.background_workers):
            background_futures.append(
                executor.submit(
                    background_worker,
                    manager,
                    hot_targets,
                    background_tasks,
                    strategy,
                    config,
                    stop_at,
                    counters,
                    lock,
                    worker,
                    background_stop,
                    background_admission,
                )
            )
        try:
            for future in agent_futures:
                future.result()
        finally:
            background_stop.set()
        for future in background_futures:
            future.result()

    elapsed_s = max(0.001, time.perf_counter() - started_at)
    completed = max(0, int(counters.completed_agent_tasks))
    failed = max(0, int(counters.failed_agent_tasks))
    submitted = completed + failed
    agent_attempt_abort_rate = counters.agent_aborts / counters.agent_attempts if counters.agent_attempts else 0.0
    agent_abort_rate = counters.agent_aborts / completed if completed else 0.0
    avg_ops = average(counters.agent_operation_counts)
    avg_tokens = (1.0 + agent_abort_rate) * avg_ops * int(config.tokens_per_operation) if avg_ops else 0.0
    reservation_diagnostics = manager.reservations.snapshot_diagnostics()
    reservation_waiter_target_sizes = tuple(
        reservation_diagnostics.pop("reservation_waiter_target_sizes", ())
    )
    row = {
        "cc": strategy,
        "elapsed_s": elapsed_s,
        "agent_attempts": counters.agent_attempts,
        "agent_commits": counters.agent_commits,
        "agent_aborts": counters.agent_aborts,
        "agent_completed_tasks": completed,
        "agent_failed_tasks": failed,
        "agent_submitted_tasks": submitted,
        "background_attempts": counters.background_attempts,
        "background_commits": counters.background_commits,
        "background_aborts": counters.background_aborts,
        "background_retries": counters.background_retries,
        "agent_tps": counters.agent_commits / elapsed_s,
        "agent_task_tps": completed / elapsed_s,
        "background_tps": counters.background_commits / elapsed_s,
        "total_tps": (counters.agent_commits + counters.background_commits) / elapsed_s,
        "agent_commit_rate": counters.agent_commits / counters.agent_attempts if counters.agent_attempts else 0.0,
        "agent_task_completion_rate": completed / submitted if submitted else 0.0,
        "agent_abort_rate": agent_abort_rate,
        "agent_attempt_abort_rate": agent_attempt_abort_rate,
        "background_commit_rate": counters.background_commits / counters.background_attempts if counters.background_attempts else 0.0,
        "agent_avg_retry_count": average(counters.agent_retry_counts),
        "agent_p50_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 50),
        "agent_p95_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 95),
        "agent_p99_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 99),
        "agent_p9999_latency_ms": percentile(counters.agent_end_to_end_latencies_ms, 99.99),
        "agent_avg_latency_ms": average(counters.agent_end_to_end_latencies_ms),
        "agent_avg_operations": avg_ops,
        "agent_avg_tokens": avg_tokens,
        "agent_total_tokens": avg_tokens * completed,
        "agent_guard_wait_ms": counters.agent_reservation_wait_s * 1000.0,
        "background_guard_wait_ms": counters.background_reservation_wait_s * 1000.0,
        "guard_wait_ms": (counters.agent_reservation_wait_s + counters.background_reservation_wait_s) * 1000.0,
        "agent_reservation_wait_ms": counters.agent_reservation_wait_s * 1000.0,
        "background_reservation_wait_ms": counters.background_reservation_wait_s * 1000.0,
        "reservation_wait_ms": (counters.agent_reservation_wait_s + counters.background_reservation_wait_s) * 1000.0,
        "wasted_reasoning_ms": counters.wasted_reasoning_ms,
        "read_conflicts": counters.read_conflicts,
        "write_conflicts": counters.write_conflicts,
        "action_counts": dict(sorted(counters.action_counts.items())),
        "atcc_hot_rw_k": int(config.atcc_hot_rw_k),
        "atcc_bp_background_threshold": int(config.atcc_bp_background_threshold),
        "atcc_bp_queue_pressure_threshold": int(config.atcc_bp_queue_pressure_threshold),
        "atcc_bp_min_windows": int(config.atcc_bp_min_windows),
        "atcc_agent_guardrail": bool(config.atcc_agent_guardrail),
        "atcc_agent_guardrail_queue_threshold": int(config.atcc_agent_guardrail_queue_threshold),
        "atcc_full_reservation_fallback_ratio": float(config.atcc_full_reservation_fallback_ratio),
        "atcc_pure_policy": bool(config.atcc_pure_policy),
        "background_admission_cap": int(config.background_admission_cap),
    }
    row.update(reservation_diagnostics)
    add_distribution_fields(
        row,
        "agent_task_guard_wait_ms",
        counters.agent_task_reservation_waits_ms,
    )
    add_distribution_fields(
        row,
        "reservation_waiter_target_set_size",
        reservation_waiter_target_sizes,
        include_histogram=True,
    )
    add_distribution_fields(
        row,
        "reservation_action_target_set_size",
        counters.reservation_target_sizes,
        include_histogram=True,
    )
    add_distribution_fields(
        row,
        "reserve_read_write_set_target_size",
        counters.reserve_read_write_set_target_sizes,
        include_histogram=True,
    )
    add_distribution_fields(
        row,
        "reserve_read_write_set_hot_target_count",
        counters.reserve_read_write_set_hot_target_counts,
        include_histogram=True,
    )
    add_distribution_fields(
        row,
        "reserve_read_write_set_hot_coverage_ratio",
        counters.reserve_read_write_set_hot_coverage_ratios,
    )
    add_distribution_fields(
        row,
        "reserve_hot_rw_k_target_size",
        counters.reserve_hot_rw_k_target_sizes,
        include_histogram=True,
    )
    row["reserve_read_write_set_attempts"] = len(counters.reserve_read_write_set_target_sizes)
    row["reserve_read_write_set_unique_target_count"] = len(
        counters.reserve_read_write_set_unique_targets
    )
    row["reserve_read_write_set_unique_hot_target_count"] = len(
        counters.reserve_read_write_set_unique_hot_targets
    )
    row["reserve_read_write_set_unique_hot_targets"] = sorted(
        counters.reserve_read_write_set_unique_hot_targets
    )
    row["reserve_hot_rw_k_attempts"] = len(counters.reserve_hot_rw_k_target_sizes)
    row["reserve_hot_rw_k_unique_target_count"] = len(counters.reserve_hot_rw_k_unique_targets)
    row["reserve_hot_rw_k_unique_targets"] = sorted(counters.reserve_hot_rw_k_unique_targets)
    return row


def agent_worker(
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    config: MixedBenchmarkConfig,
    stop_at: float,
    counters: MixedCounters,
    lock: threading.Lock,
    worker: int,
) -> None:
    rng = random.Random(config.seed + worker)
    profile = ReasoningProfile(config.reasoning_profile, config.reasoning_scale)
    index = worker
    while time.perf_counter() < stop_at:
        task = tasks[index % len(tasks)]
        index += max(1, config.agent_workers)
        task_started_at = time.perf_counter()
        final_result: Dict[str, Any] = {"committed": False}
        task_reservation_wait_s = 0.0
        max_attempts = int(config.max_attempts_per_task) if config.retry_until_commit else int(config.retries) + 1
        attempt = 0
        attempts_done = 0
        while attempt < max_attempts and (time.perf_counter() < stop_at or config.retry_until_commit):
            planned = plan_task_phases(task, attempt=attempt, profile=profile)
            try:
                result, action, wait_s, attempt_diagnostics = run_agent_attempt(
                    manager,
                    planned,
                    strategy,
                    ttl_s=config.reservation_ttl_s,
                    jitter_ms=rng.randint(0, 5),
                    retry_count=attempt,
                    background_workers=config.background_workers,
                    config=config,
                )
            except LockConflict:
                result, action, wait_s, attempt_diagnostics = (
                    {"committed": False, "wasted_reasoning_ms": planned.total_reasoning_delay_ms},
                    strategy,
                    0.0,
                    {},
                )
            final_result = result
            task_reservation_wait_s += float(wait_s)
            with lock:
                counters.agent_attempts += 1
                counters.agent_reservation_wait_s += float(wait_s)
                counters.add_action(action)
                counters.add_atcc_diagnostics(attempt_diagnostics)
                counters.read_conflicts += int(result.get("read_conflicts", 0) or 0)
                counters.write_conflicts += int(result.get("write_conflicts", 0) or 0)
                if result.get("committed"):
                    counters.agent_commits += 1
                else:
                    counters.agent_aborts += 1
                    counters.wasted_reasoning_ms += int(result.get("wasted_reasoning_ms", 0))
            attempts_done += 1
            if final_result.get("committed") or (time.perf_counter() >= stop_at and not config.retry_until_commit):
                break
            if config.retry_until_commit:
                backoff_ms = rng.randint(
                    int(config.agent_retry_backoff_min_ms),
                    int(config.agent_retry_backoff_max_ms),
                )
                sleep_for_reasoning(backoff_ms)
            attempt += 1
        task_elapsed_ms = (time.perf_counter() - task_started_at) * 1000.0
        with lock:
            counters.agent_operation_counts.append(len(getattr(task, "operations", ()) or ()))
            counters.agent_task_reservation_waits_ms.append(task_reservation_wait_s * 1000.0)
            if final_result.get("committed"):
                counters.completed_agent_tasks += 1
                counters.agent_end_to_end_latencies_ms.append(task_elapsed_ms)
                counters.agent_retry_counts.append(max(0, attempts_done - 1))
            else:
                counters.failed_agent_tasks += 1


def run_agent_attempt(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    *,
    ttl_s: float,
    jitter_ms: int,
    retry_count: int,
    background_workers: int,
    config: MixedBenchmarkConfig,
) -> tuple[Dict[str, Any], str, float, Dict[str, Any]]:
    sleep_for_reasoning(jitter_ms)
    strategy_impl = manager.cc_registry.resolve(strategy)
    decision = None
    if should_use_low_conflict_atcc_runtime_fast_path(strategy_impl, planned.task, retry_count=retry_count):
        metadata = mixed_transaction_metadata(
            planned,
            retry_count=retry_count,
            background_workers=background_workers,
            strategy=strategy,
            decision=None,
        )
        result, action, wait_s = run_agent_with_low_conflict_optimistic_fast_path(
            manager,
            planned,
            metadata,
        )
        diagnostics = {
            "action": action,
            "target_size": 0,
            "targets": (),
            "runtime_fast_path": "low-conflict-optimistic",
        }
        return result, action, wait_s, diagnostics

    if getattr(strategy_impl, "family", "") == "atcc":
        agentic_features = {
            "phase_count": planned.phase_count,
            "reasoning_delay_ms": planned.total_reasoning_delay_ms,
            "retry_delay_ms": planned.retry_delay_ms,
            "background_workers": int(background_workers),
            "target_selection_seed": stable_task_seed(planned.task, retry_count=retry_count),
        }
        features = extract_task_features(
            planned.task,
            retry_count=retry_count,
            agentic=agentic_features,
        )
        pressure_targets = unique_targets(
            tuple(features.hot_targets)
            + tuple(features.hot_read_targets)
            + tuple(features.write_targets)
            + tuple(features.read_targets)
        )
        agentic_features.update(manager.reservations.snapshot_pressure(pressure_targets))
        features = extract_task_features(
            planned.task,
            retry_count=retry_count,
            agentic=agentic_features,
        )
        decision = strategy_impl.decide(features)
        decision = apply_atcc_experiment_overrides(
            planned.task,
            decision,
            features,
            config,
            retry_count=retry_count,
        )
    action = str(getattr(decision, "action", "") or "")
    attempt_diagnostics = atcc_decision_diagnostics(action, decision)
    metadata = mixed_transaction_metadata(
        planned,
        retry_count=retry_count,
        background_workers=background_workers,
        strategy=strategy,
        decision=decision,
    )
    traditional_plan = mixed_traditional_begin_lock_plan(strategy_impl, planned.task)
    if traditional_plan is not None:
        if can_defer_transaction_begin(planned):
            result, action, wait_s = run_agent_with_traditional_deferred_commit_lock(
                manager,
                planned,
                strategy,
                metadata,
                lock_table=str(traditional_plan.metadata.get("lock_table", "")),
                targets=traditional_plan.lock_targets,
                policy=str(traditional_plan.metadata.get("policy", "")),
                priority=0,
            )
            return result, action, wait_s, attempt_diagnostics
        result, action, wait_s = run_agent_with_traditional_commit_lock(
            manager,
            planned,
            strategy,
            metadata,
        )
        return result, action, wait_s, attempt_diagnostics
    if decision is not None and decision.begins_locked:
        result, action, wait_s = run_agent_with_begin_lock(
            manager,
            planned,
            strategy,
            metadata,
            lock_table="exclusive",
            targets=tuple(decision.targets),
            policy="",
            priority=int(decision.priority),
        )
        return result, action, wait_s, attempt_diagnostics
    if action in {OCC, WRITE_VALIDATE} and decision is not None and can_defer_read_heavy_transaction_begin(planned):
        result, action, wait_s = run_agent_with_deferred_read_optimistic(
            manager,
            planned,
            atcc_concrete_commit_strategy(strategy_impl, action, strategy),
            metadata,
            decision=decision,
        )
        return result, action, wait_s, attempt_diagnostics
    if action in {RESERVE_HOT, RESERVE_HOT_RW, RESERVE_HOT_RW_K, RESERVE_READ_WRITE_SET} and decision is not None:
        if action in {RESERVE_HOT_RW, RESERVE_HOT_RW_K, RESERVE_READ_WRITE_SET} and can_defer_read_heavy_transaction_begin(planned):
            result, action, wait_s = run_agent_with_deferred_read_reservation(
                manager,
                planned,
                strategy,
                metadata,
                decision=decision,
                background_workers=background_workers,
                ttl_s=ttl_s,
            )
            return result, action, wait_s, attempt_diagnostics
        owner = SimpleNamespace(started_at=time.perf_counter())
        with manager.reservations.reserve(
            decision.targets,
            owner=owner,
            ttl_s=ttl_s,
            wait=True,
            timeout_s=ttl_s,
            priority=int(getattr(decision, "priority", 0) or 0),
        ) as wait_s:
            # Acquire the agent snapshot after reservation so background writes
            # cannot stale the expensive reasoning window.
            txn = begin_planned_transaction(manager, planned, metadata)
            for phase in planned.phases:
                execute_phase(txn, phase)
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": float(wait_s) * 1000.0,
                "lock_hold_ms": float(planned.total_reasoning_delay_ms),
                "skipped_reasoning_ms": 0.0,
                "background_aborts": estimated_background_abort_cost(background_workers, planned.total_reasoning_delay_ms),
                "background_tps_loss": float(background_workers) if wait_s > 0 else 0.0,
            }
            result = txn.commit(strategy)
    elif action == LOCK_BEFORE_COMMIT and decision is not None:
        if can_defer_transaction_begin(planned):
            result, action, wait_s = run_agent_with_deferred_commit_lock(
                manager,
                planned,
                strategy,
                metadata,
                decision=decision,
                background_workers=background_workers,
                ttl_s=ttl_s,
            )
            return result, action, wait_s, attempt_diagnostics
        txn = begin_planned_transaction(manager, planned, metadata)
        before_commit, commit_phases = split_commit_phases(planned)
        for phase in before_commit:
            execute_phase(txn, phase)
        with manager.reservations.reserve(
            decision.targets,
            owner=txn,
            ttl_s=ttl_s,
            wait=True,
            timeout_s=ttl_s,
            priority=int(getattr(decision, "priority", 0) or 0),
        ) as wait_s:
            conflicts = planned_write_conflicts(manager, txn, planned)
            if conflicts:
                skipped = sum(phase.reasoning_delay_ms for phase in commit_phases)
                txn.metadata["atcc_runtime"] = {
                    "lock_wait_ms": float(wait_s) * 1000.0,
                    "lock_hold_ms": 0.0,
                    "skipped_reasoning_ms": float(skipped),
                    "background_aborts": 0.0,
                    "background_tps_loss": 0.0,
                }
                result = txn.abort(
                    "early version conflict before mixed commit phase",
                    strategy=strategy,
                )
                observe_aborted_atcc(strategy_impl, decision, txn, result)
            else:
                for phase in commit_phases:
                    execute_phase(txn, phase)
                txn.metadata["atcc_runtime"] = {
                    "lock_wait_ms": float(wait_s) * 1000.0,
                    "lock_hold_ms": float(sum(phase.reasoning_delay_ms for phase in commit_phases)),
                    "skipped_reasoning_ms": 0.0,
                    "background_aborts": estimated_background_abort_cost(
                        background_workers,
                        sum(phase.reasoning_delay_ms for phase in commit_phases),
                    ),
                    "background_tps_loss": float(background_workers) if wait_s > 0 else 0.0,
                }
                result = txn.commit(strategy)
    else:
        wait_s = 0.0
        txn = begin_planned_transaction(manager, planned, metadata)
        for phase in planned.phases:
            execute_phase(txn, phase)
        result = txn.commit(atcc_concrete_commit_strategy(strategy_impl, action, strategy))
    return {
        "committed": bool(result.committed),
        "wasted_reasoning_ms": 0 if result.committed else planned.total_reasoning_delay_ms,
        "read_conflicts": conflict_counts(getattr(result, "conflict_object_ids", ()), txn)[0],
        "write_conflicts": conflict_counts(getattr(result, "conflict_object_ids", ()), txn)[1],
    }, action, float(wait_s), attempt_diagnostics


def should_use_low_conflict_atcc_runtime_fast_path(
    strategy_impl: Any,
    task: AgentTask,
    *,
    retry_count: int,
) -> bool:
    if getattr(strategy_impl, "family", "") != "atcc":
        return False
    context = dict(getattr(task, "context", {}) or {})
    if str(context.get("level", "")).strip().lower() != "low":
        return False
    policy = getattr(strategy_impl, "policy", None)
    if bool(getattr(policy, "training", False)):
        return False
    trainable_actions = tuple(getattr(policy, "trainable_actions", ()) or ())
    if trainable_actions and OCC not in {normalize_action(action) for action in trainable_actions}:
        return False
    return True


def run_agent_with_low_conflict_optimistic_fast_path(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    metadata: Dict[str, Any],
) -> tuple[Dict[str, Any], str, float]:
    metadata = dict(metadata)
    metadata["atcc_runtime_fast_path"] = "low-conflict-optimistic"
    txn = begin_planned_transaction(manager, planned, metadata)
    for phase in planned.phases:
        execute_phase(txn, phase)
    result = txn.commit(OCC)
    return attempt_result(result, planned, txn), OCC, 0.0


def run_agent_with_deferred_read_optimistic(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
    *,
    decision: Any,
) -> tuple[Dict[str, Any], str, float]:
    action = str(getattr(decision, "action", "") or OCC)
    before_commit, commit_phases = split_commit_phases(planned)
    deferred_read_delay_ms = sleep_phase_reasoning(before_commit)
    deferred_commit_delay_ms = sleep_phase_reasoning(commit_phases)
    txn = begin_planned_transaction(manager, planned, metadata)
    for phase in before_commit:
        execute_deferred_phase(
            txn,
            phase,
            deferred_before_begin=True,
            deferred_read_replay=True,
        )
    for phase in commit_phases:
        execute_deferred_phase(txn, phase, deferred_commit_reasoning=True)
    txn.metadata["atcc_runtime"] = {
        "lock_wait_ms": 0.0,
        "lock_hold_ms": 0.0,
        "skipped_reasoning_ms": 0.0,
        "deferred_read_begin": True,
        "deferred_read_reasoning_ms": float(deferred_read_delay_ms),
        "deferred_commit_reasoning_ms": float(deferred_commit_delay_ms),
        "background_aborts": 0.0,
        "background_tps_loss": 0.0,
    }
    result = txn.commit(strategy)
    return attempt_result(result, planned, txn), action, 0.0


def atcc_concrete_commit_strategy(strategy_impl: Any, action: str, default: str) -> str:
    if getattr(strategy_impl, "family", "") != "atcc":
        return default
    policy = getattr(strategy_impl, "policy", None)
    if not bool(getattr(policy, "frozen", False)):
        return default
    normalized = str(action).strip().lower()
    if normalized == WRITE_VALIDATE:
        return "mvcc"
    if normalized == OCC:
        return "occ"
    return default


def run_agent_with_deferred_read_reservation(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
    *,
    decision: Any,
    background_workers: int,
    ttl_s: float,
) -> tuple[Dict[str, Any], str, float]:
    action = str(getattr(decision, "action", "") or "")
    before_commit, commit_phases = split_commit_phases(planned)
    deferred_read_delay_ms = sleep_phase_reasoning(before_commit)
    deferred_commit_delay_ms = sleep_phase_reasoning(commit_phases)
    owner = SimpleNamespace(started_at=time.perf_counter())
    with manager.reservations.reserve(
        decision.targets,
        owner=owner,
        ttl_s=ttl_s,
        wait=True,
        timeout_s=ttl_s,
        priority=int(getattr(decision, "priority", 0) or 0),
    ) as wait_s:
        lock_started_at = time.perf_counter()
        txn = begin_planned_transaction(manager, planned, metadata)
        for phase in before_commit:
            execute_deferred_phase(
                txn,
                phase,
                deferred_before_begin=True,
                deferred_read_replay=True,
            )
        for phase in commit_phases:
            execute_deferred_phase(txn, phase, deferred_commit_reasoning=True)
        lock_hold_ms = (time.perf_counter() - lock_started_at) * 1000.0
        txn.metadata["atcc_runtime"] = {
            "lock_wait_ms": float(wait_s) * 1000.0,
            "lock_hold_ms": lock_hold_ms,
            "skipped_reasoning_ms": 0.0,
            "deferred_read_begin": True,
            "deferred_read_reasoning_ms": float(deferred_read_delay_ms),
            "deferred_commit_reasoning_ms": float(deferred_commit_delay_ms),
            "background_aborts": estimated_background_abort_cost(background_workers, lock_hold_ms),
            "background_tps_loss": float(background_workers) if wait_s > 0 else 0.0,
        }
        result = txn.commit(strategy)
    return attempt_result(result, planned, txn), action, float(wait_s)


def run_agent_with_deferred_commit_lock(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
    *,
    decision: Any,
    background_workers: int,
    ttl_s: float,
) -> tuple[Dict[str, Any], str, float]:
    action = str(getattr(decision, "action", "") or "")
    before_commit, commit_phases = split_commit_phases(planned)
    deferred_before_delay_ms = sleep_phase_reasoning(before_commit)
    deferred_commit_delay_ms = sleep_phase_reasoning(commit_phases)
    owner = SimpleNamespace(started_at=time.perf_counter())
    with manager.reservations.reserve(
        decision.targets,
        owner=owner,
        ttl_s=ttl_s,
        wait=True,
        timeout_s=ttl_s,
        priority=int(getattr(decision, "priority", 0) or 0),
    ) as wait_s:
        lock_started_at = time.perf_counter()
        txn = begin_planned_transaction(manager, planned, metadata)
        for phase in before_commit:
            execute_deferred_phase(txn, phase, deferred_before_begin=True)
        for phase in commit_phases:
            execute_deferred_phase(txn, phase, deferred_commit_reasoning=True)
        lock_hold_ms = (time.perf_counter() - lock_started_at) * 1000.0
        txn.metadata["atcc_runtime"] = {
            "lock_wait_ms": float(wait_s) * 1000.0,
            "lock_hold_ms": lock_hold_ms,
            "skipped_reasoning_ms": 0.0,
            "deferred_before_begin_ms": float(deferred_before_delay_ms),
            "deferred_commit_reasoning_ms": float(deferred_commit_delay_ms),
            "background_aborts": estimated_background_abort_cost(background_workers, lock_hold_ms),
            "background_tps_loss": float(background_workers) if wait_s > 0 else 0.0,
        }
        result = txn.commit(strategy)
    return attempt_result(result, planned, txn), action, float(wait_s)


def mixed_transaction_metadata(
    planned: PlannedTask,
    *,
    retry_count: int,
    background_workers: int,
    strategy: str,
    decision: Any = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "workload": planned.task.workload,
        "task_type": planned.task.task_type,
        "context": dict(planned.task.context),
        "retry_count": int(retry_count),
        "agentic": {
            "phase_count": planned.phase_count,
            "reasoning_delay_ms": planned.total_reasoning_delay_ms,
            "retry_delay_ms": planned.retry_delay_ms,
            "background_workers": int(background_workers),
        },
    }
    if decision is not None:
        metadata["atcc_preplan"] = atcc_preplan_from_decision(strategy, decision)
    return metadata


def atcc_decision_diagnostics(action: str, decision: Any) -> Dict[str, Any]:
    if decision is None:
        return {}
    normalized_action = str(action or "").strip().lower()
    targets = tuple(sorted(set(str(target) for target in getattr(decision, "targets", ()) if str(target))))
    diagnostics: Dict[str, Any] = {
        "action": normalized_action,
        "target_size": len(targets),
        "targets": targets,
    }
    if normalized_action in {RESERVE_HOT, RESERVE_HOT_RW, RESERVE_HOT_RW_K, RESERVE_READ_WRITE_SET}:
        diagnostics["reservation_target_size"] = len(targets)
    if normalized_action == RESERVE_HOT_RW_K:
        metadata = dict(getattr(decision, "metadata", {}) or {})
        queue_lengths = dict(metadata.get("reservation_queue_lengths", {}) or {})
        diagnostics["selected_target_queue_lengths"] = {
            target: int(queue_lengths.get(target, 0) or 0)
            for target in targets
        }
    if normalized_action == RESERVE_READ_WRITE_SET:
        metadata = dict(getattr(decision, "metadata", {}) or {})
        hot_targets = set(str(target) for target in metadata.get("hot_targets", ()) if str(target))
        hot_targets.update(str(target) for target in metadata.get("hot_read_targets", ()) if str(target))
        covered_hot_targets = tuple(sorted(set(targets) & hot_targets))
        diagnostics.update(
            {
                "hot_target_count": len(covered_hot_targets),
                "hot_targets": covered_hot_targets,
            }
        )
    return diagnostics


def apply_atcc_experiment_overrides(
    task: AgentTask,
    decision: Any,
    features: Any,
    config: MixedBenchmarkConfig,
    *,
    retry_count: int,
) -> Any:
    if decision is None:
        return None
    action = str(getattr(decision, "action", "") or "")
    if action != RESERVE_HOT_RW_K:
        return decision
    fallback_ratio = float(config.atcc_full_reservation_fallback_ratio)
    if fallback_ratio > 0.0 and stable_task_fraction(
        task,
        retry_count=retry_count,
        salt="full-reservation-fallback",
    ) < fallback_ratio:
        targets = tuple(sorted(set(features.read_targets) | set(features.write_targets)))
        if targets:
            metadata = dict(getattr(decision, "metadata", {}) or {})
            metadata.update(
                {
                    "mixed_experiment_override": "full-reservation-fallback",
                    "full_reservation_fallback_ratio": fallback_ratio,
                    "pre_experiment_action": action,
                    "pre_experiment_target_count": len(tuple(getattr(decision, "targets", ()) or ())),
                    "target_size": len(targets),
                }
            )
            return dataclasses.replace(
                decision,
                action=RESERVE_READ_WRITE_SET,
                targets=targets,
                lock_scope="read-write-set",
                lock_phase="reserve",
                metadata=metadata,
            )

    if not bool(config.atcc_agent_guardrail):
        return decision
    threshold = int(config.atcc_agent_guardrail_queue_threshold)
    original_targets = tuple(str(target) for target in getattr(decision, "targets", ()) if str(target))
    if not original_targets:
        return decision
    scored_targets = [
        (reservation_target_pressure(features, target), index, target)
        for index, target in enumerate(original_targets)
    ]
    max_pressure = max(score for score, _index, _target in scored_targets)
    if max_pressure < threshold:
        return decision
    kept = [
        target
        for score, _index, target in scored_targets
        if score < threshold
    ]
    if not kept:
        kept = [target for _score, _index, target in sorted(scored_targets)[:1]]
    if tuple(kept) == original_targets:
        return decision
    metadata = dict(getattr(decision, "metadata", {}) or {})
    metadata.update(
        {
            "mixed_experiment_override": "agent-guardrail",
            "agent_guardrail_queue_threshold": threshold,
            "agent_guardrail_max_selected_pressure": max_pressure,
            "pre_experiment_action": action,
            "pre_experiment_target_count": len(original_targets),
            "post_experiment_target_count": len(kept),
            "target_size": len(kept),
        }
    )
    return dataclasses.replace(
        decision,
        targets=tuple(kept),
        metadata=metadata,
    )


def reservation_target_pressure(features: Any, target: str) -> int:
    text = str(target)
    queue_lengths = dict(getattr(features, "reservation_queue_lengths", {}) or {})
    owner_targets = set(getattr(features, "reservation_owner_targets", ()) or ())
    writer_targets = set(getattr(features, "reservation_writer_targets", ()) or ())
    return (
        int(queue_lengths.get(text, 0) or 0)
        + (1 if text in owner_targets else 0)
        + (1 if text in writer_targets else 0)
    )


def stable_task_fraction(task: AgentTask, *, retry_count: int = 0, salt: str = "") -> float:
    payload = f"{salt}:{getattr(task, 'task_id', '')}:{int(retry_count)}".encode(
        "utf-8",
        errors="ignore",
    )
    return float(zlib.crc32(payload) % 10_000) / 10_000.0


def begin_planned_transaction(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    metadata: Dict[str, Any],
) -> Any:
    return manager.begin(
        planned.task.task_id,
        metadata,
        snapshot_object_ids=task_targets(planned.task),
    )


def mixed_traditional_begin_lock_plan(strategy_impl: Any, task: AgentTask) -> CCPlan | None:
    name = str(getattr(strategy_impl, "name", ""))
    if not name.startswith("2pl-"):
        return None
    return CCPlan(
        strategy=name,
        family=str(getattr(strategy_impl, "family", "")),
        lock_targets=task_targets(task),
        metadata={
            "lock_table": "2pl",
            "policy": getattr(strategy_impl, "policy", "nowait"),
        },
    )


def run_agent_with_begin_lock(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
    *,
    lock_table: str,
    targets: Sequence[str],
    policy: str,
    priority: int,
) -> tuple[Dict[str, Any], str, float]:
    target_tuple = tuple(str(target) for target in targets if str(target))
    action = atcc_action(metadata) or strategy
    if not target_tuple:
        txn = begin_planned_transaction(manager, planned, metadata)
        for phase in planned.phases:
            execute_phase(txn, phase)
        result = txn.commit(strategy)
        return attempt_result(result, planned, txn), action, 0.0

    owner = SimpleNamespace(started_at=time.perf_counter())
    wait_started_at = time.perf_counter()
    try:
        with mixed_lock_context(
            manager,
            lock_table=lock_table,
            targets=target_tuple,
            owner=owner,
            policy=policy,
            wait=True,
            priority=priority,
        ):
            wait_s = time.perf_counter() - wait_started_at
            lock_started_at = time.perf_counter()
            txn = begin_planned_transaction(manager, planned, metadata)
            txn.metadata["prelocked_lock_table"] = lock_table
            txn.metadata["prelocked_targets"] = target_tuple
            for phase in planned.phases:
                execute_phase(txn, phase)
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": wait_s * 1000.0,
                "lock_hold_ms": (time.perf_counter() - lock_started_at) * 1000.0,
                "skipped_reasoning_ms": 0.0,
            }
            result = txn.commit(strategy)
            txn.metadata["atcc_runtime"]["lock_hold_ms"] = (
                time.perf_counter() - lock_started_at
            ) * 1000.0
            return attempt_result(result, planned, txn), action, wait_s
    except LockConflict as exc:
        wait_s = time.perf_counter() - wait_started_at
        reads, writes = task_conflict_counts(exc.targets, planned.task)
        return {
            "committed": False,
            "wasted_reasoning_ms": 0,
            "read_conflicts": reads,
            "write_conflicts": writes,
        }, action, wait_s


def run_agent_with_traditional_commit_lock(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
) -> tuple[Dict[str, Any], str, float]:
    txn = begin_planned_transaction(manager, planned, metadata)
    for phase in planned.phases:
        execute_phase(txn, phase)
    result = txn.commit(strategy)
    return attempt_result(result, planned, txn), strategy, float(getattr(result, "lock_wait_s", 0.0) or 0.0)


def run_agent_with_traditional_deferred_commit_lock(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    metadata: Dict[str, Any],
    *,
    lock_table: str,
    targets: Sequence[str],
    policy: str,
    priority: int,
) -> tuple[Dict[str, Any], str, float]:
    before_commit, commit_phases = split_commit_phases(planned)
    txn = begin_planned_transaction(manager, planned, metadata)
    before_delay_ms = 0
    for phase in before_commit:
        execute_deferred_phase(txn, phase, deferred_before_commit_lock=True)
        delay_ms = int(phase.reasoning_delay_ms)
        before_delay_ms += delay_ms
        sleep_for_reasoning(delay_ms)
    owner = SimpleNamespace(started_at=time.perf_counter())
    wait_started_at = time.perf_counter()
    try:
        with mixed_lock_context(
            manager,
            lock_table=lock_table,
            targets=targets,
            owner=owner,
            policy=policy,
            wait=True,
            priority=priority,
        ):
            wait_s = time.perf_counter() - wait_started_at
            lock_started_at = time.perf_counter()
            txn.metadata["prelocked_lock_table"] = lock_table
            txn.metadata["prelocked_targets"] = tuple(str(target) for target in targets if str(target))
            for phase in commit_phases:
                execute_phase(txn, phase)
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": wait_s * 1000.0,
                "lock_hold_ms": (time.perf_counter() - lock_started_at) * 1000.0,
                "skipped_reasoning_ms": 0.0,
                "deferred_before_begin_ms": float(before_delay_ms),
            }
            result = txn.commit(strategy)
            return attempt_result(result, planned, txn), strategy, wait_s
    except LockConflict as exc:
        wait_s = time.perf_counter() - wait_started_at
        reads, writes = task_conflict_counts(exc.targets, planned.task)
        return {
            "committed": False,
            "wasted_reasoning_ms": int(before_delay_ms),
            "read_conflicts": reads,
            "write_conflicts": writes,
        }, strategy, wait_s


@contextlib.contextmanager
def mixed_lock_context(
    manager: AgentTransactionManager,
    *,
    lock_table: str,
    targets: Sequence[str],
    owner: Any,
    policy: str = "",
    wait: bool = True,
    priority: int = 0,
):
    table = str(lock_table)
    if table == "2pl":
        with manager.two_phase_locks.acquire(
            targets,
            owner=owner,
            mode="x",
            policy=str(policy or "nowait"),
        ):
            yield
        return
    if table == "exclusive":
        with manager.exclusive_locks.acquire(
            targets,
            owner=owner,
            wait=bool(wait),
            priority=int(priority),
        ):
            yield
        return
    yield


def attempt_result(result: Any, planned: PlannedTask, txn: Any) -> Dict[str, Any]:
    return {
        "committed": bool(result.committed),
        "wasted_reasoning_ms": 0 if result.committed else planned.total_reasoning_delay_ms,
        "read_conflicts": conflict_counts(getattr(result, "conflict_object_ids", ()), txn)[0],
        "write_conflicts": conflict_counts(getattr(result, "conflict_object_ids", ()), txn)[1],
    }


def atcc_action(metadata: Dict[str, Any]) -> str:
    preplan = dict(metadata.get("atcc_preplan", {}) or {})
    return str(preplan.get("action", "") or "")


def atcc_preplan_from_decision(strategy: str, decision: Any | None) -> Dict[str, Any]:
    if decision is None:
        return {}
    return {
        "strategy": str(strategy),
        "action": str(decision.action),
        "targets": tuple(decision.targets),
        "priority": int(decision.priority),
        "state_key": str(decision.state_key),
        "reason": str(decision.reason),
        "lock_scope": str(decision.lock_scope),
        "lock_phase": str(decision.lock_phase),
        "metadata": dict(decision.metadata),
    }


def estimated_background_abort_cost(background_workers: int, hold_ms: int | float) -> float:
    if float(hold_ms) <= 0:
        return 0.0
    return max(0.0, float(background_workers)) * min(1.0, float(hold_ms) / 1000.0)


def conflict_counts(conflict_object_ids: Iterable[str], txn: Any) -> tuple[int, int]:
    write_targets = {str(key) for key in getattr(txn, "write_set", {}).keys()}
    conflicts = {str(value) for value in conflict_object_ids}
    write_conflicts = sum(1 for value in conflicts if value in write_targets)
    read_conflicts = max(0, len(conflicts) - write_conflicts)
    return read_conflicts, write_conflicts


def task_conflict_counts(conflict_object_ids: Iterable[str], task: AgentTask) -> tuple[int, int]:
    writes = {
        str(operation.object_id)
        for operation in getattr(task, "operations", ()) or ()
        if str(getattr(operation, "kind", "")) == "write"
    }
    conflicts = {str(value) for value in conflict_object_ids}
    write_conflicts = sum(1 for value in conflicts if value in writes)
    read_conflicts = max(0, len(conflicts) - write_conflicts)
    return read_conflicts, write_conflicts


def observe_aborted_atcc(strategy_impl: Any, decision: Any, txn: Any, result: Any) -> None:
    observer = getattr(strategy_impl, "observe", None)
    if observer is None:
        return
    plan = CCPlan(
        strategy=str(getattr(strategy_impl, "name", "")),
        family=str(getattr(strategy_impl, "family", "")),
        lock_targets=tuple(getattr(decision, "targets", ()) or ()),
        validate_reads=True,
        validate_writes=True,
        metadata={
            "atcc_action": str(getattr(decision, "action", "") or ""),
            "atcc_state_key": str(getattr(decision, "state_key", "") or ""),
            "atcc_reason": str(getattr(decision, "reason", "") or ""),
            "atcc_lock_scope": str(getattr(decision, "lock_scope", "") or ""),
            "atcc_lock_phase": str(getattr(decision, "lock_phase", "") or ""),
        },
    )
    observer(plan, result, txn)


def background_worker(
    manager: AgentTransactionManager,
    hot_targets: Sequence[str],
    background_tasks: Sequence[AgentTask],
    strategy: str,
    config: MixedBenchmarkConfig,
    stop_at: float,
    counters: MixedCounters,
    lock: threading.Lock,
    worker: int,
    background_stop: threading.Event,
    background_admission: threading.BoundedSemaphore | None = None,
) -> None:
    rng = random.Random(config.seed + 1000 + worker)
    task_index = worker
    while time.perf_counter() < stop_at and not background_stop.is_set():
        started_wait = time.perf_counter()
        txn = None
        try:
            if config.background_mode == "procedure":
                task = background_tasks[task_index % len(background_tasks)]
                task_index += max(1, config.background_workers)
                targets = task_targets(task)
                txn = manager.begin(
                    f"bg-{worker}-{task.task_id}-{rng.randrange(10_000_000)}",
                    {
                        "workload": config.workload,
                        "task_type": f"background-{task.task_type}",
                        "context": dict(task.context),
                    },
                    snapshot_object_ids=targets,
                )
                with background_write_guard(
                    manager,
                    operation_write_targets(task),
                    strategy,
                    config,
                    owner=txn,
                    admission=background_admission,
                ) as wait_s:
                    for operation in task.operations:
                        apply_operation(txn, operation)
                    result = txn.commit("occ")
            else:
                target = hot_targets[rng.randrange(len(hot_targets))]
                txn = manager.begin(
                    f"bg-{worker}-{rng.randrange(10_000_000)}",
                    {"workload": config.workload, "task_type": "background-hot-write", "context": {"level": config.level}},
                    snapshot_object_ids=(target,),
                )
                with background_write_guard(
                    manager,
                    (target,),
                    strategy,
                    config,
                    owner=txn,
                    admission=background_admission,
                ) as wait_s:
                    current = txn.read(target).value
                    txn.write(target, f"bg:{worker}:{current}:{rng.randrange(10_000_000)}")
                    result = txn.commit("occ")
            committed = bool(result.committed)
            aborted = not committed
        except (LockConflict, KeyError, ValueError):
            wait_s = time.perf_counter() - started_wait
            committed = False
            aborted = True
        with lock:
            counters.background_attempts += 1
            counters.background_reservation_wait_s += wait_s
            if committed:
                counters.background_commits += 1
            elif aborted:
                counters.background_aborts += 1
        if aborted:
            backoff_ms = rng.randint(
                int(config.background_retry_backoff_min_ms),
                int(config.background_retry_backoff_max_ms),
            )
            with lock:
                counters.background_retries += 1
            sleep_for_reasoning(backoff_ms)


@contextlib.contextmanager
def background_write_guard(
    manager: AgentTransactionManager,
    targets: Sequence[str],
    strategy: str,
    config: MixedBenchmarkConfig,
    *,
    owner: Any,
    admission: threading.BoundedSemaphore | None = None,
):
    started_at = time.perf_counter()
    strategy_impl = manager.cc_registry.resolve(strategy)
    target_tuple = tuple(str(target) for target in targets if str(target))
    with contextlib.ExitStack() as stack:
        if admission is not None:
            stack.enter_context(semaphore_context(admission))
        stack.enter_context(
            manager.reservations.write_guard(
                target_tuple,
                owner=owner,
                wait=bool(config.background_wait),
                timeout_s=0.010,
            )
        )
        name = str(getattr(strategy_impl, "name", ""))
        family = str(getattr(strategy_impl, "family", ""))
        if name.startswith("2pl-"):
            stack.enter_context(
                manager.two_phase_locks.acquire(
                    target_tuple,
                    owner=owner,
                    mode="x",
                    policy=str(getattr(strategy_impl, "policy", "nowait")),
                )
            )
        elif family in {"silo", "tictoc"}:
            stack.enter_context(
                manager.exclusive_locks.acquire(
                    target_tuple,
                    owner=owner,
                    wait=bool(config.background_wait),
                    priority=0,
                )
            )
        yield time.perf_counter() - started_at


@contextlib.contextmanager
def semaphore_context(semaphore: threading.BoundedSemaphore):
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def execute_phase(txn: Any, phase: Any) -> None:
    execute_phase_operations(txn, phase)
    sleep_for_reasoning(phase.reasoning_delay_ms)


def execute_phase_operations(txn: Any, phase: Any) -> None:
    for operation in phase.operations:
        apply_operation(txn, operation)


def sleep_phase_reasoning(phases: Sequence[Any]) -> int:
    total_ms = 0
    for phase in phases:
        delay_ms = int(phase.reasoning_delay_ms)
        total_ms += delay_ms
        sleep_for_reasoning(delay_ms)
    return total_ms


def execute_deferred_phase(txn: Any, phase: Any, **flags: Any) -> None:
    detail = {
        "name": phase.name,
        "operations": len(phase.operations),
        "reasoning_delay_ms": int(phase.reasoning_delay_ms),
    }
    detail.update(flags)
    txn._event("phase", detail)
    execute_phase_operations(txn, phase)


def split_commit_phases(planned: PlannedTask) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    before = []
    commit = []
    commit_started = False
    for phase in planned.phases:
        if str(phase.name) == "commit":
            commit_started = True
        if commit_started:
            commit.append(phase)
        else:
            before.append(phase)
    return tuple(before), tuple(commit)


def can_defer_transaction_begin(planned: PlannedTask) -> bool:
    before_commit, _commit_phases = split_commit_phases(planned)
    return all(not phase.operations for phase in before_commit)


def can_defer_read_heavy_transaction_begin(planned: PlannedTask) -> bool:
    before_commit, _commit_phases = split_commit_phases(planned)
    operations = [
        operation
        for phase in before_commit
        for operation in phase.operations
    ]
    if not operations:
        return False
    return all(str(getattr(operation, "kind", "")) == "read" for operation in operations)


def planned_write_conflicts(
    manager: AgentTransactionManager,
    txn: Any,
    planned: PlannedTask,
) -> tuple[str, ...]:
    conflicts = []
    _before, commit_phases = split_commit_phases(planned)
    for phase in commit_phases:
        for operation in phase.operations:
            if str(getattr(operation, "kind", "")) != "write":
                continue
            object_id = str(operation.object_id)
            snapshot = getattr(txn, "snapshot", {}).get(object_id)
            if snapshot is not None and int(manager.store.get_version(object_id)) != int(snapshot.version):
                conflicts.append(object_id)
    return tuple(sorted(set(conflicts)))


def hot_write_targets(tasks: Sequence[AgentTask]) -> tuple[str, ...]:
    targets = []
    for task in tasks:
        context = dict(task.context)
        for operation in task.operations:
            if str(operation.kind) != "write":
                continue
            object_id = str(operation.object_id)
            if "next_order_id" in object_id:
                targets.append(object_id)
            elif object_id.endswith(":orders"):
                targets.append(object_id)
            elif ":stock:" in object_id:
                targets.append(object_id)
            elif context.get("hot_record_count") and ":record:" in object_id:
                parts = object_id.split(":")
                try:
                    index = parts.index("record") + 1
                    if int(parts[index]) < int(context.get("hot_record_count") or 0):
                        targets.append(object_id)
                except (ValueError, IndexError):
                    pass
    return tuple(sorted(set(targets)))


def write_targets(tasks: Sequence[AgentTask]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(operation.object_id)
                for task in tasks
                for operation in task.operations
                if str(operation.kind) == "write"
            }
        )
    )


def task_targets(task: AgentTask) -> tuple[str, ...]:
    return unique_targets(operation.object_id for operation in task.operations)


def stable_task_seed(task: AgentTask, *, retry_count: int = 0) -> int:
    payload = f"{getattr(task, 'task_id', '')}:{int(retry_count)}".encode(
        "utf-8",
        errors="ignore",
    )
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def operation_write_targets(task: AgentTask) -> tuple[str, ...]:
    return unique_targets(
        operation.object_id
        for operation in task.operations
        if str(getattr(operation, "kind", "")) == "write"
    )


def expand_cc(value: str) -> List[str]:
    registry = ConcurrencyControlRegistry()
    return registry.expand(value)


def registry_for(config: MixedBenchmarkConfig) -> ConcurrencyControlRegistry:
    atcc_options = {
        "hot_rw_k_target_limit": int(config.atcc_hot_rw_k),
        "bp_background_threshold": int(config.atcc_bp_background_threshold),
        "bp_queue_pressure_threshold": int(config.atcc_bp_queue_pressure_threshold),
        "bp_min_windows": int(config.atcc_bp_min_windows),
        "runtime_guards_enabled": not bool(config.atcc_pure_policy),
    }
    if config.policy is not None and hasattr(config.policy, "set_mode"):
        config.policy.set_mode(config.policy_mode)
        return ConcurrencyControlRegistry(atcc_policy=config.policy, atcc_options=atcc_options)
    registry = ConcurrencyControlRegistry.from_policy_file(config.policy, atcc_options=atcc_options)
    for name in registry.strategies().keys():
        impl = registry.resolve(name)
        policy = getattr(impl, "policy", None)
        if hasattr(policy, "set_mode"):
            policy.set_mode(config.policy_mode)
    return registry


def average(values: Sequence[int | float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def percentile(values: Sequence[int | float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def add_distribution_fields(
    row: Dict[str, Any],
    prefix: str,
    values: Sequence[int | float],
    *,
    include_histogram: bool = False,
) -> None:
    samples = tuple(float(value) for value in values)
    row[f"{prefix}_count"] = len(samples)
    row[f"{prefix}_mean"] = average(samples)
    row[f"{prefix}_p50"] = percentile(samples, 50)
    row[f"{prefix}_p95"] = percentile(samples, 95)
    row[f"{prefix}_p99"] = percentile(samples, 99)
    row[f"{prefix}_max"] = max(samples) if samples else 0.0
    if include_histogram:
        histogram: Dict[str, int] = {}
        for value in samples:
            key = str(int(round(value)))
            histogram[key] = histogram.get(key, 0) + 1
        row[f"{prefix}_hist"] = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
