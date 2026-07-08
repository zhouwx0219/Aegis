"""Barrier-batch concurrent benchmark runner."""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

from agent.benchmarks.config import BenchmarkConfig
from agent.benchmarks.metrics import BenchmarkAttempt, aggregate_metrics
from agent.benchmarks.phases import PlannedPhase, ReasoningProfile, PlannedTask, plan_task_phases, sleep_for_reasoning
from agent.cc import ConcurrencyControlRegistry, LockConflict
from agent.cc.atcc.actions import LOCK_BEFORE_COMMIT
from agent.cc.atcc.features import extract_task_features
from agent.cc.base import CCPlan, unique_targets
from agent.runtime import AgentTransactionManager
from agent.workloads import (
    AgentTask,
    apply_operation,
    build_workload,
    register_workload,
)


def run_cc_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    config = config.normalized()
    registry = registry_for(config)
    strategies = registry.expand(config.cc)
    rows = []
    for strategy in strategies:
        rows.append(
            run_strategy_benchmark(
                config=config,
                strategy=strategy,
            )
        )
    return {
        "mode": "cc-benchmark",
        "workload": config.workload,
        "level": config.level,
        "workload_profile": config.workload_profile,
        "tasks": int(config.tasks),
        "workers": int(config.workers),
        "retries": int(config.retries),
        "seed": int(config.seed),
        "reasoning_profile": config.reasoning_profile,
        "reasoning_scale": float(config.reasoning_scale),
        "policy_mode": config.policy_mode,
        "policy": str(config.policy) if config.policy else "",
        "strategies": strategies,
        "cc_results": rows,
    }


def run_strategy_benchmark(
    *,
    config: BenchmarkConfig,
    strategy: str,
) -> Dict[str, Any]:
    workload = build_workload(config.workload, config.level, config.workload_profile)
    manager = AgentTransactionManager(
        cc_registry=registry_for(config)
    )
    register_workload(manager, workload)
    tasks = list(workload.generate_tasks(config.tasks, seed=config.seed))

    pending = list(tasks)
    attempts: List[BenchmarkAttempt] = []
    started_at = time.perf_counter()
    for attempt_index in range(config.retries + 1):
        if not pending:
            break
        failed: List[AgentTask] = []
        for batch in batched(pending, config.workers):
            results = execute_batch(
                config=config,
                manager=manager,
                tasks=batch,
                strategy=strategy,
                attempt_index=attempt_index,
            )
            attempts.extend(results)
            failed.extend(
                task
                for task, result in zip(batch, results)
                if not result.committed
            )
        pending = failed
    elapsed_s = time.perf_counter() - started_at
    return aggregate_metrics(
        cc=strategy,
        task_count=len(tasks),
        attempts=attempts,
        elapsed_s=elapsed_s,
    ).to_dict()


def execute_batch(
    *,
    config: BenchmarkConfig,
    manager: AgentTransactionManager,
    tasks: Sequence[AgentTask],
    strategy: str,
    attempt_index: int,
) -> List[BenchmarkAttempt]:
    if not tasks:
        return []
    task_list = list(tasks)
    profile = ReasoningProfile(
        name=config.reasoning_profile,
        scale=float(config.reasoning_scale),
    )
    planned_tasks = [
        plan_task_phases(task, attempt=attempt_index, profile=profile)
        for task in task_list
    ]
    prelocks = [
        plan_prelock(
            manager=manager,
            task=planned,
            strategy=strategy,
            attempt_index=attempt_index,
        )
        for planned in planned_tasks
    ]
    optimistic_count = sum(1 for prelock in prelocks if prelock["plan"] is None)
    commit_barrier = threading.Barrier(optimistic_count) if optimistic_count > 1 else None
    with ThreadPoolExecutor(max_workers=len(task_list)) as executor:
        futures = [
            executor.submit(
                execute_one,
                manager,
                planned,
                strategy,
                attempt_index,
                prelock,
                commit_barrier,
            )
            for planned, prelock in zip(planned_tasks, prelocks)
        ]
        return [future.result() for future in futures]


def execute_one(
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    attempt_index: int,
    prelock: Dict[str, Any],
    commit_barrier: threading.Barrier | None,
) -> BenchmarkAttempt:
    task = planned.task
    started_at = time.perf_counter()
    if prelock["plan"] is None:
        if decision_locks_before_commit(prelock) and can_defer_transaction_begin(planned):
            return execute_deferred_commit_phase_lock(
                manager=manager,
                planned=planned,
                strategy=strategy,
                attempt_index=attempt_index,
                prelock=prelock,
                commit_barrier=commit_barrier,
                started_at=started_at,
            )
        txn = manager.begin(
            task.task_id,
            transaction_metadata(task, attempt_index, planned=planned),
        )
        attach_preplanned_atcc(txn, strategy=strategy, prelock=prelock)
        if decision_locks_before_commit(prelock):
            return execute_with_commit_phase_lock(
                manager=manager,
                txn=txn,
                planned=planned,
                strategy=strategy,
                attempt_index=attempt_index,
                prelock=prelock,
                commit_barrier=commit_barrier,
            )
        execute_planned_task(txn, planned)
        if commit_barrier is not None:
            commit_barrier.wait()
        result = txn.commit(strategy)
        attempt = BenchmarkAttempt.from_result(result, attempt=attempt_index)
        return dataclasses_replace_attempt(attempt, planned=planned, txn=txn)

    try:
        wait_started_at = time.perf_counter()
        with prelock_context(manager, owner=task, plan=prelock["plan"]):
            prelock_wait_s = time.perf_counter() - wait_started_at
            lock_started_at = time.perf_counter()
            txn = manager.begin(
                task.task_id,
                transaction_metadata(task, attempt_index, planned=planned),
            )
            attach_preplanned_atcc(txn, strategy=strategy, prelock=prelock)
            txn.metadata["prelocked_lock_table"] = prelock["plan"].metadata.get(
                "lock_table",
                "",
            )
            txn.metadata["prelocked_targets"] = tuple(prelock["plan"].lock_targets)
            execute_planned_task(txn, planned)
            lock_hold_s = time.perf_counter() - lock_started_at
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": prelock_wait_s * 1000.0,
                "lock_hold_ms": lock_hold_s * 1000.0,
                "skipped_reasoning_ms": 0.0,
            }
            result = txn.commit(strategy)
            lock_hold_s = time.perf_counter() - lock_started_at
            txn.metadata["atcc_runtime"]["lock_hold_ms"] = lock_hold_s * 1000.0
        return BenchmarkAttempt(
            task_id=result.task_id,
            attempt=attempt_index,
            committed=bool(result.committed),
            reason=result.reason,
            elapsed_s=time.perf_counter() - started_at,
            lock_wait_s=prelock_wait_s + float(result.lock_wait_s),
            conflict_object_ids=tuple(result.conflict_object_ids),
            read_count=result.read_count,
            write_count=result.write_count,
            phase_count=planned.phase_count,
            reasoning_delay_ms=planned.total_reasoning_delay_ms,
            lock_hold_s=lock_hold_s,
            atcc_action=atcc_action_from_prelock(prelock),
        )
    except LockConflict as exc:
        return BenchmarkAttempt(
            task_id=task.task_id,
            attempt=attempt_index,
            committed=False,
            reason=exc.reason,
            elapsed_s=time.perf_counter() - started_at,
            lock_wait_s=time.perf_counter() - started_at,
            conflict_object_ids=tuple(exc.targets),
            read_count=0,
            write_count=0,
            phase_count=planned.phase_count,
            reasoning_delay_ms=planned.total_reasoning_delay_ms,
            atcc_action=atcc_action_from_prelock(prelock),
        )


def plan_prelock(
    *,
    manager: AgentTransactionManager,
    task: AgentTask | PlannedTask,
    strategy: str,
    attempt_index: int,
) -> Dict[str, Any]:
    strategy_impl = manager.cc_registry.resolve(strategy)
    agent_task = task.task if isinstance(task, PlannedTask) else task
    name = str(strategy_impl.name)
    if name.startswith("2pl-"):
        targets = task_targets(agent_task)
        return {
            "owner": None,
            "plan": CCPlan(
                strategy=name,
                family=strategy_impl.family,
                lock_targets=targets,
                metadata={
                    "lock_table": "2pl",
                    "policy": getattr(strategy_impl, "policy", "nowait"),
                },
            ),
        }
    if getattr(strategy_impl, "family", "") != "atcc":
        return {"owner": None, "plan": None, "atcc_decision": None}

    decision = strategy_impl.decide(
        extract_task_features(
            agent_task,
            retry_count=attempt_index,
            agentic=agentic_metadata(task) if isinstance(task, PlannedTask) else None,
        )
    )
    if not decision.begins_locked:
        return {"owner": None, "plan": None, "atcc_decision": decision}
    return {
        "owner": None,
        "atcc_decision": decision,
        "plan": CCPlan(
            strategy=name,
            family=strategy_impl.family,
            lock_targets=tuple(decision.targets),
            validate_reads=True,
            validate_writes=True,
            metadata={
                "lock_table": "exclusive",
                "wait": True,
                "priority": int(decision.priority),
                "atcc_action": decision.action,
                "atcc_state_key": decision.state_key,
                "atcc_reason": decision.reason,
                "atcc_lock_scope": decision.lock_scope,
                "atcc_lock_phase": decision.lock_phase,
            },
        ),
    }


def attach_preplanned_atcc(txn: Any, *, strategy: str, prelock: Dict[str, Any]) -> None:
    decision = prelock.get("atcc_decision")
    if decision is None:
        return
    txn.metadata["atcc_preplan"] = {
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


def execute_planned_task(txn: Any, planned: PlannedTask) -> None:
    sleep_for_reasoning(planned.retry_delay_ms)
    for phase in planned.phases:
        execute_phase(txn, phase)


def execute_phase(txn: Any, phase: PlannedPhase) -> None:
    for operation in phase.operations:
        apply_operation(txn, operation)
    txn._event(
        "phase",
        {
            "name": phase.name,
            "operations": len(phase.operations),
            "reasoning_delay_ms": int(phase.reasoning_delay_ms),
        },
    )
    sleep_for_reasoning(phase.reasoning_delay_ms)


def decision_locks_before_commit(prelock: Dict[str, Any]) -> bool:
    decision = prelock.get("atcc_decision")
    return (
        decision is not None
        and str(getattr(decision, "action", "")) == LOCK_BEFORE_COMMIT
    )


def execute_with_commit_phase_lock(
    *,
    manager: AgentTransactionManager,
    txn: Any,
    planned: PlannedTask,
    strategy: str,
    attempt_index: int,
    prelock: Dict[str, Any],
    commit_barrier: threading.Barrier | None,
) -> BenchmarkAttempt:
    decision = prelock.get("atcc_decision")
    targets = tuple(getattr(decision, "targets", ()) or ())
    sleep_for_reasoning(planned.retry_delay_ms)
    before_commit, commit_phases = split_commit_phases(planned)
    for phase in before_commit:
        execute_phase(txn, phase)
    if commit_barrier is not None:
        commit_barrier.wait()
    wait_started_at = time.perf_counter()
    try:
        with manager.exclusive_locks.acquire(
            targets,
            owner=txn,
            wait=True,
            priority=int(getattr(decision, "priority", 0) or 0),
        ):
            lock_wait_s = time.perf_counter() - wait_started_at
            lock_started_at = time.perf_counter()
            conflicts = planned_write_conflicts(manager, txn, planned)
            if conflicts:
                skipped = sum(phase.reasoning_delay_ms for phase in commit_phases)
                txn.metadata["atcc_runtime"] = {
                    "lock_wait_ms": lock_wait_s * 1000.0,
                    "lock_hold_ms": (time.perf_counter() - lock_started_at) * 1000.0,
                    "skipped_reasoning_ms": skipped,
                }
                result = txn.abort(
                    "early version conflict before commit phase",
                    strategy=strategy,
                )
                observe_strategy(manager, strategy, txn, prelock, result)
                return BenchmarkAttempt(
                    task_id=result.task_id,
                    attempt=attempt_index,
                    committed=False,
                    reason=result.reason,
                    elapsed_s=result.elapsed_s,
                    lock_wait_s=lock_wait_s,
                    conflict_object_ids=tuple(conflicts),
                    read_count=result.read_count,
                    write_count=result.write_count,
                    phase_count=planned.phase_count,
                    reasoning_delay_ms=planned.total_reasoning_delay_ms,
                    lock_hold_s=time.perf_counter() - lock_started_at,
                    early_abort=True,
                    skipped_reasoning_ms=skipped,
                    atcc_action=atcc_action_from_prelock(prelock),
                )
            txn.metadata["prelocked_lock_table"] = "exclusive"
            txn.metadata["prelocked_targets"] = tuple(targets)
            for phase in commit_phases:
                execute_phase(txn, phase)
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": lock_wait_s * 1000.0,
                "lock_hold_ms": (time.perf_counter() - lock_started_at) * 1000.0,
                "skipped_reasoning_ms": 0.0,
            }
            result = txn.commit(strategy)
            return BenchmarkAttempt(
                task_id=result.task_id,
                attempt=attempt_index,
                committed=bool(result.committed),
                reason=result.reason,
                elapsed_s=result.elapsed_s,
                lock_wait_s=lock_wait_s + float(result.lock_wait_s),
                conflict_object_ids=tuple(result.conflict_object_ids),
                read_count=result.read_count,
                write_count=result.write_count,
                phase_count=planned.phase_count,
                reasoning_delay_ms=planned.total_reasoning_delay_ms,
                lock_hold_s=time.perf_counter() - lock_started_at,
                atcc_action=atcc_action_from_prelock(prelock),
            )
    except LockConflict as exc:
        return BenchmarkAttempt(
            task_id=planned.task.task_id,
            attempt=attempt_index,
            committed=False,
            reason=exc.reason,
            elapsed_s=time.perf_counter() - txn.started_at,
            lock_wait_s=time.perf_counter() - wait_started_at,
            conflict_object_ids=tuple(exc.targets),
            read_count=len(getattr(txn, "read_set", {}) or {}),
            write_count=len(getattr(txn, "write_set", {}) or {}),
            phase_count=planned.phase_count,
            reasoning_delay_ms=planned.total_reasoning_delay_ms,
            atcc_action=atcc_action_from_prelock(prelock),
        )


def execute_deferred_commit_phase_lock(
    *,
    manager: AgentTransactionManager,
    planned: PlannedTask,
    strategy: str,
    attempt_index: int,
    prelock: Dict[str, Any],
    commit_barrier: threading.Barrier | None,
    started_at: float,
) -> BenchmarkAttempt:
    decision = prelock.get("atcc_decision")
    targets = tuple(getattr(decision, "targets", ()) or ())
    owner = SimpleNamespace(started_at=float(started_at))
    sleep_for_reasoning(planned.retry_delay_ms)
    before_commit, commit_phases = split_commit_phases(planned)
    for phase in before_commit:
        sleep_for_reasoning(phase.reasoning_delay_ms)
    if commit_barrier is not None:
        commit_barrier.wait()
    wait_started_at = time.perf_counter()
    try:
        with manager.exclusive_locks.acquire(
            targets,
            owner=owner,
            wait=True,
            priority=int(getattr(decision, "priority", 0) or 0),
        ):
            lock_wait_s = time.perf_counter() - wait_started_at
            lock_started_at = time.perf_counter()
            txn = manager.begin(
                planned.task.task_id,
                transaction_metadata(planned.task, attempt_index, planned=planned),
            )
            attach_preplanned_atcc(txn, strategy=strategy, prelock=prelock)
            txn.metadata["prelocked_lock_table"] = "exclusive"
            txn.metadata["prelocked_targets"] = tuple(targets)
            for phase in before_commit:
                txn._event(
                    "phase",
                    {
                        "name": phase.name,
                        "operations": 0,
                        "reasoning_delay_ms": int(phase.reasoning_delay_ms),
                        "deferred_before_begin": True,
                    },
                )
            for phase in commit_phases:
                execute_phase(txn, phase)
            txn.metadata["atcc_runtime"] = {
                "lock_wait_ms": lock_wait_s * 1000.0,
                "lock_hold_ms": (time.perf_counter() - lock_started_at) * 1000.0,
                "skipped_reasoning_ms": 0.0,
            }
            result = txn.commit(strategy)
            return BenchmarkAttempt(
                task_id=result.task_id,
                attempt=attempt_index,
                committed=bool(result.committed),
                reason=result.reason,
                elapsed_s=time.perf_counter() - started_at,
                lock_wait_s=lock_wait_s + float(result.lock_wait_s),
                conflict_object_ids=tuple(result.conflict_object_ids),
                read_count=result.read_count,
                write_count=result.write_count,
                phase_count=planned.phase_count,
                reasoning_delay_ms=planned.total_reasoning_delay_ms,
                lock_hold_s=time.perf_counter() - lock_started_at,
                atcc_action=atcc_action_from_prelock(prelock),
            )
    except LockConflict as exc:
        return BenchmarkAttempt(
            task_id=planned.task.task_id,
            attempt=attempt_index,
            committed=False,
            reason=exc.reason,
            elapsed_s=time.perf_counter() - started_at,
            lock_wait_s=time.perf_counter() - wait_started_at,
            conflict_object_ids=tuple(exc.targets),
            read_count=0,
            write_count=0,
            phase_count=planned.phase_count,
            reasoning_delay_ms=planned.total_reasoning_delay_ms,
            atcc_action=atcc_action_from_prelock(prelock),
        )


def can_defer_transaction_begin(planned: PlannedTask) -> bool:
    before_commit, _commit_phases = split_commit_phases(planned)
    return all(not phase.operations for phase in before_commit)


def split_commit_phases(planned: PlannedTask) -> tuple[tuple[PlannedPhase, ...], tuple[PlannedPhase, ...]]:
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


def observe_strategy(
    manager: AgentTransactionManager,
    strategy: str,
    txn: Any,
    prelock: Dict[str, Any],
    result: Any,
) -> None:
    strategy_impl = manager.cc_registry.resolve(strategy)
    plan = CCPlan(
        strategy=str(strategy),
        family=getattr(strategy_impl, "family", ""),
        metadata={
            "atcc_action": atcc_action_from_prelock(prelock),
            "atcc_state_key": str(getattr(prelock.get("atcc_decision"), "state_key", "") or ""),
        },
    )
    observer = getattr(strategy_impl, "observe", None)
    if observer is not None:
        observer(plan, result, txn)


def atcc_action_from_prelock(prelock: Dict[str, Any]) -> str:
    decision = prelock.get("atcc_decision")
    if decision is None:
        return ""
    return str(getattr(decision, "action", "") or "")


def transaction_metadata(
    task: AgentTask,
    attempt_index: int,
    *,
    planned: PlannedTask | None = None,
) -> Dict[str, Any]:
    context = {
        **dict(task.context),
        "retry_count": attempt_index,
        "benchmark_attempt": attempt_index,
    }
    metadata = {
        "workload": task.workload,
        "task_type": task.task_type,
        "context": context,
        "retry_count": int(attempt_index),
    }
    if planned is not None:
        metadata["agentic"] = agentic_metadata(planned)
    return metadata


def agentic_metadata(planned: PlannedTask) -> Dict[str, Any]:
    context = dict(planned.task.context)
    return {
        "phase_count": planned.phase_count,
        "reasoning_delay_ms": planned.total_reasoning_delay_ms,
        "retry_delay_ms": planned.retry_delay_ms,
        "phase_names": [phase.name for phase in planned.phases],
        "agent_cost_class": str(context.get("agent_cost_class", "")),
        "phase_shape": str(context.get("phase_shape", "")),
        "side_effect_cost_ms": int(context.get("side_effect_cost_ms", 0) or 0),
    }


def dataclasses_replace_attempt(
    attempt: BenchmarkAttempt,
    *,
    planned: PlannedTask,
    txn: Any,
) -> BenchmarkAttempt:
    runtime = dict(getattr(txn, "metadata", {}).get("atcc_runtime", {}) or {})
    preplan = dict(getattr(txn, "metadata", {}).get("atcc_preplan", {}) or {})
    return dataclasses.replace(
        attempt,
        phase_count=planned.phase_count,
        reasoning_delay_ms=planned.total_reasoning_delay_ms,
        lock_hold_s=float(runtime.get("lock_hold_ms", 0.0) or 0.0) / 1000.0,
        skipped_reasoning_ms=int(runtime.get("skipped_reasoning_ms", 0.0) or 0.0),
        atcc_action=str(preplan.get("action", "") or ""),
    )


@contextlib.contextmanager
def prelock_context(
    manager: AgentTransactionManager,
    *,
    owner: Any,
    plan: CCPlan | None,
):
    if plan is None:
        yield
        return
    lock_table = str(plan.metadata.get("lock_table", ""))
    if lock_table == "2pl":
        with manager.two_phase_locks.acquire(
            plan.lock_targets,
            owner=owner,
            mode="x",
            policy=str(plan.metadata.get("policy", "nowait")),
        ):
            yield
        return
    if lock_table == "exclusive":
        with manager.exclusive_locks.acquire(
            plan.lock_targets,
            owner=owner,
            wait=bool(plan.metadata.get("wait", True)),
            priority=int(plan.metadata.get("priority", 0) or 0),
        ):
            yield
        return
    yield


def task_targets(task: AgentTask) -> tuple[str, ...]:
    return unique_targets(operation.object_id for operation in task.operations)


def batched(values: Sequence[AgentTask], size: int) -> Iterable[Sequence[AgentTask]]:
    for offset in range(0, len(values), max(1, int(size))):
        yield values[offset : offset + max(1, int(size))]


def registry_for(config: BenchmarkConfig) -> ConcurrencyControlRegistry:
    if config.atcc_policy is not None:
        config.atcc_policy.set_mode(config.policy_mode)
        return ConcurrencyControlRegistry(atcc_policy=config.atcc_policy)
    registry = ConcurrencyControlRegistry.from_policy_file(config.policy)
    for strategy in registry.strategies().keys():
        impl = registry.resolve(strategy)
        policy = getattr(impl, "policy", None)
        if hasattr(policy, "set_mode"):
            policy.set_mode(config.policy_mode)
    return registry
