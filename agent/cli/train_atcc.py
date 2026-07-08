"""Train a cost-aware dynamic ATCC policy artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TextIO

from agent.benchmarks import BenchmarkConfig, MixedBenchmarkConfig, run_cc_benchmark, run_mixed_benchmark
from agent.cc import ATCCPolicyTable
from agent.cc.atcc.actions import MIXED_TRAINABLE_ACTIONS, all_actions
from agent.cc.atcc.reward import ATCCRewardConfig


ATCC_STRATEGY = "dynamic-atcc"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("concurrent", "mixed"), default="concurrent")
    parser.add_argument("--workload", "-w", choices=("ycsb", "tpcc"), default="tpcc")
    parser.add_argument("--level", "-l", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--workload-profile", choices=("small", "paper"), default="small")
    parser.add_argument("--workloads", help="Comma-separated workloads for one shared policy, or all.")
    parser.add_argument("--levels", help="Comma-separated conflict levels for one shared policy, or all.")
    parser.add_argument("--episodes", "-e", type=int, default=5)
    parser.add_argument("--tasks", "-n", type=int, default=100)
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--duration", "-d", type=float, default=1.0)
    parser.add_argument("--clients", "-c", type=int, default=0, help="Total clients for mixed training. When set, derives agents/background from --agent-ratio.")
    parser.add_argument("--agent-ratio", type=float, default=0.80)
    parser.add_argument("--agents", "-a", type=int, default=2)
    parser.add_argument("--background", "-b", type=int, default=8)
    parser.add_argument("--background-mode", choices=("hotspot", "procedure"), default="hotspot")
    parser.add_argument("--retries", "-r", type=int, default=0)
    parser.add_argument("--retry-until-commit", action="store_true")
    parser.add_argument("--max-attempts-per-task", type=int, default=100)
    parser.add_argument("--agent-retry-backoff-ms", default="500,5000")
    parser.add_argument("--background-retry-backoff-ms", default="10,30")
    parser.add_argument("--tokens-per-operation", type=int, default=2703)
    parser.add_argument("--seed", type=int, default=920104)
    parser.add_argument("--abort-threshold", type=float, default=0.20)
    parser.add_argument("--min-visits", type=int, default=5)
    parser.add_argument("--protect-cost-threshold-ms", type=float, default=10.0)
    parser.add_argument("--low-conflict-safe-abort-rate", type=float, default=0.50)
    parser.add_argument("--disable-low-conflict-occ-guard", action="store_true")
    parser.add_argument("--disable-sparse-state-risk-prior", action="store_true")
    parser.add_argument("--commit-value", type=float, default=100.0)
    parser.add_argument("--abort-penalty", type=float, default=80.0)
    parser.add_argument("--reasoning-weight", type=float, default=1.0)
    parser.add_argument("--lock-wait-weight", type=float, default=0.5)
    parser.add_argument("--latency-weight", type=float, default=0.1)
    parser.add_argument("--lock-hold-weight", type=float, default=0.05)
    parser.add_argument("--background-abort-weight", type=float, default=2.0)
    parser.add_argument("--background-tps-loss-weight", type=float, default=0.1)
    parser.add_argument("--ucb-c", type=float, default=1.5)
    parser.add_argument(
        "--actions",
        default="auto",
        help="Comma-separated ATCC actions. Use auto for benchmark-specific defaults.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        help="Optional wall-clock training budget. Overrides --episodes by deriving episode count from --duration.",
    )
    parser.add_argument(
        "--reasoning-profile",
        choices=("none", "light", "agentic", "heavy"),
        default="agentic",
    )
    parser.add_argument("--reasoning-scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser


def train_policy(
    *,
    benchmark: str,
    workload: str,
    level: str,
    workload_profile: str,
    episodes: int,
    tasks: int,
    workers: int,
    duration_s: float,
    agents: int,
    background: int,
    clients: int = 0,
    agent_ratio: float = 0.80,
    background_mode: str = "hotspot",
    retries: int = 0,
    retry_until_commit: bool = False,
    max_attempts_per_task: int = 100,
    agent_retry_backoff_min_ms: int = 500,
    agent_retry_backoff_max_ms: int = 5000,
    background_retry_backoff_min_ms: int = 10,
    background_retry_backoff_max_ms: int = 30,
    tokens_per_operation: int = 2703,
    seed: int = 920104,
    abort_threshold: float = 0.20,
    min_visits: int = 5,
    protect_cost_threshold_ms: float = 10.0,
    low_conflict_safe_abort_rate: float = 0.50,
    low_conflict_occ_guard: bool = True,
    sparse_state_risk_prior: bool = True,
    commit_value: float = 100.0,
    abort_penalty: float = 80.0,
    reasoning_weight: float = 1.0,
    lock_wait_weight: float = 0.5,
    latency_weight: float = 0.1,
    lock_hold_weight: float = 0.05,
    background_abort_weight: float = 2.0,
    background_tps_loss_weight: float = 0.1,
    ucb_c: float = 1.5,
    reasoning_profile: str = "agentic",
    reasoning_scale: float = 1.0,
    actions: str | Sequence[str] | None = "auto",
    budget_seconds: float | None = None,
) -> Dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if tasks < 0:
        raise ValueError("task count must be non-negative")

    benchmark_name = str(benchmark).strip().lower()
    trainable_actions = resolve_trainable_actions(benchmark_name, actions)
    if budget_seconds is not None:
        if float(budget_seconds) <= 0:
            raise ValueError("budget seconds must be positive")
        episode_width = float(duration_s) if benchmark_name == "mixed" else 1.0
        episodes = max(1, int(float(budget_seconds) // max(0.001, episode_width)))

    policy = make_policy(
        abort_threshold=float(abort_threshold),
        min_visits=int(min_visits),
        protect_cost_threshold_ms=float(protect_cost_threshold_ms),
        low_conflict_occ_guard=bool(low_conflict_occ_guard),
        low_conflict_safe_abort_rate=float(low_conflict_safe_abort_rate),
        sparse_state_risk_prior=bool(sparse_state_risk_prior),
        commit_value=float(commit_value),
        abort_penalty=float(abort_penalty),
        reasoning_weight=float(reasoning_weight),
        lock_wait_weight=float(lock_wait_weight),
        latency_weight=float(latency_weight),
        lock_hold_weight=float(lock_hold_weight),
        background_abort_weight=float(background_abort_weight),
        background_tps_loss_weight=float(background_tps_loss_weight),
        trainable_actions=trainable_actions,
        exploration_coefficient=float(ucb_c),
    )
    episode_rows = []
    started_at = time.perf_counter()
    for episode in range(int(episodes)):
        if benchmark_name == "mixed":
            report = run_mixed_benchmark(
                MixedBenchmarkConfig(
                    workload=workload,
                    level=level,
                    workload_profile=workload_profile,
                    cc=ATCC_STRATEGY,
                    duration_s=float(duration_s),
                    agent_workers=int(agents),
                    background_workers=int(background),
                    clients=int(clients),
                    agent_ratio=float(agent_ratio),
                    background_mode=background_mode,
                    retries=int(retries),
                    retry_until_commit=bool(retry_until_commit),
                    max_attempts_per_task=int(max_attempts_per_task),
                    agent_retry_backoff_min_ms=int(agent_retry_backoff_min_ms),
                    agent_retry_backoff_max_ms=int(agent_retry_backoff_max_ms),
                    background_retry_backoff_min_ms=int(background_retry_backoff_min_ms),
                    background_retry_backoff_max_ms=int(background_retry_backoff_max_ms),
                    tokens_per_operation=int(tokens_per_operation),
                    seed=int(seed) + episode,
                    reasoning_profile=reasoning_profile,
                    reasoning_scale=reasoning_scale,
                    policy_mode="train",
                    policy=policy,
                )
            )
        else:
            report = run_cc_benchmark(
                BenchmarkConfig(
                    workload=workload,
                    level=level,
                    workload_profile=workload_profile,
                    cc=ATCC_STRATEGY,
                    tasks=tasks,
                    workers=workers,
                    retries=0,
                    seed=int(seed) + episode,
                    reasoning_profile=reasoning_profile,
                    reasoning_scale=reasoning_scale,
                    policy_mode="train",
                    atcc_policy=policy,
                )
            )
        row = report["cc_results"][0]
        episode_rows.append(training_episode_row(episode, row, policy_states=len(policy.rows)))

    elapsed_s = time.perf_counter() - started_at
    effective_agents, effective_background = effective_client_mix(
        agents=agents,
        background=background,
        clients=clients,
        agent_ratio=agent_ratio,
    )
    return {
        "mode": "train-atcc",
        "benchmark": benchmark_name,
        "strategy": ATCC_STRATEGY,
        "workload": workload,
        "level": level,
        "workload_profile": workload_profile,
        "episodes": int(episodes),
        "tasks": int(tasks),
        "workers": int(workers),
        "duration_s": float(duration_s),
        "clients": int(clients),
        "agent_ratio": float(agent_ratio),
        "agents": int(effective_agents),
        "background": int(effective_background),
        "background_mode": background_mode,
        "retries": int(retries),
        "retry_until_commit": bool(retry_until_commit),
        "max_attempts_per_task": int(max_attempts_per_task),
        "agent_retry_backoff_ms": [
            int(agent_retry_backoff_min_ms),
            int(agent_retry_backoff_max_ms),
        ],
        "background_retry_backoff_ms": [
            int(background_retry_backoff_min_ms),
            int(background_retry_backoff_max_ms),
        ],
        "tokens_per_operation": int(tokens_per_operation),
        "seed": int(seed),
        "abort_threshold": float(abort_threshold),
        "min_visits": int(min_visits),
        "protect_cost_threshold_ms": float(protect_cost_threshold_ms),
        "low_conflict_occ_guard": bool(low_conflict_occ_guard),
        "low_conflict_safe_abort_rate": float(low_conflict_safe_abort_rate),
        "sparse_state_risk_prior": bool(sparse_state_risk_prior),
        "reward_config": policy.reward_config.to_dict(),
        "ucb_c": float(ucb_c),
        "reasoning_profile": reasoning_profile,
        "reasoning_scale": float(reasoning_scale),
        "actions": list(trainable_actions),
        "budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "elapsed_s": elapsed_s,
        "policy_states": len(policy.rows),
        "episodes_detail": episode_rows,
        "policy": policy.to_dict(),
    }


def train_policy_matrix(
    *,
    benchmark: str,
    workloads: Sequence[str],
    levels: Sequence[str],
    workload_profile: str,
    episodes: int,
    tasks: int,
    workers: int,
    duration_s: float,
    agents: int,
    background: int,
    clients: int = 0,
    agent_ratio: float = 0.80,
    background_mode: str = "hotspot",
    retries: int = 0,
    retry_until_commit: bool = False,
    max_attempts_per_task: int = 100,
    agent_retry_backoff_min_ms: int = 500,
    agent_retry_backoff_max_ms: int = 5000,
    background_retry_backoff_min_ms: int = 10,
    background_retry_backoff_max_ms: int = 30,
    tokens_per_operation: int = 2703,
    seed: int = 920104,
    abort_threshold: float = 0.20,
    min_visits: int = 5,
    protect_cost_threshold_ms: float = 10.0,
    low_conflict_safe_abort_rate: float = 0.50,
    low_conflict_occ_guard: bool = True,
    sparse_state_risk_prior: bool = True,
    commit_value: float = 100.0,
    abort_penalty: float = 80.0,
    reasoning_weight: float = 1.0,
    lock_wait_weight: float = 0.5,
    latency_weight: float = 0.1,
    lock_hold_weight: float = 0.05,
    background_abort_weight: float = 2.0,
    background_tps_loss_weight: float = 0.1,
    ucb_c: float = 1.5,
    reasoning_profile: str = "agentic",
    reasoning_scale: float = 1.0,
    actions: str | Sequence[str] | None = "auto",
    budget_seconds: float | None = None,
) -> Dict[str, Any]:
    workload_names = expand_training_values(workloads, allowed=("ycsb", "tpcc"), all_values=("ycsb", "tpcc"), field="workloads")
    level_names = expand_training_values(levels, allowed=("low", "medium", "high"), all_values=("low", "medium", "high"), field="levels")
    combos = [(workload, level) for workload in workload_names for level in level_names]
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not combos:
        raise ValueError("training matrix must not be empty")

    benchmark_name = str(benchmark).strip().lower()
    trainable_actions = resolve_trainable_actions(benchmark_name, actions)
    if budget_seconds is not None:
        if float(budget_seconds) <= 0:
            raise ValueError("budget seconds must be positive")
        episode_width = float(duration_s) if benchmark_name == "mixed" else 1.0
        episodes = max(1, int(float(budget_seconds) // max(0.001, episode_width * len(combos))))

    policy = make_policy(
        abort_threshold=float(abort_threshold),
        min_visits=int(min_visits),
        protect_cost_threshold_ms=float(protect_cost_threshold_ms),
        low_conflict_occ_guard=bool(low_conflict_occ_guard),
        low_conflict_safe_abort_rate=float(low_conflict_safe_abort_rate),
        sparse_state_risk_prior=bool(sparse_state_risk_prior),
        commit_value=float(commit_value),
        abort_penalty=float(abort_penalty),
        reasoning_weight=float(reasoning_weight),
        lock_wait_weight=float(lock_wait_weight),
        latency_weight=float(latency_weight),
        lock_hold_weight=float(lock_hold_weight),
        background_abort_weight=float(background_abort_weight),
        background_tps_loss_weight=float(background_tps_loss_weight),
        trainable_actions=trainable_actions,
        exploration_coefficient=float(ucb_c),
    )
    episode_rows = []
    started_at = time.perf_counter()
    run_index = 0
    for episode in range(int(episodes)):
        for workload, level in combos:
            report = run_training_episode(
                benchmark_name=benchmark_name,
                strategy=ATCC_STRATEGY,
                workload=workload,
                level=level,
                workload_profile=workload_profile,
                episode=run_index,
                tasks=tasks,
                workers=workers,
                duration_s=duration_s,
                agents=agents,
                background=background,
                clients=clients,
                agent_ratio=agent_ratio,
                background_mode=background_mode,
                retries=retries,
                retry_until_commit=retry_until_commit,
                max_attempts_per_task=max_attempts_per_task,
                agent_retry_backoff_min_ms=agent_retry_backoff_min_ms,
                agent_retry_backoff_max_ms=agent_retry_backoff_max_ms,
                background_retry_backoff_min_ms=background_retry_backoff_min_ms,
                background_retry_backoff_max_ms=background_retry_backoff_max_ms,
                tokens_per_operation=tokens_per_operation,
                seed=seed,
                reasoning_profile=reasoning_profile,
                reasoning_scale=reasoning_scale,
                policy=policy,
            )
            row = report["cc_results"][0]
            detail = training_episode_row(run_index, row, policy_states=len(policy.rows))
            detail["workload"] = workload
            detail["level"] = level
            detail["matrix_round"] = episode
            episode_rows.append(detail)
            run_index += 1

    elapsed_s = time.perf_counter() - started_at
    effective_agents, effective_background = effective_client_mix(
        agents=agents,
        background=background,
        clients=clients,
        agent_ratio=agent_ratio,
    )
    return {
        "mode": "train-atcc",
        "benchmark": benchmark_name,
        "strategy": ATCC_STRATEGY,
        "training_scope": "matrix",
        "workloads": list(workload_names),
        "levels": list(level_names),
        "workload_profile": workload_profile,
        "episodes": int(episodes),
        "runs": int(run_index),
        "tasks": int(tasks),
        "workers": int(workers),
        "duration_s": float(duration_s),
        "clients": int(clients),
        "agent_ratio": float(agent_ratio),
        "agents": int(effective_agents),
        "background": int(effective_background),
        "background_mode": background_mode,
        "retries": int(retries),
        "retry_until_commit": bool(retry_until_commit),
        "max_attempts_per_task": int(max_attempts_per_task),
        "agent_retry_backoff_ms": [
            int(agent_retry_backoff_min_ms),
            int(agent_retry_backoff_max_ms),
        ],
        "background_retry_backoff_ms": [
            int(background_retry_backoff_min_ms),
            int(background_retry_backoff_max_ms),
        ],
        "tokens_per_operation": int(tokens_per_operation),
        "seed": int(seed),
        "abort_threshold": float(abort_threshold),
        "min_visits": int(min_visits),
        "protect_cost_threshold_ms": float(protect_cost_threshold_ms),
        "low_conflict_occ_guard": bool(low_conflict_occ_guard),
        "low_conflict_safe_abort_rate": float(low_conflict_safe_abort_rate),
        "sparse_state_risk_prior": bool(sparse_state_risk_prior),
        "reward_config": policy.reward_config.to_dict(),
        "ucb_c": float(ucb_c),
        "reasoning_profile": reasoning_profile,
        "reasoning_scale": float(reasoning_scale),
        "actions": list(trainable_actions),
        "budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "elapsed_s": elapsed_s,
        "policy_states": len(policy.rows),
        "episodes_detail": episode_rows,
        "policy": policy.to_dict(),
    }


def make_policy(
    *,
    abort_threshold: float,
    min_visits: int,
    protect_cost_threshold_ms: float,
    low_conflict_occ_guard: bool,
    low_conflict_safe_abort_rate: float,
    sparse_state_risk_prior: bool,
    commit_value: float,
    abort_penalty: float,
    reasoning_weight: float,
    lock_wait_weight: float,
    latency_weight: float,
    lock_hold_weight: float,
    background_abort_weight: float,
    background_tps_loss_weight: float,
    trainable_actions: Sequence[str],
    exploration_coefficient: float,
) -> ATCCPolicyTable:
    return ATCCPolicyTable(
        abort_threshold=float(abort_threshold),
        min_visits=int(min_visits),
        protect_cost_threshold_ms=float(protect_cost_threshold_ms),
        low_conflict_occ_guard=bool(low_conflict_occ_guard),
        low_conflict_safe_abort_rate=float(low_conflict_safe_abort_rate),
        sparse_state_risk_prior=bool(sparse_state_risk_prior),
        reward_config=ATCCRewardConfig(
            commit_value=float(commit_value),
            abort_penalty=float(abort_penalty),
            reasoning_weight=float(reasoning_weight),
            lock_wait_weight=float(lock_wait_weight),
            latency_weight=float(latency_weight),
            lock_hold_weight=float(lock_hold_weight),
            background_abort_weight=float(background_abort_weight),
            background_tps_loss_weight=float(background_tps_loss_weight),
        ),
        trainable_actions=tuple(trainable_actions),
        exploration_coefficient=float(exploration_coefficient),
    )


def run_training_episode(
    *,
    benchmark_name: str,
    strategy: str,
    workload: str,
    level: str,
    workload_profile: str,
    episode: int,
    tasks: int,
    workers: int,
    duration_s: float,
    agents: int,
    background: int,
    clients: int,
    agent_ratio: float,
    background_mode: str,
    retries: int,
    retry_until_commit: bool,
    max_attempts_per_task: int,
    agent_retry_backoff_min_ms: int,
    agent_retry_backoff_max_ms: int,
    background_retry_backoff_min_ms: int,
    background_retry_backoff_max_ms: int,
    tokens_per_operation: int,
    seed: int,
    reasoning_profile: str,
    reasoning_scale: float,
    policy: ATCCPolicyTable,
) -> Dict[str, Any]:
    if benchmark_name == "mixed":
        return run_mixed_benchmark(
            MixedBenchmarkConfig(
                workload=workload,
                level=level,
                workload_profile=workload_profile,
                cc=strategy,
                duration_s=float(duration_s),
                agent_workers=int(agents),
                background_workers=int(background),
                clients=int(clients),
                agent_ratio=float(agent_ratio),
                background_mode=background_mode,
                retries=int(retries),
                retry_until_commit=bool(retry_until_commit),
                max_attempts_per_task=int(max_attempts_per_task),
                agent_retry_backoff_min_ms=int(agent_retry_backoff_min_ms),
                agent_retry_backoff_max_ms=int(agent_retry_backoff_max_ms),
                background_retry_backoff_min_ms=int(background_retry_backoff_min_ms),
                background_retry_backoff_max_ms=int(background_retry_backoff_max_ms),
                tokens_per_operation=int(tokens_per_operation),
                seed=int(seed) + int(episode),
                reasoning_profile=reasoning_profile,
                reasoning_scale=reasoning_scale,
                policy_mode="train",
                policy=policy,
            )
        )
    return run_cc_benchmark(
        BenchmarkConfig(
            workload=workload,
            level=level,
            workload_profile=workload_profile,
            cc=strategy,
            tasks=tasks,
            workers=workers,
            retries=0,
            seed=int(seed) + int(episode),
            reasoning_profile=reasoning_profile,
            reasoning_scale=reasoning_scale,
            policy_mode="train",
            atcc_policy=policy,
        )
    )


def resolve_trainable_actions(
    benchmark: str,
    actions: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if actions is None:
        return all_actions(None)
    if isinstance(actions, str) and actions.strip().lower() == "auto":
        if str(benchmark).strip().lower() == "mixed":
            return MIXED_TRAINABLE_ACTIONS
        return all_actions(None)
    return all_actions(actions)


def expand_training_values(
    values: Sequence[str],
    *,
    allowed: Sequence[str],
    all_values: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    raw = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not raw:
        raise ValueError(f"{field} must not be empty")
    if raw == ("all",):
        return tuple(all_values)
    unknown = [value for value in raw if value not in set(allowed)]
    if unknown:
        raise ValueError(f"unsupported {field}: {','.join(unknown)}")
    return tuple(dict.fromkeys(raw))


def training_episode_row(episode: int, row: Dict[str, Any], *, policy_states: int) -> Dict[str, Any]:
    if "agent_tps" in row:
        return {
            "episode": episode,
            "agent_attempts": row["agent_attempts"],
            "agent_commits": row["agent_commits"],
            "agent_aborts": row["agent_aborts"],
            "agent_commit_rate": row["agent_commit_rate"],
            "agent_tps": row["agent_tps"],
            "agent_task_tps": row.get("agent_task_tps", row["agent_tps"]),
            "agent_completed_tasks": row.get("agent_completed_tasks", row["agent_commits"]),
            "agent_failed_tasks": row.get("agent_failed_tasks", 0),
            "agent_task_completion_rate": row.get("agent_task_completion_rate", row["agent_commit_rate"]),
            "agent_abort_rate": row.get("agent_abort_rate", 0.0),
            "agent_avg_retry_count": row.get("agent_avg_retry_count", 0.0),
            "agent_p50_latency_ms": row.get("agent_p50_latency_ms", 0.0),
            "agent_p95_latency_ms": row.get("agent_p95_latency_ms", 0.0),
            "agent_p99_latency_ms": row.get("agent_p99_latency_ms", 0.0),
            "agent_p9999_latency_ms": row.get("agent_p9999_latency_ms", 0.0),
            "agent_avg_latency_ms": row.get("agent_avg_latency_ms", 0.0),
            "agent_avg_operations": row.get("agent_avg_operations", 0.0),
            "agent_avg_tokens": row.get("agent_avg_tokens", 0.0),
            "agent_total_tokens": row.get("agent_total_tokens", 0.0),
            "background_tps": row["background_tps"],
            "background_retries": row.get("background_retries", 0),
            "total_tps": row["total_tps"],
            "wasted_reasoning_ms": row["wasted_reasoning_ms"],
            "action_counts": row["action_counts"],
            "policy_states": policy_states,
        }
    return {
        "episode": episode,
        "tasks": row["tasks"],
        "attempts": row["attempts"],
        "committed_tasks": row["committed_tasks"],
        "abort_count": row["abort_count"],
        "commit_rate": row["task_commit_rate"],
        "throughput": row["throughput"],
        "wasted_reasoning_ms": row["wasted_reasoning_ms"],
        "action_counts": row.get("action_counts", {}),
        "policy_states": policy_states,
    }


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_range(value: str, *, field: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{field} must be min,max")
    low, high = (int(parts[0]), int(parts[1]))
    if low < 0 or high < 0 or low > high:
        raise ValueError(f"{field} must be non-negative min<=max")
    return low, high


def effective_client_mix(
    *,
    agents: int,
    background: int,
    clients: int,
    agent_ratio: float,
) -> tuple[int, int]:
    if int(clients) <= 0:
        return int(agents), int(background)
    agent_workers = max(1, int(round(int(clients) * float(agent_ratio))))
    background_workers = max(1, int(clients) - agent_workers)
    return agent_workers, background_workers


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
) -> int:
    args = build_parser().parse_args(argv)
    agent_backoff = parse_range(args.agent_retry_backoff_ms, field="agent-retry-backoff-ms")
    background_backoff = parse_range(args.background_retry_backoff_ms, field="background-retry-backoff-ms")
    common = {
        "benchmark": args.benchmark,
        "workload_profile": args.workload_profile,
        "episodes": args.episodes,
        "tasks": args.tasks,
        "workers": args.workers,
        "duration_s": args.duration,
        "clients": args.clients,
        "agent_ratio": args.agent_ratio,
        "agents": args.agents,
        "background": args.background,
        "background_mode": args.background_mode,
        "retries": args.retries,
        "retry_until_commit": args.retry_until_commit,
        "max_attempts_per_task": args.max_attempts_per_task,
        "agent_retry_backoff_min_ms": agent_backoff[0],
        "agent_retry_backoff_max_ms": agent_backoff[1],
        "background_retry_backoff_min_ms": background_backoff[0],
        "background_retry_backoff_max_ms": background_backoff[1],
        "tokens_per_operation": args.tokens_per_operation,
        "seed": args.seed,
        "abort_threshold": args.abort_threshold,
        "min_visits": args.min_visits,
        "protect_cost_threshold_ms": args.protect_cost_threshold_ms,
        "low_conflict_safe_abort_rate": args.low_conflict_safe_abort_rate,
        "low_conflict_occ_guard": not args.disable_low_conflict_occ_guard,
        "sparse_state_risk_prior": not args.disable_sparse_state_risk_prior,
        "commit_value": args.commit_value,
        "abort_penalty": args.abort_penalty,
        "reasoning_weight": args.reasoning_weight,
        "lock_wait_weight": args.lock_wait_weight,
        "latency_weight": args.latency_weight,
        "lock_hold_weight": args.lock_hold_weight,
        "background_abort_weight": args.background_abort_weight,
        "background_tps_loss_weight": args.background_tps_loss_weight,
        "ucb_c": args.ucb_c,
        "reasoning_profile": args.reasoning_profile,
        "reasoning_scale": args.reasoning_scale,
        "actions": args.actions,
        "budget_seconds": args.budget_seconds,
    }
    if args.workloads or args.levels:
        report = train_policy_matrix(
            workloads=split_csv(args.workloads or args.workload),
            levels=split_csv(args.levels or args.level),
            **common,
        )
    else:
        report = train_policy(
            workload=args.workload,
            level=args.level,
            **common,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report["policy"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["output"] = str(args.output)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    out.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
