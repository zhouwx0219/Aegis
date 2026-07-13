"""Multi-seed benchmark matrix aggregation."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from agent.benchmarks.mixed import MixedBenchmarkConfig, run_mixed_benchmark


@dataclasses.dataclass(frozen=True)
class MixedMatrixConfig:
    workloads: tuple[str, ...] = ("ycsb", "tpcc")
    levels: tuple[str, ...] = ("low", "medium", "high")
    seeds: tuple[int, ...] = (920104, 920105, 920106)
    client_counts: tuple[int, ...] = ()
    workload_profile: str = "small"
    ycsb_zipf_theta: float | None = None
    cc: str = "occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,bamboo,polaris,dynamic-atcc"
    duration_s: float = 3.0
    agent_workers: int = 2
    background_workers: int = 8
    clients: int = 0
    agent_ratio: float = 0.80
    reasoning_profile: str = "agentic"
    reasoning_scale: float = 2.0
    retries: int = 0
    retry_until_commit: bool = False
    max_attempts_per_task: int = 100
    agent_retry_backoff_min_ms: int = 500
    agent_retry_backoff_max_ms: int = 5000
    background_retry_backoff_min_ms: int = 10
    background_retry_backoff_max_ms: int = 30
    tokens_per_operation: int = 2703
    background_wait: bool = False
    background_mode: str = "hotspot"
    reservation_ttl_s: float = 5.0
    policy: Path | None = None
    policy_mode: str = "eval"
    atcc_hot_rw_k: int = 3
    atcc_bp_background_threshold: int = 6
    atcc_bp_queue_pressure_threshold: int = 2
    atcc_bp_min_windows: int = 3
    atcc_agent_guardrail: bool = False
    atcc_agent_guardrail_queue_threshold: int = 1
    atcc_full_reservation_fallback_ratio: float = 0.0
    atcc_pure_policy: bool = False
    background_admission_cap: int = 0

    def normalized(self) -> "MixedMatrixConfig":
        workloads = normalize_names(self.workloads, allowed={"ycsb", "tpcc"}, field="workloads")
        levels = normalize_names(self.levels, allowed={"low", "medium", "high"}, field="levels")
        seeds = tuple(int(seed) for seed in self.seeds)
        client_counts = tuple(dict.fromkeys(int(value) for value in self.client_counts))
        if not seeds:
            raise ValueError("at least one seed is required")
        if any(value < 2 for value in client_counts):
            raise ValueError("client counts must be at least 2")
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
        background_mode = str(self.background_mode).strip().lower()
        if background_mode not in {"hotspot", "procedure"}:
            raise ValueError(f"unsupported background mode: {self.background_mode}")
        policy_mode = str(self.policy_mode).strip().lower() or "eval"
        if policy_mode not in {"train", "eval", "online"}:
            raise ValueError(f"unsupported policy mode: {self.policy_mode}")
        return dataclasses.replace(
            self,
            workloads=workloads,
            levels=levels,
            seeds=seeds,
            client_counts=client_counts,
            workload_profile=str(self.workload_profile).strip().lower() or "small",
            ycsb_zipf_theta=self.ycsb_zipf_theta,
            cc=str(self.cc).strip() or "occ",
            agent_workers=agent_workers,
            background_workers=background_workers,
            clients=clients,
            agent_ratio=agent_ratio,
            reasoning_profile=str(self.reasoning_profile).strip().lower() or "agentic",
            background_mode=background_mode,
            policy_mode=policy_mode,
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


def run_mixed_matrix(config: MixedMatrixConfig) -> Dict[str, Any]:
    config = config.normalized()
    runs = []
    client_counts = effective_client_counts(config)
    worker_mix = client_worker_mix_rows(config, client_counts)
    top_level_agent_workers = worker_mix[0]["agent_workers"] if len(worker_mix) == 1 else int(config.agent_workers)
    top_level_background_workers = worker_mix[0]["background_workers"] if len(worker_mix) == 1 else int(config.background_workers)
    for workload in config.workloads:
        for level in config.levels:
            for client_count in client_counts:
                for seed in config.seeds:
                    report = run_mixed_benchmark(
                        MixedBenchmarkConfig(
                            workload=workload,
                            level=level,
                            workload_profile=config.workload_profile,
                            ycsb_zipf_theta=config.ycsb_zipf_theta,
                            cc=config.cc,
                            duration_s=config.duration_s,
                            agent_workers=config.agent_workers,
                            background_workers=config.background_workers,
                            clients=client_count,
                            agent_ratio=config.agent_ratio,
                            reasoning_profile=config.reasoning_profile,
                            reasoning_scale=config.reasoning_scale,
                            seed=seed,
                            retries=config.retries,
                            retry_until_commit=config.retry_until_commit,
                            max_attempts_per_task=config.max_attempts_per_task,
                            agent_retry_backoff_min_ms=config.agent_retry_backoff_min_ms,
                            agent_retry_backoff_max_ms=config.agent_retry_backoff_max_ms,
                            background_retry_backoff_min_ms=config.background_retry_backoff_min_ms,
                            background_retry_backoff_max_ms=config.background_retry_backoff_max_ms,
                            tokens_per_operation=config.tokens_per_operation,
                            background_wait=config.background_wait,
                            background_mode=config.background_mode,
                            reservation_ttl_s=config.reservation_ttl_s,
                            policy=config.policy,
                            policy_mode=config.policy_mode,
                            atcc_hot_rw_k=config.atcc_hot_rw_k,
                            atcc_bp_background_threshold=config.atcc_bp_background_threshold,
                            atcc_bp_queue_pressure_threshold=config.atcc_bp_queue_pressure_threshold,
                            atcc_bp_min_windows=config.atcc_bp_min_windows,
                            atcc_agent_guardrail=config.atcc_agent_guardrail,
                            atcc_agent_guardrail_queue_threshold=config.atcc_agent_guardrail_queue_threshold,
                            atcc_full_reservation_fallback_ratio=config.atcc_full_reservation_fallback_ratio,
                            atcc_pure_policy=config.atcc_pure_policy,
                            background_admission_cap=config.background_admission_cap,
                        )
                    )
                    for row in report["cc_results"]:
                        run = dict(row)
                        run["workload"] = workload
                        run["level"] = level
                        run["seed"] = int(seed)
                        run["clients"] = int(report["clients"])
                        run["agent_ratio"] = float(report["agent_ratio"])
                        run["agent_workers"] = int(report["agent_workers"])
                        run["background_workers"] = int(report["background_workers"])
                        runs.append(run)
    summary = summarize_runs(runs)
    return {
        "mode": "mixed-starvation-matrix",
        "workloads": list(config.workloads),
        "levels": list(config.levels),
        "seeds": list(config.seeds),
        "client_counts": list(client_counts),
        "client_worker_mix": worker_mix,
        "workload_profile": config.workload_profile,
        "ycsb_zipf_theta": config.ycsb_zipf_theta,
        "cc": config.cc,
        "duration_s": float(config.duration_s),
        "clients": int(config.clients),
        "agent_ratio": float(config.agent_ratio),
        "agent_workers": int(top_level_agent_workers),
        "background_workers": int(top_level_background_workers),
        "reasoning_profile": config.reasoning_profile,
        "reasoning_scale": float(config.reasoning_scale),
        "retries": int(config.retries),
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
        "background_mode": config.background_mode,
        "policy_mode": config.policy_mode,
        "policy": str(config.policy) if config.policy else "",
        "atcc_hot_rw_k": int(config.atcc_hot_rw_k),
        "atcc_bp_background_threshold": int(config.atcc_bp_background_threshold),
        "atcc_bp_queue_pressure_threshold": int(config.atcc_bp_queue_pressure_threshold),
        "atcc_bp_min_windows": int(config.atcc_bp_min_windows),
        "atcc_agent_guardrail": bool(config.atcc_agent_guardrail),
        "atcc_agent_guardrail_queue_threshold": int(config.atcc_agent_guardrail_queue_threshold),
        "atcc_full_reservation_fallback_ratio": float(config.atcc_full_reservation_fallback_ratio),
        "atcc_pure_policy": bool(config.atcc_pure_policy),
        "background_admission_cap": int(config.background_admission_cap),
        "paper_figures": paper_figure_rows(summary),
        "summary": summary,
        "runs": runs,
    }


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int, str], list[Dict[str, Any]]] = {}
    for row in runs:
        key = (str(row["workload"]), str(row["level"]), int(row.get("clients", 0) or 0), str(row["cc"]))
        grouped.setdefault(key, []).append(row)

    baselines: Dict[tuple[str, str, int], Dict[str, float]] = {}
    for (workload, level, clients, cc), rows in grouped.items():
        if cc == "occ":
            baselines[(workload, level, clients)] = {
                "agent_tps": average(row_float(rows, "agent_tps")),
                "total_tps": average(row_float(rows, "total_tps")),
                "agent_commit_rate": average(row_float(rows, "agent_commit_rate")),
                "background_tps": average(row_float(rows, "background_tps")),
            }

    summary = []
    for (workload, level, clients, cc), rows in sorted(grouped.items()):
        baseline = baselines.get((workload, level, clients), {})
        agent_tps = row_float(rows, "agent_tps")
        total_tps = row_float(rows, "total_tps")
        commit_rates = row_float(rows, "agent_commit_rate")
        completion_rates = row_float(rows, "agent_task_completion_rate")
        background_tps = row_float(rows, "background_tps")
        row = {
            "workload": workload,
            "level": level,
            "clients": clients,
            "agent_workers": int(rows[0].get("agent_workers", 0) or 0),
            "background_workers": int(rows[0].get("background_workers", 0) or 0),
            "cc": cc,
            "runs": len(rows),
            "atcc_hot_rw_k": int(rows[0].get("atcc_hot_rw_k", 0) or 0),
            "atcc_bp_background_threshold": int(rows[0].get("atcc_bp_background_threshold", 0) or 0),
            "atcc_bp_queue_pressure_threshold": int(rows[0].get("atcc_bp_queue_pressure_threshold", 0) or 0),
            "atcc_bp_min_windows": int(rows[0].get("atcc_bp_min_windows", 0) or 0),
            "atcc_agent_guardrail": bool(rows[0].get("atcc_agent_guardrail", False)),
            "atcc_agent_guardrail_queue_threshold": int(
                rows[0].get("atcc_agent_guardrail_queue_threshold", 0) or 0
            ),
            "atcc_full_reservation_fallback_ratio": float(
                rows[0].get("atcc_full_reservation_fallback_ratio", 0.0) or 0.0
            ),
            "atcc_pure_policy": bool(rows[0].get("atcc_pure_policy", False)),
            "background_admission_cap": int(rows[0].get("background_admission_cap", 0) or 0),
            "agent_tps_mean": average(agent_tps),
            "agent_tps_std": stddev(agent_tps),
            "total_tps_mean": average(total_tps),
            "total_tps_std": stddev(total_tps),
            "bottom_txn_attempt_tps_mean": average(row_float(rows, "bottom_txn_attempt_tps")),
            "bottom_txn_commit_tps_mean": average(row_float(rows, "bottom_txn_commit_tps")),
            "underlying_txn_attempt_tps_mean": average(row_float(rows, "underlying_txn_attempt_tps")),
            "underlying_txn_commit_tps_mean": average(row_float(rows, "underlying_txn_commit_tps")),
            "native_throughput_mean": average(row_float(rows, "native_throughput")),
            "background_tps_mean": average(background_tps),
            "background_tps_std": stddev(background_tps),
            "agent_commit_rate_mean": average(commit_rates),
            "agent_commit_rate_std": stddev(commit_rates),
            "agent_task_tps_mean": average(row_float(rows, "agent_task_tps")),
            "agent_task_completion_rate_mean": average(completion_rates),
            "agent_abort_rate_mean": average(row_float(rows, "agent_abort_rate")),
            "agent_attempt_abort_rate_mean": average(row_float(rows, "agent_attempt_abort_rate")),
            "agent_avg_retry_count_mean": average(row_float(rows, "agent_avg_retry_count")),
            "agent_p50_latency_ms_mean": average(row_float(rows, "agent_p50_latency_ms")),
            "agent_p95_latency_ms_mean": average(row_float(rows, "agent_p95_latency_ms")),
            "agent_p99_latency_ms_mean": average(row_float(rows, "agent_p99_latency_ms")),
            "agent_p9999_latency_ms_mean": average(row_float(rows, "agent_p9999_latency_ms")),
            "agent_avg_tokens_mean": average(row_float(rows, "agent_avg_tokens")),
            "agent_total_tokens_mean": average(row_float(rows, "agent_total_tokens")),
            "agent_aborts_mean": average(row_float(rows, "agent_aborts")),
            "background_aborts_mean": average(row_float(rows, "background_aborts")),
            "wasted_reasoning_ms_mean": average(row_float(rows, "wasted_reasoning_ms")),
        }
        for metric in diagnostic_metric_keys():
            row[f"{metric}_mean"] = average(row_float(rows, metric))
        row["agent_tps_speedup_vs_occ"] = safe_ratio(row["agent_tps_mean"], baseline.get("agent_tps", 0.0))
        row["total_tps_speedup_vs_occ"] = safe_ratio(row["total_tps_mean"], baseline.get("total_tps", 0.0))
        row["background_tps_ratio_vs_occ"] = safe_ratio(row["background_tps_mean"], baseline.get("background_tps", 0.0))
        row["agent_commit_rate_ratio_vs_occ"] = safe_ratio(
            row["agent_commit_rate_mean"],
            baseline.get("agent_commit_rate", 0.0),
        )
        row["agent_commit_rate_delta_vs_occ"] = (
            row["agent_commit_rate_mean"] - baseline["agent_commit_rate"]
            if "agent_commit_rate" in baseline
            else None
        )
        summary.append(row)
    return summary


def diagnostic_metric_keys() -> tuple[str, ...]:
    return (
        "agent_task_guard_wait_ms_mean",
        "agent_task_guard_wait_ms_p50",
        "agent_task_guard_wait_ms_p95",
        "agent_task_guard_wait_ms_p99",
        "agent_task_guard_wait_ms_max",
        "reservation_waiter_count",
        "reservation_unique_target_count",
        "reservation_waiter_target_set_size_mean",
        "reservation_waiter_target_set_size_p50",
        "reservation_waiter_target_set_size_p95",
        "reservation_waiter_target_set_size_p99",
        "reservation_waiter_target_set_size_max",
        "reservation_all_or_nothing_failed_grant_checks",
        "reservation_all_or_nothing_not_front_wait_ms",
        "reservation_front_queue_wait_ms",
        "reservation_owner_blocked_checks",
        "reservation_writer_blocked_checks",
        "reservation_blocked_target_checks",
        "background_writer_waiter_blocked_checks",
        "background_writer_waiter_blocked_targets",
        "background_writer_reservation_blocked_checks",
        "version_conflict_count",
        "guarded_conflict_checks",
        "conflict_pressure_count",
        "reserve_read_write_set_attempts",
        "reserve_read_write_set_target_size_mean",
        "reserve_read_write_set_target_size_p50",
        "reserve_read_write_set_target_size_p95",
        "reserve_read_write_set_target_size_p99",
        "reserve_read_write_set_target_size_max",
        "reserve_read_write_set_hot_target_count_mean",
        "reserve_read_write_set_hot_target_count_p50",
        "reserve_read_write_set_hot_target_count_p95",
        "reserve_read_write_set_hot_target_count_p99",
        "reserve_read_write_set_hot_target_count_max",
        "reserve_read_write_set_hot_coverage_ratio_mean",
        "reserve_read_write_set_unique_target_count",
        "reserve_read_write_set_unique_hot_target_count",
        "reserve_hot_rw_k_attempts",
        "reserve_hot_rw_k_target_size_mean",
        "reserve_hot_rw_k_target_size_p50",
        "reserve_hot_rw_k_target_size_p95",
        "reserve_hot_rw_k_target_size_p99",
        "reserve_hot_rw_k_target_size_max",
        "reserve_hot_rw_k_unique_target_count",
    )


def effective_client_counts(config: MixedMatrixConfig) -> tuple[int, ...]:
    if config.client_counts:
        return tuple(int(value) for value in config.client_counts)
    if int(config.clients) > 0:
        return (int(config.clients),)
    return (0,)


def client_worker_mix_rows(config: MixedMatrixConfig, client_counts: Sequence[int]) -> list[Dict[str, int]]:
    rows = []
    for clients in client_counts:
        if int(clients) > 0:
            agent_workers = max(1, int(round(int(clients) * float(config.agent_ratio))))
            background_workers = max(0, int(clients) - agent_workers)
        else:
            agent_workers = int(config.agent_workers)
            background_workers = int(config.background_workers)
        rows.append(
            {
                "clients": int(clients),
                "agent_workers": int(agent_workers),
                "background_workers": int(background_workers),
            }
        )
    return rows


def paper_figure_rows(summary: Sequence[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    figures: Dict[str, list[Dict[str, Any]]] = {
        "agent_throughput": [],
        "total_throughput": [],
        "avg_tokens": [],
        "p9999_latency_ms": [],
    }
    for row in summary:
        base = {
            "workload": row["workload"],
            "level": row["level"],
            "clients": row["clients"],
            "cc": row["cc"],
        }
        figures["agent_throughput"].append({**base, "value": row["agent_task_tps_mean"]})
        figures["total_throughput"].append({**base, "value": row["total_tps_mean"]})
        figures["avg_tokens"].append({**base, "value": row["agent_avg_tokens_mean"]})
        figures["p9999_latency_ms"].append({**base, "value": row["agent_p9999_latency_ms_mean"]})
    return figures


def normalize_names(values: Iterable[str], *, allowed: set[str], field: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(value).strip().lower() for value in values if str(value).strip()))
    if not names:
        raise ValueError(f"{field} must not be empty")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unsupported {field}: {','.join(unknown)}")
    return names


def row_float(rows: Sequence[Dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def average(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = average(values)
    return math.sqrt(sum((float(value) - avg) ** 2 for value in values) / (len(values) - 1))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)
