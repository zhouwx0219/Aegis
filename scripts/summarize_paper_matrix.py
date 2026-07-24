#!/usr/bin/env python3
"""Build one paper-ready aggregate CSV from a unified trace matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


GROUP_FIELDS = (
    "run_id",
    "experiment",
    "parameter",
    "parameter_value",
    "workload",
    "workload_variant",
    "level",
    "ycsb_zipf_theta",
    "tpcc_warehouses",
    "client_mix",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "agent_count",
    "worker_count",
    "cc_label",
    "cc_family",
    "source_system",
    "system",
    "cc",
    "paper_switching",
    "paper_priority",
    "paper_performance_guards",
    "paper_delayed_write_apply",
    "paper_policy_mode",
    "paper_policy_path",
    "atcc_retry_cache_enabled",
    "paper_deferred_replay_enabled",
    "max_attempts",
    "retry_budget",
    "priority_quantum_scale",
    "policy_invocation_ops",
    "throughput_metric",
)

METRICS = (
    "throughput",
    "total_tps",
    "agent_tps",
    "agent_task_tps",
    "background_tps",
    "bottom_txn_attempt_tps",
    "bottom_txn_commit_tps",
    "underlying_txn_attempt_tps",
    "underlying_txn_commit_tps",
    "native_throughput",
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
    "background_commit_rate",
    "agent_avg_tokens",
    "agent_total_tokens",
    "agent_committed_reasoning_tokens",
    "agent_wasted_reasoning_tokens",
    "agent_tokens_per_committed_txn",
    "agent_wasted_tokens_per_commit",
    "agent_wasted_token_ratio",
    "agent_initial_reasoning_invocations",
    "agent_retry_reasoning_invocations",
    "agent_cached_retry_replays",
    "agent_initial_reasoning_tokens",
    "agent_retry_reasoning_tokens",
    "agent_retry_cache_saved_tokens",
    "agent_counterfactual_no_cache_tokens",
    "agent_avg_tokens_without_retry_cache",
    "agent_retry_cache_savings_ratio",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_retries",
    "wasted_reasoning_ms",
    "wasted_reasoning_ms_per_commit",
    "read_conflicts",
    "write_conflicts",
    "version_conflict_count",
    "conflict_abort_count",
    "agent_admission_deferrals",
    "agent_admission_deferral_rate",
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
    "paper_background_fast_publishes",
    "paper_background_fast_publish_failures",
    "paper_background_native_batch_attempts",
    "paper_background_native_batch_commits",
    "paper_background_native_batch_validation_failures",
    "paper_background_native_batch_admission_fallbacks",
    "paper_background_native_batch_pin_fallbacks",
    "paper_background_publish_fallbacks",
    "paper_background_publish_fallback_active_reader",
    "paper_background_publish_fallback_active_writer",
    "paper_background_publish_fallback_version_mismatch",
    "paper_version_private_prepares",
    "paper_version_private_discards",
    "paper_version_atomic_publishes",
    "paper_version_native_publishes",
    "paper_version_background_version_change_events",
    "paper_retry_validation_conflicts",
    "paper_retry_mask_escalations",
    "paper_retry_full_observed_escalations",
    "paper_retry_inherited_attempts",
    "paper_retry_conflict_hot_read",
    "paper_retry_conflict_cold_read",
    "paper_retry_conflict_hot_write",
    "paper_retry_conflict_cold_write",
    "paper_retry_conflict_read_before_write",
    "paper_retry_conflict_blind_write",
    "paper_commit_timing_lock_ms_mean",
    "paper_commit_timing_validate_ms_mean",
    "paper_commit_timing_install_ms_mean",
    "paper_commit_timing_publish_ms_mean",
    "paper_commit_timing_gc_ms_mean",
    "elapsed_s",
)

PRIMARY_SEED_METRICS = (
    "throughput",
    "agent_tps",
    "total_tps",
    "background_tps",
    "agent_attempt_abort_rate",
    "agent_p99_latency_ms",
    "agent_total_tokens",
    "agent_wasted_reasoning_tokens",
    "agent_wasted_tokens_per_commit",
    "agent_wasted_token_ratio",
    "agent_initial_reasoning_tokens",
    "agent_retry_reasoning_tokens",
    "agent_retry_cache_saved_tokens",
    "agent_counterfactual_no_cache_tokens",
)

SUM_ACROSS_SEEDS = {
    "agent_total_tokens",
    "agent_wasted_reasoning_tokens",
    "agent_attempts",
    "agent_commits",
    "agent_aborts",
    "agent_completed_tasks",
    "agent_failed_tasks",
    "background_attempts",
    "background_commits",
    "background_aborts",
    "background_retries",
    "wasted_reasoning_ms",
    "read_conflicts",
    "write_conflicts",
    "version_conflict_count",
    "conflict_abort_count",
}

COMPARISON_FIELDS = (
    "best_baseline_cc",
    "best_baseline_throughput_mean",
    "throughput_speedup_vs_best",
    "throughput_gain_pct_vs_best",
    "best_agent_baseline_cc",
    "best_agent_baseline_agent_tps_mean",
    "agent_tps_speedup_vs_best",
    "agent_tps_gain_pct_vs_best",
    "best_total_baseline_cc",
    "best_total_baseline_total_tps_mean",
    "total_tps_speedup_vs_best",
    "total_tps_gain_pct_vs_best",
    "best_background_baseline_cc",
    "best_background_baseline_background_tps_mean",
    "background_tps_speedup_vs_best",
    "background_tps_gain_pct_vs_best",
    "abort_rate_delta_vs_best_agent_baseline",
    "p99_reduction_vs_best_agent_baseline",
    "best_agent_baseline_wasted_tokens_per_commit_mean",
    "wasted_tokens_reduction_factor_vs_best_agent_baseline",
    "best_agent_baseline_wasted_reasoning_ms_per_commit_mean",
    "wasted_reasoning_reduction_factor_vs_best_agent_baseline",
)

CURVE_FIELDS = (
    "previous_clients",
    "throughput_ratio_vs_previous_client",
    "throughput_change_pct_vs_previous_client",
    "throughput_curve_drop_gt_10pct",
    "agent_tps_ratio_vs_previous_client",
    "agent_tps_change_pct_vs_previous_client",
    "agent_curve_drop_gt_10pct",
    "total_tps_ratio_vs_previous_client",
    "background_tps_ratio_vs_previous_client",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=0)
    parser.add_argument("--expected-strategies", default="")
    args = parser.parse_args()

    raw_rows = read_csv(args.raw)
    if not raw_rows:
        raise SystemExit(f"no rows in {args.raw}")
    summaries = aggregate(raw_rows)
    add_baseline_comparisons(summaries)
    add_curve_diagnostics(summaries)
    validate_coverage(
        summaries,
        expected_seeds=max(0, int(args.expected_seeds)),
        expected_strategies={
            value.strip() for value in args.expected_strategies.split(",") if value.strip()
        },
    )
    summaries.sort(key=sort_key)
    fields = output_fields()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in summaries)
    print(f"output={args.output}")
    print(f"raw_rows={len(raw_rows)} summary_rows={len(summaries)} fields={len(fields)}")
    return 0


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in GROUP_FIELDS)].append(row)

    output: list[dict[str, object]] = []
    for group in grouped.values():
        first = group[0]
        ok_rows = [row for row in group if str(row.get("status", "")).lower() == "ok"]
        seeds = sorted({str(row.get("seed", "")) for row in ok_rows}, key=seed_key)
        summary: dict[str, object] = {field: first.get(field, "") for field in GROUP_FIELDS}
        summary["cc_label"] = system_label(first)
        summary["cc_family"] = (
            "atcc" if is_atcc(first) else "traditional"
        )
        summary.update(
            {
                "n_rows": len(group),
                "n_ok": len(ok_rows),
                "n_error": len(group) - len(ok_rows),
                "n_seeds": len(seeds),
                "seeds": ",".join(seeds),
                "status": "ok" if ok_rows and len(ok_rows) == len(group) else "partial",
                "errors": " | ".join(
                    str(row.get("error", "")) for row in group if row.get("error")
                ),
            }
        )
        for metric in METRICS:
            values = [
                value
                for row in ok_rows
                if (value := metric_value(row, metric)) is not None
            ]
            metric_stats(summary, metric, values)
            if metric in SUM_ACROSS_SEEDS:
                summary[f"{metric}_sum_across_seeds"] = fmt(sum(values)) if values else ""
        for metric in PRIMARY_SEED_METRICS:
            by_seed = {
                str(row.get("seed", "")): number(row.get(metric))
                for row in ok_rows
                if number(row.get(metric)) is not None
            }
            summary[f"{metric}_by_seed_json"] = json.dumps(by_seed, sort_keys=True)
        actions: Counter[int] = Counter()
        for row in ok_rows:
            try:
                raw_actions = json.loads(row.get("raw_action_counts", "{}") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            for label, count in raw_actions.items():
                if str(label).startswith("paper-action-"):
                    actions[int(str(label).rsplit("-", 1)[-1])] += int(count)
        summary["raw_action_counts_sum_json"] = json.dumps(
            {f"paper-action-{action}": count for action, count in sorted(actions.items())}
        )
        for action in range(16):
            summary[f"paper_action_{action}_count_sum"] = actions[action]
        output.append(summary)
    return output


def add_baseline_comparisons(rows: list[dict[str, object]]) -> None:
    configs: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        configs[config_key(row)].append(row)
        for field in COMPARISON_FIELDS:
            row[field] = ""
    for config_rows in configs.values():
        atcc = next((row for row in config_rows if is_atcc(row)), None)
        baselines = [row for row in config_rows if not is_atcc(row)]
        if atcc is None or not baselines:
            continue
        best_agent = best_row(baselines, "agent_tps_mean")
        best_total = best_row(baselines, "total_tps_mean")
        best_background = best_row(baselines, "background_tps_mean")
        add_speedup(atcc, "", best_agent, "throughput")
        add_speedup(atcc, "agent", best_agent, "agent_tps")
        add_speedup(atcc, "total", best_total, "total_tps")
        if number(atcc.get("background_workers")):
            add_speedup(atcc, "background", best_background, "background_tps")
        atcc_abort = number(atcc.get("agent_attempt_abort_rate_mean"))
        baseline_abort = number(best_agent.get("agent_attempt_abort_rate_mean"))
        if atcc_abort is not None and baseline_abort is not None:
            atcc["abort_rate_delta_vs_best_agent_baseline"] = fmt(atcc_abort - baseline_abort)
        atcc_p99 = number(atcc.get("agent_p99_latency_ms_mean"))
        baseline_p99 = number(best_agent.get("agent_p99_latency_ms_mean"))
        if atcc_p99 is not None and baseline_p99 not in (None, 0.0):
            atcc["p99_reduction_vs_best_agent_baseline"] = fmt(
                (baseline_p99 - atcc_p99) / baseline_p99
            )
        add_reduction_factor(
            atcc,
            best_agent,
            metric="agent_wasted_tokens_per_commit",
            baseline_field="best_agent_baseline_wasted_tokens_per_commit_mean",
            factor_field="wasted_tokens_reduction_factor_vs_best_agent_baseline",
        )
        add_reduction_factor(
            atcc,
            best_agent,
            metric="wasted_reasoning_ms_per_commit",
            baseline_field="best_agent_baseline_wasted_reasoning_ms_per_commit_mean",
            factor_field="wasted_reasoning_reduction_factor_vs_best_agent_baseline",
        )


def add_speedup(
    atcc: dict[str, object],
    prefix: str,
    baseline: dict[str, object],
    metric: str,
) -> None:
    atcc_value = number(atcc.get(f"{metric}_mean"))
    baseline_value = number(baseline.get(f"{metric}_mean"))
    field_prefix = f"{prefix}_" if prefix else ""
    atcc[f"best_{field_prefix}baseline_cc"] = system_label(baseline)
    atcc[f"best_{field_prefix}baseline_{metric}_mean"] = fmt(baseline_value)
    if atcc_value is None or baseline_value in (None, 0.0):
        return
    speedup = atcc_value / baseline_value
    atcc[f"{metric}_speedup_vs_best"] = fmt(speedup)
    atcc[f"{metric}_gain_pct_vs_best"] = fmt((speedup - 1.0) * 100.0)


def add_reduction_factor(
    atcc: dict[str, object],
    baseline: dict[str, object],
    *,
    metric: str,
    baseline_field: str,
    factor_field: str,
) -> None:
    atcc_value = number(atcc.get(f"{metric}_mean"))
    baseline_value = number(baseline.get(f"{metric}_mean"))
    atcc[baseline_field] = fmt(baseline_value)
    if baseline_value in (None, 0.0) or atcc_value is None:
        return
    atcc[factor_field] = "inf" if atcc_value == 0.0 else fmt(
        baseline_value / atcc_value
    )


def add_curve_diagnostics(rows: list[dict[str, object]]) -> None:
    curves: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for field in CURVE_FIELDS:
            row[field] = ""
        curves[
            (
                str(row.get("experiment", "")),
                str(row.get("parameter", "")),
                str(row.get("workload_variant", "")),
                str(row.get("agent_ratio", "")),
                system_label(row),
            )
        ].append(row)
    for curve in curves.values():
        curve.sort(
            key=lambda row: number(row.get("parameter_value"))
            if number(row.get("parameter_value")) is not None
            else number(row.get("clients")) or 0
        )
        for previous, current in zip(curve, curve[1:]):
            current["previous_clients"] = previous.get("clients", "")
            for metric in ("throughput", "agent_tps", "total_tps", "background_tps"):
                old = number(previous.get(f"{metric}_mean"))
                new = number(current.get(f"{metric}_mean"))
                ratio_value = new / old if new is not None and old not in (None, 0.0) else None
                current[f"{metric}_ratio_vs_previous_client"] = fmt(ratio_value)
                if metric == "throughput" and ratio_value is not None:
                    current["throughput_change_pct_vs_previous_client"] = fmt(
                        (ratio_value - 1.0) * 100.0
                    )
                    current["throughput_curve_drop_gt_10pct"] = str(
                        ratio_value < 0.90
                    ).lower()
                if metric == "agent_tps" and ratio_value is not None:
                    current["agent_tps_change_pct_vs_previous_client"] = fmt(
                        (ratio_value - 1.0) * 100.0
                    )
                    current["agent_curve_drop_gt_10pct"] = str(ratio_value < 0.90).lower()


def validate_coverage(
    rows: list[dict[str, object]],
    *,
    expected_seeds: int,
    expected_strategies: set[str],
) -> None:
    if expected_seeds:
        incomplete = [row for row in rows if int(row.get("n_seeds", 0) or 0) != expected_seeds]
        if incomplete:
            raise SystemExit(f"seed coverage failure: {len(incomplete)} summary rows")
    if expected_strategies:
        by_config: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in rows:
            by_config[config_key(row)].add(str(row.get("cc_label", "")))
        missing = [key for key, labels in by_config.items() if labels != expected_strategies]
        if missing:
            raise SystemExit(f"strategy coverage failure: {len(missing)} configurations")


def output_fields() -> list[str]:
    fields = list(GROUP_FIELDS)
    fields.extend(("n_rows", "n_ok", "n_error", "n_seeds", "seeds", "status", "errors"))
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_cv"))
        if metric in SUM_ACROSS_SEEDS:
            fields.append(f"{metric}_sum_across_seeds")
    fields.extend(f"{metric}_by_seed_json" for metric in PRIMARY_SEED_METRICS)
    fields.append("raw_action_counts_sum_json")
    fields.extend(f"paper_action_{action}_count_sum" for action in range(16))
    fields.extend(COMPARISON_FIELDS)
    fields.extend(CURVE_FIELDS)
    return fields


def metric_stats(target: dict[str, object], metric: str, values: list[float]) -> None:
    if not values:
        target[f"{metric}_mean"] = ""
        target[f"{metric}_std"] = ""
        target[f"{metric}_cv"] = ""
        return
    average = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    target[f"{metric}_mean"] = fmt(average)
    target[f"{metric}_std"] = fmt(deviation)
    target[f"{metric}_cv"] = fmt(deviation / abs(average)) if average else ""


def best_row(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    return max(rows, key=lambda row: number(row.get(metric)) or float("-inf"))


def config_key(row: dict[str, object]) -> tuple[str, ...]:
    return (
        str(row.get("experiment", "")),
        str(row.get("parameter", "")),
        str(row.get("parameter_value", "")),
        str(row.get("workload_variant", "")),
        str(row.get("agent_ratio", "")),
        str(row.get("clients", "")),
        str(row.get("agent_count", "")),
        str(row.get("worker_count", "")),
    )


def system_label(row: dict[str, object]) -> str:
    explicit = str(row.get("cc_label", "") or "").strip()
    if explicit:
        return explicit
    system = str(row.get("system", "") or "").strip()
    cc = str(row.get("cc", "") or "").strip()
    if system.lower() in {"cast-das", "castdas"} and cc:
        return cc
    return system or cc


def is_atcc(row: dict[str, object]) -> bool:
    family = str(row.get("cc_family", "") or "").strip().lower()
    label = system_label(row).strip().lower()
    cc = str(row.get("cc", "") or "").strip().lower()
    return bool(
        family == "atcc"
        or label in {"aegis", "atcc"}
        or cc.startswith("paper-atcc")
    )


def sort_key(row: dict[str, object]) -> tuple[object, ...]:
    variants = {"ycsb_low": 0, "ycsb_medium_z07": 1, "ycsb_high_z099": 2,
                "tpcc_low_w100": 3, "tpcc_high_w1": 4}
    strategies = {"ATCC": 0, "OCC": 1, "2PL-wait-die": 2, "Silo": 3, "Polaris": 4}
    return (
        str(row.get("experiment", "")),
        str(row.get("parameter", "")),
        number(row.get("parameter_value")) or 0.0,
        variants.get(str(row.get("workload_variant", "")), 99),
        int(number(row.get("clients")) or 0),
        -float(number(row.get("agent_ratio")) or 0.0),
        strategies.get(system_label(row), 99),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric_value(row: dict[str, object], metric: str) -> float | None:
    if metric == "wasted_reasoning_ms_per_commit":
        wasted = number(row.get("wasted_reasoning_ms"))
        commits = number(row.get("agent_completed_tasks"))
        if wasted is None or commits in (None, 0.0):
            return None
        return wasted / commits
    return number(row.get(metric))


def number(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)  # type: ignore[arg-type]
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def fmt(value: object) -> str:
    parsed = number(value)
    return "" if parsed is None else f"{parsed:.10g}"


def seed_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 0, value


if __name__ == "__main__":
    raise SystemExit(main())
