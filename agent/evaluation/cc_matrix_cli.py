"""Command-line runner for deterministic CC strategy matrix experiments."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, TextIO, Tuple

from agent.evaluation import run_strategy_matrix_repeated
from agent.runtime import AdaptivePolicyTable, AgentTransactionManager
from agent.workloads import (
    TPCCConfig,
    YCSBConfig,
    build_agent_workload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASTRA agent workload CC strategy matrices."
    )
    parser.add_argument("--workload", choices=("ycsb", "tpcc"), default="ycsb")
    parser.add_argument(
        "--workload-layer",
        choices=("faithful", "semantic"),
        default="semantic",
        help="Agent-executable workload layer. Use astra-dbx1000-native for native DBx1000.",
    )
    parser.add_argument(
        "--strategies",
        default="semantic,adaptive,occ,2pl",
        help="Comma-separated CC strategies to compare.",
    )
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--contention-window", type=int, default=1)
    parser.add_argument(
        "--adaptive-policy",
        choices=("default", "new-order"),
        default="default",
        help="Policy table used when a strategy resolves to adaptive/atcc.",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--csv-section", choices=("runs", "aggregates"), default="runs")
    parser.add_argument("--output", type=Path)

    ycsb = parser.add_argument_group("YCSB options")
    ycsb.add_argument("--records", type=int, default=1000)
    ycsb.add_argument("--fields", type=int, default=10)
    ycsb.add_argument("--requests-per-task", type=int, default=4)
    ycsb.add_argument("--candidates", type=int, default=3)
    ycsb.add_argument("--read-weight", type=float, default=0.5)
    ycsb.add_argument("--update-weight", type=float, default=0.5)
    ycsb.add_argument("--zipf-theta", type=float, default=0.6)

    tpcc = parser.add_argument_group("TPC-C options")
    tpcc.add_argument("--warehouses", type=int, default=1)
    tpcc.add_argument("--districts-per-warehouse", type=int, default=10)
    tpcc.add_argument("--customers-per-district", type=int, default=300)
    tpcc.add_argument("--items", type=int, default=1000)
    tpcc.add_argument("--initial-stock", type=int, default=100)
    tpcc.add_argument("--order-lines", type=int, default=5)
    tpcc.add_argument(
        "--transaction-mix",
        default="new_order:0.45,payment:0.43,order_status:0.04,delivery:0.04,stock_level:0.04",
        help="Comma-separated TPC-C mix entries like new_order:0.45,payment:0.43.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_report(args)
    emit_report(
        report,
        fmt=args.format,
        csv_section=args.csv_section,
        output=args.output,
        stdout=stdout or sys.stdout,
    )
    return 0


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    strategies = parse_strategies(args.strategies)
    workload, workload_config = build_workload(args)
    seeds = tuple(args.seed + offset for offset in range(args.repeats))
    summaries, aggregates = run_strategy_matrix_repeated(
        workload,
        strategies,
        task_count=args.task_count,
        seeds=seeds,
        contention_window=args.contention_window,
        manager_factory=build_manager_factory(args.adaptive_policy),
    )
    return {
        "workload": workload.name,
        "workload_kind": args.workload,
        "workload_layer": args.workload_layer,
        "workload_config": workload_config,
        "workload_manifest": workload.manifest().to_dict(),
        "adaptive_policy": args.adaptive_policy,
        "strategies": list(strategies),
        "task_count": args.task_count,
        "seed": args.seed,
        "seeds": list(seeds),
        "repeats": args.repeats,
        "contention_window": args.contention_window,
        "summaries": [summary.to_dict() for summary in summaries],
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
    }


def emit_report(
    report: Dict[str, Any],
    *,
    fmt: str,
    output: Optional[Path],
    stdout: TextIO,
    csv_section: str = "runs",
) -> None:
    if fmt == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
        if output is None:
            stdout.write(text + "\n")
        else:
            output.write_text(text + "\n", encoding="utf-8")
        return

    if fmt != "csv":
        raise ValueError(f"unsupported output format: {fmt}")
    rows, fieldnames = _csv_rows_and_fields(report, csv_section)
    if output is None:
        writer = csv.DictWriter(stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_workload(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    if args.workload == "ycsb":
        config = YCSBConfig(
            record_count=args.records,
            field_count=args.fields,
            requests_per_task=args.requests_per_task,
            candidates_per_task=args.candidates,
            read_weight=args.read_weight,
            update_weight=args.update_weight,
            zipf_theta=args.zipf_theta,
        )
        return build_agent_workload(
            "ycsb", args.workload_layer, ycsb_config=config
        ), dataclasses.asdict(config)

    config = TPCCConfig(
        warehouses=args.warehouses,
        districts_per_warehouse=args.districts_per_warehouse,
        customers_per_district=args.customers_per_district,
        items=args.items,
        initial_stock=args.initial_stock,
        order_lines=args.order_lines,
        candidates_per_task=args.candidates,
        transaction_mix=parse_tpcc_mix(args.transaction_mix),
    )
    return build_agent_workload(
        "tpcc", args.workload_layer, tpcc_config=config
    ), dataclasses.asdict(config)


def build_manager_factory(policy_name: str):
    if policy_name == "default":
        return AgentTransactionManager
    if policy_name == "new-order":
        return lambda: AgentTransactionManager(
            adaptive_policy=AdaptivePolicyTable.new_order()
        )
    raise ValueError(f"unknown adaptive policy: {policy_name}")


def parse_strategies(value: str) -> Tuple[str, ...]:
    strategies = tuple(part.strip() for part in value.split(",") if part.strip())
    if not strategies:
        raise ValueError("at least one strategy is required")
    return strategies


def parse_tpcc_mix(value: str) -> Tuple[Tuple[str, float], ...]:
    entries = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid transaction mix entry: {item}")
        name, weight = item.split(":", 1)
        entries.append((name.strip(), float(weight)))
    if not entries:
        raise ValueError("transaction mix must not be empty")
    return tuple(entries)


def _csv_rows_and_fields(
    report: Dict[str, Any], csv_section: str
) -> Tuple[Iterable[Dict[str, Any]], Sequence[str]]:
    if csv_section == "runs":
        return _csv_run_rows(report), [
            "workload",
            "benchmark_family",
            "source_system",
            "strategy",
            "seed",
            "task_count",
            "contention_window",
            "committed",
            "rejected",
            "aborted",
            "n_merge",
            "n_reselect",
            "n_regen",
            "elapsed_s",
            "action_counts",
            "selected_cc_counts",
        ]
    if csv_section == "aggregates":
        return _csv_aggregate_rows(report), [
            "workload",
            "benchmark_family",
            "source_system",
            "strategy",
            "seeds",
            "runs",
            "task_count_per_run",
            "total_task_count",
            "contention_window",
            "committed_total",
            "rejected_total",
            "aborted_total",
            "committed_mean",
            "rejected_mean",
            "aborted_mean",
            "n_merge_total",
            "n_reselect_total",
            "n_regen_total",
            "n_merge_mean",
            "n_reselect_mean",
            "n_regen_mean",
            "elapsed_s_total",
            "elapsed_s_mean",
            "action_counts_total",
            "selected_cc_counts_total",
        ]
    raise ValueError(f"unsupported CSV section: {csv_section}")


def _csv_run_rows(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for summary in report["summaries"]:
        row = dict(summary)
        manifest = row.pop("workload_manifest", {})
        row["benchmark_family"] = manifest.get("benchmark_family", "")
        row["source_system"] = manifest.get("source_system", "")
        row["action_counts"] = json.dumps(row["action_counts"], sort_keys=True)
        row["selected_cc_counts"] = json.dumps(row["selected_cc_counts"], sort_keys=True)
        yield row


def _csv_aggregate_rows(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for aggregate in report["aggregates"]:
        row = dict(aggregate)
        manifest = row.pop("workload_manifest", {})
        row["benchmark_family"] = manifest.get("benchmark_family", "")
        row["source_system"] = manifest.get("source_system", "")
        row["seeds"] = json.dumps(row["seeds"])
        row["action_counts_total"] = json.dumps(
            row["action_counts_total"], sort_keys=True
        )
        row["selected_cc_counts_total"] = json.dumps(
            row["selected_cc_counts_total"], sort_keys=True
        )
        yield row


if __name__ == "__main__":
    raise SystemExit(main())
