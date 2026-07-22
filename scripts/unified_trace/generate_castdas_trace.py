#!/usr/bin/env python3
"""Generate fixed CAST-DAS workload traces for unified CC comparisons.

The trace is intentionally workload-engine neutral.  Each transaction is a
concrete ordered read/write set plus deterministic agent reasoning delays.  The
same file can be replayed by the Python CAST-DAS runtime and by patched
DBx1000-family systems such as Polaris/Bamboo.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmarks.phases import ReasoningProfile, plan_task_phases
from agent.workloads import AgentTask, build_workload
from agent.workloads.tpcc import TPCCWorkload
from agent.workloads.ycsb import YCSBWorkload


VARIANTS: dict[str, dict[str, Any]] = {
    "ycsb_low": {
        "workload": "ycsb",
        "level": "low",
        "ycsb_zipf_theta": 0.0,
        "tpcc_warehouses": None,
    },
    "ycsb_medium_z07": {
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": 0.7,
        "tpcc_warehouses": None,
    },
    "ycsb_medium_z08": {
        "workload": "ycsb",
        "level": "medium",
        "ycsb_zipf_theta": 0.8,
        "tpcc_warehouses": None,
    },
    "ycsb_high_z099": {
        "workload": "ycsb",
        "level": "high",
        "ycsb_zipf_theta": 0.99,
        "tpcc_warehouses": None,
    },
    "tpcc_low_w100": {
        "workload": "tpcc",
        "level": "low",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": 100,
    },
    "tpcc_medium": {
        "workload": "tpcc",
        "level": "medium",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": None,
    },
    "tpcc_high_w1": {
        "workload": "tpcc",
        "level": "high",
        "ycsb_zipf_theta": None,
        "tpcc_warehouses": 1,
    },
}

FIELDS = [
    "trace_id",
    "workload_variant",
    "workload",
    "level",
    "ycsb_zipf_theta",
    "tpcc_warehouses",
    "clients",
    "agent_ratio",
    "agent_workers",
    "background_workers",
    "seed",
    "repeat",
    "worker_id",
    "client_type",
    "sequence",
    "task_id",
    "task_type",
    "operation_count",
    "read_count",
    "write_count",
    "ops_json",
    "object_keys_json",
    "explore_delay_ms",
    "refine_delay_ms",
    "commit_delay_ms",
    "retry_delay_ms",
    "retry_delays_json",
    "total_reasoning_delay_ms",
    "context_json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-id", default="")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--agent-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--transactions-per-worker", type=int, default=128)
    parser.add_argument("--background-trace-transactions-per-worker", type=int, default=0)
    parser.add_argument("--reasoning-profile", default="agentic")
    parser.add_argument("--reasoning-scale", type=float, default=1.0)
    parser.add_argument("--ycsb-access-distribution", choices=("", "zipfian", "hotspot"), default="")
    parser.add_argument("--ycsb-zipf-theta", type=float, default=None)
    parser.add_argument("--ycsb-hotset-size", type=int, default=0)
    parser.add_argument("--ycsb-hotspot-access-probability", type=float, default=None)
    parser.add_argument("--ycsb-operations", type=int, default=0)
    parser.add_argument("--ycsb-write-ratio", type=float, default=None)
    parser.add_argument("--tpcc-order-lines", type=int, default=0)
    parser.add_argument("--policy-invocation-ops", type=int, default=0)
    parser.add_argument(
        "--ycsb-post-write-reasoning-ms",
        type=int,
        default=0,
        help="Move YCSB writes before a commit-phase reasoning suffix (DWA ablation).",
    )
    parser.add_argument(
        "--ycsb-dwa-role-mix", action="store_true",
        help="One shared-key writer and read-only YCSB readers (DWA scenario).",
    )
    parser.add_argument("--ycsb-dwa-writers", type=int, default=2)
    parser.add_argument("--ycsb-dwa-shards", type=int, default=2)
    parser.add_argument("--ycsb-dwa-heterogeneous-writers", action="store_true")
    parser.add_argument("--ycsb-dwa-reader-reasoning-ms", type=int, default=0)
    parser.add_argument("--client-think-ms", type=int, default=0)
    args = parser.parse_args()

    if args.clients < 1:
        raise SystemExit("--clients must be positive")
    if not 0.0 < args.agent_ratio <= 1.0:
        raise SystemExit("--agent-ratio must be > 0 and <= 1")
    if args.transactions_per_worker <= 0:
        raise SystemExit("--transactions-per-worker must be positive")
    if args.background_trace_transactions_per_worker < 0:
        raise SystemExit("--background-trace-transactions-per-worker must be non-negative")
    if args.ycsb_hotset_size < 0 or args.ycsb_operations < 0 or args.tpcc_order_lines < 0:
        raise SystemExit("workload size overrides must be non-negative")
    if args.policy_invocation_ops < 0:
        raise SystemExit("--policy-invocation-ops must be non-negative")
    if args.ycsb_write_ratio is not None and not 0.0 <= args.ycsb_write_ratio <= 1.0:
        raise SystemExit("--ycsb-write-ratio must be in [0, 1]")
    if (
        args.ycsb_hotspot_access_probability is not None
        and not 0.0 <= args.ycsb_hotspot_access_probability <= 1.0
    ):
        raise SystemExit("--ycsb-hotspot-access-probability must be in [0, 1]")

    variant = VARIANTS[args.variant]
    agent_workers = max(1, int(round(args.clients * args.agent_ratio)))
    background_workers = max(0, args.clients - agent_workers)
    background_transactions_per_worker = resolve_background_trace_length(
        args.transactions_per_worker,
        args.background_trace_transactions_per_worker,
    )
    trace_id = args.trace_id or (
        f"{args.variant}_c{args.clients}_a{args.agent_ratio:g}_r{args.repeat}_s{args.seed}"
    )

    workload = build_workload(
        variant["workload"],
        variant["level"],
        "paper",
        ycsb_zipf_theta=variant["ycsb_zipf_theta"],
        tpcc_warehouses=variant["tpcc_warehouses"],
    )
    workload = paper_trace_workload(workload)
    workload = apply_experiment_overrides(
        workload,
        ycsb_access_distribution=args.ycsb_access_distribution,
        ycsb_zipf_theta=args.ycsb_zipf_theta,
        ycsb_hotset_size=args.ycsb_hotset_size,
        ycsb_hotspot_access_probability=args.ycsb_hotspot_access_probability,
        ycsb_operations=args.ycsb_operations,
        ycsb_write_ratio=args.ycsb_write_ratio,
        tpcc_order_lines=args.tpcc_order_lines,
    )
    experiment_context = workload_experiment_context(
        workload,
        policy_invocation_ops=args.policy_invocation_ops,
    )
    experiment_context["client_think_ms"] = max(0, int(args.client_think_ms))
    # Generate enough tasks to match the original mixed benchmark's worker
    # stride pattern without cycling for the requested fixed trace length.
    task_count = max(256, agent_workers * args.transactions_per_worker)
    bg_task_count = max(
        512,
        max(1, background_workers) * background_transactions_per_worker,
    )
    agent_tasks = list(workload.generate_tasks(task_count, seed=args.seed))
    background_tasks = list(workload.generate_tasks(bg_task_count, seed=args.seed + 700_000))
    profile = ReasoningProfile(args.reasoning_profile, args.reasoning_scale)

    object_key_map: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for worker in range(agent_workers):
        rows.extend(
            task_rows(
                trace_id=trace_id,
                variant_name=args.variant,
                variant=variant,
                clients=args.clients,
                agent_ratio=args.agent_ratio,
                agent_workers=agent_workers,
                background_workers=background_workers,
                seed=args.seed,
                repeat=args.repeat,
                worker_id=worker,
                client_type="agent",
                tasks=agent_tasks,
                stride=agent_workers,
                transactions_per_worker=args.transactions_per_worker,
                profile=profile,
                object_key_map=object_key_map,
                experiment_context=experiment_context,
                ycsb_post_write_reasoning_ms=args.ycsb_post_write_reasoning_ms,
                ycsb_dwa_role_mix=args.ycsb_dwa_role_mix,
                ycsb_dwa_writers=args.ycsb_dwa_writers,
                ycsb_dwa_shards=args.ycsb_dwa_shards,
                ycsb_dwa_heterogeneous_writers=args.ycsb_dwa_heterogeneous_writers,
                ycsb_dwa_reader_reasoning_ms=args.ycsb_dwa_reader_reasoning_ms,
            )
        )
    for worker in range(background_workers):
        rows.extend(
            task_rows(
                trace_id=trace_id,
                variant_name=args.variant,
                variant=variant,
                clients=args.clients,
                agent_ratio=args.agent_ratio,
                agent_workers=agent_workers,
                background_workers=background_workers,
                seed=args.seed,
                repeat=args.repeat,
                worker_id=agent_workers + worker,
                client_type="backend",
                tasks=background_tasks,
                stride=max(1, background_workers),
                transactions_per_worker=background_transactions_per_worker,
                profile=profile,
                object_key_map=object_key_map,
                experiment_context=experiment_context,
                ycsb_post_write_reasoning_ms=args.ycsb_post_write_reasoning_ms,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    meta = {
        "trace_id": trace_id,
        "workload_variant": args.variant,
        "variant": variant,
        "clients": args.clients,
        "agent_ratio": args.agent_ratio,
        "agent_workers": agent_workers,
        "background_workers": background_workers,
        "seed": args.seed,
        "repeat": args.repeat,
        "transactions_per_worker": args.transactions_per_worker,
        "background_trace_transactions_per_worker": background_transactions_per_worker,
        "reasoning_profile": args.reasoning_profile,
        "reasoning_scale": args.reasoning_scale,
        "workload_config": experiment_context,
        "reasoning_timing": "fixed_seed_per_operation_delay",
        "retry_timing": "fixed_seed_per_attempt_delay",
        "transaction_count": len(rows),
        "object_count": len(object_key_map),
        "trace_csv": str(args.output),
        "object_key_map": object_key_map,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"transactions={len(rows)} objects={len(object_key_map)}")
    return 0


def resolve_background_trace_length(
    agent_transactions_per_worker: int,
    configured_background_transactions_per_worker: int,
) -> int:
    configured = int(configured_background_transactions_per_worker)
    return configured if configured > 0 else int(agent_transactions_per_worker)


def paper_trace_workload(workload: Any) -> Any:
    """Use paper-scale logical schemas without materializing the full database."""
    if isinstance(workload, YCSBWorkload):
        logical_records = int(workload.config.logical_record_count or workload.config.record_count)
        return YCSBWorkload(
            dataclasses.replace(
                workload.config,
                record_count=logical_records,
                field_count=1,
            )
        )
    if isinstance(workload, TPCCWorkload):
        return TPCCWorkload(
            dataclasses.replace(
                workload.config,
                districts_per_warehouse=10,
                customers_per_district=3_000,
                items=100_000,
                order_lines=10,
                trace_mode=True,
            )
        )
    return workload


def apply_experiment_overrides(
    workload: Any,
    *,
    ycsb_access_distribution: str = "",
    ycsb_zipf_theta: float | None = None,
    ycsb_hotset_size: int = 0,
    ycsb_hotspot_access_probability: float | None = None,
    ycsb_operations: int = 0,
    ycsb_write_ratio: float | None = None,
    tpcc_order_lines: int = 0,
) -> Any:
    if isinstance(workload, YCSBWorkload):
        config = workload.config
        changes: dict[str, Any] = {}
        if ycsb_operations > 0:
            changes["operations_per_task"] = int(ycsb_operations)
        if ycsb_write_ratio is not None:
            changes["read_weight"] = 1.0 - float(ycsb_write_ratio)
            changes["update_weight"] = float(ycsb_write_ratio)
        distribution = str(ycsb_access_distribution).strip().lower()
        if ycsb_zipf_theta is not None:
            changes["zipf_theta"] = float(ycsb_zipf_theta)
        if ycsb_hotset_size > 0:
            changes["hotspot_fraction"] = min(
                1.0,
                float(ycsb_hotset_size) / max(1, int(config.record_count)),
            )
        if ycsb_hotspot_access_probability is not None:
            changes["hotspot_access_probability"] = float(
                ycsb_hotspot_access_probability
            )
        if distribution:
            changes["access_distribution"] = distribution
        elif ycsb_hotset_size > 0:
            changes["access_distribution"] = "hotspot"
        elif ycsb_zipf_theta is not None:
            changes["access_distribution"] = "zipfian"
        return YCSBWorkload(dataclasses.replace(config, **changes))
    if isinstance(workload, TPCCWorkload) and tpcc_order_lines > 0:
        return TPCCWorkload(
            dataclasses.replace(workload.config, order_lines=int(tpcc_order_lines))
        )
    return workload


def workload_experiment_context(
    workload: Any,
    *,
    policy_invocation_ops: int = 0,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "policy_invocation_ops": max(0, int(policy_invocation_ops)),
    }
    if isinstance(workload, YCSBWorkload):
        config = workload.config
        context.update(
            {
                "access_distribution": config.access_distribution,
                "ycsb_zipf_theta": float(config.zipf_theta),
                "ycsb_hotset_size": int(workload._hot_record_count()),
                "ycsb_hotspot_access_probability": float(
                    config.hotspot_access_probability
                ),
                "transaction_length": int(config.operations_per_task),
                "read_ratio": float(config.read_weight),
                "write_ratio": float(config.update_weight),
            }
        )
    elif isinstance(workload, TPCCWorkload):
        context.update(
            {
                "tpcc_warehouses": int(workload.config.warehouses),
                "tpcc_order_lines": int(workload.config.order_lines),
                "transaction_length": int(workload.config.order_lines),
            }
        )
    return context


def task_rows(
    *,
    trace_id: str,
    variant_name: str,
    variant: dict[str, Any],
    clients: int,
    agent_ratio: float,
    agent_workers: int,
    background_workers: int,
    seed: int,
    repeat: int,
    worker_id: int,
    client_type: str,
    tasks: list[AgentTask],
    stride: int,
    transactions_per_worker: int,
    profile: ReasoningProfile,
    object_key_map: dict[str, int],
    experiment_context: dict[str, Any] | None = None,
    ycsb_post_write_reasoning_ms: int = 0,
    ycsb_dwa_role_mix: bool = False,
    ycsb_dwa_writers: int = 2,
    ycsb_dwa_shards: int = 2,
    ycsb_dwa_heterogeneous_writers: bool = False,
    ycsb_dwa_reader_reasoning_ms: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    start = worker_id if client_type == "agent" else worker_id - agent_workers
    for sequence in range(transactions_per_worker):
        task = tasks[(start + sequence * stride) % len(tasks)]
        trace_context = {**dict(task.context), **dict(experiment_context or {})}
        planned = plan_task_phases(task, attempt=0, profile=profile)
        phase_by_operation = {
            id(operation): phase.name
            for phase in planned.phases
            for operation in phase.operations
        }
        delay_by_operation = {
            id(operation): int(delay_ms)
            for phase in planned.phases
            for operation, delay_ms in zip(
                phase.operations,
                phase.operation_delays_ms or (0,) * len(phase.operations),
            )
        }
        ops = []
        for operation in task.operations:
            key = object_key_map.setdefault(operation.object_id, len(object_key_map))
            ops.append(
                {
                    "kind": operation.kind,
                    "object_id": operation.object_id,
                    "key": key,
                    "value": operation.value,
                    "phase": phase_by_operation.get(id(operation), "commit"),
                    "delay_ms": delay_by_operation.get(id(operation), 0),
                }
            )
        if ycsb_post_write_reasoning_ms > 0 and variant["workload"] == "ycsb":
            # Execute writes after observed reads, then reason before commit.
            # This is the Agent interaction pattern DWA is intended to shorten.
            for operation in ops:
                if operation["kind"] == "write":
                    operation["phase"] = "refine"
        if ycsb_dwa_role_mix and variant["workload"] == "ycsb":
            # Two independent update shards avoid writer/writer commit
            # serialization while retaining read/write overlap per shard.
            writer_count = max(1, min(int(ycsb_dwa_writers), clients))
            shard_count = max(1, int(ycsb_dwa_shards))
            shard = worker_id % shard_count
            shared = f"ycsb:record:dwa-shared-{shard}:field:0"
            key = object_key_map.setdefault(shared, len(object_key_map))
            if worker_id < writer_count:
                ops = [
                    {"kind": "read", "object_id": shared, "key": key,
                     "value": "", "phase": "explore", "delay_ms": 0},
                    {"kind": "read", "object_id": shared, "key": key,
                     "value": "", "phase": "refine", "delay_ms": 0},
                    {"kind": "write", "object_id": shared, "key": key,
                     "value": f"dwa-{sequence}", "phase": "refine", "delay_ms": 0},
                ]
            else:
                ops = [{"kind": "read", "object_id": shared, "key": key,
                        "value": "", "phase": "explore", "delay_ms": 0}]
        phase_delays = {phase.name: int(phase.reasoning_delay_ms) for phase in planned.phases}
        retry_delays = [
            profile.retry_delay_ms(
                level=str(dict(task.context).get("level", variant["level"])),
                task_id=task.task_id,
                attempt=attempt,
            )
            for attempt in range(6)
        ]
        read_count = sum(1 for op in ops if op["kind"] == "read")
        write_count = sum(1 for op in ops if op["kind"] == "write")
        rows.append(
            {
                "trace_id": trace_id,
                "workload_variant": variant_name,
                "workload": variant["workload"],
                "level": variant["level"],
                "ycsb_zipf_theta": trace_context.get(
                    "ycsb_zipf_theta",
                    "" if variant["ycsb_zipf_theta"] is None else variant["ycsb_zipf_theta"],
                ),
                "tpcc_warehouses": trace_context.get(
                    "tpcc_warehouses",
                    "" if variant["tpcc_warehouses"] is None else variant["tpcc_warehouses"],
                ),
                "clients": clients,
                "agent_ratio": agent_ratio,
                "agent_workers": agent_workers,
                "background_workers": background_workers,
                "seed": seed,
                "repeat": repeat,
                "worker_id": worker_id,
                "client_type": client_type,
                "sequence": sequence,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "operation_count": len(ops),
                "read_count": read_count,
                "write_count": write_count,
                "ops_json": json.dumps(ops, separators=(",", ":"), sort_keys=True),
                "object_keys_json": json.dumps([op["key"] for op in ops], separators=(",", ":")),
                "explore_delay_ms": phase_delays.get("explore", 0),
                "refine_delay_ms": phase_delays.get("refine", 0),
                "commit_delay_ms": (
                    int(ycsb_post_write_reasoning_ms) * (
                        2
                        if ycsb_dwa_heterogeneous_writers and worker_id % 2 == 0
                        else 1
                    )
                    if (
                        ycsb_post_write_reasoning_ms > 0
                        and variant["workload"] == "ycsb"
                        and (not ycsb_dwa_role_mix or worker_id < max(1, int(ycsb_dwa_writers)))
                    )
                    else (
                        int(ycsb_dwa_reader_reasoning_ms)
                        if ycsb_dwa_role_mix
                        and variant["workload"] == "ycsb"
                        and worker_id >= max(1, int(ycsb_dwa_writers))
                        else phase_delays.get("commit", 0)
                    )
                ),
                "retry_delay_ms": int(retry_delays[1]),
                "retry_delays_json": json.dumps(retry_delays, separators=(",", ":")),
                "total_reasoning_delay_ms": int(planned.total_reasoning_delay_ms),
                "context_json": json.dumps(trace_context, separators=(",", ":"), sort_keys=True),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
