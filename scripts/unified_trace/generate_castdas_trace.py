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
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmarks.phases import ReasoningProfile, plan_task_phases
from agent.workloads import AgentTask, build_workload


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
    parser.add_argument("--reasoning-profile", default="agentic")
    parser.add_argument("--reasoning-scale", type=float, default=2.0)
    args = parser.parse_args()

    if args.clients < 1:
        raise SystemExit("--clients must be positive")
    if not 0.0 < args.agent_ratio <= 1.0:
        raise SystemExit("--agent-ratio must be > 0 and <= 1")
    if args.transactions_per_worker <= 0:
        raise SystemExit("--transactions-per-worker must be positive")

    variant = VARIANTS[args.variant]
    agent_workers = max(1, int(round(args.clients * args.agent_ratio)))
    background_workers = max(0, args.clients - agent_workers)
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
    # Generate enough tasks to match the original mixed benchmark's worker
    # stride pattern without cycling for the requested fixed trace length.
    task_count = max(256, agent_workers * args.transactions_per_worker)
    bg_task_count = max(512, max(1, background_workers) * args.transactions_per_worker)
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
                transactions_per_worker=args.transactions_per_worker,
                profile=profile,
                object_key_map=object_key_map,
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
        "reasoning_profile": args.reasoning_profile,
        "reasoning_scale": args.reasoning_scale,
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
) -> list[dict[str, Any]]:
    rows = []
    start = worker_id if client_type == "agent" else worker_id - agent_workers
    for sequence in range(transactions_per_worker):
        task = tasks[(start + sequence * stride) % len(tasks)]
        planned = plan_task_phases(task, attempt=0, profile=profile)
        ops = []
        for operation in task.operations:
            key = object_key_map.setdefault(operation.object_id, len(object_key_map))
            ops.append(
                {
                    "kind": operation.kind,
                    "object_id": operation.object_id,
                    "key": key,
                    "value": operation.value,
                }
            )
        phase_delays = {phase.name: int(phase.reasoning_delay_ms) for phase in planned.phases}
        read_count = sum(1 for op in ops if op["kind"] == "read")
        write_count = sum(1 for op in ops if op["kind"] == "write")
        rows.append(
            {
                "trace_id": trace_id,
                "workload_variant": variant_name,
                "workload": variant["workload"],
                "level": variant["level"],
                "ycsb_zipf_theta": "" if variant["ycsb_zipf_theta"] is None else variant["ycsb_zipf_theta"],
                "tpcc_warehouses": "" if variant["tpcc_warehouses"] is None else variant["tpcc_warehouses"],
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
                "commit_delay_ms": phase_delays.get("commit", 0),
                "retry_delay_ms": int(planned.retry_delay_ms),
                "total_reasoning_delay_ms": int(planned.total_reasoning_delay_ms),
                "context_json": json.dumps(dict(task.context), separators=(",", ":"), sort_keys=True),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
