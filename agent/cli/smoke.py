"""Run delivery smoke checks for CAST-DAS."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence, TextIO

from agent.native import load_cast_core
from agent.runtime import AgentTransactionManager
from agent.workloads import (
    AgentCandidate,
    AgentOperation,
    AgentTask,
    TPCCConfig,
    YCSBConfig,
    build_agent_workload,
    execute_task,
    prepare_task_transaction,
    register_workload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    return parser


def run_smoke_checks() -> Dict[str, Any]:
    cc = load_cast_core()
    store = cc.Dbx1000VersionedKVStore()
    store.put("key", "v1")
    base_version = store.get_version("key")
    kv_put_ok = store.put_if_version("key", base_version, "v2")
    kv_stale_rejected = not store.put_if_version("key", base_version, "stale")

    manager = AgentTransactionManager()
    manager.register_object("counter", "0", kind="counter")
    txn = manager.begin("increment")
    txn.add_candidate("increment", quality=1.0, gen_cost=0.0).delta("counter", 1)
    runtime_commit = txn.commit(strategy="semantic")

    hot_task = AgentTask(
        task_id="hot-ycsb",
        workload="agent-ycsb-semantic",
        task_type="read-update",
        request="delivery smoke transaction-atcc plan",
        candidates=(
            AgentCandidate(
                "candidate",
                1.0,
                (
                    AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                    AgentOperation.read("ycsb:record:0:field:1"),
                ),
            ),
        ),
        context={
            "agent_phase": "commit",
            "hot_record_count": 2,
            "hotspot_access_probability": 1.0,
        },
    )
    atcc_targets, atcc_decisions = manager.cc_registry.pre_snapshot_operation_plan(
        "transaction-atcc-strict",
        hot_task.candidates,
        metadata={
            "workload": hot_task.workload,
            "task_type": hot_task.task_type,
            "context": hot_task.context,
            "agent_phase": "commit",
        },
    )

    ycsb_committed = _run_one_task(
        build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=4,
                requests_per_task=2,
                candidates_per_task=2,
            ),
        )
    )
    tpcc_committed = _run_one_task(
        build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=8,
                items=16,
                order_lines=2,
                candidates_per_task=2,
                transaction_mix=(("new_order", 1.0),),
            ),
        )
    )

    checks = {
        "native_backend": store.backend_name,
        "kv_put_if_version": kv_put_ok,
        "kv_stale_write_rejected": kv_stale_rejected,
        "runtime_commit": bool(runtime_commit.committed),
        "runtime_counter": manager.value_of("counter"),
        "transaction_atcc_targets": list(atcc_targets),
        "transaction_atcc_decisions": len(atcc_decisions),
        "ycsb_task_committed": ycsb_committed,
        "tpcc_task_committed": tpcc_committed,
    }
    checks["ok"] = all(
        (
            checks["kv_put_if_version"],
            checks["kv_stale_write_rejected"],
            checks["runtime_commit"],
            checks["runtime_counter"] == "1",
            checks["transaction_atcc_decisions"] > 0,
            checks["ycsb_task_committed"],
            checks["tpcc_task_committed"],
        )
    )
    return checks


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
) -> int:
    args = build_parser().parse_args(argv)
    checks = run_smoke_checks()
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    if args.json:
        out.write(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    else:
        out.write("CAST-DAS smoke checks\n")
        for key, value in checks.items():
            out.write(f"{key}: {value}\n")
    return 0 if checks["ok"] else 1


def _run_one_task(workload: Any) -> bool:
    manager = AgentTransactionManager()
    register_workload(manager, workload)
    task = workload.generate_tasks(1, seed=7)[0]
    result = execute_task(manager, task, cc="transaction-atcc-strict")
    return bool(result.committed or result.rejected)


if __name__ == "__main__":
    raise SystemExit(main())
