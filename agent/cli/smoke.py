"""Run CAST-DAS smoke checks."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence, TextIO

from agent.native import load_cast_core
from agent.runtime import AgentTransactionManager
from agent.workloads import build_workload, execute_task, register_workload


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
    txn.read("counter")
    txn.write("counter", "1")
    runtime_commit = txn.commit("occ")

    ycsb_ok = _run_one_task("ycsb")
    tpcc_ok = _run_one_task("tpcc")

    checks = {
        "native_backend": store.backend_name,
        "kv_put_if_version": kv_put_ok,
        "kv_stale_write_rejected": kv_stale_rejected,
        "runtime_commit": bool(runtime_commit.committed),
        "runtime_counter": manager.value_of("counter"),
        "ycsb_task_committed": ycsb_ok,
        "tpcc_task_committed": tpcc_ok,
    }
    checks["ok"] = all(
        (
            checks["kv_put_if_version"],
            checks["kv_stale_write_rejected"],
            checks["runtime_commit"],
            checks["runtime_counter"] == "1",
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


def _run_one_task(workload_name: str) -> bool:
    manager = AgentTransactionManager()
    workload = build_workload(workload_name, "low")
    register_workload(manager, workload)
    task = workload.generate_tasks(1, seed=7)[0]
    result = execute_task(manager, task, cc="dynamic-atcc")
    return bool(result.committed)


if __name__ == "__main__":
    raise SystemExit(main())
