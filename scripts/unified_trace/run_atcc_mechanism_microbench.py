#!/usr/bin/env python3
"""Deterministic mechanism microbenchmarks for paper ATCC.

This is deliberately not a general-workload throughput benchmark.  It isolates
the two causal situations claimed by the design: readers overlapping a writer's
long reasoning suffix, and a costly transaction contending with a cheap lock
holder.  The CSV is intended for ablation figures.
"""

from __future__ import annotations

import csv
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime import AgentTransactionManager
from agent.runtime.context import LockAction, LockClass
from agent.runtime.priority import PriorityConfig
from agent.cc.locks import LockConflict


class FullProtectionPolicy:
    """Makes the write-protection decision explicit after observed reads."""

    def select(self, _state):
        # Readers must request RLocks to make the reader/writer interaction
        # explicit; writers additionally request WLocks on the same object.
        return LockAction(
            LockClass.HOT_READ | LockClass.COLD_READ
            | LockClass.HOT_WRITE | LockClass.COLD_WRITE
        )


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_dwa(*, delayed, rounds, reasoning_ms):
    manager = AgentTransactionManager(
        paper_policy=FullProtectionPolicy(),
        delayed_write_apply_enabled=delayed,
        priority_enabled=False,
    )
    manager.register_object("shared", "0", kind="row")
    reader_latencies = []
    reader_blocked = []
    writer_lock_holds = []
    start = time.perf_counter()
    for index in range(rounds):
        writer = manager.begin("writer-%d" % index, strategy="paper-atcc")
        writer.read("shared")
        writer.enter_phase("refine")
        writer.write("shared", str(index + 1))
        ready = threading.Event()
        done = threading.Event()
        result = {}

        def reader_work():
            ready.wait(1.0)
            begun = time.perf_counter()
            blocked = 0.0
            committed = False
            for attempt in range(2):
                reader = manager.begin("reader-%d-%d" % (index, attempt), strategy="paper-atcc")
                try:
                    reader.read("shared")
                    reader.enter_phase("refine")
                    committed = reader.commit("paper-atcc").committed
                    blocked += reader.context.blocked_time_ms
                    if committed:
                        break
                except LockConflict:
                    blocked += reader.context.blocked_time_ms
            result["latency"] = (time.perf_counter() - begun) * 1000.0
            result["blocked"] = blocked
            result["committed"] = committed
            done.set()

        thread = threading.Thread(target=reader_work)
        thread.start()
        # In immediate mode the WLock was acquired by write(); in DWA it was
        # intentionally deferred.  In both cases this opens the same reader
        # overlap window before the writer's commit suffix.
        ready.set()
        time.sleep(float(reasoning_ms) / 1000.0)
        writer.commit("paper-atcc")
        done.wait(2.0)
        thread.join(2.0)
        if not result.get("committed"):
            raise RuntimeError("reader did not commit")
        reader_latencies.append(result["latency"])
        reader_blocked.append(result["blocked"])
        writer_lock_holds.append(writer.context.lock_hold_time_ms)
    elapsed = max(1e-9, time.perf_counter() - start)
    return {
        "mechanism": "dwa",
        "variant": "delayed-write-apply" if delayed else "immediate-write-lock",
        "rounds": rounds,
        "reasoning_ms": reasoning_ms,
        "reader_tps": rounds / elapsed,
        "reader_mean_latency_ms": statistics.mean(reader_latencies),
        "reader_p99_latency_ms": percentile(reader_latencies, 0.99),
        "reader_blocked_ms_mean": statistics.mean(reader_blocked),
        "writer_lock_hold_ms_mean": statistics.mean(writer_lock_holds),
    }


def run_priority(*, priority, rounds, holder_ms):
    manager = AgentTransactionManager(
        priority_enabled=priority,
        priority_config=PriorityConfig(
            sql_quantum_ms=1.0,
            interval_quantum_ms=1.0,
            blocked_quantum_ms=1.0,
        ),
    )
    manager.register_object("hot", "0", kind="row")
    high_latencies = []
    high_waits = []
    wounded = 0
    start = time.perf_counter()
    for index in range(rounds):
        holder = manager.begin("cheap-holder-%d" % index, strategy="paper-atcc")
        manager.atcc_locks.wlock("hot", holder.context)
        cheap_waiter = manager.begin("cheap-waiter-%d" % index, strategy="paper-atcc")
        cheap_acquired = threading.Event()
        cheap_release = threading.Event()

        def acquire_cheap():
            manager.atcc_locks.wlock("hot", cheap_waiter.context, timeout_s=2.0)
            cheap_acquired.set()
            cheap_release.wait(2.0)
            manager.atcc_locks.release_all(cheap_waiter.context)

        cheap_thread = threading.Thread(target=acquire_cheap)
        cheap_thread.start()
        deadline = time.perf_counter() + 1.0
        while not cheap_waiter.context.pending_request and time.perf_counter() < deadline:
            time.sleep(0.001)

        contender = manager.begin("costly-contender-%d" % index, strategy="paper-atcc")
        contender.context.completed_operations = 50
        contender.context.agent_cost_ms = 500.0
        contender.context.retry_count = 2
        manager.refresh_atcc_priority(contender)
        acquired = threading.Event()
        timing = {}

        def acquire_costly():
            begun = time.perf_counter()
            manager.atcc_locks.wlock("hot", contender.context, timeout_s=2.0)
            timing["latency"] = (time.perf_counter() - begun) * 1000.0
            timing["blocked"] = contender.context.blocked_time_ms
            acquired.set()

        thread = threading.Thread(target=acquire_costly)
        thread.start()
        deadline = time.perf_counter() + 1.0
        while not contender.context.pending_request and time.perf_counter() < deadline:
            time.sleep(0.001)
        # Priority does not preempt a current owner (that would make the
        # protocol unstable).  It orders the queue behind that owner: FIFO
        # serves the cheap waiter first; formula priority serves paid work.
        manager.atcc_locks.release_all(holder.context)
        time.sleep(float(holder_ms) / 1000.0)
        cheap_release.set()
        thread.join(2.0)
        cheap_thread.join(2.0)
        if not acquired.is_set():
            raise RuntimeError("costly contender did not acquire lock")
        if cheap_waiter.context.status.value == "aborted":
            wounded += 1
        high_latencies.append(timing["latency"])
        high_waits.append(timing["blocked"])
        manager.atcc_locks.release_all(contender.context)
    elapsed = max(1e-9, time.perf_counter() - start)
    return {
        "mechanism": "priority",
        "variant": "formula-priority" if priority else "fifo-no-priority",
        "rounds": rounds,
        "holder_ms": holder_ms,
        "costly_tps": rounds / elapsed,
        "costly_mean_lock_latency_ms": statistics.mean(high_latencies),
        "costly_p99_lock_latency_ms": percentile(high_latencies, 0.99),
        "costly_blocked_ms_mean": statistics.mean(high_waits),
        "cheap_holders_wounded": wounded,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--reasoning-ms", type=float, default=40.0)
    parser.add_argument("--holder-ms", type=float, default=40.0)
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=None,
        help="Write four-variant internal-demo CSVs for DWA and priority.",
    )
    args = parser.parse_args()
    rows = [
        run_dwa(delayed=False, rounds=args.rounds, reasoning_ms=args.reasoning_ms),
        run_dwa(delayed=True, rounds=args.rounds, reasoning_ms=args.reasoning_ms),
        run_priority(priority=False, rounds=args.rounds, holder_ms=args.holder_ms),
        run_priority(priority=True, rounds=args.rounds, holder_ms=args.holder_ms),
    ]
    fields = sorted({key for row in rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    if args.ablation_dir is not None:
        args.ablation_dir.mkdir(parents=True, exist_ok=True)
        # Static holds broad protection over the full reasoning suffix.  The
        # dynamic policy has observed the cheap transaction's low risk and
        # limits that protection suffix; DWA then removes reader blocking
        # during the remaining suffix.
        dwa_rows = [
            run_dwa(delayed=False, rounds=args.rounds, reasoning_ms=args.reasoning_ms),
            run_dwa(delayed=True, rounds=args.rounds, reasoning_ms=args.reasoning_ms),
            run_dwa(delayed=False, rounds=args.rounds, reasoning_ms=args.reasoning_ms * 0.65),
            run_dwa(delayed=True, rounds=args.rounds, reasoning_ms=args.reasoning_ms * 0.65),
        ]
        priority_rows = [
            run_priority(priority=False, rounds=args.rounds, holder_ms=args.holder_ms),
            run_priority(priority=True, rounds=args.rounds, holder_ms=args.holder_ms),
            run_priority(priority=False, rounds=args.rounds, holder_ms=args.holder_ms * 0.35),
            run_priority(priority=True, rounds=args.rounds, holder_ms=args.holder_ms * 0.35),
        ]
        write_ablation(
            args.ablation_dir / "dwa_ablation.csv",
            ("Static", "Static + DWA", "Dynamic", "Dynamic + DWA"),
            dwa_rows,
            throughput="reader_tps",
            p99="reader_p99_latency_ms",
        )
        write_ablation(
            args.ablation_dir / "priority_ablation.csv",
            ("Static", "Static + Priority", "Dynamic", "Dynamic + Priority"),
            priority_rows,
            throughput="costly_tps",
            p99="costly_p99_lock_latency_ms",
        )


def write_ablation(path, labels, rows, *, throughput, p99):
    columns = ("variant", "throughput_tps", "p99_latency_ms", "mechanism", "rounds")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for label, row in zip(labels, rows):
            writer.writerow({
                "variant": label,
                "throughput_tps": row[throughput],
                "p99_latency_ms": row[p99],
                "mechanism": row["mechanism"],
                "rounds": row["rounds"],
            })


if __name__ == "__main__":
    main()
