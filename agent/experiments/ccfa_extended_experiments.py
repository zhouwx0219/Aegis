"""CCFA extended experiments for baseline coverage and workload scale.

This script complements the earlier prototype figures with experiments aimed at
two common reviewer concerns:

1. Are the baselines broad enough?
2. Are the workloads larger than toy-scale demos?

It intentionally uses only the Python standard library. It writes CSV files that
`paper_figures.py` turns into SVG figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    m = statistics.mean(vals)
    half = t95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / (len(vals) ** 0.5)
    return m, half


def write_csv(name: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def stable_rng(seed: int, *parts: int) -> random.Random:
    x = seed
    for p in parts:
        x = (x * 1103515245 + 12345 + p * 1000003) & 0x7FFFFFFF
    return random.Random(x)


def sample_objects(
    rng: random.Random,
    n_obj: int,
    count: int,
    *,
    hot_fraction: float = 1.0,
    hot_bias: float = 0.0,
) -> List[int]:
    count = min(count, n_obj)
    if count <= 0:
        return []
    if hot_bias <= 0.0 or hot_fraction >= 1.0:
        return rng.sample(range(n_obj), count)

    hot_n = max(count, min(n_obj, int(n_obj * hot_fraction)))
    picks: List[int] = []
    seen = set()
    attempts = 0
    while len(picks) < count and attempts < count * 32:
        attempts += 1
        limit = hot_n if rng.random() < hot_bias else n_obj
        oid = rng.randrange(limit)
        if oid not in seen:
            seen.add(oid)
            picks.append(oid)
    if len(picks) < count:
        rest = [oid for oid in range(n_obj) if oid not in seen]
        picks.extend(rng.sample(rest, count - len(picks)))
    return picks


@dataclass(frozen=True)
class MixedCandidate:
    reads: Tuple[int, ...]
    writes: Tuple[Tuple[int, str], ...]  # (object id, "delta"|"strict")


@dataclass(frozen=True)
class MixedTask:
    candidates: Tuple[MixedCandidate, ...]


def make_mixed_tasks(
    n_tasks: int,
    n_obj: int,
    k: int,
    seed: int,
    p_merge: float = 0.6,
    reads_per_task: int = 1,
    writes_per_task: int = 2,
    hot_fraction: float = 1.0,
    hot_bias: float = 0.0,
) -> Tuple[List[MixedTask], int]:
    n_counter = max(1, int(n_obj * p_merge))
    tasks: List[MixedTask] = []
    for tid in range(n_tasks):
        rng = stable_rng(seed, tid)
        cands = []
        for cid in range(k):
            picks = sample_objects(
                rng,
                n_obj,
                reads_per_task + writes_per_task,
                hot_fraction=hot_fraction,
                hot_bias=hot_bias,
            )
            w_objs = picks[:writes_per_task]
            r_objs = picks[writes_per_task:writes_per_task + reads_per_task]
            writes = tuple((o, "delta" if o < n_counter else "strict") for o in w_objs)
            cands.append(MixedCandidate(tuple(r_objs), writes))
        tasks.append(MixedTask(tuple(cands)))
    return tasks, n_counter


def mixed_validation_policy(policy: str) -> str:
    if policy in ("OCC-K1", "OCC+K"):
        return "OCC"
    if policy == "HYBRID-K1":
        return "HYBRID"
    return policy


def mixed_aborts(policy: str, cand: MixedCandidate, changed: set[int]) -> bool:
    policy = mixed_validation_policy(policy)
    read_changed = bool(set(cand.reads) & changed)
    write_objs = {o for o, _ in cand.writes}
    write_changed = bool(write_objs & changed)
    strict_changed = bool({o for o, kind in cand.writes if kind == "strict"} & changed)
    if policy in ("OCC", "Silo"):
        return read_changed or write_changed
    if policy in ("TicToc", "MVCC"):
        return write_changed
    if policy == "HYBRID":
        return strict_changed
    return False


def run_mixed(
    policy: str,
    *,
    n_tasks: int,
    n_obj: int,
    threads: int,
    k: int,
    seed: int,
    c_gen: float,
    p_merge: float = 0.6,
    hot_fraction: float = 1.0,
    hot_bias: float = 0.0,
) -> Dict[str, float]:
    tasks, n_counter = make_mixed_tasks(
        n_tasks,
        n_obj,
        k,
        seed,
        p_merge=p_merge,
        hot_fraction=hot_fraction,
        hot_bias=hot_bias,
    )
    versions = [0] * n_obj
    values = [100000 if i < n_counter else 0 for i in range(n_obj)]
    store_lock = threading.Lock()
    object_locks = [threading.Lock() for _ in range(n_obj)]
    q: queue.Queue[int] = queue.Queue()
    for i in range(n_tasks):
        q.put(i)

    stats = {
        "committed": 0,
        "regen": 0,
        "reselect": 0,
        "merge": 0,
        "latency": [],
    }
    stats_lock = threading.Lock()

    def snapshot(cand: MixedCandidate) -> Dict[int, int]:
        objs = set(cand.reads)
        objs.update(o for o, _ in cand.writes)
        return {o: versions[o] for o in objs}

    def apply(cand: MixedCandidate, changed: set[int], count_merge: bool) -> int:
        merges = 0
        for o, kind in cand.writes:
            if kind == "delta":
                values[o] -= 1
                if count_merge and o in changed:
                    merges += 1
            else:
                values[o] += 1
            versions[o] += 1
        return merges

    def worker() -> None:
        while True:
            try:
                tid = q.get_nowait()
            except queue.Empty:
                return
            raw_task = tasks[tid]
            candidates = raw_task.candidates[:1] if policy in ("OCC-K1", "HYBRID-K1") else raw_task.candidates
            task = MixedTask(candidates)
            t0 = time.perf_counter()
            if policy == "2PL":
                cand = task.candidates[0]
                objs = sorted(set(cand.reads) | {o for o, _ in cand.writes})
                locks = [object_locks[o] for o in objs]
                for lock in locks:
                    lock.acquire()
                try:
                    time.sleep(c_gen)
                    with store_lock:
                        apply(cand, set(), False)
                finally:
                    for lock in reversed(locks):
                        lock.release()
                with stats_lock:
                    stats["committed"] += 1
                    stats["latency"].append(time.perf_counter() - t0)
                q.task_done()
                continue

            with store_lock:
                bases = [snapshot(c) for c in task.candidates]
            time.sleep(c_gen)
            committed = False
            local_merge = 0
            local_reselect = 0
            with store_lock:
                for idx, cand in enumerate(task.candidates):
                    base = bases[idx]
                    changed = {o for o, v in base.items() if versions[o] != v}
                    if not mixed_aborts(policy, cand, changed):
                        local_merge = apply(cand, changed, policy == "HYBRID")
                        local_reselect = 1 if idx > 0 else 0
                        committed = True
                        break
            local_regen = 0
            if not committed:
                time.sleep(c_gen)
                local_regen = 1
                with store_lock:
                    apply(task.candidates[0], set(), False)
            with stats_lock:
                stats["committed"] += 1
                stats["regen"] += local_regen
                stats["reselect"] += local_reselect
                stats["merge"] += local_merge
                stats["latency"].append(time.perf_counter() - t0)
            q.task_done()

    t_start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - t_start
    lat = stats["latency"]
    regen_per_task = float(stats["regen"]) / max(1, stats["committed"])
    return {
        "throughput": stats["committed"] / wall if wall else 0.0,
        "latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
        "wall_s": wall,
        "regen": float(stats["regen"]),
        "reselect": float(stats["reselect"]),
        "merge": float(stats["merge"]),
        "regen_per_task": regen_per_task,
        "generation_calls_per_task": 1.0 + regen_per_task,
        "regen_waste_ms_per_task": regen_per_task * c_gen * 1000,
        "commit_efficiency": 1.0 / (1.0 + regen_per_task),
    }


@dataclass(frozen=True)
class BookingTask:
    candidates: Tuple[int, ...]


def make_booking_tasks(
    n_tasks: int,
    n_flights: int,
    k: int,
    seed: int,
    hot_fraction: float = 0.25,
    hot_bias: float = 0.75,
) -> List[BookingTask]:
    hot_n = max(1, int(n_flights * hot_fraction))
    tasks = []
    for tid in range(n_tasks):
        rng = stable_rng(seed, tid, 17)
        pool = list(range(hot_n if rng.random() < hot_bias else n_flights))
        if len(pool) >= k:
            cand = tuple(rng.sample(pool, k))
        else:
            cand = tuple(pool)
        tasks.append(BookingTask(cand))
    return tasks


def run_booking(
    policy: str,
    *,
    n_tasks: int,
    n_flights: int,
    seats_per_flight: int,
    threads: int,
    k: int,
    seed: int,
    c_gen: float,
) -> Dict[str, float]:
    limit = 1 if policy == "HYBRID-K1" else k
    tasks = make_booking_tasks(n_tasks, n_flights, limit, seed)
    seats = [seats_per_flight] * n_flights
    versions = [0] * n_flights
    store_lock = threading.Lock()
    object_locks = [threading.Lock() for _ in range(n_flights)]
    q: queue.Queue[int] = queue.Queue()
    for i in range(n_tasks):
        q.put(i)
    stats = {
        "booked": 0,
        "no_seat": 0,
        "oversell": 0,
        "regen": 0,
        "reselect": 0,
        "merge": 0,
        "latency": [],
    }
    stats_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                tid = q.get_nowait()
            except queue.Empty:
                return
            cand = tasks[tid].candidates
            if not cand:
                q.task_done()
                continue
            t0 = time.perf_counter()
            if policy == "2PL":
                oid = cand[0]
                object_locks[oid].acquire()
                try:
                    time.sleep(c_gen)
                    with store_lock:
                        if seats[oid] > 0:
                            seats[oid] -= 1
                            versions[oid] += 1
                            booked = 1
                            no_seat = 0
                        else:
                            booked = 0
                            no_seat = 1
                finally:
                    object_locks[oid].release()
                with stats_lock:
                    stats["booked"] += booked
                    stats["no_seat"] += no_seat
                    stats["latency"].append(time.perf_counter() - t0)
                q.task_done()
                continue

            with store_lock:
                base = {oid: versions[oid] for oid in cand}
            time.sleep(c_gen)

            booked = no_seat = oversell = regen = reselect = merge = 0
            if policy == "merge-all":
                oid = cand[0]
                with store_lock:
                    seats[oid] -= 1
                    versions[oid] += 1
                    booked = 1
                    oversell = 1 if seats[oid] < 0 else 0
            elif policy in ("OCC", "OCC+K"):
                search = cand[:1] if policy == "OCC" else cand
                committed = False
                with store_lock:
                    for idx, oid in enumerate(search):
                        if versions[oid] == base[oid] and seats[oid] > 0:
                            seats[oid] -= 1
                            versions[oid] += 1
                            booked = 1
                            reselect = 1 if idx > 0 else 0
                            committed = True
                            break
                    if not committed:
                        has_capacity = any(seats[oid] > 0 for oid in search)
                        if not has_capacity:
                            no_seat = 1
                if not committed and not no_seat:
                    time.sleep(c_gen)
                    with store_lock:
                        chosen = next((oid for oid in search if seats[oid] > 0), None)
                        regen = 1
                        if chosen is None:
                            no_seat = 1
                        else:
                            seats[chosen] -= 1
                            versions[chosen] += 1
                            booked = 1
            else:  # HYBRID / HYBRID-K1
                with store_lock:
                    for idx, oid in enumerate(cand):
                        if seats[oid] > 0:
                            if versions[oid] != base[oid]:
                                merge += 1
                            seats[oid] -= 1
                            versions[oid] += 1
                            booked = 1
                            reselect = 1 if idx > 0 else 0
                            break
                    if not booked:
                        no_seat = 1
            with stats_lock:
                stats["booked"] += booked
                stats["no_seat"] += no_seat
                stats["oversell"] += oversell
                stats["regen"] += regen
                stats["reselect"] += reselect
                stats["merge"] += merge
                stats["latency"].append(time.perf_counter() - t0)
            q.task_done()

    t_start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - t_start
    lat = stats["latency"]
    regen_per_task = float(stats["regen"]) / max(1, n_tasks)
    gen_calls = n_tasks + float(stats["regen"])
    return {
        "throughput": stats["booked"] / wall if wall else 0.0,
        "task_throughput": n_tasks / wall if wall else 0.0,
        "latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
        "booked": float(stats["booked"]),
        "no_seat": float(stats["no_seat"]),
        "oversell": float(stats["oversell"]),
        "regen": float(stats["regen"]),
        "reselect": float(stats["reselect"]),
        "merge": float(stats["merge"]),
        "wall_s": wall,
        "regen_per_task": regen_per_task,
        "generation_calls_per_task": gen_calls / max(1, n_tasks),
        "regen_waste_ms_per_task": regen_per_task * c_gen * 1000,
        "booked_per_generation_call": float(stats["booked"]) / max(1.0, gen_calls),
    }


def aggregate_runs(runs: Iterable[Dict[str, float]], metric_names: Sequence[str]) -> Dict[str, float]:
    runs = list(runs)
    out: Dict[str, float] = {}
    for m in metric_names:
        values = [r[m] for r in runs]
        avg, ci = mean_ci(values)
        out[m] = avg
        out[m + "_ci"] = ci
    return out


def experiment_baseline_family(args) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    policies = ["OCC", "Silo", "TicToc", "MVCC", "2PL", "HYBRID"]
    for n_obj in args.baseline_objects:
        print(f"[baseline-family] n_obj={n_obj}")
        for policy in policies:
            runs = [
                run_mixed(
                    policy,
                    n_tasks=args.baseline_tasks,
                    n_obj=n_obj,
                    threads=args.baseline_threads,
                    k=args.k,
                    seed=s,
                    c_gen=args.c_gen,
                    p_merge=args.p_merge,
                )
                for s in args.seeds
            ]
            agg = aggregate_runs(runs, ["throughput", "latency_ms", "regen_per_task", "merge"])
            rows.append({
                "n_obj": n_obj,
                "policy": policy,
                "n_tasks": args.baseline_tasks,
                "threads": args.baseline_threads,
                "k": args.k,
                **{k: round(v, 4) for k, v in agg.items()},
            })
            print(f"  {policy:8} tp={agg['throughput']:.1f} regen/task={agg['regen_per_task']:.3f}")
    write_csv("ccfa_baseline_family.csv", rows)
    return rows


def experiment_scale(args) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    thread_rows: List[Dict[str, object]] = []
    task_rows: List[Dict[str, object]] = []
    thread_policies = ["OCC", "MVCC", "2PL", "HYBRID"]
    for threads in args.scale_threads:
        print(f"[scale-threads] threads={threads}")
        for policy in thread_policies:
            runs = [
                run_mixed(
                    policy,
                    n_tasks=args.scale_tasks,
                    n_obj=args.scale_objects,
                    threads=threads,
                    k=args.k,
                    seed=s,
                    c_gen=args.c_gen,
                    p_merge=args.p_merge,
                )
                for s in args.seeds
            ]
            agg = aggregate_runs(runs, ["throughput", "latency_ms", "regen_per_task"])
            thread_rows.append({
                "threads": threads,
                "policy": policy,
                "n_tasks": args.scale_tasks,
                "n_obj": args.scale_objects,
                "k": args.k,
                **{k: round(v, 4) for k, v in agg.items()},
            })
            print(f"  {policy:6} tp={agg['throughput']:.1f} latency={agg['latency_ms']:.2f}ms")

    for n_tasks in args.task_counts:
        print(f"[scale-tasks] n_tasks={n_tasks}")
        for policy in ["OCC", "HYBRID"]:
            runs = [
                run_mixed(
                    policy,
                    n_tasks=n_tasks,
                    n_obj=args.scale_objects,
                    threads=args.baseline_threads,
                    k=args.k,
                    seed=s,
                    c_gen=args.c_gen,
                    p_merge=args.p_merge,
                )
                for s in args.seeds
            ]
            agg = aggregate_runs(runs, ["throughput", "latency_ms", "regen_per_task"])
            task_rows.append({
                "n_tasks": n_tasks,
                "policy": policy,
                "threads": args.baseline_threads,
                "n_obj": args.scale_objects,
                "k": args.k,
                **{k: round(v, 4) for k, v in agg.items()},
            })
            print(f"  {policy:6} tp={agg['throughput']:.1f} regen/task={agg['regen_per_task']:.3f}")
    write_csv("ccfa_scale_threads.csv", thread_rows)
    write_csv("ccfa_scale_tasks.csv", task_rows)
    return thread_rows, task_rows


def experiment_agent_aware(args) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    policies = ["OCC", "OCC+K", "2PL", "merge-all", "HYBRID-K1", "HYBRID"]
    for policy in policies:
        print(f"[agent-aware] {policy}")
        runs = [
            run_booking(
                policy,
                n_tasks=args.booking_tasks,
                n_flights=args.booking_flights,
                seats_per_flight=args.seats_per_flight,
                threads=args.booking_threads,
                k=args.k,
                seed=s,
                c_gen=args.c_gen,
            )
            for s in args.seeds
        ]
        agg = aggregate_runs(runs, [
            "throughput",
            "task_throughput",
            "latency_ms",
            "booked",
            "no_seat",
            "oversell",
            "regen",
            "reselect",
            "merge",
            "regen_per_task",
            "generation_calls_per_task",
            "regen_waste_ms_per_task",
            "booked_per_generation_call",
        ])
        rows.append({
            "policy": policy,
            "n_tasks": args.booking_tasks,
            "n_flights": args.booking_flights,
            "seats_per_flight": args.seats_per_flight,
            "threads": args.booking_threads,
            "k": args.k,
            **{k: round(v, 4) for k, v in agg.items()},
        })
        print(f"  tp={agg['throughput']:.1f} booked={agg['booked']:.0f} oversell={agg['oversell']:.0f} regen={agg['regen']:.0f}")
    write_csv("ccfa_agent_aware.csv", rows)
    return rows


def experiment_hotspot(args) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    policies = ["OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL"]
    workloads = [
        ("uniform-mid", args.scale_objects, 1.0, 0.0, args.p_merge),
        ("hotspot-mixed", args.hotspot_objects, args.hot_fraction, args.hot_bias, args.hotspot_p_merge),
    ]
    metric_names = [
        "throughput",
        "latency_ms",
        "regen_per_task",
        "generation_calls_per_task",
        "regen_waste_ms_per_task",
        "commit_efficiency",
        "merge",
        "reselect",
    ]
    for workload, n_obj, hot_fraction, hot_bias, p_merge in workloads:
        print(f"[hotspot] workload={workload} n_obj={n_obj} hot_fraction={hot_fraction} hot_bias={hot_bias}")
        workload_rows: List[Dict[str, object]] = []
        for policy in policies:
            runs = [
                run_mixed(
                    policy,
                    n_tasks=args.hotspot_tasks,
                    n_obj=n_obj,
                    threads=args.hotspot_threads,
                    k=args.k,
                    seed=s,
                    c_gen=args.hotspot_c_gen,
                    p_merge=p_merge,
                    hot_fraction=hot_fraction,
                    hot_bias=hot_bias,
                )
                for s in args.seeds
            ]
            agg = aggregate_runs(runs, metric_names)
            row: Dict[str, object] = {
                "workload": workload,
                "policy": policy,
                "n_tasks": args.hotspot_tasks,
                "n_obj": n_obj,
                "threads": args.hotspot_threads,
                "k": args.k,
                "p_merge": p_merge,
                "hot_fraction": hot_fraction,
                "hot_bias": hot_bias,
                "c_gen": args.hotspot_c_gen,
                **{k: round(v, 4) for k, v in agg.items()},
            }
            workload_rows.append(row)
            print(
                f"  {policy:10} tp={agg['throughput']:.1f} "
                f"regen/task={agg['regen_per_task']:.3f} gen/task={agg['generation_calls_per_task']:.3f}"
            )

        by_policy = {str(r["policy"]): r for r in workload_rows}
        base_occ = float(by_policy["OCC-K1"]["throughput"])
        fair_occ = float(by_policy["OCC+K"]["throughput"])
        base_regen = float(by_policy["OCC-K1"]["regen_per_task"])
        for row in workload_rows:
            tp = float(row["throughput"])
            regen = float(row["regen_per_task"])
            row["speedup_vs_occ_k1"] = round(tp / base_occ, 4) if base_occ else 0.0
            row["speedup_vs_occ_k"] = round(tp / fair_occ, 4) if fair_occ else 0.0
            row["regen_reduction_vs_occ_k1_pct"] = round((base_regen - regen) * 100 / base_regen, 2) if base_regen else 0.0
        rows.extend(workload_rows)

    write_csv("ccfa_hotspot_mixed.csv", rows)
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "baseline", "scale", "agent", "hotspot"], default="all")
    ap.add_argument("--profile", choices=["quick", "large"], default="quick")
    ap.add_argument("--c-gen", type=float, default=0.0005)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p-merge", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    args = ap.parse_args()

    if args.profile == "large":
        args.baseline_tasks = 6000
        args.scale_tasks = 10000
        args.booking_tasks = 10000
        args.hotspot_tasks = 12000
        args.task_counts = [1000, 5000, 10000, 50000]
        args.scale_threads = [1, 2, 4, 8, 16, 32, 64]
    else:
        args.baseline_tasks = 2000
        args.scale_tasks = 5000
        args.booking_tasks = 5000
        args.hotspot_tasks = 4000
        args.task_counts = [1000, 5000, 10000]
        args.scale_threads = [1, 2, 4, 8, 16, 32]

    args.baseline_objects = [24, 96, 384]
    args.baseline_threads = 16
    args.scale_objects = 96
    args.hotspot_objects = 48
    args.hotspot_threads = 32 if args.profile == "large" else 16
    args.hot_fraction = 0.25
    args.hot_bias = 0.88
    args.hotspot_p_merge = 0.75
    args.hotspot_c_gen = max(args.c_gen, 0.001)
    args.booking_flights = 64
    args.seats_per_flight = 50
    args.booking_threads = 16
    return args


def main():
    args = parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    manifest = {
        "profile": args.profile,
        "c_gen": args.c_gen,
        "hotspot_c_gen": args.hotspot_c_gen,
        "k": args.k,
        "p_merge": args.p_merge,
        "seeds": args.seeds,
    }
    if args.mode in ("all", "baseline"):
        experiment_baseline_family(args)
    if args.mode in ("all", "scale"):
        experiment_scale(args)
    if args.mode in ("all", "agent"):
        experiment_agent_aware(args)
    if args.mode in ("all", "hotspot"):
        experiment_hotspot(args)
    with open(os.path.join(RESULTS, "ccfa_extended_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved", os.path.join(RESULTS, "ccfa_extended_manifest.json"))


if __name__ == "__main__":
    main()
