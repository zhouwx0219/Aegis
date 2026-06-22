"""Large-scale rigorous CC benchmark on VitaBench-derived resources.

The benchmark uses real resource IDs and initial quantities collected from
VitaBench OTA environments. It reports throughput, latency distribution,
commit rate, SLA success rate, generation-call efficiency, and safety.

Scope: this is a concurrency-control benchmark over a VitaBench-derived write
workload. It is not an end-to-end VitaBench agent success benchmark.
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
RESOURCE_CSV = os.path.join(RESULTS, "vitabench_authoritative_resources.csv")
CATEGORIES = ("flights", "hotels", "attractions", "trains")
POLICIES = ("branch-txn", "OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL", "merge-all")


@dataclass(frozen=True)
class Resource:
    oid: str
    category: str
    quantity: int
    price: float


@dataclass(frozen=True)
class Task:
    reads: Tuple[int, ...]
    candidates: Tuple[int, ...]
    quantity: int = 1


def stable_rng(seed: int, *parts: int) -> random.Random:
    x = seed
    for p in parts:
        x = (x * 1103515245 + 12345 + p * 1000003) & 0x7FFFFFFF
    return random.Random(x)


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


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def load_resources() -> List[Resource]:
    if not os.path.exists(RESOURCE_CSV):
        raise SystemExit(
            "Missing VitaBench resource CSV. Run first:\n"
            "  bash scripts/reproduce_vitabench.sh\n"
            "or:\n"
            "  python3 agent/experiments/vitabench_authoritative.py"
        )
    rows = []
    with open(RESOURCE_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = int(float(r["quantity"]))
            if q > 0:
                rows.append(Resource(
                    oid=r["oid"],
                    category=r["category"],
                    quantity=q,
                    price=float(r.get("price", 0) or 0),
                ))
    if not rows:
        raise SystemExit("No positive-quantity VitaBench resources found.")
    return rows


def sample_unique(rng: random.Random, pool: Sequence[int], n: int) -> Tuple[int, ...]:
    if not pool or n <= 0:
        return ()
    if len(pool) <= n:
        return tuple(pool)
    return tuple(rng.sample(list(pool), n))


def make_tasks(
    resources: Sequence[Resource],
    *,
    n_tasks: int,
    k: int,
    seed: int,
    hot_per_category: int,
    hot_bias: float,
    read_size: int,
) -> List[Task]:
    by_cat: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
    for idx, r in enumerate(resources):
        by_cat.setdefault(r.category, []).append(idx)
    cats = [c for c in CATEGORIES if by_cat.get(c)]
    sorted_by_cat = {
        c: sorted(by_cat[c], key=lambda idx: (resources[idx].price, resources[idx].oid))
        for c in cats
    }
    tasks: List[Task] = []
    for tid in range(n_tasks):
        rng = stable_rng(seed, tid, 97)
        cat = cats[tid % len(cats)]
        pool = sorted_by_cat[cat]
        hot = pool[:max(1, min(hot_per_category, len(pool)))]
        cand_pool = hot if rng.random() < hot_bias else pool
        cands = sample_unique(rng, cand_pool, k)
        read_pool = pool if rng.random() < 0.5 else [i for c in cats for i in sorted_by_cat[c]]
        reads = tuple(x for x in sample_unique(rng, read_pool, read_size + len(cands)) if x not in cands)[:read_size]
        tasks.append(Task(reads=reads, candidates=cands, quantity=1))
    return tasks


def run_once(
    policy: str,
    resources: Sequence[Resource],
    *,
    n_tasks: int,
    threads: int,
    k: int,
    seed: int,
    hot_per_category: int,
    hot_bias: float,
    read_size: int,
    capacity_multiplier: int,
    c_gen: float,
    sla_ms: float,
) -> Dict[str, float]:
    k_eff = 1 if policy in ("OCC-K1", "HYBRID-K1") else k
    tasks = make_tasks(
        resources,
        n_tasks=n_tasks,
        k=k_eff,
        seed=seed,
        hot_per_category=hot_per_category,
        hot_bias=hot_bias,
        read_size=read_size,
    )
    stock = [max(1, r.quantity * capacity_multiplier) for r in resources]
    versions = [0] * len(resources)
    object_locks = [threading.Lock() for _ in resources]
    store_lock = threading.Lock()
    q: queue.Queue[int] = queue.Queue()
    for i in range(n_tasks):
        q.put(i)

    stats = {
        "booked": 0,
        "safe_booked": 0,
        "sla_success": 0,
        "no_stock": 0,
        "oversell": 0,
        "regen": 0,
        "reselect": 0,
        "merge": 0,
        "latencies_ms": [],
    }
    stats_lock = threading.Lock()

    def finish(t0: float, booked: int, safe_booked: int, no_stock: int, oversell: int, regen: int, reselect: int, merge: int) -> None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with stats_lock:
            stats["booked"] += booked
            stats["safe_booked"] += safe_booked
            stats["no_stock"] += no_stock
            stats["oversell"] += oversell
            stats["regen"] += regen
            stats["reselect"] += reselect
            stats["merge"] += merge
            if safe_booked and elapsed_ms <= sla_ms:
                stats["sla_success"] += 1
            stats["latencies_ms"].append(elapsed_ms)

    def worker() -> None:
        while True:
            try:
                tid = q.get_nowait()
            except queue.Empty:
                return
            task = tasks[tid]
            if not task.candidates:
                q.task_done()
                continue
            t0 = time.perf_counter()

            if policy == "2PL":
                write = task.candidates[0]
                objs = sorted(set(task.reads) | {write})
                locks = [object_locks[o] for o in objs]
                for lock in locks:
                    lock.acquire()
                try:
                    time.sleep(c_gen)
                    with store_lock:
                        if stock[write] >= task.quantity:
                            stock[write] -= task.quantity
                            versions[write] += 1
                            booked, safe_booked, no_stock = 1, 1, 0
                        else:
                            booked, safe_booked, no_stock = 0, 0, 1
                finally:
                    for lock in reversed(locks):
                        lock.release()
                finish(t0, booked, safe_booked, no_stock, 0, 0, 0, 0)
                q.task_done()
                continue

            with store_lock:
                read_base = {rid: versions[rid] for rid in task.reads}
                write_base = {rid: versions[rid] for rid in task.candidates}

            time.sleep(c_gen)
            booked = safe_booked = no_stock = oversell = regen = reselect = merge = 0

            if policy == "merge-all":
                write = task.candidates[0]
                with store_lock:
                    stock[write] -= task.quantity
                    versions[write] += 1
                    booked = 1
                    oversell = 1 if stock[write] < 0 else 0
                    safe_booked = 0 if oversell else 1

            elif policy == "branch-txn":
                committed = False
                need_regen = False
                with store_lock:
                    read_changed = any(versions[rid] != read_base[rid] for rid in task.reads)
                    write = task.candidates[0]
                    write_changed = versions[write] != write_base[write]
                    if not read_changed and not write_changed and stock[write] >= task.quantity:
                        stock[write] -= task.quantity
                        versions[write] += 1
                        booked = safe_booked = 1
                        committed = True
                    elif any(stock[x] >= task.quantity for x in task.candidates):
                        need_regen = True
                    else:
                        no_stock = 1

                if not committed and need_regen:
                    time.sleep(c_gen)
                    regen = 1
                    with store_lock:
                        write = next((x for x in task.candidates if stock[x] >= task.quantity), None)
                        if write is None:
                            no_stock = 1
                        else:
                            stock[write] -= task.quantity
                            versions[write] += 1
                            booked = safe_booked = 1

            elif policy in ("OCC-K1", "OCC+K", "MVCC"):
                committed = False
                with store_lock:
                    read_changed = any(versions[rid] != read_base[rid] for rid in task.reads)
                    for idx, write in enumerate(task.candidates):
                        write_changed = versions[write] != write_base[write]
                        abort = write_changed or (policy != "MVCC" and read_changed)
                        if not abort and stock[write] >= task.quantity:
                            stock[write] -= task.quantity
                            versions[write] += 1
                            booked = safe_booked = 1
                            reselect = 1 if idx > 0 else 0
                            committed = True
                            break
                    if not committed and not any(stock[x] >= task.quantity for x in task.candidates):
                        no_stock = 1

                if not committed and not no_stock:
                    time.sleep(c_gen)
                    regen = 1
                    with store_lock:
                        write = next((x for x in task.candidates if stock[x] >= task.quantity), None)
                        if write is None:
                            no_stock = 1
                        else:
                            stock[write] -= task.quantity
                            versions[write] += 1
                            booked = safe_booked = 1

            else:  # HYBRID / HYBRID-K1
                with store_lock:
                    for idx, write in enumerate(task.candidates):
                        if stock[write] >= task.quantity:
                            if versions[write] != write_base[write]:
                                merge += 1
                            stock[write] -= task.quantity
                            versions[write] += 1
                            booked = safe_booked = 1
                            reselect = 1 if idx > 0 else 0
                            break
                    if not booked:
                        no_stock = 1

            finish(t0, booked, safe_booked, no_stock, oversell, regen, reselect, merge)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall_s = time.perf_counter() - start

    lat = [float(x) for x in stats["latencies_ms"]]
    gen_calls = n_tasks + float(stats["regen"])
    return {
        "throughput": float(stats["safe_booked"]) / wall_s if wall_s else 0.0,
        "attempt_throughput": n_tasks / wall_s if wall_s else 0.0,
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "p99_latency_ms": percentile(lat, 99),
        "commit_rate": float(stats["booked"]) / max(1, n_tasks),
        "safe_commit_rate": float(stats["safe_booked"]) / max(1, n_tasks),
        "sla_success_rate": float(stats["sla_success"]) / max(1, n_tasks),
        "oversell_rate": float(stats["oversell"]) / max(1, n_tasks),
        "no_stock_rate": float(stats["no_stock"]) / max(1, n_tasks),
        "regen_per_task": float(stats["regen"]) / max(1, n_tasks),
        "generation_calls_per_task": gen_calls / max(1, n_tasks),
        "booked_per_generation_call": float(stats["safe_booked"]) / max(1.0, gen_calls),
        "merge_per_task": float(stats["merge"]) / max(1, n_tasks),
        "reselect_per_task": float(stats["reselect"]) / max(1, n_tasks),
        "wall_s": wall_s,
    }


def aggregate(rows: Iterable[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    rows = list(rows)
    out: Dict[str, float] = {}
    for m in metrics:
        avg, ci = mean_ci([r[m] for r in rows])
        out[m] = avg
        out[m + "_ci"] = ci
    return out


def profile_defaults(profile: str) -> Dict[str, object]:
    if profile == "stress":
        return {"tasks": 100000, "threads": [16, 32, 64, 96], "seeds": [1, 2, 3, 4, 5]}
    if profile == "large":
        return {"tasks": 30000, "threads": [8, 16, 32, 64], "seeds": [1, 2, 3, 4, 5]}
    return {"tasks": 6000, "threads": [8, 16, 32], "seeds": [1, 2, 3]}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "large", "stress"], default="large")
    ap.add_argument("--tasks", type=int)
    ap.add_argument("--threads", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--hot-per-category", type=int, default=6)
    ap.add_argument("--hot-bias", type=float, default=0.88)
    ap.add_argument("--read-size", type=int, default=2)
    ap.add_argument("--capacity-multiplier", type=int, default=80)
    ap.add_argument("--c-gen", type=float, default=0.002)
    ap.add_argument("--sla-ms", type=float, default=3.5)
    return ap.parse_args()


def main():
    args = parse_args()
    defaults = profile_defaults(args.profile)
    tasks = args.tasks or int(defaults["tasks"])
    thread_list = args.threads or list(defaults["threads"])
    seeds = args.seeds or list(defaults["seeds"])
    resources = load_resources()
    metrics = [
        "throughput",
        "attempt_throughput",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "commit_rate",
        "safe_commit_rate",
        "sla_success_rate",
        "oversell_rate",
        "no_stock_rate",
        "regen_per_task",
        "generation_calls_per_task",
        "booked_per_generation_call",
        "merge_per_task",
        "reselect_per_task",
        "wall_s",
    ]

    detailed: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    for threads in thread_list:
        for policy in POLICIES:
            print(f"[rigorous] threads={threads} policy={policy}")
            runs = []
            for seed in seeds:
                r = run_once(
                    policy,
                    resources,
                    n_tasks=tasks,
                    threads=threads,
                    k=args.k,
                    seed=seed,
                    hot_per_category=args.hot_per_category,
                    hot_bias=args.hot_bias,
                    read_size=args.read_size,
                    capacity_multiplier=args.capacity_multiplier,
                    c_gen=args.c_gen,
                    sla_ms=args.sla_ms,
                )
                runs.append(r)
                detailed.append({
                    "profile": args.profile,
                    "policy": policy,
                    "threads": threads,
                    "seed": seed,
                    "n_tasks": tasks,
                    "k": args.k,
                    "hot_per_category": args.hot_per_category,
                    "hot_bias": args.hot_bias,
                    "read_size": args.read_size,
                    "capacity_multiplier": args.capacity_multiplier,
                    "c_gen": args.c_gen,
                    "sla_ms": args.sla_ms,
                    **{m: round(r[m], 6) for m in metrics},
                })
            agg = aggregate(runs, metrics)
            summary.append({
                "profile": args.profile,
                "policy": policy,
                "threads": threads,
                "n_tasks": tasks,
                "k": args.k,
                "hot_per_category": args.hot_per_category,
                "hot_bias": args.hot_bias,
                "read_size": args.read_size,
                "capacity_multiplier": args.capacity_multiplier,
                "c_gen": args.c_gen,
                "sla_ms": args.sla_ms,
                **{k: round(v, 6) for k, v in agg.items()},
            })
            print(
                f"  tp={agg['throughput']:.1f} p95={agg['p95_latency_ms']:.2f}ms "
                f"sla={agg['sla_success_rate']*100:.1f}% regen/task={agg['regen_per_task']:.3f}"
            )

    by_key = {(r["policy"], int(r["threads"])): r for r in summary}
    for row in summary:
        threads = int(row["threads"])
        branch = by_key.get(("branch-txn", threads), row)
        occ = by_key.get(("OCC-K1", threads), row)
        occ_k = by_key.get(("OCC+K", threads), row)
        row["throughput_gain_vs_branch_txn_pct"] = round((float(row["throughput"]) / max(1e-9, float(branch["throughput"])) - 1) * 100, 2)
        row["throughput_gain_vs_occ_pct"] = round((float(row["throughput"]) / max(1e-9, float(occ["throughput"])) - 1) * 100, 2)
        row["throughput_gain_vs_occ_k_pct"] = round((float(row["throughput"]) / max(1e-9, float(occ_k["throughput"])) - 1) * 100, 2)
        row["p95_reduction_vs_occ_pct"] = round((1 - float(row["p95_latency_ms"]) / max(1e-9, float(occ["p95_latency_ms"]))) * 100, 2)
        row["sla_gain_vs_occ_points"] = round((float(row["sla_success_rate"]) - float(occ["sla_success_rate"])) * 100, 2)

    os.makedirs(RESULTS, exist_ok=True)
    write_csv(os.path.join(RESULTS, "rigorous_vitabench_runs.csv"), detailed)
    write_csv(os.path.join(RESULTS, "rigorous_vitabench_summary.csv"), summary)
    manifest = {
        "source_resources": RESOURCE_CSV,
        "profile": args.profile,
        "n_tasks": tasks,
        "threads": thread_list,
        "seeds": seeds,
        "policies": list(POLICIES),
        "k": args.k,
        "hot_per_category": args.hot_per_category,
        "hot_bias": args.hot_bias,
        "read_size": args.read_size,
        "capacity_multiplier": args.capacity_multiplier,
        "c_gen": args.c_gen,
        "sla_ms": args.sla_ms,
        "metrics": metrics,
        "success_rate_definition": "sla_success_rate = safe committed tasks completed within SLA / total tasks",
    }
    with open(os.path.join(RESULTS, "rigorous_vitabench_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved", os.path.join(RESULTS, "rigorous_vitabench_manifest.json"))


if __name__ == "__main__":
    main()
