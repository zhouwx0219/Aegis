"""Semantic workload benchmark for ASTRA/HYBRID CC.

This benchmark complements the OTA booking experiments. The OTA matrix mostly
exercises constrained DELTA; this script adds APPEND, CAS, mixed-intent, and
private/control workloads so the semantic-aware dispatcher is evaluated across
the intent taxonomy.

The simulator is intentionally small and deterministic. It models an in-memory
versioned object store, real worker threads, wall-clock generation cost, and a
common task stream shared by all policies.
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

POLICIES = (
    "branch-txn",
    "OCC-K1",
    "OCC+K",
    "MVCC",
    "HYBRID-K1",
    "HYBRID",
    "2PL",
    "merge-all",
)
WORKLOADS = ("append_log", "cas_claim", "mixed_checkout", "private_strict")


@dataclass(frozen=True)
class Op:
    kind: str
    oid: str
    amount: int = 0
    payload: str = ""
    expected: str = ""
    value: str = ""
    lower_bound: int = 0
    commutative: bool = False


@dataclass(frozen=True)
class Candidate:
    ops: Tuple[Op, ...]
    quality: float


@dataclass(frozen=True)
class Task:
    candidates: Tuple[Candidate, ...]


@dataclass
class Obj:
    value: object
    version: int = 0


def stable_rng(seed: int, *parts: int) -> random.Random:
    x = seed & 0x7FFFFFFF
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
    return m, t95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / (len(vals) ** 0.5)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def snapshot(store: Dict[str, Obj], cand: Candidate, include_reads: bool = True) -> Dict[str, Tuple[int, object]]:
    out: Dict[str, Tuple[int, object]] = {}
    for op in cand.ops:
        if op.kind == "read" and not include_reads:
            continue
        obj = store[op.oid]
        out[op.oid] = (obj.version, obj.value)
    return out


def syntactic_abort(policy: str, cand: Candidate, base: Dict[str, Tuple[int, object]], store: Dict[str, Obj]) -> bool:
    for op in cand.ops:
        if op.kind == "read" and policy == "MVCC":
            continue
        if store[op.oid].version != base[op.oid][0]:
            return True
    return False


def feasible_after_regen(cand: Candidate, store: Dict[str, Obj]) -> bool:
    for op in cand.ops:
        if op.kind == "delta":
            if int(store[op.oid].value) + op.amount < op.lower_bound:
                return False
        elif op.kind == "cas":
            if str(store[op.oid].value) != op.expected:
                return False
    return True


def apply_ops(cand: Candidate, store: Dict[str, Obj], *, unsafe: bool = False) -> Dict[str, int]:
    changed = {"append": 0, "delta": 0, "cas": 0, "overwrite": 0, "oversell": 0, "cas_violation": 0}
    for op in cand.ops:
        if op.kind == "read":
            continue
        obj = store[op.oid]
        if op.kind == "append":
            obj.value = int(obj.value) + 1
            changed["append"] += 1
        elif op.kind == "delta":
            obj.value = int(obj.value) + op.amount
            changed["delta"] += 1
            if int(obj.value) < op.lower_bound:
                changed["oversell"] += 1
        elif op.kind == "cas":
            if str(obj.value) != op.expected and unsafe:
                changed["cas_violation"] += 1
            obj.value = op.value
            changed["cas"] += 1
        elif op.kind == "overwrite":
            obj.value = op.value
            changed["overwrite"] += 1
        obj.version += 1
    return changed


def semantic_validate(cand: Candidate, base: Dict[str, Tuple[int, object]], store: Dict[str, Obj]) -> Tuple[bool, str, int]:
    merges = 0
    for op in cand.ops:
        obj = store[op.oid]
        base_version = base[op.oid][0]
        changed = obj.version != base_version
        if op.kind == "read":
            continue
        if op.kind == "append":
            if op.commutative:
                merges += 1 if changed else 0
                continue
            if changed:
                return False, "strict_append_conflict", merges
        elif op.kind == "delta":
            if int(obj.value) + op.amount < op.lower_bound:
                return False, "delta_bound", merges
            merges += 1 if changed else 0
        elif op.kind == "cas":
            if str(obj.value) != op.expected:
                return False, "cas_failed", merges
            merges += 1 if changed else 0
        elif op.kind == "overwrite":
            if changed:
                return False, "strict_conflict", merges
    return True, "", merges


def maybe_noise(workload: str, task: Task, store: Dict[str, Obj], rng: random.Random, noise_prob: float) -> None:
    if workload != "cas_claim" or noise_prob <= 0.0:
        return
    # Metadata churn: bump the row version while leaving slot.status == free.
    for cand in task.candidates:
        for op in cand.ops:
            if op.kind == "cas" and str(store[op.oid].value) == "free" and rng.random() < noise_prob:
                store[op.oid].version += 1


def make_store_and_tasks(
    workload: str,
    *,
    n_tasks: int,
    threads: int,
    k: int,
    seed: int,
    hot_objects: int,
    object_pool: int,
    p_strict: float,
) -> Tuple[Dict[str, Obj], List[Task], Dict[str, object]]:
    store: Dict[str, Obj] = {}
    tasks: List[Task] = []
    meta: Dict[str, object] = {}

    if workload == "append_log":
        logs = [f"log:{i}" for i in range(object_pool)]
        users = [f"user:{i}" for i in range(max(16, object_pool))]
        for oid in logs + users:
            store[oid] = Obj(0)
        for tid in range(n_tasks):
            rng = stable_rng(seed, tid, 11)
            cands = []
            for cid in range(k):
                log_pool = logs[:hot_objects] if rng.random() < 0.9 else logs
                log = rng.choice(log_pool)
                user = rng.choice(users)
                ops = (
                    Op("read", f"user:{rng.randrange(len(users))}"),
                    Op("append", log, payload=f"event:{tid}:{cid}", commutative=True),
                    Op("append", user, payload=f"cart:{tid}:{cid}", commutative=True),
                )
                cands.append(Candidate(ops, quality=float(k - cid)))
            tasks.append(Task(tuple(cands)))
        meta["semantic_mix"] = {"APPEND": 2, "READ": 1}

    elif workload == "cas_claim":
        # Slots are scoped to a concurrency window. This keeps the workload from
        # degenerating into global resource exhaustion while still creating real
        # conflicts among tasks that are likely to overlap in time.
        group_size = max(8, threads)
        slots_per_group = max(k * 4, hot_objects * 3)
        groups = (n_tasks + group_size - 1) // group_size
        slots_by_group = []
        for gid in range(groups):
            slots = [f"slot:{gid}:{i}" for i in range(slots_per_group)]
            slots_by_group.append(slots)
            for oid in slots:
                store[oid] = Obj("free")
        for tid in range(n_tasks):
            rng = stable_rng(seed, tid, 23)
            slots = slots_by_group[tid // group_size]
            hot = slots[:max(k, min(hot_objects, len(slots)))]
            pool = hot if rng.random() < 0.85 else slots
            chosen = rng.sample(pool if len(pool) >= k else slots, min(k, len(slots)))
            cands = []
            for cid, slot in enumerate(chosen):
                ops = (Op("cas", slot, expected="free", value="taken"),)
                cands.append(Candidate(ops, quality=float(k - cid)))
            tasks.append(Task(tuple(cands)))
        meta["semantic_mix"] = {"CAS": 1, "group_size": group_size, "slots_per_group": slots_per_group}

    elif workload == "mixed_checkout":
        stocks = [f"stock:{i}" for i in range(object_pool)]
        logs = [f"audit:{i}" for i in range(max(8, object_pool // 2))]
        carts = [f"cart:{i}" for i in range(max(32, object_pool))]
        coupons = [f"coupon:{tid}:{cid}" for tid in range(n_tasks) for cid in range(k)]
        prefs = [f"pref:{i}" for i in range(max(16, object_pool // 2))]
        for oid in stocks:
            store[oid] = Obj(400)
        for oid in logs + carts:
            store[oid] = Obj(0)
        for oid in coupons:
            store[oid] = Obj("unused")
        for oid in prefs:
            store[oid] = Obj("v0")
        for tid in range(n_tasks):
            rng = stable_rng(seed, tid, 37)
            cands = []
            for cid in range(k):
                hot_stock = stocks[:hot_objects] if rng.random() < 0.88 else stocks
                stock = rng.choice(hot_stock)
                cart = rng.choice(carts)
                log = rng.choice(logs[:max(2, hot_objects // 2)])
                coupon = f"coupon:{tid}:{cid}"
                pref = rng.choice(prefs[:max(2, hot_objects // 2)])
                ops: List[Op] = [
                    Op("read", pref),
                    Op("delta", stock, amount=-1, lower_bound=0),
                    Op("append", cart, payload=f"item:{tid}:{cid}", commutative=True),
                    Op("append", log, payload=f"evt:{tid}:{cid}", commutative=True),
                    Op("cas", coupon, expected="unused", value="used"),
                ]
                if rng.random() < p_strict:
                    ops.append(Op("overwrite", pref, value=f"pref:{tid}:{cid}"))
                cands.append(Candidate(tuple(ops), quality=float(k - cid)))
            tasks.append(Task(tuple(cands)))
        meta["semantic_mix"] = {
            "DELTA_CONSTRAINED": 1,
            "APPEND": 2,
            "CAS": 1,
            "READ": 1,
            "OVERWRITE_probability": p_strict,
        }

    elif workload == "private_strict":
        for tid in range(n_tasks):
            store[f"order:{tid}"] = Obj("new")
            store[f"user:{tid % max(1, object_pool)}"] = Obj("profile")
        for tid in range(n_tasks):
            ops = (
                Op("read", f"user:{tid % max(1, object_pool)}"),
                Op("overwrite", f"order:{tid}", value=f"confirmed:{tid}"),
            )
            tasks.append(Task((Candidate(ops, quality=1.0),)))
        meta["semantic_mix"] = {"OVERWRITE": 1, "READ": 1, "private": True}

    else:
        raise ValueError(f"unknown workload: {workload}")

    return store, tasks, meta


def run_once(
    workload: str,
    policy: str,
    *,
    n_tasks: int,
    threads: int,
    k: int,
    seed: int,
    c_gen: float,
    hot_objects: int,
    object_pool: int,
    p_strict: float,
    noise_prob: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    k_eff = 1 if policy in ("OCC-K1", "HYBRID-K1") else k
    store, tasks, meta = make_store_and_tasks(
        workload,
        n_tasks=n_tasks,
        threads=threads,
        k=k_eff,
        seed=seed,
        hot_objects=hot_objects,
        object_pool=object_pool,
        p_strict=p_strict,
    )
    q: queue.Queue[int] = queue.Queue()
    for i in range(len(tasks)):
        q.put(i)
    store_lock = threading.Lock()
    object_locks = {oid: threading.Lock() for oid in store}
    stats = {
        "committed": 0,
        "safe_committed": 0,
        "failed": 0,
        "regen": 0,
        "reselect": 0,
        "merge": 0,
        "oversell": 0,
        "cas_violation": 0,
        "appends": 0,
        "deltas": 0,
        "cas": 0,
        "overwrites": 0,
        "lat_ms": [],
    }
    stats_lock = threading.Lock()

    def finish(t0: float, local: Dict[str, int]) -> None:
        elapsed = (time.perf_counter() - t0) * 1000.0
        with stats_lock:
            for key, val in local.items():
                stats[key] += val
            stats["lat_ms"].append(elapsed)

    def worker(worker_id: int) -> None:
        rng = stable_rng(seed, worker_id, 991)
        while True:
            try:
                tid = q.get_nowait()
            except queue.Empty:
                return
            task = tasks[tid]
            t0 = time.perf_counter()
            local = {k: 0 for k in (
                "committed", "safe_committed", "failed", "regen", "reselect", "merge",
                "oversell", "cas_violation", "appends", "deltas", "cas", "overwrites"
            )}

            if policy == "2PL":
                cand = task.candidates[0]
                objs = sorted({op.oid for op in cand.ops})
                locks = [object_locks[o] for o in objs]
                for lock in locks:
                    lock.acquire()
                try:
                    time.sleep(c_gen)
                    with store_lock:
                        if feasible_after_regen(cand, store):
                            changed = apply_ops(cand, store)
                            local["committed"] = local["safe_committed"] = 1
                            local["appends"] += changed["append"]
                            local["deltas"] += changed["delta"]
                            local["cas"] += changed["cas"]
                            local["overwrites"] += changed["overwrite"]
                        else:
                            local["failed"] = 1
                finally:
                    for lock in reversed(locks):
                        lock.release()
                finish(t0, local)
                q.task_done()
                continue

            with store_lock:
                bases = [snapshot(store, cand) for cand in task.candidates]
            time.sleep(c_gen)

            committed = False
            if policy == "merge-all":
                cand = task.candidates[0]
                with store_lock:
                    changed = apply_ops(cand, store, unsafe=True)
                    local["committed"] = 1
                    local["safe_committed"] = 0 if changed["oversell"] or changed["cas_violation"] else 1
                    local["oversell"] += changed["oversell"]
                    local["cas_violation"] += changed["cas_violation"]
                    local["appends"] += changed["append"]
                    local["deltas"] += changed["delta"]
                    local["cas"] += changed["cas"]
                    local["overwrites"] += changed["overwrite"]
                committed = True

            elif policy in ("branch-txn", "OCC-K1", "OCC+K", "MVCC"):
                search = task.candidates[:1] if policy in ("branch-txn", "OCC-K1") else task.candidates
                with store_lock:
                    maybe_noise(workload, task, store, rng, noise_prob)
                    for idx, cand in enumerate(search):
                        if not syntactic_abort("MVCC" if policy == "MVCC" else "OCC", cand, bases[idx], store):
                            if feasible_after_regen(cand, store):
                                changed = apply_ops(cand, store)
                                local["committed"] = local["safe_committed"] = 1
                                local["reselect"] = 1 if idx > 0 else 0
                                local["appends"] += changed["append"]
                                local["deltas"] += changed["delta"]
                                local["cas"] += changed["cas"]
                                local["overwrites"] += changed["overwrite"]
                                committed = True
                                break
                if not committed:
                    time.sleep(c_gen)
                    local["regen"] = 1
                    with store_lock:
                        # Regeneration sees the latest state and can choose the first feasible known alternative.
                        for idx, cand in enumerate(search):
                            if feasible_after_regen(cand, store):
                                changed = apply_ops(cand, store)
                                local["committed"] = local["safe_committed"] = 1
                                local["reselect"] = 1 if idx > 0 else 0
                                local["appends"] += changed["append"]
                                local["deltas"] += changed["delta"]
                                local["cas"] += changed["cas"]
                                local["overwrites"] += changed["overwrite"]
                                committed = True
                                break
                        if not committed:
                            local["failed"] = 1

            else:  # HYBRID / HYBRID-K1
                with store_lock:
                    maybe_noise(workload, task, store, rng, noise_prob)
                    hard_conflict = False
                    for idx, cand in enumerate(task.candidates):
                        ok, reason, merges = semantic_validate(cand, bases[idx], store)
                        if ok:
                            changed = apply_ops(cand, store)
                            local["committed"] = local["safe_committed"] = 1
                            local["reselect"] = 1 if idx > 0 else 0
                            local["merge"] += merges
                            local["appends"] += changed["append"]
                            local["deltas"] += changed["delta"]
                            local["cas"] += changed["cas"]
                            local["overwrites"] += changed["overwrite"]
                            committed = True
                            break
                        hard_conflict = hard_conflict or reason in ("strict_conflict", "strict_append_conflict")
                    if not committed and not hard_conflict:
                        local["failed"] = 1
                if not committed and local["failed"] == 0:
                    time.sleep(c_gen)
                    local["regen"] = 1
                    with store_lock:
                        for idx, cand in enumerate(task.candidates):
                            if feasible_after_regen(cand, store):
                                changed = apply_ops(cand, store)
                                local["committed"] = local["safe_committed"] = 1
                                local["reselect"] = 1 if idx > 0 else 0
                                local["appends"] += changed["append"]
                                local["deltas"] += changed["delta"]
                                local["cas"] += changed["cas"]
                                local["overwrites"] += changed["overwrite"]
                                committed = True
                                break
                        if not committed:
                            local["failed"] = 1

            finish(t0, local)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall_s = time.perf_counter() - start
    lat = [float(v) for v in stats["lat_ms"]]
    gen_calls = n_tasks + float(stats["regen"])
    result = {
        "throughput": float(stats["safe_committed"]) / wall_s if wall_s else 0.0,
        "attempt_throughput": n_tasks / wall_s if wall_s else 0.0,
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "p95_latency_ms": percentile(lat, 95),
        "safe_commit_rate": float(stats["safe_committed"]) / max(1, n_tasks),
        "commit_rate": float(stats["committed"]) / max(1, n_tasks),
        "failed_rate": float(stats["failed"]) / max(1, n_tasks),
        "regen_per_task": float(stats["regen"]) / max(1, n_tasks),
        "generation_calls_per_task": gen_calls / max(1, n_tasks),
        "reselect_per_task": float(stats["reselect"]) / max(1, n_tasks),
        "merge_per_task": float(stats["merge"]) / max(1, n_tasks),
        "oversell": float(stats["oversell"]),
        "cas_violation": float(stats["cas_violation"]),
        "append_ops": float(stats["appends"]),
        "delta_ops": float(stats["deltas"]),
        "cas_ops": float(stats["cas"]),
        "overwrite_ops": float(stats["overwrites"]),
        "wall_s": wall_s,
    }
    return result, meta


def aggregate(rows: Iterable[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    runs = list(rows)
    out: Dict[str, float] = {}
    for m in metrics:
        avg, ci = mean_ci([r[m] for r in runs])
        out[m] = avg
        out[m + "_ci"] = ci
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def profile_defaults(profile: str) -> Dict[str, object]:
    if profile == "large":
        return {"tasks": 8000, "threads": 32, "seeds": [1, 2, 3, 4, 5]}
    return {"tasks": 2500, "threads": 24, "seeds": [1, 2, 3]}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "large"], default="quick")
    ap.add_argument("--tasks", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--c-gen", type=float, default=0.002)
    ap.add_argument("--hot-objects", type=int, default=6)
    ap.add_argument("--object-pool", type=int, default=64)
    ap.add_argument("--p-strict", type=float, default=0.25)
    ap.add_argument("--noise-prob", type=float, default=0.35)
    ap.add_argument("--workloads", nargs="*", choices=WORKLOADS, default=list(WORKLOADS))
    ap.add_argument("--policies", nargs="*", choices=POLICIES, default=list(POLICIES))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    defaults = profile_defaults(args.profile)
    n_tasks = args.tasks or int(defaults["tasks"])
    threads = args.threads or int(defaults["threads"])
    seeds = args.seeds or list(defaults["seeds"])
    metrics = [
        "throughput",
        "attempt_throughput",
        "mean_latency_ms",
        "p95_latency_ms",
        "safe_commit_rate",
        "commit_rate",
        "failed_rate",
        "regen_per_task",
        "generation_calls_per_task",
        "reselect_per_task",
        "merge_per_task",
        "oversell",
        "cas_violation",
        "append_ops",
        "delta_ops",
        "cas_ops",
        "overwrite_ops",
        "wall_s",
    ]

    detailed: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    manifest_mix: Dict[str, object] = {}
    for workload in args.workloads:
        for policy in args.policies:
            print(f"[semantic] workload={workload} policy={policy}")
            runs = []
            for seed in seeds:
                result, meta = run_once(
                    workload,
                    policy,
                    n_tasks=n_tasks,
                    threads=threads,
                    k=args.k,
                    seed=seed,
                    c_gen=args.c_gen,
                    hot_objects=args.hot_objects,
                    object_pool=args.object_pool,
                    p_strict=args.p_strict,
                    noise_prob=args.noise_prob,
                )
                manifest_mix[workload] = meta
                runs.append(result)
                detailed.append({
                    "profile": args.profile,
                    "workload": workload,
                    "policy": policy,
                    "seed": seed,
                    "n_tasks": n_tasks,
                    "threads": threads,
                    "k": args.k,
                    "c_gen": args.c_gen,
                    "hot_objects": args.hot_objects,
                    "object_pool": args.object_pool,
                    "p_strict": args.p_strict,
                    "noise_prob": args.noise_prob,
                    **{m: round(result[m], 6) for m in metrics},
                })
            agg = aggregate(runs, metrics)
            row: Dict[str, object] = {
                "profile": args.profile,
                "workload": workload,
                "policy": policy,
                "n_tasks": n_tasks,
                "threads": threads,
                "k": args.k,
                "c_gen": args.c_gen,
                "hot_objects": args.hot_objects,
                "object_pool": args.object_pool,
                "p_strict": args.p_strict,
                "noise_prob": args.noise_prob,
                **{k: round(v, 6) for k, v in agg.items()},
            }
            summary.append(row)
            print(
                f"  tp={agg['throughput']:.1f}/s p95={agg['p95_latency_ms']:.2f}ms "
                f"regen/task={agg['regen_per_task']:.3f} merge/task={agg['merge_per_task']:.3f}"
            )

    by_key = {(r["workload"], r["policy"]): r for r in summary}
    for row in summary:
        workload = str(row["workload"])
        branch = by_key.get((workload, "branch-txn"), row)
        occ = by_key.get((workload, "OCC-K1"), row)
        occ_k = by_key.get((workload, "OCC+K"), row)
        hy_k1 = by_key.get((workload, "HYBRID-K1"), row)
        tp = float(row["throughput"])
        row["speedup_vs_branch_txn"] = round(tp / max(1e-9, float(branch["throughput"])), 4)
        row["speedup_vs_occ_k1"] = round(tp / max(1e-9, float(occ["throughput"])), 4)
        row["speedup_vs_occ_k"] = round(tp / max(1e-9, float(occ_k["throughput"])), 4)
        row["speedup_vs_hybrid_k1"] = round(tp / max(1e-9, float(hy_k1["throughput"])), 4)

    os.makedirs(RESULTS, exist_ok=True)
    write_csv(os.path.join(RESULTS, "semantic_workloads_runs.csv"), detailed)
    write_csv(os.path.join(RESULTS, "semantic_workloads_summary.csv"), summary)
    manifest = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "threads": threads,
        "seeds": seeds,
        "k": args.k,
        "c_gen": args.c_gen,
        "hot_objects": args.hot_objects,
        "object_pool": args.object_pool,
        "p_strict": args.p_strict,
        "noise_prob": args.noise_prob,
        "workloads": list(args.workloads),
        "policies": list(args.policies),
        "semantic_mix": manifest_mix,
        "scope": "in-memory semantic CC workload benchmark; no persistence or recovery claims",
    }
    with open(os.path.join(RESULTS, "semantic_workloads_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved", os.path.join(RESULTS, "semantic_workloads_manifest.json"))


if __name__ == "__main__":
    main()
