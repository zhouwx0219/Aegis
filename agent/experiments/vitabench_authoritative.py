"""VitaBench environment-derived authoritative write workload.

This benchmark uses real VitaBench OTA task environments to collect shared
resources (flight seats, hotel rooms, attraction tickets, train seats) and to
verify that the official order tool mutates shared quantity fields. It then
runs a deterministic multi-threaded CC benchmark over those real resource IDs
and quantities.

Scope: this is a concurrency-control benchmark derived from VitaBench data and
tool semantics. It is not an end-to-end VitaBench task-success evaluation.
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
CATEGORIES = ("flights", "hotels", "attractions", "trains")
POLICIES = ("OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL", "merge-all")


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


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


@dataclass(frozen=True)
class Resource:
    oid: str
    category: str
    item_id: str
    product_id: str
    quantity: int
    price: float


@dataclass(frozen=True)
class BookingTask:
    candidates: Tuple[int, ...]
    quantity: int


def require_vitabench():
    try:
        from vita.domains.ota.environment import get_environment, get_tasks
        from deepdiff import DeepDiff
    except Exception as e:  # pragma: no cover - depends on external package
        raise SystemExit(
            "VitaBench/deepdiff is not installed. Run:\n"
            "  bash agent/integrations/setup_vitabench.sh\n"
            "Then rerun:\n"
            "  python3 agent/experiments/vitabench_authoritative.py\n"
            f"Original import error: {type(e).__name__}: {e}"
        )
    return get_environment, get_tasks, DeepDiff


def collect_resources(limit_per_category: int) -> Tuple[List[Resource], Dict[str, object]]:
    _, get_tasks, _ = require_vitabench()
    tasks = get_tasks("english")
    seen = set()
    resources: List[Resource] = []
    for task in tasks:
        env = task.environment
        for category in CATEGORIES:
            current = [r for r in resources if r.category == category]
            if len(current) >= limit_per_category:
                continue
            for item_id, obj in (env.get(category) or {}).items():
                for product in (obj.get("products") or []):
                    quantity = int(product.get("quantity", 0) or 0)
                    if quantity <= 0:
                        continue
                    product_id = str(product.get("product_id"))
                    oid = f"{category}:{item_id}:{product_id}"
                    if oid in seen:
                        continue
                    seen.add(oid)
                    resources.append(Resource(
                        oid=oid,
                        category=category,
                        item_id=str(item_id),
                        product_id=product_id,
                        quantity=quantity,
                        price=float(product.get("price", 0) or 0),
                    ))
                    if len([r for r in resources if r.category == category]) >= limit_per_category:
                        break
                if len([r for r in resources if r.category == category]) >= limit_per_category:
                    break
    by_cat = {c: sum(1 for r in resources if r.category == c) for c in CATEGORIES}
    manifest = {
        "source": "real VitaBench OTA task environments",
        "n_tasks_scanned": len(tasks),
        "resources_total": len(resources),
        "resources_by_category": by_cat,
    }
    return resources, manifest


def verify_quantity_decrement() -> Dict[str, object]:
    get_environment, get_tasks, DeepDiff = require_vitabench()
    tasks = get_tasks("english")
    task = next(t for t in tasks if t.environment.get("flights"))
    env = get_environment(task.environment, "english")
    db = env.tools.db
    flight_id, flight = next(iter(task.environment["flights"].items()))
    product = flight["products"][0]
    before = json.loads(db.model_dump_json())
    env.use_tool(
        "create_flight_order",
        flight_id=flight_id,
        seat_id=product["product_id"],
        user_id=task.environment.get("user_id", "U1"),
        date=str(product.get("date", "2026-08-01"))[:10],
        quantity=1,
    )
    after = json.loads(db.model_dump_json())
    diff = DeepDiff(before, after, verbose_level=2)
    quantity_changes = []
    for path, change in (diff.get("values_changed", {}) or {}).items():
        if "quantity" in str(path):
            old = change.get("old_value")
            new = change.get("new_value")
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) and new < old:
                quantity_changes.append({"path": str(path), "old": old, "new": new, "delta": new - old})
    return {
        "tool": "create_flight_order",
        "verified": bool(quantity_changes),
        "quantity_changes": quantity_changes[:5],
    }


def write_resources(resources: List[Resource]) -> None:
    rows = [{
        "oid": r.oid,
        "category": r.category,
        "item_id": r.item_id,
        "product_id": r.product_id,
        "quantity": r.quantity,
        "price": r.price,
    } for r in resources]
    write_csv(os.path.join(RESULTS, "vitabench_authoritative_resources.csv"), rows)


def make_tasks(
    resources: Sequence[Resource],
    *,
    n_tasks: int,
    k: int,
    seed: int,
    hot_per_category: int,
    hot_bias: float,
) -> List[BookingTask]:
    by_cat: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
    for idx, r in enumerate(resources):
        by_cat[r.category].append(idx)
    cats = [c for c, ids in by_cat.items() if ids]
    tasks: List[BookingTask] = []
    for tid in range(n_tasks):
        rng = stable_rng(seed, tid, 31)
        cat = cats[tid % len(cats)]
        pool = sorted(by_cat[cat], key=lambda idx: (resources[idx].price, resources[idx].oid))
        hot = pool[:max(1, min(hot_per_category, len(pool)))]
        base = hot if rng.random() < hot_bias else pool
        if len(base) >= k:
            cand = tuple(rng.sample(base, k))
        else:
            cand = tuple(base)
        tasks.append(BookingTask(candidates=cand, quantity=1))
    return tasks


def run_policy(
    policy: str,
    resources: Sequence[Resource],
    *,
    n_tasks: int,
    threads: int,
    k: int,
    seed: int,
    hot_per_category: int,
    hot_bias: float,
    capacity_multiplier: int,
    c_gen: float,
) -> Dict[str, float]:
    limit = 1 if policy in ("OCC-K1", "HYBRID-K1") else k
    tasks = make_tasks(
        resources,
        n_tasks=n_tasks,
        k=limit,
        seed=seed,
        hot_per_category=hot_per_category,
        hot_bias=hot_bias,
    )
    stock = [max(1, r.quantity * capacity_multiplier) for r in resources]
    versions = [0] * len(resources)
    locks = [threading.Lock() for _ in resources]
    store_lock = threading.Lock()
    q: queue.Queue[int] = queue.Queue()
    for i in range(n_tasks):
        q.put(i)

    stats = {
        "booked": 0,
        "no_stock": 0,
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
            task = tasks[tid]
            if not task.candidates:
                q.task_done()
                continue
            t0 = time.perf_counter()

            if policy == "2PL":
                rid = task.candidates[0]
                locks[rid].acquire()
                try:
                    time.sleep(c_gen)
                    with store_lock:
                        if stock[rid] >= task.quantity:
                            stock[rid] -= task.quantity
                            versions[rid] += 1
                            booked, no_stock = 1, 0
                        else:
                            booked, no_stock = 0, 1
                finally:
                    locks[rid].release()
                with stats_lock:
                    stats["booked"] += booked
                    stats["no_stock"] += no_stock
                    stats["latency"].append(time.perf_counter() - t0)
                q.task_done()
                continue

            with store_lock:
                base = {rid: versions[rid] for rid in task.candidates}
            time.sleep(c_gen)
            booked = no_stock = oversell = regen = reselect = merge = 0

            if policy == "merge-all":
                rid = task.candidates[0]
                with store_lock:
                    stock[rid] -= task.quantity
                    versions[rid] += 1
                    booked = 1
                    oversell = 1 if stock[rid] < 0 else 0
            elif policy in ("OCC-K1", "OCC+K", "MVCC"):
                committed = False
                with store_lock:
                    for idx, rid in enumerate(task.candidates):
                        if versions[rid] == base[rid] and stock[rid] >= task.quantity:
                            stock[rid] -= task.quantity
                            versions[rid] += 1
                            booked = 1
                            reselect = 1 if idx > 0 else 0
                            committed = True
                            break
                    if not committed and not any(stock[rid] >= task.quantity for rid in task.candidates):
                        no_stock = 1
                if not committed and not no_stock:
                    time.sleep(c_gen)
                    regen = 1
                    with store_lock:
                        rid = next((x for x in task.candidates if stock[x] >= task.quantity), None)
                        if rid is None:
                            no_stock = 1
                        else:
                            stock[rid] -= task.quantity
                            versions[rid] += 1
                            booked = 1
            else:  # HYBRID / HYBRID-K1: constrained delta merge with stock check
                with store_lock:
                    for idx, rid in enumerate(task.candidates):
                        if stock[rid] >= task.quantity:
                            if versions[rid] != base[rid]:
                                merge += 1
                            stock[rid] -= task.quantity
                            versions[rid] += 1
                            booked = 1
                            reselect = 1 if idx > 0 else 0
                            break
                    if not booked:
                        no_stock = 1

            with stats_lock:
                stats["booked"] += booked
                stats["no_stock"] += no_stock
                stats["oversell"] += oversell
                stats["regen"] += regen
                stats["reselect"] += reselect
                stats["merge"] += merge
                stats["latency"].append(time.perf_counter() - t0)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - start
    gen_calls = n_tasks + stats["regen"]
    lat = stats["latency"]
    return {
        "throughput": stats["booked"] / wall if wall else 0.0,
        "task_throughput": n_tasks / wall if wall else 0.0,
        "latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
        "booked": float(stats["booked"]),
        "no_stock": float(stats["no_stock"]),
        "oversell": float(stats["oversell"]),
        "regen": float(stats["regen"]),
        "reselect": float(stats["reselect"]),
        "merge": float(stats["merge"]),
        "regen_per_task": float(stats["regen"]) / max(1, n_tasks),
        "generation_calls_per_task": gen_calls / max(1, n_tasks),
        "booked_per_generation_call": float(stats["booked"]) / max(1, gen_calls),
        "wall_s": wall,
    }


def aggregate_runs(runs: Iterable[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    runs = list(runs)
    out: Dict[str, float] = {}
    for metric in metrics:
        avg, ci = mean_ci([r[metric] for r in runs])
        out[metric] = avg
        out[metric + "_ci"] = ci
    return out


def run_benchmark(args, resources: List[Resource]) -> List[Dict[str, object]]:
    metrics = [
        "throughput",
        "task_throughput",
        "latency_ms",
        "booked",
        "no_stock",
        "oversell",
        "regen",
        "reselect",
        "merge",
        "regen_per_task",
        "generation_calls_per_task",
        "booked_per_generation_call",
    ]
    rows: List[Dict[str, object]] = []
    for policy in POLICIES:
        print(f"[vitabench-authoritative] {policy}")
        runs = [
            run_policy(
                policy,
                resources,
                n_tasks=args.tasks,
                threads=args.threads,
                k=args.k,
                seed=s,
                hot_per_category=args.hot_per_category,
                hot_bias=args.hot_bias,
                capacity_multiplier=args.capacity_multiplier,
                c_gen=args.c_gen,
            )
            for s in args.seeds
        ]
        agg = aggregate_runs(runs, metrics)
        row: Dict[str, object] = {
            "policy": policy,
            "n_tasks": args.tasks,
            "threads": args.threads,
            "k": args.k,
            "hot_per_category": args.hot_per_category,
            "hot_bias": args.hot_bias,
            "capacity_multiplier": args.capacity_multiplier,
            "c_gen": args.c_gen,
            **{k: round(v, 4) for k, v in agg.items()},
        }
        rows.append(row)
        print(
            f"  tp={agg['throughput']:.1f} gen/task={agg['generation_calls_per_task']:.3f} "
            f"regen={agg['regen']:.0f} oversell={agg['oversell']:.0f}"
        )

    by_policy = {str(r["policy"]): r for r in rows}
    base_occ = float(by_policy["OCC-K1"]["throughput"])
    fair_occ = float(by_policy["OCC+K"]["throughput"])
    unsafe = float(by_policy["merge-all"]["oversell"])
    for row in rows:
        tp = float(row["throughput"])
        row["speedup_vs_occ_k1"] = round(tp / base_occ, 4) if base_occ else 0.0
        row["speedup_vs_occ_k"] = round(tp / fair_occ, 4) if fair_occ else 0.0
        row["unsafe_oversell_reference"] = unsafe
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--limit-per-category", type=int, default=24)
    ap.add_argument("--hot-per-category", type=int, default=6)
    ap.add_argument("--hot-bias", type=float, default=0.85)
    ap.add_argument("--capacity-multiplier", type=int, default=20)
    ap.add_argument("--c-gen", type=float, default=0.002)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    resources, manifest = collect_resources(args.limit_per_category)
    if not resources:
        raise SystemExit("No VitaBench OTA resources with positive quantity were collected.")
    verification = verify_quantity_decrement()
    write_resources(resources)
    rows = run_benchmark(args, resources)
    write_csv(os.path.join(RESULTS, "vitabench_authoritative.csv"), rows)
    manifest.update({
        "benchmark": {
            "tasks": args.tasks,
            "threads": args.threads,
            "k": args.k,
            "seeds": args.seeds,
            "hot_per_category": args.hot_per_category,
            "hot_bias": args.hot_bias,
            "capacity_multiplier": args.capacity_multiplier,
            "c_gen": args.c_gen,
            "policies": list(POLICIES),
        },
        "quantity_decrement_verification": verification,
    })
    with open(os.path.join(RESULTS, "vitabench_authoritative_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("saved", os.path.join(RESULTS, "vitabench_authoritative_manifest.json"))
    print("quantity decrement verified:", verification["verified"])


if __name__ == "__main__":
    main()
