"""Real LLM closed-loop semantic workload experiment.

Pipeline:

1. LLM creates user tasks.
2. LLM proposes K candidate plans for each task.
3. The database/runtime materializes those candidates as speculative branches.
4. LLM selects the preferred winner/ranking.
5. Policies replay the same trace:
   branch-txn, OCC, MVCC, TicToc, Silo, 2PL, merge-all, HYBRID.

`OCC+K` is intentionally not part of the main traditional-baseline table. It is
an agent-aware ablation, not a traditional CC scheme.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.llm import deepseek_client as ds
from agent.runtime import AgentTransactionManager


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "llm_semantic_closed_loop_cache.json")
RUNS_CSV = os.path.join(RESULTS, "llm_semantic_closed_loop_runs.csv")
SUMMARY_CSV = os.path.join(RESULTS, "llm_semantic_closed_loop_summary.csv")
SUMMARY_MD = os.path.join(RESULTS, "llm_semantic_closed_loop_report.md")
MANIFEST = os.path.join(RESULTS, "llm_semantic_closed_loop_manifest.json")

WORKLOADS = ("append_log", "cas_claim", "mixed_checkout", "private_strict")
POLICIES = ("branch-txn", "OCC", "MVCC", "TicToc", "Silo", "2PL", "merge-all", "HYBRID")
TRADITIONAL_SAFE = ("branch-txn", "OCC", "MVCC", "TicToc", "Silo", "2PL")


@dataclass
class Obj:
    value: str
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
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def safe_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}


def call_llm(messages: List[Dict[str, str]], *, model: str, mock: bool, seed: int, purpose: str) -> Dict[str, Any]:
    if mock or not ds.have_key():
        rng = stable_rng(seed, len(purpose))
        return {
            "text": "{}",
            "latency_s": 0.0,
            "usage": {"total_tokens": 0},
            "mock_random": rng.random(),
        }
    return ds.chat(messages, model=model, temperature=0.25, max_tokens=900, response_json=True, timeout=90)


def build_catalog(tasks_per_workload: int, k: int, threads: int) -> Dict[str, Obj]:
    store: Dict[str, Obj] = {}
    for i in range(64):
        store[f"audit:{i}"] = Obj("base")
        store[f"cart:{i}"] = Obj("base")
        store[f"pref:{i}"] = Obj("v0")
    for i in range(64):
        store[f"stock:{i}"] = Obj("500")
    group_size = max(8, threads)
    groups = (tasks_per_workload * len(WORKLOADS) + group_size - 1) // group_size
    slots_per_group = max(k * 4, 18)
    for gid in range(groups):
        for sid in range(slots_per_group):
            store[f"slot:{gid}:{sid}"] = Obj("free")
    # Coupon ids use the global task id after all workload batches are merged,
    # so over-provision by the number of supported workload families.
    for tid in range(tasks_per_workload * len(WORKLOADS)):
        store[f"order:{tid}"] = Obj("new")
        for cid in range(k):
            store[f"coupon:{tid}:{cid}"] = Obj("unused")
    return store


def workload_options(workload: str, task_index: int, k: int, threads: int) -> Dict[str, Any]:
    rng = stable_rng(17, task_index, len(workload))
    if workload == "append_log":
        return {
            "audit_logs": [f"audit:{i}" for i in range(8)],
            "carts": [f"cart:{(task_index + i) % 16}" for i in range(8)],
        }
    if workload == "cas_claim":
        group_size = max(8, threads)
        gid = task_index // group_size
        slots = [f"slot:{gid}:{i}" for i in range(max(k * 4, 18))]
        return {"slots": slots[:max(k * 2, 8)]}
    if workload == "mixed_checkout":
        hot = [f"stock:{i}" for i in range(8)]
        rng.shuffle(hot)
        return {
            "products": hot,
            "cart": f"cart:{task_index % 16}",
            "audit_logs": [f"audit:{i}" for i in range(6)],
            "coupons": [f"coupon:{task_index}:{i}" for i in range(k)],
            "pref": f"pref:{task_index % 16}",
        }
    if workload == "private_strict":
        return {
            "order": f"order:{task_index}",
            "profile": f"pref:{task_index % 64}",
        }
    raise ValueError(workload)


def generate_task_batch(workload: str, n: int, *, model: str, mock: bool, batch_seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if mock or not ds.have_key():
        tasks = [
            {
                "request_id": f"{workload}-{i}",
                "user_request": f"{workload} request {i}",
                "preference": ["fast", "cheap", "reliable"][i % 3],
            }
            for i in range(n)
        ]
        return tasks, {"latency_s": 0.0, "usage": {"total_tokens": 0}, "raw": "MOCK"}

    prompt = (
        f"Generate {n} concise user requests for an agent benchmark workload named {workload}. "
        "Return strict JSON: {\"tasks\":[{\"request_id\":\"...\",\"user_request\":\"...\","
        "\"preference\":\"...\"}]}. Keep requests realistic and varied."
    )
    resp = call_llm(
        [
            {"role": "system", "content": "You create benchmark user tasks as strict JSON."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        mock=mock,
        seed=batch_seed,
        purpose=f"task:{workload}",
    )
    obj = safe_json(resp["text"])
    tasks = obj.get("tasks", []) if isinstance(obj, dict) else []
    out: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks[:n]):
        out.append({
            "request_id": str(task.get("request_id") or f"{workload}-{i}"),
            "user_request": str(task.get("user_request") or f"{workload} request {i}"),
            "preference": str(task.get("preference") or "balanced"),
        })
    while len(out) < n:
        i = len(out)
        out.append({
            "request_id": f"{workload}-{i}",
            "user_request": f"{workload} fallback request {i}",
            "preference": "balanced",
        })
    return out, {"latency_s": resp["latency_s"], "usage": resp.get("usage", {}), "raw": resp["text"]}


def candidate_prompt(workload: str, task: Dict[str, Any], options: Dict[str, Any], k: int) -> str:
    schema = {
        "append_log": "candidate_id, audit_log_id, cart_id, event_label, reason",
        "cas_claim": "candidate_id, slot_id, reason",
        "mixed_checkout": "candidate_id, product_id, audit_log_id, coupon_id, update_pref, reason",
        "private_strict": "candidate_id, order_value, reason",
    }[workload]
    return (
        f"User request: {task['user_request']}\nPreference: {task['preference']}\n"
        f"Workload: {workload}\nAllowed resources:\n{json.dumps(options, ensure_ascii=False)}\n"
        f"Return up to {k} distinct candidate plans. Use only allowed resource IDs. "
        f"Return strict JSON: {{\"candidates\":[{{{schema}}}]}}."
    )


def fallback_candidates(workload: str, task_id: int, options: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(k):
        cid = f"c{i}"
        if workload == "append_log":
            out.append({
                "candidate_id": cid,
                "audit_log_id": options["audit_logs"][i % len(options["audit_logs"])],
                "cart_id": options["carts"][i % len(options["carts"])],
                "event_label": f"event-{task_id}-{i}",
                "reason": "fallback append candidate",
            })
        elif workload == "cas_claim":
            out.append({
                "candidate_id": cid,
                "slot_id": options["slots"][i % len(options["slots"])],
                "reason": "fallback slot candidate",
            })
        elif workload == "mixed_checkout":
            out.append({
                "candidate_id": cid,
                "product_id": options["products"][i % len(options["products"])],
                "audit_log_id": options["audit_logs"][i % len(options["audit_logs"])],
                "coupon_id": options["coupons"][i % len(options["coupons"])],
                "update_pref": i == 0,
                "reason": "fallback checkout candidate",
            })
        else:
            out.append({
                "candidate_id": cid,
                "order_value": f"confirmed-{task_id}-{i}",
                "reason": "fallback private strict candidate",
            })
    return out


def normalize_candidates(workload: str, raw: List[Dict[str, Any]], options: Dict[str, Any], k: int, task_id: int) -> List[Dict[str, Any]]:
    fallback = fallback_candidates(workload, task_id, options, k)
    allowed = {v for value in options.values() for v in (value if isinstance(value, list) else [value])}
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, candidate in enumerate(raw):
        c = dict(candidate)
        cid = str(c.get("candidate_id") or f"c{i}")
        c["candidate_id"] = cid
        ok = True
        if workload == "append_log":
            ok = c.get("audit_log_id") in allowed and c.get("cart_id") in allowed
            c.setdefault("event_label", f"event-{task_id}-{i}")
        elif workload == "cas_claim":
            ok = c.get("slot_id") in allowed
        elif workload == "mixed_checkout":
            ok = c.get("product_id") in allowed and c.get("audit_log_id") in allowed and c.get("coupon_id") in allowed
            c["update_pref"] = bool(c.get("update_pref", False))
        elif workload == "private_strict":
            c.setdefault("order_value", f"confirmed-{task_id}-{i}")
        key = json.dumps(c, sort_keys=True)
        if ok and key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= k:
            break
    for candidate in fallback:
        if len(out) >= k:
            break
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out[:k]


def build_ops(workload: str, task_id: int, options: Dict[str, Any], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    if workload == "append_log":
        label = str(candidate.get("event_label") or f"event-{task_id}")
        return [
            {"kind": "read", "oid": f"pref:{task_id % 64}"},
            {"kind": "append", "oid": candidate["audit_log_id"], "payload": f"|audit:{label}", "commutative": True},
            {"kind": "append", "oid": candidate["cart_id"], "payload": f"|cart:{label}", "commutative": True},
        ]
    if workload == "cas_claim":
        return [
            {"kind": "cas", "oid": candidate["slot_id"], "expected": "free", "value": "taken"},
        ]
    if workload == "mixed_checkout":
        ops = [
            {"kind": "read", "oid": options["pref"]},
            {"kind": "delta", "oid": candidate["product_id"], "amount": -1, "lower_bound": 0},
            {"kind": "append", "oid": options["cart"], "payload": f"|item:{task_id}:{candidate['candidate_id']}", "commutative": True},
            {"kind": "append", "oid": candidate["audit_log_id"], "payload": f"|evt:{task_id}:{candidate['candidate_id']}", "commutative": True},
            {"kind": "cas", "oid": candidate["coupon_id"], "expected": "unused", "value": "used"},
        ]
        if bool(candidate.get("update_pref")):
            ops.append({"kind": "overwrite", "oid": options["pref"], "value": f"pref:{task_id}:{candidate['candidate_id']}"})
        return ops
    if workload == "private_strict":
        return [
            {"kind": "read", "oid": options["profile"]},
            {"kind": "overwrite", "oid": options["order"], "value": str(candidate.get("order_value") or f"confirmed-{task_id}")},
        ]
    raise ValueError(workload)


def generate_candidates_and_winner(
    task: Dict[str, Any],
    *,
    model: str,
    mock: bool,
    k: int,
    threads: int,
) -> Dict[str, Any]:
    workload = task["workload"]
    task_id = int(task["task_id"])
    option_task_id = int(task.get("local_task_id", task_id)) if workload == "cas_claim" else task_id
    options = workload_options(workload, option_task_id, k, threads)

    if mock or not ds.have_key():
        candidates = fallback_candidates(workload, task_id, options, k)
        cand_meta = {"latency_s": 0.0, "usage": {"total_tokens": 0}, "raw": "MOCK"}
    else:
        resp = call_llm(
            [
                {"role": "system", "content": "You are an agent that proposes executable candidate plans as strict JSON."},
                {"role": "user", "content": candidate_prompt(workload, task, options, k)},
            ],
            model=model,
            mock=mock,
            seed=task_id,
            purpose="candidates",
        )
        obj = safe_json(resp["text"])
        candidates = normalize_candidates(workload, obj.get("candidates", []), options, k, task_id)
        cand_meta = {"latency_s": resp["latency_s"], "usage": resp.get("usage", {}), "raw": resp["text"]}

    if mock or not ds.have_key():
        ranked = [c["candidate_id"] for c in candidates]
        win_meta = {"latency_s": 0.0, "usage": {"total_tokens": 0}, "raw": "MOCK"}
    else:
        winner_prompt = (
            f"User request: {task['user_request']}\nPreference: {task['preference']}\n"
            f"Candidate plans:\n{json.dumps(candidates, ensure_ascii=False)}\n"
            "The database has materialized these as speculative branches. Pick the best winner and fallback order. "
            "Return strict JSON: {\"winner\":\"candidate_id\",\"ranked\":[\"candidate_id\",...]}"
        )
        resp = call_llm(
            [
                {"role": "system", "content": "You choose a winner among already generated candidates. Return strict JSON."},
                {"role": "user", "content": winner_prompt},
            ],
            model=model,
            mock=mock,
            seed=task_id + 100000,
            purpose="winner",
        )
        obj = safe_json(resp["text"])
        ranked = [str(x) for x in obj.get("ranked", []) if str(x) in {c["candidate_id"] for c in candidates}]
        winner = str(obj.get("winner") or "")
        if winner and winner in {c["candidate_id"] for c in candidates} and winner not in ranked:
            ranked.insert(0, winner)
        for c in candidates:
            if c["candidate_id"] not in ranked:
                ranked.append(c["candidate_id"])
        win_meta = {"latency_s": resp["latency_s"], "usage": resp.get("usage", {}), "raw": resp["text"]}

    by_id = {c["candidate_id"]: c for c in candidates}
    ranked_candidates = [by_id[cid] for cid in ranked if cid in by_id]
    candidates_with_ops = []
    for rank, candidate in enumerate(ranked_candidates):
        c = dict(candidate)
        c["rank"] = rank
        c["ops"] = build_ops(workload, task_id, options, c)
        candidates_with_ops.append(c)

    return {
        **task,
        "options": options,
        "candidates": candidates_with_ops,
        "winner": ranked_candidates[0]["candidate_id"] if ranked_candidates else "",
        "ranked": [c["candidate_id"] for c in ranked_candidates],
        "candidate_call": cand_meta,
        "winner_call": win_meta,
        "c_gen": float(cand_meta["latency_s"]) + float(win_meta["latency_s"]) + float(task.get("task_gen_latency_s", 0.0)),
        "usage": {
            "total_tokens": int(cand_meta.get("usage", {}).get("total_tokens", 0) or 0)
            + int(win_meta.get("usage", {}).get("total_tokens", 0) or 0)
            + int(task.get("task_gen_tokens", 0) or 0)
        },
    }


def register_runtime_objects(runtime: AgentTransactionManager, store: Dict[str, Obj]) -> None:
    for oid, obj in store.items():
        if oid.startswith(("stock:",)):
            kind = "counter"
        elif oid.startswith(("audit:", "cart:")):
            kind = "text"
        else:
            kind = "generic"
        runtime.register_object(oid, obj.value, kind=kind)


def add_record_to_txn(txn: Any, record: Dict[str, Any], model: str) -> None:
    txn.record_model_call(
        model=model,
        latency_s=float(record.get("c_gen", 0.0)),
        usage=record.get("usage", {}),
        candidates=len(record.get("candidates", [])),
    )
    txn.record_tool_call("materialize_candidates", args={"workload": record["workload"]}, outcome="prepared")
    total = len(record.get("candidates", []))
    for idx, candidate in enumerate(record.get("candidates", [])):
        branch = txn.add_candidate(
            f"{record['task_id']}:{candidate['candidate_id']}",
            quality=float(total - idx),
            gen_cost=float(record.get("c_gen", 0.0)),
            metadata={"workload": record["workload"], "candidate_id": candidate["candidate_id"]},
        )
        for op in candidate["ops"]:
            kind = op["kind"]
            if kind == "read":
                continue
            if kind == "append":
                branch.append(op["oid"], op["payload"], commutative=bool(op.get("commutative", False)))
            elif kind == "delta":
                branch.delta(op["oid"], int(op["amount"]), constrained=True, lower_bound=int(op.get("lower_bound", 0)))
            elif kind == "cas":
                branch.cas(op["oid"], op["expected"], op["value"])
            elif kind == "overwrite":
                branch.overwrite(op["oid"], op["value"])


def run_live_hybrid(records: List[Dict[str, Any]], initial_store: Dict[str, Obj], *, threads: int, model: str) -> Dict[str, Any]:
    runtime = AgentTransactionManager(c_gen=1.0, c_merge=0.0)
    register_runtime_objects(runtime, initial_store)
    q: queue.Queue[Dict[str, Any]] = queue.Queue()
    for record in records:
        q.put(record)
    results = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                record = q.get_nowait()
            except queue.Empty:
                return
            txn = runtime.begin(record["task_id"], {"workload": record["workload"], "request": record["user_request"]})
            add_record_to_txn(txn, record, model)
            result = txn.commit("cast") if txn.candidates else txn.abort("no candidates")
            with lock:
                results.append(result)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - start
    committed = sum(1 for r in results if r.committed)
    values = runtime.values()
    oversell = sum(1 for k, v in values.items() if k.startswith("stock:") and int(v) < 0)
    lat = [r.elapsed_s for r in results]
    return {
        "throughput": committed / wall if wall else 0.0,
        "wall_s": wall,
        "committed": committed,
        "rejected": sum(1 for r in results if r.rejected),
        "regen": sum(r.n_regen for r in results),
        "merge": sum(r.n_merge for r in results),
        "reselect": sum(r.n_reselect for r in results),
        "oversell": oversell,
        "mean_latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
        "trace_count": len(runtime.traces()),
    }


def copy_store(store: Dict[str, Obj]) -> Dict[str, Obj]:
    return {k: Obj(v.value, v.version) for k, v in store.items()}


def default_obj_for_oid(oid: str) -> Obj:
    if oid.startswith("stock:"):
        return Obj("500")
    if oid.startswith(("audit:", "cart:")):
        return Obj("base")
    if oid.startswith("slot:"):
        return Obj("free")
    if oid.startswith("coupon:"):
        return Obj("unused")
    if oid.startswith("order:"):
        return Obj("new")
    if oid.startswith("pref:"):
        return Obj("v0")
    return Obj("0")


def ensure_store_covers_records(store: Dict[str, Obj], records: Sequence[Dict[str, Any]]) -> Dict[str, Obj]:
    for record in records:
        for candidate in record.get("candidates", []):
            for op in candidate.get("ops", []):
                oid = op["oid"]
                if oid not in store:
                    store[oid] = default_obj_for_oid(oid)
    return store


def op_oids(candidate: Dict[str, Any], include_reads: bool = True) -> List[str]:
    out = []
    for op in candidate["ops"]:
        if op["kind"] == "read" and not include_reads:
            continue
        out.append(op["oid"])
    return out


def write_ops(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [op for op in candidate["ops"] if op["kind"] != "read"]


def feasible(candidate: Dict[str, Any], store: Dict[str, Obj]) -> bool:
    for op in write_ops(candidate):
        obj = store[op["oid"]]
        if op["kind"] == "delta" and int(obj.value) + int(op["amount"]) < int(op.get("lower_bound", 0)):
            return False
        if op["kind"] == "cas" and obj.value != op["expected"]:
            return False
    return True


def apply_candidate(candidate: Dict[str, Any], store: Dict[str, Obj], *, unsafe: bool = False) -> Dict[str, int]:
    local = {"oversell": 0, "cas_violation": 0}
    for op in write_ops(candidate):
        obj = store[op["oid"]]
        if op["kind"] == "append":
            obj.value = obj.value + op["payload"]
        elif op["kind"] == "delta":
            obj.value = str(int(obj.value) + int(op["amount"]))
            if int(obj.value) < int(op.get("lower_bound", 0)):
                local["oversell"] += 1
        elif op["kind"] == "cas":
            if obj.value != op["expected"] and unsafe:
                local["cas_violation"] += 1
            obj.value = op["value"]
        elif op["kind"] == "overwrite":
            obj.value = op["value"]
        obj.version += 1
    return local


def aborts(policy: str, candidate: Dict[str, Any], base: Dict[str, int], store: Dict[str, Obj]) -> bool:
    if policy in ("OCC", "Silo", "branch-txn"):
        return any(store[oid].version != ver for oid, ver in base.items())
    if policy in ("MVCC", "TicToc"):
        write_ids = {op["oid"] for op in write_ops(candidate)}
        return any(store[oid].version != ver for oid, ver in base.items() if oid in write_ids)
    return False


def run_baseline(
    policy: str,
    records: List[Dict[str, Any]],
    initial_store: Dict[str, Obj],
    *,
    threads: int,
    speed: float,
    seed: int,
) -> Dict[str, float]:
    store = copy_store(initial_store)
    locks = {oid: threading.Lock() for oid in store}
    store_lock = threading.Lock()
    q: queue.Queue[Dict[str, Any]] = queue.Queue()
    for record in records:
        q.put(record)
    stats = {
        "committed": 0,
        "rejected": 0,
        "regen": 0,
        "reselect": 0,
        "oversell": 0,
        "cas_violation": 0,
        "lat_ms": [],
    }
    stats_lock = threading.Lock()

    def choose_after_regen(record: Dict[str, Any], store_ref: Dict[str, Obj]) -> Optional[Dict[str, Any]]:
        # This represents a traditional expensive regeneration call. It may find
        # a feasible plan, but it pays another c_gen.
        for candidate in record["candidates"]:
            if feasible(candidate, store_ref):
                return candidate
        return None

    def finish(t0: float, local: Dict[str, int]) -> None:
        elapsed = (time.perf_counter() - t0) * 1000
        with stats_lock:
            for key, val in local.items():
                stats[key] += val
            stats["lat_ms"].append(elapsed)

    def worker() -> None:
        while True:
            try:
                record = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            local = {k: 0 for k in ("committed", "rejected", "regen", "reselect", "oversell", "cas_violation")}
            if not record.get("candidates"):
                local["rejected"] = 1
                finish(t0, local)
                q.task_done()
                continue
            winner = record["candidates"][0]
            sleep_s = max(0.0, float(record.get("c_gen", 0.0)) / speed)

            if policy == "2PL":
                ids = sorted(set(op_oids(winner, include_reads=True)))
                held = [locks[oid] for oid in ids]
                for lock in held:
                    lock.acquire()
                try:
                    time.sleep(sleep_s)
                    with store_lock:
                        if feasible(winner, store):
                            changed = apply_candidate(winner, store)
                            local["committed"] = 1
                            local["oversell"] += changed["oversell"]
                            local["cas_violation"] += changed["cas_violation"]
                        else:
                            local["rejected"] = 1
                finally:
                    for lock in reversed(held):
                        lock.release()
                finish(t0, local)
                q.task_done()
                continue

            if policy == "merge-all":
                time.sleep(sleep_s)
                with store_lock:
                    changed = apply_candidate(winner, store, unsafe=True)
                    local["committed"] = 1
                    local["oversell"] += changed["oversell"]
                    local["cas_violation"] += changed["cas_violation"]
                finish(t0, local)
                q.task_done()
                continue

            with store_lock:
                base = {oid: store[oid].version for oid in op_oids(winner, include_reads=True)}
            time.sleep(sleep_s)
            conflict = False
            with store_lock:
                conflict = aborts(policy, winner, base, store)
                if not conflict and feasible(winner, store):
                    changed = apply_candidate(winner, store)
                    local["committed"] = 1
                    local["oversell"] += changed["oversell"]
                    local["cas_violation"] += changed["cas_violation"]
                elif not conflict:
                    local["rejected"] = 1
            if conflict:
                time.sleep(sleep_s)
                with store_lock:
                    local["regen"] = 1
                    candidate = choose_after_regen(record, store)
                    if candidate is None:
                        local["rejected"] = 1
                    else:
                        if candidate["candidate_id"] != winner["candidate_id"]:
                            # This is after an expensive regeneration, not a free OCC+K reselect.
                            local["reselect"] = 1
                        changed = apply_candidate(candidate, store)
                        local["committed"] = 1
                        local["oversell"] += changed["oversell"]
                        local["cas_violation"] += changed["cas_violation"]
            finish(t0, local)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - start
    lat = [float(v) for v in stats["lat_ms"]]
    return {
        "throughput": stats["committed"] / wall if wall else 0.0,
        "wall_s": wall,
        "committed": float(stats["committed"]),
        "rejected": float(stats["rejected"]),
        "regen": float(stats["regen"]),
        "reselect": float(stats["reselect"]),
        "oversell": float(stats["oversell"]),
        "cas_violation": float(stats["cas_violation"]),
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "p95_latency_ms": percentile(lat, 95),
        "generation_calls_per_task": (len(records) + stats["regen"]) / max(1, len(records)),
    }


def run_hybrid_replay(
    records: List[Dict[str, Any]],
    initial_store: Dict[str, Obj],
    *,
    threads: int,
    speed: float,
    seed: int,
    model: str,
) -> Dict[str, float]:
    runtime = AgentTransactionManager(c_gen=1.0, c_merge=0.0)
    register_runtime_objects(runtime, initial_store)
    q: queue.Queue[Dict[str, Any]] = queue.Queue()
    for record in records:
        q.put(record)
    results = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                record = q.get_nowait()
            except queue.Empty:
                return
            txn = runtime.begin(f"replay-{seed}-{record['task_id']}", {"workload": record["workload"]})
            time.sleep(max(0.0, float(record.get("c_gen", 0.0)) / speed))
            add_record_to_txn(txn, record, model)
            result = txn.commit("cast") if txn.candidates else txn.abort("no candidates")
            with lock:
                results.append(result)
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - start
    committed = sum(1 for r in results if r.committed)
    values = runtime.values()
    oversell = sum(1 for k, v in values.items() if k.startswith("stock:") and int(v) < 0)
    lat = [r.elapsed_s * 1000 for r in results]
    regen = sum(r.n_regen for r in results)
    return {
        "throughput": committed / wall if wall else 0.0,
        "wall_s": wall,
        "committed": float(committed),
        "rejected": float(sum(1 for r in results if r.rejected)),
        "regen": float(regen),
        "reselect": float(sum(r.n_reselect for r in results)),
        "merge": float(sum(r.n_merge for r in results)),
        "oversell": float(oversell),
        "cas_violation": 0.0,
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "p95_latency_ms": percentile(lat, 95),
        "generation_calls_per_task": (len(records) + regen) / max(1, len(records)),
    }


def generate_live_trace(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(RESULTS, exist_ok=True)
    initial_store = build_catalog(args.tasks_per_workload, args.k, args.threads)
    all_tasks = []
    task_generation = []
    task_id = 0
    for workload in args.workloads:
        tasks, meta = generate_task_batch(workload, args.tasks_per_workload, model=args.model, mock=args.mock, batch_seed=task_id)
        task_generation.append({"workload": workload, **meta})
        latency_each = float(meta.get("latency_s", 0.0)) / max(1, len(tasks))
        tokens_each = int(meta.get("usage", {}).get("total_tokens", 0) or 0) / max(1, len(tasks))
        for i, task in enumerate(tasks):
            all_tasks.append({
                "task_id": task_id,
                "local_task_id": i,
                "workload": workload,
                "request_id": task["request_id"],
                "user_request": task["user_request"],
                "preference": task["preference"],
                "task_gen_latency_s": latency_each,
                "task_gen_tokens": tokens_each,
            })
            task_id += 1

    q: queue.Queue[Dict[str, Any]] = queue.Queue()
    for task in all_tasks:
        q.put(task)
    records: List[Optional[Dict[str, Any]]] = [None] * len(all_tasks)
    errors = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                return
            try:
                record = generate_candidates_and_winner(task, model=args.model, mock=args.mock, k=args.k, threads=args.threads)
                records[int(task["task_id"])] = record
            except Exception as exc:  # noqa
                with lock:
                    errors.append({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                fallback = dict(task)
                fallback["candidates"] = []
                fallback["winner"] = ""
                fallback["ranked"] = []
                fallback["c_gen"] = float(task.get("task_gen_latency_s", 0.0))
                fallback["usage"] = {"total_tokens": int(task.get("task_gen_tokens", 0) or 0)}
                records[int(task["task_id"])] = fallback
            q.task_done()

    start = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(args.llm_concurrency)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    live_wall = time.perf_counter() - start
    good_records = [r for r in records if r is not None]
    ensure_store_covers_records(initial_store, good_records)
    live_hybrid = run_live_hybrid(good_records, initial_store, threads=args.threads, model=args.model)
    cache = {
        "records": good_records,
        "initial_store": {k: {"value": v.value, "version": v.version} for k, v in initial_store.items()},
        "task_generation": task_generation,
        "errors": errors,
        "live_wall_s": live_wall,
        "live_hybrid": live_hybrid,
        "model": args.model,
        "mock": args.mock,
        "params": vars(args),
    }
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    return cache


def load_initial_store(cache: Dict[str, Any]) -> Dict[str, Obj]:
    store = {k: Obj(str(v["value"]), int(v.get("version", 0))) for k, v in cache["initial_store"].items()}
    return ensure_store_covers_records(store, cache.get("records", []))


def aggregate_runs(rows: Iterable[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    rows = list(rows)
    out: Dict[str, float] = {}
    for metric in metrics:
        avg, ci = mean_ci([r[metric] for r in rows])
        out[metric] = avg
        out[metric + "_ci"] = ci
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def replay(cache: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = cache["records"]
    initial_store = load_initial_store(cache)
    metrics = [
        "throughput",
        "committed",
        "rejected",
        "regen",
        "reselect",
        "oversell",
        "cas_violation",
        "mean_latency_ms",
        "p95_latency_ms",
        "generation_calls_per_task",
        "wall_s",
    ]
    detailed: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for policy in args.policies:
        runs = []
        for seed in range(1, args.replay_seeds + 1):
            if policy == "HYBRID":
                result = run_hybrid_replay(records, initial_store, threads=args.threads, speed=args.speed, seed=seed, model=cache["model"])
            else:
                result = run_baseline(policy, records, initial_store, threads=args.threads, speed=args.speed, seed=seed)
            runs.append(result)
            detailed.append({
                "policy": policy,
                "seed": seed,
                "n_tasks": len(records),
                "threads": args.threads,
                "speed": args.speed,
                **{m: round(float(result.get(m, 0.0)), 6) for m in metrics},
                "merge": round(float(result.get("merge", 0.0)), 6),
            })
        agg = aggregate_runs(runs, metrics)
        merge_avg, merge_ci = mean_ci([r.get("merge", 0.0) for r in runs])
        summary.append({
            "policy": policy,
            "n_tasks": len(records),
            "threads": args.threads,
            "speed": args.speed,
            "replay_seeds": args.replay_seeds,
            **{k: round(float(v), 6) for k, v in agg.items()},
            "merge": round(merge_avg, 6),
            "merge_ci": round(merge_ci, 6),
        })

    by = {row["policy"]: row for row in summary}
    safe = [p for p in TRADITIONAL_SAFE if p in by]
    best_safe = max(safe, key=lambda p: float(by[p]["throughput"]))
    best_tp = float(by[best_safe]["throughput"])
    branch_tp = float(by.get("branch-txn", by[best_safe])["throughput"])
    for row in summary:
        tp = float(row["throughput"])
        row["speedup_vs_best_traditional"] = round(tp / max(1e-9, best_tp), 4)
        row["speedup_vs_branch_txn"] = round(tp / max(1e-9, branch_tp), 4)
        row["best_traditional_policy"] = best_safe
    os.makedirs(RESULTS, exist_ok=True)
    write_csv(RUNS_CSV, detailed)
    write_csv(SUMMARY_CSV, summary)
    return detailed, summary


def write_report(cache: Dict[str, Any], summary: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    records = cache["records"]
    lat = [float(r.get("c_gen", 0.0)) for r in records if float(r.get("c_gen", 0.0)) > 0]
    tokens = sum(int(r.get("usage", {}).get("total_tokens", 0) or 0) for r in records)
    intent_counts: Dict[str, int] = {}
    for record in records:
        for candidate in record.get("candidates", []):
            for op in candidate.get("ops", []):
                intent_counts[op["kind"]] = intent_counts.get(op["kind"], 0) + 1
    by = {row["policy"]: row for row in summary}
    hy = by["HYBRID"]
    best = hy["best_traditional_policy"]
    api_calls = len(cache.get("task_generation", [])) + 2 * len(records)
    lines = [
        "# Real LLM Semantic Closed-Loop Experiment",
        "",
        "This experiment runs the full loop: LLM task creation, LLM candidate generation, DB candidate materialization, LLM winner selection, and final commit.",
        "",
        "## Parameters",
        "",
        "Reproduction commands:",
        "",
        "```bash",
        "bash build.sh",
        "export DEEPSEEK_API_KEY=...",
        "python3 agent/experiments/llm_semantic_closed_loop.py all --tasks-per-workload 60 --k 4 --llm-concurrency 12 --threads 32 --replay-seeds 5 --speed 20",
        "```",
        "",
        f"- Model: `{cache['model']}`",
        f"- Workloads: `{', '.join(args.workloads)}`",
        f"- Tasks per workload: `{args.tasks_per_workload}`",
        f"- Total tasks: `{len(records)}`",
        f"- K candidates: `{args.k}`",
        f"- LLM concurrency: `{args.llm_concurrency}`",
        f"- Replay policies: `{', '.join(args.policies)}`",
        f"- Replay seeds: `{args.replay_seeds}`",
        f"- Replay threads: `{args.threads}`",
        f"- Replay speed: `{args.speed}`",
        "",
        "## LLM Trace",
        "",
        f"- Real API calls: `{api_calls}` (`{len(cache.get('task_generation', []))}` task-batch calls + `{len(records)}` candidate calls + `{len(records)}` winner calls)",
        f"- Live LLM trace-generation wall time: `{cache['live_wall_s']:.2f}s`",
        f"- API/parse errors: `{len(cache.get('errors', []))}`",
        f"- Mean real `c_gen`: `{statistics.mean(lat):.3f}s`" if lat else "- Mean real `c_gen`: `0`",
        f"- P95 real `c_gen`: `{percentile(lat, 95):.3f}s`" if lat else "- P95 real `c_gen`: `0`",
        f"- Total tokens: `{tokens}`",
        f"- Mean parsed candidates/task: `{statistics.mean(len(r.get('candidates', [])) for r in records):.2f}`",
        f"- Intent operations in candidate branches: `{intent_counts}`",
        "",
        "HYBRID commit check on the recorded trace:",
        "",
        "This check reuses the recorded LLM trace and verifies that the actual CAST/HYBRID commit kernel can materialize and commit the generated branches. Its wall-clock throughput is not used as the fair policy comparison; the fair comparison is the replay table below, where every policy pays the same recorded generation latency.",
        "",
        f"- committed: `{cache['live_hybrid']['committed']}`",
        f"- rejected: `{cache['live_hybrid']['rejected']}`",
        f"- merge: `{cache['live_hybrid']['merge']}`",
        f"- reselect: `{cache['live_hybrid']['reselect']}`",
        f"- regen: `{cache['live_hybrid']['regen']}`",
        f"- oversell: `{cache['live_hybrid']['oversell']}`",
        "",
        "## Replay Results",
        "",
        "| Policy | Throughput | P95 latency | Regen | Gen calls/task | Merge | Reselect | Rejected | Oversell | CAS violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {float(row['throughput']):.2f} | "
            f"{float(row['p95_latency_ms']):.2f}ms | {float(row['regen']):.1f} | "
            f"{float(row['generation_calls_per_task']):.3f} | {float(row.get('merge', 0.0)):.1f} | "
            f"{float(row['reselect']):.1f} | {float(row['rejected']):.1f} | "
            f"{float(row['oversell']):.1f} | {float(row['cas_violation']):.1f} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Best traditional safe policy: `{best}`.",
        f"- HYBRID speedup vs best traditional: `{float(hy['speedup_vs_best_traditional']):.2f}x`.",
        f"- HYBRID speedup vs branch-txn: `{float(hy['speedup_vs_branch_txn']):.2f}x`.",
        "- `merge-all` is an unsafe upper bound and is not counted as a valid traditional safe baseline.",
        "- `OCC+K` is intentionally omitted from the main comparison because it is an agent-aware ablation, not a traditional CC scheme.",
    ]
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("live", "replay", "all", "ping"), default="all")
    parser.add_argument("--tasks-per-workload", type=int, default=30)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--replay-seeds", type=int, default=5)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--model", default=ds.DEFAULT_MODEL)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--workloads", nargs="*", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--policies", nargs="*", choices=POLICIES, default=list(POLICIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "ping":
        ok, latency, msg = ds.ping(args.model)
        print(f"DeepSeek ping: ok={ok} latency={latency:.2f}s msg={msg!r}")
        return
    if args.mode in ("live", "all") and not args.mock and not ds.have_key():
        raise SystemExit("DEEPSEEK_API_KEY is required unless --mock is used")

    if args.mode in ("live", "all"):
        print(f"[live] model={args.model} tasks/workload={args.tasks_per_workload} k={args.k}")
        cache = generate_live_trace(args)
        print(f"[live] records={len(cache['records'])} errors={len(cache['errors'])} wall={cache['live_wall_s']:.1f}s")
        print(f"[live] HYBRID committed={cache['live_hybrid']['committed']} merge={cache['live_hybrid']['merge']} reselect={cache['live_hybrid']['reselect']}")
    else:
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        fixed_store = load_initial_store(cache)
        cache["live_hybrid"] = run_live_hybrid(cache["records"], fixed_store, threads=args.threads, model=cache.get("model", args.model))
        cache["initial_store"] = {k: {"value": v.value, "version": v.version} for k, v in fixed_store.items()}
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    if args.mode in ("replay", "all"):
        print(f"[replay] policies={','.join(args.policies)} seeds={args.replay_seeds}")
        _, summary = replay(cache, args)
        for row in summary:
            print(
                f"  {row['policy']:>10}: tp={float(row['throughput']):.2f} "
                f"regen={float(row['regen']):.1f} merge={float(row.get('merge', 0.0)):.1f} "
                f"oversell={float(row['oversell']):.1f}"
            )
        write_report(cache, summary, args)
        manifest = {
            "cache": CACHE,
            "runs_csv": RUNS_CSV,
            "summary_csv": SUMMARY_CSV,
            "summary_md": SUMMARY_MD,
            "params": vars(args),
            "traditional_safe_policies": list(TRADITIONAL_SAFE),
            "note": "OCC+K omitted from main comparison because it is an agent-aware ablation.",
        }
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"saved {RUNS_CSV}")
        print(f"saved {SUMMARY_CSV}")
        print(f"saved {SUMMARY_MD}")
        print(f"saved {MANIFEST}")


if __name__ == "__main__":
    main()
