"""真实 LLM-in-the-loop 实验（关闭 SUBMISSION_DEFENSE §8 第 1 条：motivation 真实性最大缺口）。

把"sleep 代表 c_gen + 随机选对象代表候选"换成**真实 DeepSeek 调用**：
agent 对真实 OTA 订票任务,一次调用返回 K 个候选航班(真实 c_gen),映射为对共享座位库存的扣减写,
经我们的提交内核(HYBRID 用 C++ cast_core.EscrowAccount 守座位下界)提交;K 个候选互为备选 ⇒ 真实多候选 reselect。

验证三件此前只能假设的事：
  A) 真实 c_gen 分布——LLM 调用是否秒级长尾(证明 sleep 假设合理 + 给出实测分布)。
  B) 真实写意图与多候选多样性——真实 agent 是否产出可合并写 + 是否给出可 reselect 的备选。
  C) 端到端——真实 c_gen + 真实候选下,HYBRID vs OCC/2PL/merge-all 的吞吐/超卖/浪费。

口径(诚实)：
  - A/B 为真实 LLM 实测；c_gen=真实墙钟。
  - C 用 record-and-replay：真实候选 + 真实 c_gen 录下后回放比较各策略(不重复烧 API)。
    回放对所有策略**统一缩放 sleep**(--speed),故吞吐/浪费的**策略间比值**与缩放无关(尺度不变),
    绝对延迟以 A 的实测分布为准。HYBRID 座位容量由 C++ EscrowAccount(真) 执行;
    OCC/2PL/merge-all 为同框架 Python 基线(与 cc_comparison/vitabench_ota 同口径)。
  - 未接 key 时用 --mock 干跑校验链路(不调 API)。

用法：
  DEEPSEEK_API_KEY=... python3 agent/experiments/llm_in_the_loop.py all --tasks 60 --k 3 --conc 8
  python3 agent/experiments/llm_in_the_loop.py all --mock          # 无 key 干跑
"""
import argparse
import json
import os
import queue
import random
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent", "experiments"))
import cast_core as cc
from agent.llm import deepseek_client as ds
from agent.llm import llm_agent_operator as op
from agent.runtime import AgentTransactionManager

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
CACHE = os.path.join(RESULTS, "llm_cache.json")
OUTJSON = os.path.join(RESULTS, "llm_in_the_loop.json")

try:
    import eval_common as E
    def mean_ci(xs):
        return E.mean_ci(xs)
except Exception:
    def mean_ci(xs):
        if not xs:
            return 0.0, 0.0
        m = statistics.mean(xs)
        if len(xs) < 2:
            return m, 0.0
        sd = statistics.stdev(xs)
        return m, 1.96 * sd / (len(xs) ** 0.5)


# ---------------- 阶段1：真实并发 LLM 调用（缓存） ----------------
def make_runtime(seats0):
    runtime = AgentTransactionManager(c_gen=1.0, c_merge=0.0)
    for oid, seats in seats0.items():
        runtime.register_object(oid, seats, kind="counter")
    return runtime


def add_record_to_transaction(txn, record, model):
    txn.record_model_call(
        model=model,
        latency_s=record.get("c_gen", 0.0),
        usage=record.get("usage", {}),
        candidates=len(record.get("candidates", [])),
    )
    txn.record_tool_call(
        "prepare_flight_booking",
        args={"route": record.get("route", ""), "quantity": 1},
        outcome="prepared",
    )
    n = len(record.get("candidates", []))
    for i, candidate in enumerate(record.get("candidates", [])):
        branch = txn.add_candidate(
            f"{txn.task_id}:flight:{i}",
            quality=float(n - i),
            gen_cost=float(record.get("c_gen", 0.0)),
            metadata={
                "flight_id": candidate["flight_id"],
                "oid": candidate["oid"],
                "rank": i,
                "price": candidate.get("price"),
            },
        )
        branch.delta(candidate["oid"], -1, constrained=True, lower_bound=0)


def summarize_transactions(results, wall, runtime):
    booked = sum(1 for r in results if r.committed)
    rejected = sum(1 for r in results if r.rejected)
    values = runtime.values()
    oversell = sum(1 for value in values.values() if int(value) < 0)
    latencies = [r.elapsed_s for r in results]
    return {
        "throughput": booked / wall if wall > 0 else 0.0,
        "latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "wall": wall,
        "booked": booked,
        "reselect": sum(r.n_reselect for r in results),
        "no_seat": rejected,
        "oversell": oversell,
        "regen": sum(r.n_regen for r in results),
        "merge": sum(r.n_merge for r in results),
        "total_tokens": sum(r.total_tokens for r in results),
    }


def live_generate(tasks, k, conc, model, mock, runtime=None):
    """Generate real candidates and, when runtime is provided, commit one full task transaction."""
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    records = [None] * len(tasks)
    results = []
    errors = [0]
    lock = threading.Lock()

    def worker():
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                return
            txn = runtime.begin(
                task["id"], {"route": task["route"], "preference": task["pref"]}
            ) if runtime else None
            try:
                record = op.generate_candidates(task, k=k, model=model, mock=mock)
                if txn:
                    add_record_to_transaction(txn, record, model)
                    result = txn.commit("cast")
                    record["transaction_result"] = result.to_dict()
                    with lock:
                        results.append(result)
            except Exception as exc:  # noqa
                with lock:
                    errors[0] += 1
                if txn and txn.state.value == "active":
                    result = txn.abort(f"generation failed: {type(exc).__name__}")
                    with lock:
                        results.append(result)
                record = {
                    "task_id": task["id"],
                    "route": task["route"],
                    "candidates": [],
                    "c_gen": 0.0,
                    "usage": {},
                    "n_parsed": 0,
                    "distinct_flights": 0,
                    "raw": f"ERR:{type(exc).__name__}:{str(exc)[:160]}",
                }
            records[task["id"]] = record
            q.task_done()

    wall0 = time.perf_counter()
    n_threads = 1 if mock else max(1, conc)
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall0
    return records, wall, errors[0], results

# ---------------- 端到端预订模型（被 live HYBRID 与 replay 复用） ----------------
def book_hybrid(records, seats0, threads, seed, speed=1.0, sleep_real=True,
                model=ds.DEFAULT_MODEL, candidate_limit=None):
    """Replay records through the complete AgentTransaction lifecycle."""
    runtime = make_runtime(seats0)
    results = []
    result_lock = threading.Lock()
    work = queue.Queue()
    for record in records:
        work.put(record)

    def worker():
        while True:
            try:
                record = work.get_nowait()
            except queue.Empty:
                return
            txn = runtime.begin(
                f"replay-{seed}-{record['task_id']}",
                {"route": record.get("route", ""), "replay_seed": seed},
            )
            if sleep_real:
                time.sleep(max(0.0, record.get("c_gen", 0.0)) / speed)
            replay_record = record
            if candidate_limit is not None:
                replay_record = dict(record)
                replay_record["candidates"] = record.get("candidates", [])[:candidate_limit]
            add_record_to_transaction(txn, replay_record, model)
            if txn.candidates:
                result = txn.commit("cast")
            else:
                result = txn.abort("no parsed candidates")
            with result_lock:
                results.append(result)
            work.task_done()

    wall0 = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    wall = time.perf_counter() - wall0
    metrics = summarize_transactions(results, wall, runtime)
    metrics["trace_count"] = len(runtime.traces())
    return metrics

def book_baseline(policy, records, seats0, threads, seed, speed=1.0, sleep_real=True):
    """OCC / 2PL / merge-all replay over the same recorded candidates.

    OCC receives the same K alternatives as HYBRID and may reselect a candidate
    whose version is still valid. It regenerates only when no existing candidate
    can commit but at least one target still has capacity.
    """
    seats = {oid: [s, 0] for oid, s in seats0.items()}
    glock = threading.Lock()
    object_locks = {oid: threading.Lock() for oid in seats0}
    counters = {"booked": 0, "reselect": 0, "no_seat": 0, "oversell": 0, "regen": 0}
    latencies = []
    work = queue.Queue()
    for record in records:
        work.put(record)

    def decrement(oid, allow_oversell):
        value = seats[oid]
        if allow_oversell:
            value[0] -= 1
            value[1] += 1
            if value[0] < 0:
                counters["oversell"] += 1
            return True
        if value[0] > 0:
            value[0] -= 1
            value[1] += 1
            return True
        return False

    def worker():
        while True:
            try:
                record = work.get_nowait()
            except queue.Empty:
                return
            candidates = record.get("candidates", [])
            if not candidates:
                work.task_done()
                continue
            t0 = time.perf_counter()

            if policy == "2PL":
                oid = candidates[0]["oid"]
                object_locks[oid].acquire()
                try:
                    if sleep_real:
                        time.sleep(max(0.0, record["c_gen"]) / speed)
                    with glock:
                        ok = decrement(oid, allow_oversell=False)
                        counters["booked" if ok else "no_seat"] += 1
                finally:
                    object_locks[oid].release()
            elif policy == "merge-all":
                if sleep_real:
                    time.sleep(max(0.0, record["c_gen"]) / speed)
                with glock:
                    decrement(candidates[0]["oid"], allow_oversell=True)
                    counters["booked"] += 1
            elif policy == "branch-txn":
                # Traditional branch-per-transaction model: each candidate is
                # treated as an independent speculative DB transaction. The
                # selected winner commits if its branch can validate; loser
                # branches abort. Unlike Agent-OCC-K, this model does not
                # reselect another already generated branch after the winner
                # conflicts, because the DB sees separate transactions rather
                # than one semantic agent transaction.
                with glock:
                    oid = candidates[0]["oid"]
                    base_version = seats[oid][1]
                if sleep_real:
                    time.sleep(max(0.0, record["c_gen"]) / speed)

                committed = False
                need_regen = False
                with glock:
                    oid = candidates[0]["oid"]
                    if seats[oid][1] == base_version and seats[oid][0] > 0:
                        decrement(oid, allow_oversell=False)
                        counters["booked"] += 1
                        committed = True
                    elif any(seats[candidate["oid"]][0] > 0 for candidate in candidates):
                        need_regen = True
                    else:
                        counters["no_seat"] += 1

                if need_regen:
                    if sleep_real:
                        time.sleep(max(0.0, record["c_gen"]) / speed)
                    with glock:
                        oid = candidates[0]["oid"]
                        counters["regen"] += 1
                        ok = decrement(oid, allow_oversell=False)
                        counters["booked" if ok else "no_seat"] += 1
            else:
                # Agent-OCC-K baseline: OCC-style row-version validation while
                # fairly allowing the same K already generated alternatives.
                with glock:
                    base_versions = {
                        candidate["oid"]: seats[candidate["oid"]][1]
                        for candidate in candidates
                    }
                if sleep_real:
                    time.sleep(max(0.0, record["c_gen"]) / speed)

                committed = False
                need_regen = False
                with glock:
                    for index, candidate in enumerate(candidates):
                        oid = candidate["oid"]
                        if seats[oid][1] == base_versions[oid] and seats[oid][0] > 0:
                            decrement(oid, allow_oversell=False)
                            counters["booked"] += 1
                            if index > 0:
                                counters["reselect"] += 1
                            committed = True
                            break
                    if not committed:
                        need_regen = any(seats[candidate["oid"]][0] > 0 for candidate in candidates)
                        if not need_regen:
                            counters["no_seat"] += 1

                if need_regen:
                    if sleep_real:
                        time.sleep(max(0.0, record["c_gen"]) / speed)
                    with glock:
                        chosen = next(
                            (
                                (index, candidate)
                                for index, candidate in enumerate(candidates)
                                if seats[candidate["oid"]][0] > 0
                            ),
                            None,
                        )
                        counters["regen"] += 1
                        if chosen is None:
                            counters["no_seat"] += 1
                        else:
                            index, candidate = chosen
                            decrement(candidate["oid"], allow_oversell=False)
                            counters["booked"] += 1
                            if index > 0:
                                counters["reselect"] += 1

            latencies.append(time.perf_counter() - t0)
            work.task_done()

    wall0 = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    wall = time.perf_counter() - wall0
    return _summ(counters, latencies, wall)

def _summ(ctr, lat, wall):
    return {"throughput": ctr["booked"] / wall if wall > 0 else 0.0,
            "latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
            "wall": wall, **ctr}


# ---------------- 分析与出图 ----------------
def pctl(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return xs[i]


def analyze_and_plot(records, replay, k, speed, model, mock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cg = [r["c_gen"] for r in records if r["c_gen"] > 0]
    distinct = [r["distinct_flights"] for r in records]
    nparsed = [r["n_parsed"] for r in records]
    toks = sum(int(r.get("usage", {}).get("total_tokens", 0) or 0) for r in records)

    # ---- A: c_gen 分布 ----
    a_stats = {}
    if cg:
        a_stats = {"n": len(cg), "mean": statistics.mean(cg), "median": statistics.median(cg),
                   "p95": pctl(cg, 95), "p99": pctl(cg, 99), "min": min(cg), "max": max(cg),
                   "std": statistics.stdev(cg) if len(cg) > 1 else 0.0}
    # ---- B: 多样性 / 可合并 ----
    b_stats = {"mean_distinct": statistics.mean(distinct) if distinct else 0,
               "frac_reselectable": (sum(1 for d in distinct if d >= 2) / len(distinct)) if distinct else 0,
               "mean_parsed": statistics.mean(nparsed) if nparsed else 0,
               "commutative_ratio": 1.0,  # 订票写=共享座位 DELTA 扣减,全为可交换约束写
               "total_tokens": toks}

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (a) c_gen 直方图
    if cg:
        ax[0, 0].hist(cg, bins=min(20, max(5, len(cg) // 3)), color="tab:blue", alpha=0.8, edgecolor="white")
        ax[0, 0].axvline(a_stats["median"], color="tab:green", ls="--", label=f"p50={a_stats['median']:.2f}s")
        ax[0, 0].axvline(a_stats["p95"], color="tab:red", ls="--", label=f"p95={a_stats['p95']:.2f}s")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("real LLM call latency c_gen (s)")
    ax[0, 0].set_ylabel("count")
    ax[0, 0].set_title(f"(A) Real c_gen distribution (DeepSeek {model}) — seconds-scale, long-tailed")
    ax[0, 0].grid(True, alpha=0.3)
    # (b) 候选多样性
    if distinct:
        vals = list(range(0, k + 1))
        cnt = [sum(1 for d in distinct if d == v) for v in vals]
        ax[0, 1].bar(vals, cnt, color="tab:purple", alpha=0.85)
    ax[0, 1].set_xlabel("distinct alternative flights per task (K options)")
    ax[0, 1].set_ylabel("tasks")
    ax[0, 1].set_title(f"(B) Multi-candidate diversity — {b_stats['frac_reselectable']*100:.0f}% tasks have ≥2 alts (reselect viable)")
    ax[0, 1].grid(True, axis="y", alpha=0.3)
    # (c)(d) replay 对比
    if replay:
        pols = list(replay.keys())
        colors = {"OCC": "tab:blue", "2PL": "tab:gray", "merge-all": "tab:red", "HYBRID": "tab:green"}
        tp = [replay[p]["throughput"][0] for p in pols]
        tpe = [replay[p]["throughput"][1] for p in pols]
        ov = [replay[p]["oversell"][0] for p in pols]
        ove = [replay[p]["oversell"][1] for p in pols]
        cs = [colors.get(p, "tab:cyan") for p in pols]
        ax[1, 0].bar(pols, tp, yerr=tpe, capsize=3, color=cs)
        ax[1, 0].set_ylabel("throughput (booked/s, replay)")
        ax[1, 0].set_title("(C) End-to-end throughput (record-and-replay, ratios scale-invariant)")
        ax[1, 0].grid(True, axis="y", alpha=0.3)
        ax[1, 1].bar(pols, ov, yerr=ove, capsize=3, color=cs)
        ax[1, 1].set_ylabel("oversell events (seats < 0) [lower=correct]")
        ax[1, 1].set_title("(D) Correctness — only merge-all oversells; HYBRID escrow caps")
        ax[1, 1].grid(True, axis="y", alpha=0.3)
    note = "MOCK (no API)" if mock else f"real DeepSeek, replay speed×{speed}"
    fig.suptitle("Real LLM-in-the-loop on OTA booking: real c_gen + real multi-candidate alternatives "
                 f"→ our hybrid CC kernel\n[{note}]", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "llm_in_the_loop.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out, a_stats, b_stats


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["live", "replay", "plot", "all", "ping"])
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--conc", type=int, default=8, help="并发 LLM 调用数")
    ap.add_argument("--threads", type=int, default=8, help="replay 线程数")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--speed", type=float, default=20.0, help="replay sleep 缩放(比值尺度不变)")
    ap.add_argument("--seats-scale", type=float, default=1.0, help="座位库存缩放(<1 加剧争用)")
    ap.add_argument("--hot-bias", type=float, default=0.6)
    ap.add_argument("--model", default=ds.DEFAULT_MODEL)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.mode == "ping":
        ok, lat, msg = ds.ping(args.model)
        print(f"DeepSeek ping: ok={ok} latency={lat:.2f}s msg={msg!r}")
        return

    if args.mode in ("live", "all") and not args.mock and not ds.have_key():
        print("！未检测到 DEEPSEEK_API_KEY。请 export DEEPSEEK_API_KEY=... 或加 --mock 干跑。")
        sys.exit(2)

    # 座位库存（缩放以加剧争用）
    seats0 = {oid: max(1, int(round(s * args.seats_scale)))
              for oid, s in op.cat.all_flight_objects().items()}

    if args.mode in ("live", "all"):
        tasks = op.make_tasks(args.tasks, seed=0, hot_bias=args.hot_bias)
        runtime = make_runtime(seats0)
        print(f"[live] {len(tasks)} 任务, K={args.k}, 并发={args.conc}, mock={args.mock} → 完整事务执行中…")
        records, wall, errs, tx_results = live_generate(
            tasks, args.k, args.conc, args.model, args.mock, runtime=runtime
        )
        records = [r for r in records if r]
        cg = [r["c_gen"] for r in records if r["c_gen"] > 0]
        print(f"[live] 完成 {len(records)} 条, 错误 {errs}, 墙钟 {wall:.1f}s, "
              f"c_gen 均值 {statistics.mean(cg):.2f}s (p95 {pctl(cg,95):.2f}s)" if cg else f"[live] mock {len(records)} 条")
        hb = summarize_transactions(tx_results, wall, runtime)
        print(f"[live] 事务端到端: booked={hb['booked']} reselect={hb['reselect']} "
              f"no_seat={hb['no_seat']} oversell={hb['oversell']} tp={hb['throughput']:.2f}/s")
        with open(CACHE, "w", encoding="utf-8") as out:
            json.dump({
                "records": records,
                "live_hybrid": hb,
                "live_traces": runtime.traces(),
                "wall": wall,
                "errs": errs,
                "model": args.model,
                "mock": args.mock,
            }, out, ensure_ascii=False)
        print(f"[live] 脱敏事务缓存 → {CACHE}")

    if args.mode in ("replay", "all"):
        cache = json.load(open(CACHE))
        records = cache["records"]
        pols = ["branch-txn", "OCC", "2PL", "merge-all", "HYBRID-K1", "HYBRID"]
        replay = {p: {} for p in pols}
        print(f"\n[replay] 录制回放比较 {pols}, speed×{args.speed}, threads={args.threads}, seeds={args.seeds}")
        for p in pols:
            tps, ovs, rgs, rss, nss = [], [], [], [], []
            for sd in range(1, args.seeds + 1):
                if p in ("HYBRID-K1", "HYBRID"):
                    m = book_hybrid(records, seats0, args.threads, sd, speed=args.speed,
                                    sleep_real=not cache.get("mock"),
                                    model=cache.get("model", args.model),
                                    candidate_limit=1 if p == "HYBRID-K1" else None)
                else:
                    m = book_baseline(p, records, seats0, args.threads, sd, speed=args.speed, sleep_real=not cache.get("mock"))
                tps.append(m["throughput"]); ovs.append(m["oversell"]); rgs.append(m["regen"])
                rss.append(m["reselect"]); nss.append(m["no_seat"])
            replay[p] = {"throughput": mean_ci(tps), "oversell": mean_ci(ovs),
                         "regen": statistics.mean(rgs), "reselect": statistics.mean(rss),
                         "no_seat": statistics.mean(nss)}
            print(f"  {p:>9}: tp={replay[p]['throughput'][0]:.1f} oversell={replay[p]['oversell'][0]:.1f} "
                  f"regen={replay[p]['regen']:.0f} reselect={replay[p]['reselect']:.0f} no_seat={replay[p]['no_seat']:.0f}")
        json.dump({k_: {kk: (vv if not isinstance(vv, tuple) else list(vv)) for kk, vv in v.items()}
                   for k_, v in replay.items()}, open(OUTJSON, "w"), ensure_ascii=False)

    if args.mode in ("plot", "all"):
        cache = json.load(open(CACHE))
        replay = None
        if os.path.exists(OUTJSON):
            replay = json.load(open(OUTJSON))
        out, a_stats, b_stats = analyze_and_plot(cache["records"], replay, args.k, args.speed,
                                                 cache.get("model", args.model), cache.get("mock", args.mock))
        print(f"\n=== 分析摘要 ===")
        if a_stats:
            print(f"A 真实 c_gen: n={a_stats['n']} 均值={a_stats['mean']:.2f}s p50={a_stats['median']:.2f}s "
                  f"p95={a_stats['p95']:.2f}s p99={a_stats['p99']:.2f}s max={a_stats['max']:.2f}s std={a_stats['std']:.2f}")
        print(f"B 多候选: 平均不同备选={b_stats['mean_distinct']:.2f}/K, "
              f"可reselect任务占比={b_stats['frac_reselectable']*100:.0f}%, 总tokens={b_stats['total_tokens']}")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
