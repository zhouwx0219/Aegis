"""gap1：真多线程并发执行 + 真实墙钟计时（替换解析时间模型）。

把"makespan 公式 + t_gen 常数"换成**真并发 + 真实 time.perf_counter**：
  - N 个 worker 线程 = N 个并发 agent，从任务队列取任务；
  - "LLM 生成候选"用 time.sleep(C_GEN) 真实表示（sleep 释放 GIL → 多线程真重叠）；
  - 提交在 commit_lock 内原子完成（OCC/CAST 的串行验证点，锁内极快 ≈ c_merge）；
  - OCC 冲突 → commit_task 返回 action=regenerate → 真实补一次 sleep(C_GEN)（重跑成本）；
  - CAST merge/reselect 锁内完成、不补 sleep；2PL 持对象锁执行 sleep → 真实串行化争用。
三策略（OCC / 2PL / CAST）在**同一并发框架真跑**；多 seed 报均值±std。吞吐/延迟全是实测墙钟。
"""
import csv
import os
import queue
import random
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cast_core as cc

C_GEN = 0.004   # 一次候选生成的真实耗时（秒，代表 LLM）；c_merge≈0（锁内 KV 操作）
N_OBJ = 10
W = 2
P_MERGE = 0.6
N_TASKS = 160


def gen_tasks(seed):
    rng = random.Random(seed)
    objs = [f"o{i}" for i in range(N_OBJ)]
    n_counter = int(N_OBJ * P_MERGE)
    kind_of = {f"o{i}": ("delta" if i < n_counter else "strict") for i in range(N_OBJ)}
    tasks = []
    for _ in range(N_TASKS):
        picks = rng.sample(objs, W)
        specs = [(o, kind_of[o]) for o in picks]
        tasks.append(specs)
    return tasks


def seed_store(store):
    n_counter = int(N_OBJ * P_MERGE)
    for i in range(N_OBJ):
        store.put(f"o{i}", "1000" if i < n_counter else "v0")


def build_candidate(store, specs, tag):
    writes = []
    for oid, kind in specs:
        v = store.get(oid)
        it = cc.WriteIntent(); it.object_id = oid
        if kind == "delta":
            it.intent_type = cc.IntentType.kDelta; it.payload = "-1"; bv = str(int(v.value) - 1)
        else:
            it.intent_type = cc.IntentType.kOverwrite; bv = f"set-{tag}"
        w = cc.BranchWrite(); w.object_id = oid; w.base_value = v.value; w.base_version = v.version
        w.branch_value = bv; w.intent = it
        writes.append(w)
    b = cc.SpeculativeBranch(); b.branch_id = f"t{tag}"; b.writes = writes; b.quality = 1.0
    return b


def run(strategy_name, n_threads, seed):
    store = cc.VersionedObjectStore(); seed_store(store)
    model = cc.CostModel(1.0, 0.0)
    commit = cc.CostAsymmetricCommit(store, model)
    stats = cc.CostStats()
    commit_lock = threading.Lock()
    obj_locks = {f"o{i}": threading.Lock() for i in range(N_OBJ)}
    tasks = gen_tasks(seed)
    q = queue.Queue()
    for i, t in enumerate(tasks):
        q.put((i, t))
    latencies = []
    lat_lock = threading.Lock()
    committed = [0]
    strat_enum = cc.CommitStrategy.kCAST if strategy_name == "CAST" else cc.CommitStrategy.kStrictOCC

    def worker():
        while True:
            try:
                i, specs = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            if strategy_name == "2PL":
                objs = sorted(o for o, _ in specs)
                locks = [obj_locks[o] for o in objs]
                for l in locks:
                    l.acquire()
                try:
                    cand = build_candidate(store, specs, i)
                    time.sleep(C_GEN)                       # 持锁执行 → 争用真实串行化
                    with commit_lock:
                        commit.commit_task([cand], cc.CommitStrategy.kStrictOCC, stats)
                finally:
                    for l in reversed(locks):
                        l.release()
            else:  # OCC / CAST 乐观
                cand = build_candidate(store, specs, i)     # 读基线
                time.sleep(C_GEN)                           # 生成候选（并发重叠）
                with commit_lock:
                    out = commit.commit_task([cand], strat_enum, stats)
                if out.action == "regenerate":
                    time.sleep(C_GEN)                       # 重跑的真实成本
            dt = time.perf_counter() - t0
            with lat_lock:
                latencies.append(dt); committed[0] += 1
            q.task_done()

    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    return {"throughput": committed[0] / wall, "mean_latency": statistics.mean(latencies) * 1000,  # ms
            "wall": wall, "regen": stats.n_regen, "merge": stats.n_merge}


def main():
    RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS, exist_ok=True)
    seeds = [1, 2, 3]
    threads_list = [1, 2, 4, 8]
    strategies = ["OCC", "2PL", "CAST"]
    agg = {s: {"tp": [], "tp_sd": [], "lat": [], "lat_sd": []} for s in strategies}
    print(f"=== 真并发实测（N_TASKS={N_TASKS}, C_GEN={C_GEN}s, 写/任务={W}, 可合并={P_MERGE}, seeds={seeds}）===")
    print(f"{'threads':>7} | {'strategy':>8} | {'throughput(±sd)':>18} | {'latency_ms(±sd)':>16} | regen/merge")
    rows = []
    for nt in threads_list:
        for s in strategies:
            tps, lats, regens, merges = [], [], [], []
            for sd in seeds:
                r = run(s, nt, sd)
                tps.append(r["throughput"]); lats.append(r["mean_latency"]); regens.append(r["regen"]); merges.append(r["merge"])
            tp_m, tp_s = statistics.mean(tps), (statistics.stdev(tps) if len(tps) > 1 else 0)
            lat_m, lat_s = statistics.mean(lats), (statistics.stdev(lats) if len(lats) > 1 else 0)
            agg[s]["tp"].append(tp_m); agg[s]["tp_sd"].append(tp_s); agg[s]["lat"].append(lat_m); agg[s]["lat_sd"].append(lat_s)
            rows.append({"threads": nt, "strategy": s, "throughput": round(tp_m, 1), "tp_sd": round(tp_s, 1),
                         "latency_ms": round(lat_m, 2), "lat_sd": round(lat_s, 2),
                         "regen": round(statistics.mean(regens), 1), "merge": round(statistics.mean(merges), 1)})
            print(f"{nt:>7} | {s:>8} | {tp_m:>10.1f} ± {tp_s:>4.1f} | {lat_m:>8.2f} ± {lat_s:>4.2f} | {statistics.mean(regens):.0f}/{statistics.mean(merges):.0f}")
        print("  " + "-" * 70)

    with open(os.path.join(RESULTS, "concurrent.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    style = {"OCC": ("o-", "tab:blue"), "2PL": ("d-.", "tab:red"), "CAST": ("s-", "tab:green")}
    for s in strategies:
        st, c = style[s]
        a1.errorbar(threads_list, agg[s]["tp"], yerr=agg[s]["tp_sd"], fmt=st, color=c, label=s, linewidth=2, capsize=3)
        a2.errorbar(threads_list, agg[s]["lat"], yerr=agg[s]["lat_sd"], fmt=st, color=c, label=s, linewidth=2, capsize=3)
    a1.set_xlabel("concurrent agents (threads)"); a1.set_ylabel("throughput (committed/s, measured)"); a1.set_title("(a) throughput — higher better"); a1.legend(); a1.grid(True, alpha=0.3)
    a2.set_xlabel("concurrent agents (threads)"); a2.set_ylabel("mean latency (ms, measured)"); a2.set_title("(b) latency — lower better"); a2.legend(); a2.grid(True, alpha=0.3)
    fig.suptitle("gap1: REAL multi-threaded execution + wall-clock timing (not analytical)\n"
                 "OCC/2PL/CAST run in the same concurrent framework; 3 seeds, mean±std", fontsize=10, y=1.05)
    fig.tight_layout()
    out = os.path.join(RESULTS, "concurrent.png"); fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
