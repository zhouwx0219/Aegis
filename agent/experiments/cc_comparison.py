"""多 CC 策略对比（真并发 + 真实计时）：OCC / Silo / TicToc / MVCC-SI / 2PL / CAST。

动机：审稿会问"对比的 baseline 是不是太弱"。这里把一整族**先进 syntactic CC** 拉进来：
  - OCC / Silo：提交时验证读集 + 写集（Silo = OCC 的多核优化，单提交点模型下 abort 同 OCC）；
  - TicToc：数据驱动时间戳，读-写冲突可"挪 ts"避免 → 读放行、写写仍 abort；
  - MVCC-SI：读从快照不验证、只写写 first-committer-wins；
  - 2PL：悲观锁，无 abort 但持锁串行化；
  - CAST（ours）：读放行 + **可交换写放行（merge）**，仅 strict-strict 才是真冲突。
关键：syntactic CC 都**不懂可交换性** → 可交换写写冲突一律 abort→重跑；CAST 用语义合并消掉它。
执行：真多线程 + 真实墙钟（sleep 代表 c_gen）；abort→reselect 其他候选→都不行 regenerate(sleep)。
所有策略都给 k 个候选 + reselect（公平）；CAST 额外 merge。统一 Python 并发框架（CAST 语义与 C++ 核一致）。
注：Silo/TicToc 的多核可扩展性创新在单提交点模型不体现，此处只对比其 abort 语义。
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

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

C_GEN = 0.003
N_TASKS = 120
W = 2          # 写/候选
R = 1          # 读/候选
P_MERGE = 0.6
STRATS = ["OCC", "Silo", "TicToc", "MVCC", "2PL", "CAST"]


def make_store(n_obj):
    n_counter = int(n_obj * P_MERGE)
    return {f"o{i}": [("1000" if i < n_counter else "v0"), 0] for i in range(n_obj)}, n_counter


def kind_of(oid, n_counter):
    return "delta" if int(oid[1:]) < n_counter else "strict"


def gen_candidates(rng, n_obj, n_counter, k):
    """k 个异构候选：各写 W 个不同对象 + 读 R 个对象。"""
    objs = [f"o{i}" for i in range(n_obj)]
    cands = []
    for _ in range(k):
        picks = rng.sample(objs, min(W + R, n_obj))
        w_objs = picks[:W]
        r_objs = picks[W:W + R]
        writes = [(o, kind_of(o, n_counter)) for o in w_objs]
        cands.append({"reads": set(r_objs), "writes": writes})
    return cands


def aborts(strat, RS, WS_objs, sWS_objs, changed):
    ww = bool(WS_objs & changed); rw = bool(RS & changed); sww = bool(sWS_objs & changed)
    if strat in ("OCC", "Silo"):
        return ww or rw            # 读集+写集全严格
    if strat in ("TicToc", "MVCC"):
        return ww                  # 读放行（TicToc 挪 ts / MVCC 快照），写写仍冲突
    if strat == "CAST":
        return sww                 # 读放行 + 可交换写放行，仅 strict-strict
    return False                   # 2PL：锁，无 abort


def run(strat, n_obj, k, seed):
    rng = random.Random(seed)
    store, n_counter = make_store(n_obj)
    store_lock = threading.Lock()
    obj_locks = {o: threading.Lock() for o in store}
    q = queue.Queue()
    for i in range(N_TASKS):
        q.put(i)
    latencies = []
    committed = [0]; regens = [0]; merges = [0]
    acc_lock = threading.Lock()

    def snapshot_versions(cand):
        objs = list(cand["reads"]) + [o for o, _ in cand["writes"]]
        return {o: store[o][1] for o in objs}

    def try_commit(cand, base_ver):
        # commit_lock 内调用：判 abort；不 abort 则写入（CAST 对可交换写 merge）
        changed = {o for o, v in base_ver.items() if store[o][1] != v}
        RS = cand["reads"]; WS_objs = {o for o, _ in cand["writes"]}
        sWS = {o for o, kd in cand["writes"] if kd == "strict"}
        if aborts(strat, RS, WS_objs, sWS, changed):
            return False, 0
        m = 0
        for o, kd in cand["writes"]:
            val, ver = store[o]
            if kd == "delta":
                if o in changed:  # 可交换：在最新值 rebase（CAST 的 merge）
                    val = str(int(val) - 1); m += 1
                else:
                    val = str(int(val) - 1)
            else:
                val = f"set"
            store[o] = [val, ver + 1]
        return True, m

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            if strat == "2PL":
                cand = gen_candidates(rng, n_obj, n_counter, 1)[0]
                objs = sorted({o for o, _ in cand["writes"]} | cand["reads"])
                locks = [obj_locks[o] for o in objs]
                for l in locks: l.acquire()
                try:
                    time.sleep(C_GEN)
                    with store_lock:
                        for o, kd in cand["writes"]:
                            val, ver = store[o]
                            store[o] = [(str(int(val) - 1) if kd == "delta" else "set"), ver + 1]
                    with acc_lock: committed[0] += 1
                finally:
                    for l in reversed(locks): l.release()
            else:
                cands = gen_candidates(rng, n_obj, n_counter, k)
                base = [snapshot_versions(c) for c in cands]
                time.sleep(C_GEN)                     # 生成候选（并发重叠）
                done = False
                with store_lock:
                    for idx, c in enumerate(cands):    # winner + reselect 其他候选
                        ok, m = try_commit(c, base[idx])
                        if ok:
                            done = True
                            with acc_lock:
                                committed[0] += 1; merges[0] += m
                            break
                if not done:                           # 都 abort → regenerate
                    time.sleep(C_GEN)
                    with store_lock:
                        c = cands[0]; bv = snapshot_versions(c)
                        ok, m = try_commit(c, bv)      # 重读最新基线，必不冲突
                        with acc_lock:
                            committed[0] += 1; regens[0] += 1; merges[0] += m
            with acc_lock:
                latencies.append(time.perf_counter() - t0)
            q.task_done()

    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - wall0
    return {"throughput": committed[0] / wall, "latency_ms": statistics.mean(latencies) * 1000,
            "regen": regens[0], "merge": merges[0],
            "abort_rate": regens[0] / max(committed[0], 1)}


def sweep(param_name, values, fixed):
    seeds = [1, 2]
    data = {s: {"tp": [], "abort": []} for s in STRATS}
    print(f"\n=== 扫 {param_name} (固定 {fixed}) ===")
    print(f"{param_name:>10} | " + " | ".join(f"{s:>8}" for s in STRATS) + "   (throughput)")
    for v in values:
        line = {}
        for s in STRATS:
            kw = dict(fixed); kw[param_name] = v
            tps = [run(s, kw["n_obj"], kw["k"], sd)["throughput"] for sd in seeds]
            ar = [run(s, kw["n_obj"], kw["k"], sd)["abort_rate"] for sd in seeds]
            data[s]["tp"].append(statistics.mean(tps)); data[s]["abort"].append(statistics.mean(ar))
            line[s] = statistics.mean(tps)
        print(f"{v:>10} | " + " | ".join(f"{line[s]:>8.0f}" for s in STRATS))
    return data


def main():
    # 实验A：扫冲突等级（对象池越小越争用），固定 k=2
    contention = [("o12", 12), ("o24", 24), ("o48", 48), ("o96", 96)]
    dataA = sweep("n_obj", [c[1] for c in contention], {"k": 2})
    # 实验B：扫候选分支 k，固定中等冲突 n_obj=24
    dataB = sweep("k", [1, 2, 4, 8], {"n_obj": 24})

    style = {"OCC": ("o-", "tab:blue"), "Silo": ("v-", "tab:cyan"), "TicToc": ("^-", "tab:purple"),
             "MVCC": ("D-", "tab:olive"), "2PL": ("d-.", "tab:red"), "CAST": ("s-", "tab:green")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xa = [12, 24, 48, 96]
    for s in STRATS:
        st, c = style[s]
        axes[0].plot(xa, dataA[s]["tp"], st, color=c, label=s, linewidth=1.8)
    axes[0].set_xlabel("object-pool size (smaller = higher contention)"); axes[0].invert_xaxis()
    axes[0].set_ylabel("throughput (committed/s)"); axes[0].set_title("(a) vs contention level (k=2)"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    xb = [1, 2, 4, 8]
    for s in STRATS:
        st, c = style[s]
        axes[1].plot(xb, dataB[s]["tp"], st, color=c, label=s, linewidth=1.8)
    axes[1].set_xlabel("candidate branches k"); axes[1].set_ylabel("throughput (committed/s)")
    axes[1].set_title("(b) vs candidate branches (n_obj=24)"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Multi-CC comparison (real concurrency): OCC/Silo/TicToc/MVCC/2PL/CAST\n"
                 "syntactic CC cluster together (abort on commutative conflicts); CAST's semantic merge is orthogonal gain", fontsize=9.5, y=1.05)
    fig.tight_layout()
    out = os.path.join(RESULTS, "cc_comparison.png"); fig.savefig(out, dpi=130, bbox_inches="tight"); print("\nsaved", out)


if __name__ == "__main__":
    main()
