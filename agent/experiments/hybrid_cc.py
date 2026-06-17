"""混合 CC：per-object 去中心化提交 + per-object 自适应语义。

动机（见 GAP / 用户问 4）：CAST 当前对 strict 写仍用 OCC，且提交走**单个全局 commit-lock 串行**——
在提交有真实成本（后端持久化/验证往返）时，这个单提交点是性能瓶颈（上限≈1/t_commit）。
混合 CC 的两点：
  (1) **去中心化提交**：提交时只锁涉及的对象（per-object lock，按 oid 排序避免死锁），
      不相交事务的提交可并行 → 消除单提交点串行；
  (2) **per-object 自适应语义**：可交换对象（counter）→ 锁内累加、永不 abort；
      strict 对象 → per-object 版本校验（冲突→reselect/regen）。
本实验隔离 (1) 的收益：对比 commit_mode ∈ {global（当前 CAST）, per_object（混合）}，
其余语义相同（公平）。t_commit 代表真实后端的提交成本（内存=0，RocksDB/TiKV/网络=ms 级）。
"""
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
T_GEN, T_COMMIT = 0.003, 0.001   # 生成成本 / 提交成本（后端持久化+验证往返）
N_TASKS, W, P_MERGE = 200, 2, 0.6


def run(commit_mode, n_threads, n_obj, seed):
    rng = random.Random(seed)
    n_counter = int(n_obj * P_MERGE)
    store = {f"o{i}": [("1000" if i < n_counter else "v0"), 0] for i in range(n_obj)}
    global_lock = threading.Lock()
    obj_locks = {o: threading.Lock() for o in store}
    q = queue.Queue()
    for i in range(N_TASKS):
        q.put(i)
    committed = [0]; lat = []; acc = threading.Lock()

    def kind(o):
        return "delta" if int(o[1:]) < n_counter else "strict"

    def gen():
        picks = rng.sample(list(store), W)
        return {o: [kind(o), store[o][1]] for o in picks}   # obj -> [kind, base_ver]

    def do_commit(writes):
        for o, (kd, bv) in writes.items():
            if kd == "strict" and store[o][1] != bv:   # per-object 版本校验（strict）
                return False
        for o, (kd, bv) in writes.items():             # 可交换累加 / strict 覆盖
            val, ver = store[o]
            store[o] = [(str(int(val) - 1) if kd == "delta" else "set"), ver + 1]
        return True

    def commit(writes):
        if commit_mode == "global":
            with global_lock:
                time.sleep(T_COMMIT)                    # 单提交点串行
                return do_commit(writes)
        objs = sorted(writes); locks = [obj_locks[o] for o in objs]
        for l in locks:
            l.acquire()
        try:
            time.sleep(T_COMMIT)                        # 仅锁涉及对象 → 不相交并行
            return do_commit(writes)
        finally:
            for l in reversed(locks):
                l.release()

    def worker():
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            writes = gen(); time.sleep(T_GEN)
            if not commit(writes):                      # strict 冲突 → 重跑一次（读最新基线）
                time.sleep(T_GEN)
                commit({o: [kd, store[o][1]] for o, (kd, _) in writes.items()})
            with acc:
                committed[0] += 1; lat.append(time.perf_counter() - t0)
            q.task_done()

    w0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - w0
    return committed[0] / wall, statistics.mean(lat) * 1000


def main():
    threads_list = [1, 2, 4, 8, 16, 32]
    seeds = [1, 2]
    N_OBJ = 64   # 中低冲突，给去中心化留并行空间
    data = {"global": {"tp": [], "lat": []}, "per_object": {"tp": [], "lat": []}}
    print(f"=== 混合CC：去中心化提交 vs 全局提交点（n_obj={N_OBJ}, t_commit={T_COMMIT}s, 2 seeds）===")
    print(f"{'threads':>7} | {'global tp':>10} {'lat(ms)':>8} | {'per-object tp':>13} {'lat(ms)':>8} | speedup")
    for nt in threads_list:
        row = {}
        for mode in ("global", "per_object"):
            tps = []; lats = []
            for sd in seeds:
                tp, la = run(mode, nt, N_OBJ, sd)
                tps.append(tp); lats.append(la)
            data[mode]["tp"].append(statistics.mean(tps)); data[mode]["lat"].append(statistics.mean(lats))
            row[mode] = statistics.mean(tps)
        sp = row["per_object"] / row["global"]
        print(f"{nt:>7} | {row['global']:>10.0f} {data['global']['lat'][-1]:>8.2f} | {row['per_object']:>13.0f} {data['per_object']['lat'][-1]:>8.2f} | {sp:>5.2f}x")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(threads_list, data["global"]["tp"], "o-", color="tab:red", label="global commit-lock (current CAST)", linewidth=2)
    a1.plot(threads_list, data["per_object"]["tp"], "s-", color="tab:green", label="per-object decentralized (hybrid)", linewidth=2)
    a1.axhline(1.0 / T_COMMIT, ls=":", color="gray", label=f"global ceiling ≈ 1/t_commit = {1/T_COMMIT:.0f}/s")
    a1.set_xlabel("concurrent agents (threads)"); a1.set_ylabel("throughput (committed/s, measured)")
    a1.set_title("(a) throughput — decentralized commit removes the single-point ceiling"); a1.legend(fontsize=8); a1.grid(True, alpha=0.3)
    a2.plot(threads_list, data["global"]["lat"], "o-", color="tab:red", label="global", linewidth=2)
    a2.plot(threads_list, data["per_object"]["lat"], "s-", color="tab:green", label="per-object", linewidth=2)
    a2.set_xlabel("concurrent agents (threads)"); a2.set_ylabel("mean latency (ms)")
    a2.set_title("(b) latency"); a2.legend(fontsize=8); a2.grid(True, alpha=0.3)
    fig.suptitle("Hybrid CC: per-object decentralized commit vs single global commit-point\n"
                 "t_commit models real-backend persist/validate cost; same CAST semantics on both (fair)", fontsize=9.5, y=1.05)
    fig.tight_layout()
    out = os.path.join(RESULTS, "hybrid_cc.png"); fig.savefig(out, dpi=130, bbox_inches="tight"); print("\nsaved", out)


if __name__ == "__main__":
    main()
