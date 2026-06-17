"""统一消融（A4）：从 OCC 逐步叠加 CAST 的各机制，分离每个机制的吞吐贡献。

同一真并发框架（线程 + 真实墙钟 + sleep 代表 c_gen），固定中等冲突，多 seed。
变体（递增）：
  V0 OCC          ：读集+写集全严格（ww or rw → abort）
  V1 +read-pass   ：读放行（只 ww → abort）          —— 读不参与冲突（MVCC-SI 同级）
  V2 +merge       ：可交换写放行/合并（只 sww → abort）—— 语义合并消掉可交换写写冲突
  V3 +reselect(=CAST)：在 V2 基础上 k 候选 reselect —— strict 冲突时复用其他候选兜底
每步增量 = 该机制的独立贡献。
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
C_GEN = 0.003
N_TASKS = 120
N_OBJ = 24
W, R, P_MERGE = 2, 1, 0.6
N_COUNTER = int(N_OBJ * P_MERGE)


def kind_of(o):
    return "delta" if int(o[1:]) < N_COUNTER else "strict"


def run(rule, k, seed):
    rng = random.Random(seed)
    store = {f"o{i}": [("1000" if i < N_COUNTER else "v0"), 0] for i in range(N_OBJ)}
    store_lock = threading.Lock()
    q = queue.Queue()
    for i in range(N_TASKS):
        q.put(i)
    committed = [0]; latencies = []; acc = threading.Lock()
    objs = [f"o{i}" for i in range(N_OBJ)]

    def gen(k_):
        cs = []
        for _ in range(k_):
            picks = rng.sample(objs, W + R)
            cs.append({"reads": set(picks[W:W + R]), "writes": [(o, kind_of(o)) for o in picks[:W]]})
        return cs

    def snap(c):
        return {o: store[o][1] for o in list(c["reads"]) + [o for o, _ in c["writes"]]}

    def aborts(c, base):
        changed = {o for o, v in base.items() if store[o][1] != v}
        RS = c["reads"]; WS = {o for o, _ in c["writes"]}; sWS = {o for o, kd in c["writes"] if kd == "strict"}
        ww, rw, sww = bool(WS & changed), bool(RS & changed), bool(sWS & changed)
        if rule == "ww_rw":
            return ww or rw
        if rule == "ww":
            return ww
        return sww  # "sww"

    def commit(c):
        for o, kd in c["writes"]:
            val, ver = store[o]
            store[o] = [(str(int(val) - 1) if kd == "delta" else "set"), ver + 1]

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            cs = gen(k); base = [snap(c) for c in cs]
            time.sleep(C_GEN)
            done = False
            with store_lock:
                for idx, c in enumerate(cs):
                    if not aborts(c, base[idx]):
                        commit(c); done = True; break
            if not done:
                time.sleep(C_GEN)
                with store_lock:
                    commit(cs[0])
            with acc:
                committed[0] += 1; latencies.append(time.perf_counter() - t0)
            q.task_done()

    w0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - w0
    return committed[0] / wall


def main():
    variants = [("V0 OCC", "ww_rw", 1), ("V1 +read-pass", "ww", 1),
                ("V2 +merge", "sww", 1), ("V3 +reselect (=CAST)", "sww", 4)]
    seeds = [1, 2, 3]
    names, tps = [], []
    print("=== 统一消融（中等冲突 n_obj=24, k(V3)=4, 真并发, 3 seeds）===")
    prev = None
    for name, rule, k in variants:
        vals = [run(rule, k, sd) for sd in seeds]
        m = statistics.mean(vals)
        names.append(name); tps.append(m)
        delta = f"  (+{m - prev:.0f})" if prev is not None else ""
        print(f"  {name:24} throughput = {m:7.0f}{delta}")
        prev = m
    print(f"  => 总提升 OCC→CAST: +{tps[-1] - tps[0]:.0f}（{(tps[-1]/tps[0]-1)*100:.0f}%）")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:blue", "tab:olive", "tab:purple", "tab:green"]
    bars = ax.bar(range(len(names)), tps, color=colors)
    for i, (b, v) in enumerate(zip(bars, tps)):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}", ha="center", fontsize=9)
        if i > 0:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"+{tps[i]-tps[i-1]:.0f}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylabel("throughput (committed/s, measured)")
    ax.set_title("Ablation: contribution of each CAST mechanism (real concurrency)\n"
                 "OCC → +read-pass → +semantic-merge → +multi-candidate reselect", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "ablation.png"); fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
