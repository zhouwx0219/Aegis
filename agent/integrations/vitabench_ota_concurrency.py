"""真实 VitaBench 负载 × 真并发并发控制（完整 CC baseline 家族 + 我们的 hybrid CC）。

升级（2026-06-17）：
  - 负载来自**真实 VitaBench OTA 环境的真实库存**：航班座位 / 酒店房间 / 景点门票 / 火车座位
    四类共享资源（create_*_order 经 use_tool 实测对共享 quantity 做 DELTA 扣减，带 stock>=0）；
  - delivery 为**私有订单**（每单独立、无共享扣减）作对照；
  - **完整 baseline 家族**：OCC / Silo / TicToc / MVCC / 2PL，外加 **HYBRID（我们的 hybrid CC：
    读放行 + 可交换写放行/merge，仅 strict-strict 才判冲突）**；
  - 真多线程 + 真实墙钟；5 seeds + 95%CI；统一坐标（eval_common）。
口径：真实负载 + 真并发实测 + 5seed±95%CI；commit 临界区在内存（c_merge≈0），sleep 代表 c_gen。
注：本负载的写均为可交换 DELTA（订票扣共享库存），故 syntactic CC（OCC/Silo/TicToc/MVCC）对写写冲突
   一律 abort→重跑；HYBRID 对可交换写放行→merge。每任务附 R 个读，OCC/Silo 读冲突也 abort、MVCC/TicToc 读放行。
"""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eval_common as E

RESULTS = os.path.join(ROOT, "agent", "experiments", "results")
os.makedirs(RESULTS, exist_ok=True)

C_GEN = 0.003
N_TASKS = 120
N_THREADS = 16
HOT = 12
W, R = 1, 1
SEEDS = [1, 2, 3, 4, 5]
CATS = ["flights", "hotels", "attractions", "trains"]
STRATS = ["OCC", "Silo", "TicToc", "MVCC", "2PL", "HYBRID"]


def collect(category):
    """从真实 OTA 任务采集某类资源的真实库存 (oid, quantity, price)，截取热点池。"""
    from vita.domains.ota.environment import get_tasks
    tasks = get_tasks("english")
    pool = []
    for t in tasks:
        for _id, obj in (t.environment.get(category) or {}).items():
            for p in (obj.get("products") or []):
                q = int(p.get("quantity", 0))
                if q > 0:
                    pool.append((f"{category}:{_id}:{p['product_id']}", q, float(p.get("price", 0))))
        if len(pool) >= HOT * 4:
            break
    return pool[:HOT]


def verify_decrement():
    """真实性证据：跑一次真实 create_flight_order，确认 quantity 被扣减（DELTA）。"""
    from vita.domains.ota.environment import get_environment, get_tasks
    from deepdiff import DeepDiff
    tasks = get_tasks("english")
    task = next(t for t in tasks if t.environment.get("flights"))
    env = get_environment(task.environment, "english")
    db = env.tools.db
    fid, f = next(iter(task.environment["flights"].items()))
    p0 = f["products"][0]
    b = json.loads(db.model_dump_json())
    env.use_tool("create_flight_order", flight_id=fid, seat_id=p0["product_id"],
                 user_id=task.environment.get("user_id", "U1"),
                 date=str(p0.get("date", "2026-08-01"))[:10], quantity=2)
    a = json.loads(db.model_dump_json())
    d = DeepDiff(b, a, verbose_level=2)
    return any("quantity" in str(p) for p in (d.get("values_changed", {}) or {}))


def aborts(strat, read_changed, write_changed):
    """提交点冲突判定。本负载写均可交换、无 strict 写 → HYBRID 永不冲突（语义放行）。"""
    if strat in ("OCC", "Silo"):
        return read_changed or write_changed       # 读集+写集全严格
    if strat in ("MVCC", "TicToc"):
        return write_changed                        # 读放行，写写仍冲突
    if strat == "HYBRID":
        return False                                # 读放行 + 可交换写放行（仅 strict-strict 冲突）
    return False                                    # 2PL：锁，无 abort


def run(strat, pool, seed, private=False):
    rng = random.Random(seed)
    if private:
        store = {}
    else:
        store = {oid: [q * 50, 0] for oid, q, _ in pool}
    oids = list(store)
    glock = threading.Lock()
    objlocks = {oid: threading.Lock() for oid in store}
    q = queue.Queue()
    for i in range(N_TASKS):
        q.put(i)
    committed = [0]; regen = [0]; merge = [0]; lat = []
    acc = threading.Lock()

    def gen(tag):
        if private:
            return [], [(f"order:{seed}:{tag}", rng.choice([1, 2]))]
        wobj = [(rng.choice(oids), rng.choice([1, 2])) for _ in range(W)]
        robj = [rng.choice(oids) for _ in range(R)]
        return robj, wobj

    def apply_writes(wobj):
        for oid, dq in wobj:
            v = store[oid]; store[oid] = [v[0] - dq, v[1] + 1]

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            robj, wobj = gen(i)
            if private:
                time.sleep(C_GEN)
                with glock:
                    for oid, dq in wobj:
                        store[oid] = [-dq, 1]
                with acc:
                    committed[0] += 1
            elif strat == "2PL":
                objs = sorted(set(o for o, _ in wobj) | set(robj))
                locks = [objlocks[o] for o in objs]
                for l in locks:
                    l.acquire()
                try:
                    time.sleep(C_GEN)
                    with glock:
                        apply_writes(wobj)
                    with acc:
                        committed[0] += 1
                finally:
                    for l in reversed(locks):
                        l.release()
            else:
                rbase = {o: store[o][1] for o in robj}
                wbase = {o: store[o][1] for o, _ in wobj}
                time.sleep(C_GEN)
                did_regen = False
                with glock:
                    rch = any(store[o][1] != rbase[o] for o in robj)
                    wch = any(store[o][1] != wbase[o] for o, _ in wobj)
                    if aborts(strat, rch, wch):
                        did_regen = True
                    else:
                        apply_writes(wobj)
                        with acc:
                            committed[0] += 1
                            if strat == "HYBRID" and wch:
                                merge[0] += 1
                if did_regen:
                    time.sleep(C_GEN)
                    with glock:
                        apply_writes(wobj)
                    with acc:
                        committed[0] += 1
                        regen[0] += 1
            with acc:
                lat.append(time.perf_counter() - t0)
            q.task_done()

    w0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - w0
    n = max(committed[0], 1)
    return {"throughput": committed[0] / wall, "waste": regen[0] * C_GEN / n,
            "merge": merge[0], "regen": regen[0]}


def main():
    ok = verify_decrement()
    print(f"真实性验证：create_flight_order 实测扣减 quantity(DELTA) = {ok}")
    pools = {c: collect(c) for c in CATS}
    for c in CATS:
        print(f"  采集真实库存 [{c}]: {len(pools[c])} 个热点资源")
    cats = CATS + ["delivery(private)"]

    agg = {}
    print(f"\n=== 真实负载真并发（{N_THREADS} 线程, N={N_TASKS}, HOT={HOT}, {len(SEEDS)} seeds ±95%CI）===")
    print(f"{'category':>16} | " + " | ".join(f"{s:>7}" for s in STRATS) + "   (throughput committed/s)")
    for cat in cats:
        agg[cat] = {}
        private = cat.startswith("delivery")
        pool = [] if private else pools[cat]
        row = {}
        for s in STRATS:
            tps, ws, mg, rg = [], [], [], []
            for sd in SEEDS:
                r = run(s, pool, sd, private=private)
                tps.append(r["throughput"]); ws.append(r["waste"]); mg.append(r["merge"]); rg.append(r["regen"])
            tp_m, tp_ci = E.mean_ci(tps); w_m, w_ci = E.mean_ci(ws)
            agg[cat][s] = {"tp": (tp_m, tp_ci), "waste": (w_m, w_ci)}
            row[s] = tp_m
        print(f"{cat:>16} | " + " | ".join(f"{row[s]:>7.0f}" for s in STRATS))

    import numpy as np
    x = np.arange(len(cats)); width = 0.13
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    for j, s in enumerate(STRATS):
        c = E.color_of(s)
        off = (j - (len(STRATS) - 1) / 2) * width
        tp = [agg[cat][s]["tp"][0] for cat in cats]; tpe = [agg[cat][s]["tp"][1] for cat in cats]
        wa = [agg[cat][s]["waste"][0] for cat in cats]; wae = [agg[cat][s]["waste"][1] for cat in cats]
        a1.bar(x + off, tp, width, yerr=tpe, capsize=2, color=c, label=s)
        a2.bar(x + off, wa, width, yerr=wae, capsize=2, color=c, label=s)
    for ax, ylab, title in [(a1, "throughput (committed/s, measured)", "(a) Throughput - higher better"),
                            (a2, "wasted compute / task (c_gen units)", "(b) Wasted LLM compute - lower better")]:
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.legend(fontsize=8, ncol=2); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Real VitaBench load x real-concurrency CC family: OCC/Silo/TicToc/MVCC/2PL vs our HYBRID CC\n"
                 "4 shared-resource categories (OTA) + private delivery contrast; syntactic CC abort on commutative conflicts, HYBRID merges - " + E.CI_NOTE,
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    out = os.path.join(RESULTS, "vitabench_ota.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
