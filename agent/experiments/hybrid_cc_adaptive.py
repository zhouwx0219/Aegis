"""混合并发控制（Paper B 核心）：per-object / per-intent 自适应 CC 分发器。

核心论点（对应导师愿景"按对象/意图选 CC 策略"）：
  在一个**异构对象群**（strict 行 / 无约束可交换计数器 / 带下界约束的库存 / 只读）上，
  **没有任何单一并发协议同时做到「快」且「正确」**：
    - OCC / MVCC（统一乐观版本校验）：对可交换写一律 abort→重跑(c_gen) → 正确但慢；
    - 2PL（统一悲观锁）：串行化 → 正确但延迟高、吞吐低；
    - CAST-merge-all（统一语义合并）：可交换写全 merge → 快，但把**带约束的扣减也盲并**，
      在需求>库存时 **超卖**（违反 stock>=0，见 correctness_boundary.py 的 5-8=-3）→ 快但错；
    - **HYBRID（本工作）**：按对象意图类路由——
        READ        → 快照放行(SI，不 abort)
        COMM_FREE   → 语义合并(CRDT rebase，永不 abort)        ← 计数器/点赞
        COMM_CONSTR → escrow 额度预留(可交换但带下界守卫)        ← 库存(stock>=0)
        CAS         → 条件重绑定(提交点重检)
        STRICT      → OCC 版本校验(冲突→reselect→regen)         ← 覆盖写
      ⇒ 在可合并/escrow 路径上拿到 CAST 的速度，在约束对象上拿到 escrow 的正确性。

结论形态（Pareto）：HYBRID 是 (吞吐, 正确性=零超卖) 平面上唯一的 Pareto 最优点——
  与 CAST-merge-all 同速但零超卖；比 OCC/MVCC/2PL 快且同样正确。

测量方法（与 cc_comparison.py / concurrent_harness.py 同口径，公平）：
  真多线程 + 真实墙钟；sleep 代表 c_gen(秒级 LLM)，commit 临界区在内存(≈0)；
  每任务必提交一个候选，冲突→reselect 其他候选(≈0)→都不行→regenerate(再花 c_gen)。
  吞吐=committed/wall，延迟=每任务墙钟均值；waste=regen 次数×c_gen；多 seed 报均值。
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

C_GEN = 0.003          # 候选生成成本（秒级 LLM 的代表），merge/escrow≈0
N_TASKS = 120
W = 2                  # 写/候选
R = 1                  # 读/候选
K = 3                  # 候选分支数（乐观策略用，2PL 不投机）
# 异构对象群混合占比：strict / 无约束可交换 / 带约束可交换 / 只读
MIX = {"strict": 0.30, "commfree": 0.30, "commconstr": 0.25, "read": 0.15}
POLICIES = ["OCC", "MVCC", "2PL", "CAST-all", "HYBRID"]


def build_pool(n_obj, seed):
    """把对象池按 MIX 分类；带约束对象给定初始库存 S0（故意< 总需求 → 暴露超卖）。"""
    rng = random.Random(seed * 7919 + 17)
    classes = []
    for name, frac in MIX.items():
        classes += [name] * max(1, round(n_obj * frac))
    classes = (classes + ["strict"] * n_obj)[:n_obj]
    rng.shuffle(classes)
    cls = {f"o{i}": classes[i] for i in range(n_obj)}
    n_constr = sum(1 for c in classes if c == "commconstr")
    # 每个约束对象的预期提交扣减量；S0 设成约 0.5× → 需求约 2× 库存，CAST-all 必超卖
    demand_per = (N_TASKS * W * MIX["commconstr"]) / max(1, n_constr)
    s0 = max(3, int(0.5 * demand_per))
    store = {}
    for o, c in cls.items():
        if c == "commfree":
            store[o] = [0, 0]            # 计数器，无下界
        elif c == "commconstr":
            store[o] = [s0, 0]           # 库存，下界 0
        elif c == "strict":
            store[o] = ["v0", 0]
        else:
            store[o] = ["r", 0]          # 只读
    return store, cls, s0


def gen_candidates(rng, store, cls, k):
    objs = list(store)
    cands = []
    for _ in range(k):
        picks = rng.sample(objs, min(W + R, len(objs)))
        writes = [o for o in picks[:W] if cls[o] != "read"]
        reads = set(picks[W:W + R]) | {o for o in picks[:W] if cls[o] == "read"}
        cands.append({"reads": reads, "writes": writes})
    return cands


def snap(store, cand):
    objs = list(cand["reads"]) + cand["writes"]
    return {o: store[o][1] for o in objs}


def apply_write(store, o, cls, policy, ctr):
    """按对象类 + 策略落库；返回是否真正应用了写。记 ctr 的 merge/escrow/oversell。"""
    c = cls[o]
    val, ver = store[o]
    if c == "strict":
        store[o] = ["set", ver + 1]
        return True
    if c == "commfree":                       # 无约束可交换 → 合并(累加)，永远安全
        store[o] = [val + 1, ver + 1]
        ctr["merge"] += 1
        return True
    if c == "commconstr":                     # 带下界约束的扣减
        if policy == "CAST-all":              # 盲并：不查下界 → 可超卖
            nv = val - 1
            if nv < 0:
                ctr["oversell"] += 1          # 违反 stock>=0（正确性事故）
            store[o] = [nv, ver + 1]
            ctr["merge"] += 1
            return True
        # HYBRID=escrow / OCC / MVCC / 2PL：带守卫的扣减（库存耗尽则正确拒绝，不超卖）
        if val > 0:
            store[o] = [val - 1, ver + 1]
            if policy == "HYBRID":
                ctr["escrow_grant"] += 1
            return True
        if policy == "HYBRID":
            ctr["escrow_reject"] += 1         # 正确的缺货拒绝：非浪费、非超卖
        return True
    return True                               # read：不写


def aborts(policy, cand, changed, cls):
    """提交点是否判该候选冲突（需 reselect/regen）。"""
    RS = cand["reads"]
    WS = set(cand["writes"])
    sWS = {o for o in cand["writes"] if cls[o] == "strict"}
    if policy == "OCC":
        return bool((RS | WS) & changed)      # 读集+写集全严格
    if policy == "MVCC":
        return bool(WS & changed)             # 读放行(快照)，写写仍冲突
    if policy in ("CAST-all", "HYBRID"):
        return bool(sWS & changed)            # 读放行 + 可交换写放行，仅 strict-strict
    return False                              # 2PL 在锁内提交，无 abort


def run(policy, n_obj, n_threads, seed):
    rng_pool = random.Random(seed)
    store, cls, s0 = build_pool(n_obj, seed)
    store_lock = threading.Lock()
    obj_locks = {o: threading.Lock() for o in store}
    q = queue.Queue()
    for i in range(N_TASKS):
        q.put(i)
    committed = [0]
    lat = []
    ctr = {"regen": 0, "reselect": 0, "merge": 0, "escrow_grant": 0,
           "escrow_reject": 0, "oversell": 0}
    acc = threading.Lock()
    rng_local = threading.local()

    def rng():
        r = getattr(rng_local, "r", None)
        if r is None:
            r = rng_local.r = random.Random(seed * 1009 + threading.get_ident() % 9973)
        return r

    def commit_one(cand, base):
        # base 是【生成时】拍下的版本快照；c_gen 期间并发提交会让它过时 → 真实冲突
        changed = {o for o, v in base.items() if store[o][1] != v}
        if aborts(policy, cand, changed, cls):
            return False
        for o in cand["writes"]:
            apply_write(store, o, cls, policy, ctr)
        return True

    def worker():
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            if policy == "2PL":                       # 悲观：锁住涉及对象→生成→提交（串行化热点）
                cand = gen_candidates(rng(), store, cls, 1)[0]
                objs = sorted(set(cand["writes"]) | cand["reads"])
                locks = [obj_locks[o] for o in objs]
                for l in locks:
                    l.acquire()
                try:
                    time.sleep(C_GEN)
                    with store_lock:
                        for o in cand["writes"]:
                            apply_write(store, o, cls, policy, ctr)
                    with acc:
                        committed[0] += 1
                finally:
                    for l in reversed(locks):
                        l.release()
            else:
                cands = gen_candidates(rng(), store, cls, K)
                bases = [snap(store, c) for c in cands]   # 生成时拍快照（提交前会过时→真冲突）
                time.sleep(C_GEN)                     # 生成 K 候选（并发重叠）
                done = False
                with store_lock:
                    for idx, c in enumerate(cands):   # winner + reselect 其他候选(≈0)
                        if commit_one(c, bases[idx]):
                            done = True
                            with acc:
                                committed[0] += 1
                                if idx > 0:
                                    ctr["reselect"] += 1
                            break
                if not done:                          # 全冲突 → regenerate（再花 c_gen）
                    time.sleep(C_GEN)
                    with store_lock:
                        commit_one(cands[0], snap(store, cands[0]))  # 重读最新基线，必不冲突
                        with acc:
                            committed[0] += 1
                            ctr["regen"] += 1
            dt = time.perf_counter() - t0
            with acc:
                lat.append(dt)
            q.task_done()

    w0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - w0
    return {
        "throughput": committed[0] / wall,
        "latency_ms": statistics.mean(lat) * 1000 if lat else 0.0,
        "regen": ctr["regen"], "reselect": ctr["reselect"], "merge": ctr["merge"],
        "escrow_grant": ctr["escrow_grant"], "escrow_reject": ctr["escrow_reject"],
        "oversell": ctr["oversell"], "s0": s0,
    }


def main():
    n_obj = 24
    threads_list = [1, 2, 4, 8, 16]
    seeds = [1, 2, 3]
    agg = {p: {"tp": [], "lat": [], "oversell": []} for p in POLICIES}
    detail8 = {}   # 8 线程处的动作分解 + 正确性（取均值）

    print(f"=== 混合CC：per-object/intent 自适应分发 vs 统一单协议（n_obj={n_obj}, "
          f"mix={MIX}, K={K}, {len(seeds)} seeds）===")
    print(f"对象群：strict(覆盖)/commfree(计数器)/commconstr(库存,下界0)/read(只读)；"
          f"库存初值 S0≈需求的0.5×（故意暴露超卖）\n")
    header = f"{'threads':>7} | " + " | ".join(f"{p:>9}" for p in POLICIES) + "   (throughput committed/s)"
    print(header)
    for nt in threads_list:
        row = {}
        for p in POLICIES:
            tps, las, ovs, runs = [], [], [], []
            for sd in seeds:
                m = run(p, n_obj, nt, sd)
                tps.append(m["throughput"]); las.append(m["latency_ms"])
                ovs.append(m["oversell"]); runs.append(m)
            agg[p]["tp"].append(statistics.mean(tps))
            agg[p]["lat"].append(statistics.mean(las))
            agg[p]["oversell"].append(statistics.mean(ovs))
            row[p] = statistics.mean(tps)
            if nt == 8:
                detail8[p] = {k: statistics.mean([r[k] for r in runs])
                              for k in ("regen", "reselect", "merge",
                                        "escrow_grant", "escrow_reject", "oversell")}
                detail8[p]["throughput"] = statistics.mean(tps)
                detail8[p]["latency_ms"] = statistics.mean(las)
        print(f"{nt:>7} | " + " | ".join(f"{row[p]:>9.0f}" for p in POLICIES))

    print(f"\n=== 8 线程：正确性 + 动作分解（{len(seeds)} seeds 均值）===")
    print(f"{'policy':>9} | {'tp':>7} {'lat(ms)':>8} | {'oversell':>8} {'regen':>6} "
          f"{'merge':>6} {'escrowOK':>8} {'escrowRej':>9}  | 评价")
    verdict = {
        "OCC": "正确但慢(可交换写全重跑)",
        "MVCC": "正确但慢(写写重跑,读放行)",
        "2PL": "正确但延迟高(锁串行)",
        "CAST-all": "快但【超卖→错误】",
        "HYBRID": "★ 快 且 正确(零超卖)",
    }
    for p in POLICIES:
        d = detail8[p]
        print(f"{p:>9} | {d['throughput']:>7.0f} {d['latency_ms']:>8.2f} | "
              f"{d['oversell']:>8.1f} {d['regen']:>6.0f} {d['merge']:>6.0f} "
              f"{d['escrow_grant']:>8.0f} {d['escrow_reject']:>9.0f}  | {verdict[p]}")

    # ---- 4 面板出版风格图 ----
    style = {"OCC": ("o-", "tab:blue"), "MVCC": ("D-", "tab:olive"),
             "2PL": ("d-.", "tab:red"), "CAST-all": ("^--", "tab:orange"),
             "HYBRID": ("s-", "tab:green")}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) throughput vs threads
    for p in POLICIES:
        st, c = style[p]
        ax[0, 0].plot(threads_list, agg[p]["tp"], st, color=c, label=p, linewidth=2, markersize=6)
    ax[0, 0].set_xlabel("concurrent agents (threads)")
    ax[0, 0].set_ylabel("throughput (committed/s, measured)")
    ax[0, 0].set_title("(a) Throughput — HYBRID matches CAST-all, beats OCC/MVCC/2PL")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.3)

    # (b) oversell (correctness) vs threads
    for p in POLICIES:
        st, c = style[p]
        ax[0, 1].plot(threads_list, agg[p]["oversell"], st, color=c, label=p, linewidth=2, markersize=6)
    ax[0, 1].set_xlabel("concurrent agents (threads)")
    ax[0, 1].set_ylabel("oversell events (stock < 0)  [lower=correct]")
    ax[0, 1].set_title("(b) Correctness — only CAST-all violates the inventory bound")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, alpha=0.3)

    # (c) action breakdown @8 threads (stacked)
    labels = POLICIES
    regen = [detail8[p]["regen"] for p in labels]
    merge = [detail8[p]["merge"] for p in labels]
    escrow = [detail8[p]["escrow_grant"] for p in labels]
    resel = [detail8[p]["reselect"] for p in labels]
    x = range(len(labels))
    b1 = ax[1, 0].bar(x, merge, color="tab:green", label="merge (commutative, ~0 cost)")
    bottom = list(merge)
    b2 = ax[1, 0].bar(x, escrow, bottom=bottom, color="tab:cyan", label="escrow grant (constrained, ~0 cost)")
    bottom = [bottom[i] + escrow[i] for i in range(len(labels))]
    b3 = ax[1, 0].bar(x, resel, bottom=bottom, color="tab:purple", label="reselect (reuse candidate, ~0)")
    bottom = [bottom[i] + resel[i] for i in range(len(labels))]
    b4 = ax[1, 0].bar(x, regen, bottom=bottom, color="tab:red", label="regenerate (re-run LLM, c_gen)")
    ax[1, 0].set_xticks(list(x)); ax[1, 0].set_xticklabels(labels, fontsize=8)
    ax[1, 0].set_ylabel("resolution actions @8 threads (count)")
    ax[1, 0].set_title("(c) How each policy resolves conflicts — HYBRID uses cheap merge+escrow, not regen")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, axis="y", alpha=0.3)

    # (d) Pareto: throughput vs oversell @8 threads
    for p in POLICIES:
        st, c = style[p]
        ax[1, 1].scatter(detail8[p]["oversell"], detail8[p]["throughput"],
                         s=140, color=c, edgecolor="black", zorder=3, label=p)
        ax[1, 1].annotate(p, (detail8[p]["oversell"], detail8[p]["throughput"]),
                          textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax[1, 1].set_xlabel("oversell events  →  (right = incorrect)")
    ax[1, 1].set_ylabel("throughput  →  (up = faster)")
    ax[1, 1].set_title("(d) Pareto: HYBRID alone is fast AND correct (top-left)")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].axvline(0.5, ls=":", color="gray")

    fig.suptitle("Hybrid concurrency control: per-object / per-intent adaptive protocol selection\n"
                 "heterogeneous object pool (strict / commutative / constrained-inventory / read-only); "
                 "real multi-threaded, wall-clock measured", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS, "hybrid_cc_adaptive.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
