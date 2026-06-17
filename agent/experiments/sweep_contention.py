"""三方扫描 harness（阶段1）：strict OCC vs SCC-kS vs CAST 的【人均】浪费算力（单位 c_gen/任务）。

- OCC、CAST：真跑 C++ 核（commit_task）。
- SCC-kS：解析成本模型。SCC-kS 维护至多 k 个投机 shadow 赌序列化顺序：
    waste(SCC-kS) = (k-1)*c_gen*n_tasks      # k-1 个未被采纳的 shadow（每任务的冗余生成）
                  + #(冲突深度 d>=k)*c_gen   # 所有 shadow 没赌中 -> 重跑
  SCC-1S == OCC（对齐验证）。报告：SCC-2S（启用最小投机）与 SCC-best（k=1..8 最优）。
- 结构性冲突深度 d 来自确定性 task-plan，alignment：OCC 实测 regen 应≈ 结构冲突任务数。
- 指标用【人均】= 总浪费 / 任务数（不随任务规模缩放，便于解读与作图）。
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cast_core as cc
from agent.workloads.synthetic_contention import ContentionWorkload
from agent.scheduler.candidate_scheduler import CandidateScheduler

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_STRATS = ("OCC", "SCC_2S", "CAST")  # 作图用三条线（SCC_best 见 CSV/附注）


def run_strategy(strategy, wl, plan, c_gen, c_merge):
    model = cc.CostModel(c_gen, c_merge)
    store = cc.VersionedObjectStore()
    wl.seed_store(store)
    sched = CandidateScheduler(wl, k=1)
    stats = cc.CostStats()
    commit = cc.CostAsymmetricCommit(store, model)
    seq = 0
    for batch in plan:
        cand_lists = [sched.candidates_for(store, specs, seq + i) for i, specs in enumerate(batch)]
        seq += len(batch)
        for cands in cand_lists:
            commit.commit_task(cands, strategy, stats)
    return stats, model


def structural_depths(plan):
    depths = []
    for batch in plan:
        seen = {}
        for specs in batch:
            ws = [s[0] for s in specs]
            depths.append(max((seen.get(o, 0) for o in ws), default=0))
            for o in ws:
                seen[o] = seen.get(o, 0) + 1
    return depths


def scc_waste(depths, k, c_gen):
    n = len(depths)
    restarts = sum(1 for d in depths if d >= k)
    return (k - 1) * c_gen * n + restarts * c_gen


def scc_best(depths, c_gen, kmax=8):
    best_k, best_w = 1, scc_waste(depths, 1, c_gen)
    for k in range(2, kmax + 1):
        w = scc_waste(depths, k, c_gen)
        if w < best_w:
            best_k, best_w = k, w
    return best_k, best_w


def run_config(n_objects=10, batch_size=8, writes_per_task=3, p_mergeable=1.0,
               c_gen=1.0, c_merge=0.01, n_batches=30, seed=7):
    wl = ContentionWorkload(n_objects, writes_per_task, p_mergeable, seed)
    plan = wl.build_plan(n_batches, batch_size)
    depths = structural_depths(plan)
    occ, model = run_strategy(cc.CommitStrategy.kStrictOCC, wl, plan, c_gen, c_merge)
    cast, _ = run_strategy(cc.CommitStrategy.kCAST, wl, plan, c_gen, c_merge)
    n = max(occ.n_tasks, 1)
    sk, sw = scc_best(depths, c_gen)
    return {
        "OCC": round(occ.wasted_compute(model) / n, 4),
        "SCC_2S": round(scc_waste(depths, 2, c_gen) / n, 4),
        "SCC_best": round(sw / n, 4),
        "SCC_best_k": sk,
        "CAST": round(cast.wasted_compute(model) / n, 4),
        "occ_regen_measured": occ.n_regen,
        "struct_conflicts": sum(1 for d in depths if d > 0),
        "scc1_analytic": round(scc_waste(depths, 1, c_gen), 4),
        "occ_waste_total": round(occ.wasted_compute(model), 4),
    }


def sweep(name, xname, param, xvalues):
    rows, checks = [], []
    for x in xvalues:
        r = run_config(**{param: x})
        checks.append((r["occ_regen_measured"], r["struct_conflicts"],
                       r["occ_waste_total"], r["scc1_analytic"]))
        for strat in ("OCC", "SCC_2S", "SCC_best", "CAST"):
            rows.append({"sweep": name, "x_name": xname, "x_value": x, "strategy": strat,
                         "waste_per_task": r[strat],
                         "note": (f"k*={r['SCC_best_k']}" if strat == "SCC_best" else "")})
    path = os.path.join(RESULTS_DIR, f"sweep3_{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return name, xname, rows, checks


def print_summary(name, xname, rows, checks):
    print(f"\n=== sweep {name} ({xname}) | 人均浪费(c_gen/任务) ===")
    print(f"  {xname:>10} | {'OCC':>8} | {'SCC-2S':>8} | {'SCC-best':>8} | {'CAST':>8}")
    xs = sorted({r["x_value"] for r in rows})
    for x in xs:
        d = {r["strategy"]: r["waste_per_task"] for r in rows if r["x_value"] == x}
        kstar = next((r["note"] for r in rows if r["x_value"] == x and r["strategy"] == "SCC_best"), "")
        print(f"  {x:>10} | {d['OCC']:>8.3f} | {d['SCC_2S']:>8.3f} | {d['SCC_best']:>8.3f} ({kstar}) | {d['CAST']:>8.3f}")
    ok = all(rg == sc and abs(ow - s1) < 1e-6 for (rg, sc, ow, s1) in checks)
    print(f"  [alignment] OCC实测regen==结构冲突数 且 OCC浪费==SCC-1S解析 : {'PASS' if ok else 'CHECK'}")


def main():
    results = [
        sweep("A_concurrency", "batch_size", "batch_size", [1, 2, 4, 8, 16]),
        sweep("B_mergeable", "p_mergeable", "p_mergeable", [0.0, 0.25, 0.5, 0.75, 1.0]),
        sweep("C_asymmetry", "c_merge", "c_merge", [1.0, 0.5, 0.1, 0.01, 0.001]),
    ]
    for r in results:
        print_summary(*r)
    print(f"\nSCC-best 几乎总是退化为 k*=1（即 OCC）：每任务都付 (k-1) 份昂贵 shadow，")
    print(f"在 agent 成本下投机永不划算。CAST 用便宜的语义合并完胜。")
    print(f"CSV 输出目录: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
