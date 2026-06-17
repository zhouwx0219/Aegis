"""VitaBench-derived 负载上的三方对比（阶段1）。

复用 sweep_contention 的真跑/解析函数。报告：
  - 自然可合并写占比（由领域语义产生，非人为调参）
  - 冲突率
  - OCC / SCC-2S / SCC-best / CAST 的人均浪费算力
输出：results/vitabench.csv + results/vitabench.png（左：主配置柱状；右：并发度趋势）
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cast_core as cc
from agent.workloads.vitabench_workload import VitaBenchWorkload
from agent.experiments.sweep_contention import run_strategy, structural_depths, scc_waste, scc_best

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def mergeable_fraction(plan):
    tot = merg = cas = 0
    for batch in plan:
        for specs in batch:
            for (_oid, it, _init) in specs:
                tot += 1
                if it in (cc.IntentType.kDelta, cc.IntentType.kAppend):
                    merg += 1
                elif it == cc.IntentType.kCas:
                    cas += 1
    return merg / tot, cas / tot


def evaluate(batch_size, n_batches=40, seed=7, c_gen=1.0, c_merge=0.01):
    wl = VitaBenchWorkload(seed=seed)
    plan = wl.build_plan(n_batches, batch_size)
    depths = structural_depths(plan)
    occ, model = run_strategy(cc.CommitStrategy.kStrictOCC, wl, plan, c_gen, c_merge)
    cast, _ = run_strategy(cc.CommitStrategy.kCAST, wl, plan, c_gen, c_merge)
    n = max(occ.n_tasks, 1)
    sk, sw = scc_best(depths, c_gen)
    mf, cf = mergeable_fraction(plan)
    return {
        "batch_size": batch_size, "n_tasks": n,
        "mergeable_frac": round(mf, 4), "cas_frac": round(cf, 4),
        "conflict_rate": round(sum(1 for d in depths if d > 0) / len(depths), 4),
        "OCC": round(occ.wasted_compute(model) / n, 4),
        "SCC_2S": round(scc_waste(depths, 2, c_gen) / n, 4),
        "SCC_best": round(sw / n, 4), "SCC_best_k": sk,
        "CAST": round(cast.wasted_compute(model) / n, 4),
        "cast_merge": cast.n_merge, "cast_regen": cast.n_regen,
    }


def main():
    main_cfg = evaluate(batch_size=8)
    print("=== VitaBench-derived 负载：主配置 (batch_size=8, 40 批, 共 %d 任务) ===" % main_cfg["n_tasks"])
    print(f"  自然可合并写占比 = {main_cfg['mergeable_frac']:.1%}  (CAS 占比 = {main_cfg['cas_frac']:.1%})")
    print(f"  冲突率 = {main_cfg['conflict_rate']:.1%}")
    print(f"  人均浪费(c_gen/任务): OCC={main_cfg['OCC']:.3f}  SCC-2S={main_cfg['SCC_2S']:.3f}  "
          f"SCC-best={main_cfg['SCC_best']:.3f}(k*={main_cfg['SCC_best_k']})  CAST={main_cfg['CAST']:.3f}")
    save = (1 - main_cfg["CAST"] / main_cfg["OCC"]) * 100 if main_cfg["OCC"] else 0
    print(f"  => CAST 比 OCC 省 {save:.1f}% 浪费算力 (merge={main_cfg['cast_merge']}, regen={main_cfg['cast_regen']})")

    rows = [evaluate(b) for b in [1, 2, 4, 8, 16]]
    with open(os.path.join(RESULTS, "vitabench.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 图：左=主配置柱状；右=并发度趋势
    fig, (axb, axl) = plt.subplots(1, 2, figsize=(12, 4.3))
    strat = ["OCC", "SCC_2S", "CAST"]
    vals = [main_cfg[s] for s in strat]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    axb.bar(["OCC", "SCC-2S", "CAST"], vals, color=colors)
    for i, v in enumerate(vals):
        axb.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axb.set_ylabel("wasted compute per task (units of c_gen)")
    axb.set_title("(a) VitaBench-derived load (batch=8)\nnatural mergeable fraction = %.0f%%" % (main_cfg["mergeable_frac"] * 100), fontsize=10)
    axb.grid(True, axis="y", alpha=0.3)

    bs = [r["batch_size"] for r in rows]
    for s, c, mk in [("OCC", "tab:blue", "o-"), ("SCC_2S", "tab:orange", "^--"), ("CAST", "tab:green", "s-")]:
        axl.plot(bs, [r[s] for r in rows], mk, color=c, label=s.replace("_2S", "-2S"), linewidth=2)
    axl.set_xlabel("batch size (concurrency)")
    axl.set_ylabel("wasted compute per task (units of c_gen)")
    axl.set_title("(b) vs concurrency on VitaBench-derived load", fontsize=10)
    axl.legend()
    axl.grid(True, alpha=0.3)
    fig.suptitle("CAST on a VitaBench-derived workload (domain-natural mergeable writes)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS, "vitabench.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
