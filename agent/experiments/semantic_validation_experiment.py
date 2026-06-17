"""P3：语义感知验证分级（路 C 的核心）——OCC vs MVCC-SI vs CAST 的"减少冲突"对比。

定位修正：CAST 不只是"冲突后用便宜方式解决"，而是【按读写语义在验证阶段分级】，
让本不该算冲突的不算冲突 —— 这才是"减少冲突、提吞吐"，也才与 MVCC 区分开。

验证严格度（决定一个任务是否被判冲突；批内任务读同一基线后依次提交，与更早任务比较）：
  - OCC      ：(读集 ∪ 写集) 命中任一被更早任务写过的对象 → 冲突（读集也严格）
  - MVCC-SI  ：写集命中被更早任务写过的对象 → 冲突（读放行；但 DELTA 也算写写）
  - CAST     ：仅 strict 写命中被更早 strict 写过的对象 → 冲突（读放行 + 可交换写放行）
冲突 → 重跑（makespan += t_gen，串行）。本实验为解析模型（与 2PL/MVCC 基线同口径）；
CAST 的"可交换写放行"在 sweep/vitabench_ota 已有真跑佐证（DELTA merge 等价于验证层放行）。
"""
import csv
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
T_GEN = 1.0


def gen_plan(n_objects, n_batches, batch_size, read_size, write_size, p_mergeable, seed):
    rng = random.Random(seed)
    objs = [f"o{i}" for i in range(n_objects)]
    plan = []
    for _ in range(n_batches):
        batch = []
        for _ in range(batch_size):
            picks = rng.sample(objs, min(read_size + write_size, n_objects))
            reads = set(picks[:read_size])
            writes = {}
            for o in picks[read_size:read_size + write_size]:
                writes[o] = "delta" if rng.random() < p_mergeable else "strict"
            batch.append((reads, writes))
        plan.append(batch)
    return plan


def evaluate(strategy, plan):
    conflicts = 0
    n = 0
    makespan = 0.0
    for batch in plan:
        any_written = set()
        strict_written = set()
        bconf = 0
        for reads, writes in batch:
            n += 1
            wobjs = set(writes)
            sobjs = {o for o, k in writes.items() if k == "strict"}
            if strategy == "OCC":
                hit = bool((reads | wobjs) & any_written)
            elif strategy == "MVCC":
                hit = bool(wobjs & any_written)            # 读放行；DELTA 仍算写写
            else:  # CAST：读放行 + 可交换写放行，只 strict-strict
                hit = bool(sobjs & strict_written)
            if hit:
                conflicts += 1
                bconf += 1
            any_written |= wobjs
            strict_written |= sobjs
        makespan += T_GEN + bconf * T_GEN   # 冲突任务串行重跑
    return {"abort_rate": conflicts / n, "throughput": n / makespan, "conflicts": conflicts, "n": n}


def main():
    N, B, NB, W, PM = 24, 8, 40, 2, 0.6
    reads = [0, 1, 2, 3]
    data = {s: [] for s in ("OCC", "MVCC", "CAST")}
    print(f"=== 语义验证分级（N={N}, batch={B}, write/任务={W}, 可合并占比={PM}）===")
    print(f"{'read_size':>9} | {'OCC':>16} | {'MVCC':>16} | {'CAST':>16}   (abort率, 吞吐)")
    for r in reads:
        plan = gen_plan(N, NB, B, r, W, PM, seed=7)
        row = {}
        for s in ("OCC", "MVCC", "CAST"):
            row[s] = evaluate(s, plan)
            data[s].append(row[s])
        print(f"{r:>9} | "
              f"{row['OCC']['abort_rate']:.2f},{row['OCC']['throughput']:.2f}      | "
              f"{row['MVCC']['abort_rate']:.2f},{row['MVCC']['throughput']:.2f}      | "
              f"{row['CAST']['abort_rate']:.2f},{row['CAST']['throughput']:.2f}")

    with open(os.path.join(RESULTS, "semantic_validation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["read_size", "strategy", "abort_rate", "throughput"])
        for i, r in enumerate(reads):
            for s in ("OCC", "MVCC", "CAST"):
                w.writerow([r, s, round(data[s][i]["abort_rate"], 4), round(data[s][i]["throughput"], 4)])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    style = {"OCC": ("o-", "tab:blue"), "MVCC": ("^--", "tab:purple"), "CAST": ("s-", "tab:green")}
    for s in ("OCC", "MVCC", "CAST"):
        st, c = style[s]
        a1.plot(reads, [d["abort_rate"] for d in data[s]], st, color=c, label=s, linewidth=2)
        a2.plot(reads, [d["throughput"] for d in data[s]], st, color=c, label=s, linewidth=2)
    a1.set_xlabel("read-set size per task"); a1.set_ylabel("conflict/abort rate"); a1.set_title("(a) conflicts — lower better"); a1.legend(); a1.grid(True, alpha=0.3)
    a2.set_xlabel("read-set size per task"); a2.set_ylabel("throughput"); a2.set_title("(b) throughput — higher better"); a2.legend(); a2.grid(True, alpha=0.3)
    fig.suptitle("P3 semantic-aware validation: OCC (strict reads+writes) vs MVCC-SI (reads free) vs CAST (reads + commutative writes pass)\n"
                 "CAST reduces conflicts at the validation stage (not just cheaper resolution) -> highest throughput", fontsize=9.5, y=1.05)
    fig.tight_layout()
    out = os.path.join(RESULTS, "semantic_validation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
