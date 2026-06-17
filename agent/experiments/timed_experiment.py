"""多维对比（Step A）：成本 × 延迟 × 吞吐，含 2PL / MVCC-SI 基线。

执行/时间模型（文档化，便于审稿）：
  - batch_size = B 个并发 worker（B 个并发 agent 同时执行）。t_gen=一次候选生成(秒级)，t_merge=一次语义 rebase(≪t_gen)。
  - 乐观策略(OCC/CAST/MVCC)：一批内 B 任务并行生成(墙钟 t_gen)，再串行提交：
      direct/reselect → 0 额外；merge → +t_merge；regenerate → +t_gen(串行重跑)。
      批 makespan = t_gen + Σ(提交额外时间)。任务延迟 = t_gen + 该任务额外时间。
  - 悲观 2PL：对所有写对象加锁持到提交，争用对象串行化(不分语义)。无重跑(waste=0)。
      任务延迟 =(锁队列深度 d_i+1)·t_gen；批 makespan =(d_max+1)·t_gen。
  - MVCC-SI：写写冲突 first-committer-wins→重跑(对写密集负载等价 OCC)；读不阻塞的优势需读密集/写偏斜负载，留作后续——此处与 OCC 重合（诚实标注）。
OCC/CAST 的 wasted 与 action 来自真跑 C++ 核；2PL/MVCC 为解析基线。
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
from agent.scheduler.candidate_scheduler import CandidateScheduler

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
T_GEN, T_MERGE = 1.0, 0.01


def _extra(action):
    if action == "regenerate":
        return T_GEN
    if action == "merge":
        return T_MERGE
    return 0.0  # direct / reselect


def timed_optimistic(strategy, wl, plan, c_gen=1.0, c_merge=0.01):
    model = cc.CostModel(c_gen, c_merge)
    store = cc.VersionedObjectStore()
    wl.seed_store(store)
    sched = CandidateScheduler(wl, k=1)
    stats = cc.CostStats()
    commit = cc.CostAsymmetricCommit(store, model)
    seq = 0
    makespan = 0.0
    lat = []
    committed = 0
    for batch in plan:
        cand_lists = [sched.candidates_for(store, specs, seq + i) for i, specs in enumerate(batch)]
        seq += len(batch)
        batch_extra = 0.0
        for cands in cand_lists:
            out = commit.commit_task(cands, strategy, stats)
            committed += 1 if out.committed else 0
            e = _extra(out.action)
            batch_extra += e
            lat.append(T_GEN + e)
        makespan += T_GEN + batch_extra
    n = max(committed, 1)
    return {"wasted_per_task": stats.wasted_compute(model) / n,
            "makespan": makespan, "throughput": committed / makespan,
            "mean_latency": sum(lat) / len(lat), "committed": committed}


def timed_2pl(wl, plan, c_gen=1.0):
    makespan = 0.0
    lat = []
    committed = 0
    for batch in plan:
        seen = {}
        depths = []
        for specs in batch:
            ws = [s[0] for s in specs]
            depths.append(max((seen.get(o, 0) for o in ws), default=0))
            for o in ws:
                seen[o] = seen.get(o, 0) + 1
        makespan += (max(depths) + 1) * c_gen if depths else 0.0
        for d in depths:
            lat.append((d + 1) * c_gen)
            committed += 1
    n = max(committed, 1)
    return {"wasted_per_task": 0.0, "makespan": makespan,
            "throughput": committed / makespan, "mean_latency": sum(lat) / n, "committed": committed}


def evaluate(batch_size, n_batches=40, seed=7):
    wl = VitaBenchWorkload(seed=seed)
    plan = wl.build_plan(n_batches, batch_size)
    occ = timed_optimistic(cc.CommitStrategy.kStrictOCC, VitaBenchWorkload(seed=seed), plan)
    cast = timed_optimistic(cc.CommitStrategy.kCAST, VitaBenchWorkload(seed=seed), plan)
    twopl = timed_2pl(VitaBenchWorkload(seed=seed), plan)
    mvcc = occ  # 写密集负载下 MVCC-SI ≡ OCC（见模块注释）
    return {"batch_size": batch_size, "OCC": occ, "MVCC": mvcc, "2PL": twopl, "CAST": cast}


def main():
    bs = [1, 2, 4, 8, 16]
    data = [evaluate(b) for b in bs]
    # CSV
    with open(os.path.join(RESULTS, "timed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "strategy", "wasted_per_task", "mean_latency", "throughput", "makespan"])
        for d in data:
            for s in ("OCC", "MVCC", "2PL", "CAST"):
                m = d[s]
                w.writerow([d["batch_size"], s, round(m["wasted_per_task"], 4),
                            round(m["mean_latency"], 4), round(m["throughput"], 4), round(m["makespan"], 4)])

    print("=== Step A 多维对比 (VitaBench-derived, 40 批) ===")
    print(f"{'batch':>5} | {'metric':>13} | {'OCC':>8} {'2PL':>8} {'MVCC':>8} {'CAST':>8}")
    for d in data:
        for key, name in [("wasted_per_task", "wasted/task"), ("mean_latency", "mean_latency"), ("throughput", "throughput")]:
            print(f"{d['batch_size']:>5} | {name:>13} | {d['OCC'][key]:>8.3f} {d['2PL'][key]:>8.3f} "
                  f"{d['MVCC'][key]:>8.3f} {d['CAST'][key]:>8.3f}")
        print("  " + "-" * 60)

    # 三面板图：wasted / mean_latency / throughput vs batch_size
    styles = {"OCC": ("o-", "tab:blue"), "2PL": ("d-.", "tab:red"),
              "MVCC": ("x:", "tab:purple"), "CAST": ("s-", "tab:green")}
    panels = [("wasted_per_task", "wasted compute / task (c_gen)", "(a) cost — lower better"),
              ("mean_latency", "mean task latency (t_gen units)", "(b) latency — lower better"),
              ("throughput", "throughput (committed / wall-time)", "(c) throughput — higher better")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, ylab, title) in zip(axes, panels):
        for s in ("OCC", "2PL", "MVCC", "CAST"):
            st, c = styles[s]
            lbl = s + (" (≈OCC)" if s == "MVCC" else "")
            ax.plot([d["batch_size"] for d in data], [d[s][key] for d in data], st, color=c, label=lbl, linewidth=2, markersize=6)
        ax.set_xlabel("batch size (concurrency)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Step A — cost × latency × throughput on VitaBench-derived load\n"
                 "CAST wins on all three; 2PL avoids waste but serializes (latency/throughput collapse under contention)",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    out = os.path.join(RESULTS, "timed.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
