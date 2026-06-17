"""探索式多候选实验（Step：winner 选择 + LLM 延迟）。

证明在【纯 strict】独占资源负载上（语义合并完全用不上，n_merge=0），CAST 的收益【只】来自
"异构多候选 + reselect"：winner 资源被并发占走时，CAST 复用本轮已生成的其他候选(免费 reselect)，
而 OCC 必须重新探索(重跑 LLM)。这把"探索式多候选"卖点与语义合并彻底分离地立起来。

LLM 思考延迟：每个候选生成耗时 ~ 对数正态(median≈t_gen，有方差)；一轮 k 候选并行生成，墙钟=最慢者。
执行模型：批内 B 任务读同一基线生成 k 个异构候选，依次提交；CAST winner 冲突→reselect(本轮内)；
OCC winner 冲突→提交失败→重新探索(下一轮，重跑 k 候选)，直到成功。
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

import cast_core as cc
from agent.workloads.explore_workload import ExploreWorkload

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
MU, SIGMA = 0.0, 0.5  # 对数正态：median t_gen=1, 适度方差


def run_explore(strategy, n_resources, batch_size, n_batches=40, k=4, seed=3, lat_seed=11,
                c_gen=1.0, c_merge=0.01):
    wl = ExploreWorkload(n_resources=n_resources, k=k, seed=seed)
    latrng = random.Random(lat_seed)
    model = cc.CostModel(c_gen, c_merge)
    store = cc.VersionedObjectStore()
    wl.seed_store(store)
    stats = cc.CostStats()
    commit = cc.CostAsymmetricCommit(store, model)
    total_gen = 0
    committed = 0
    reselects = 0
    rounds_total = 0
    makespan = 0.0
    latencies = []
    seq = 0
    for _ in range(n_batches):
        wl.seed_store(store)  # 每批重置资源为 free：模拟一个独立的争用时段（批间独立）
        task_cands = [wl.gen_candidates(store, seq + i) for i in range(batch_size)]
        seq += batch_size
        batch_lat = []
        for cands in task_cands:
            cur = cands
            rounds = 0
            done = False
            tlat = 0.0
            while not done and rounds < 30:
                rounds += 1
                total_gen += len(cur)
                tlat += max(latrng.lognormvariate(MU, SIGMA) for _ in cur)  # k 并行生成，墙钟=最慢
                out = commit.commit_task(cur, strategy, stats)
                if out.committed:
                    done = True
                    if out.action == "reselect":
                        reselects += 1
                else:
                    cur = wl.gen_candidates(store, seq)  # 重新探索（读当前 store）
                    seq += 1
            committed += 1 if done else 0
            rounds_total += rounds
            batch_lat.append(tlat)
            latencies.append(tlat)
        makespan += max(batch_lat) if batch_lat else 0.0
    n = max(committed, 1)
    waste_per_task = (rounds_total - committed) * k * c_gen / n  # 额外探索轮的生成成本（1 轮固有，多出为浪费）
    return {"waste_per_task": waste_per_task, "mean_latency": sum(latencies) / len(latencies),
            "throughput": committed / makespan, "reselects": reselects,
            "mean_rounds": rounds_total / n, "n_merge": stats.n_merge, "committed": committed}


def main():
    bss = [2, 4, 8, 16]
    rows = []
    print(f"=== 探索式多候选（纯 strict 独占资源, n_resources=2*batch, k=4, 每批重置）===")
    print(f"{'batch':>5} | {'metric':>13} | {'OCC':>9} {'CAST':>9}")
    data = {}
    for bs in bss:
        occ = run_explore(cc.CommitStrategy.kStrictOCC, 2 * bs, bs)
        cast = run_explore(cc.CommitStrategy.kCAST, 2 * bs, bs)
        data[bs] = (occ, cast)
        for key, name in [("waste_per_task", "waste/task"), ("mean_latency", "mean_latency"),
                          ("throughput", "throughput"), ("mean_rounds", "explore_rounds")]:
            print(f"{bs:>5} | {name:>13} | {occ[key]:>9.3f} {cast[key]:>9.3f}")
        print(f"{bs:>5} | {'reselect/merge':>13} | OCC(rs={occ['reselects']},mg={occ['n_merge']}) "
              f"CAST(rs={cast['reselects']},mg={cast['n_merge']})")
        print("  " + "-" * 50)
        for strat, m in [("OCC", occ), ("CAST", cast)]:
            rows.append({"batch_size": bs, "strategy": strat, **{kk: round(vv, 4) for kk, vv in m.items()}})

    with open(os.path.join(RESULTS, "explore.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 三面板：waste / latency / throughput vs batch
    panels = [("waste_per_task", "wasted compute / task (c_gen)", "(a) cost — lower better"),
              ("mean_latency", "mean task latency (t_gen units)", "(b) latency — lower better"),
              ("throughput", "throughput (committed / wall-time)", "(c) throughput — higher better")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, ylab, title) in zip(axes, panels):
        ax.plot(bss, [data[b][0][key] for b in bss], "o-", color="tab:blue", label="OCC (re-explore)", linewidth=2)
        ax.plot(bss, [data[b][1][key] for b in bss], "s-", color="tab:green", label="CAST (heterogeneous reselect)", linewidth=2)
        ax.set_xlabel("batch size (concurrency)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pure-strict exclusive-resource load — CAST advantage is purely from heterogeneous multi-candidate reselect (n_merge=0)\n"
                 "LLM latency ~ lognormal; OCC must re-explore (re-run LLM), CAST reuses already-generated candidates",
                 fontsize=10, y=1.07)
    fig.tight_layout()
    out = os.path.join(RESULTS, "explore.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
