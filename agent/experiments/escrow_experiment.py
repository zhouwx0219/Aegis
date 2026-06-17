"""escrow 约束表达：把"带下界约束的扣减"从超卖边界转成可合并收益（CAST 新并发类 kEscrow）。

背景（见 ISOLATION_LEVELS.md §7 / correctness_boundary.py）：带下界约束（库存≥0）的可交换写，
纯 DELTA 放行合并会超卖；strict-CAS 不超卖但高争用下大量重跑（昂贵）。
escrow（O'Neil 1986）：把扣减表达为**额度预留**——并发事务各预留 q（只要剩余≥q），提交确认。
  - 预留可交换、互不阻塞（只要总预留≤容量）⟹ 保留并发/合并收益（不重跑）；
  - 任何使剩余<0 的预留被拒 ⟹ 不超卖（约束保持）。

三策略处理"库存 C、N 个并发各扣 q"（批内同基线依次提交）：
  - 纯 DELTA（commutative 放行）：全合并 → 终值 C−N·q，可能 <0（超卖），但 wasted=0（错误地"便宜"）。
  - strict-CAS（OCC）：版本变即冲突重跑；不超卖，但每个冲突重跑一次 c_gen（昂贵）。
  - escrow：并发预留，剩余≥q 成功否则拒绝；不超卖且 0 重跑（正确且便宜）。
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
C_GEN = 1.0


def run_delta(C, N, q):
    final = C - N * q
    return {"committed": N, "final": final, "oversold": final < 0, "wasted": 0.0, "correct": final >= 0}


def run_strict_cas(C, N, q):
    # 批内同基线 stock0=C；任务依次提交，实际 stock 已变则 OCC 冲突→重跑（读最新后再试）
    stock = C
    stock0 = C
    committed = 0
    regen = 0
    for _ in range(N):
        if stock != stock0:          # 读时基线已过期 → 版本冲突 → 重跑
            regen += 1
        if stock >= q:               # 重读最新后条件成立才扣（不超卖）
            stock -= q
            committed += 1
        # 否则售罄，任务失败（不超卖）
    return {"committed": committed, "final": stock, "oversold": stock < 0, "wasted": regen * C_GEN, "correct": stock >= 0}


def run_escrow(C, N, q):
    remaining = C
    committed = 0
    for _ in range(N):
        if remaining >= q:           # 原子预留（可交换、不看版本、不重跑）
            remaining -= q
            committed += 1
        # 否则额度不足 → 拒绝（不超卖、不重跑）
    return {"committed": committed, "final": remaining, "oversold": remaining < 0, "wasted": 0.0, "correct": remaining >= 0}


def main():
    C, q = 20, 1
    Ns = [8, 16, 24, 32, 48]
    rows = []
    print(f"=== escrow 约束表达：库存 C={C}, 每任务扣 q={q}，扫并发需求 N ===")
    print(f"{'N':>4} | {'策略':>10} | {'成功':>5} {'终库存':>6} {'超卖?':>5} {'浪费(c_gen)':>10} {'正确?':>5}")
    data = {"DELTA": [], "strict-CAS": [], "escrow": []}
    for N in Ns:
        res = {"DELTA": run_delta(C, N, q), "strict-CAS": run_strict_cas(C, N, q), "escrow": run_escrow(C, N, q)}
        for s in ("DELTA", "strict-CAS", "escrow"):
            r = res[s]
            data[s].append(r)
            rows.append({"N": N, "strategy": s, **r})
            print(f"{N:>4} | {s:>10} | {r['committed']:>5} {r['final']:>6} "
                  f"{('YES' if r['oversold'] else 'no'):>5} {r['wasted']:>10.1f} {('OK' if r['correct'] else 'BAD'):>5}")
        print("  " + "-" * 56)

    with open(os.path.join(RESULTS, "escrow.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 图：浪费算力 vs N（escrow≈0 且正确；strict-CAS 高；DELTA 0 但超卖用红叉标注）
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    style = {"DELTA": ("o--", "tab:red"), "strict-CAS": ("^-", "tab:blue"), "escrow": ("s-", "tab:green")}
    for s in ("DELTA", "strict-CAS", "escrow"):
        st, c = style[s]
        a1.plot(Ns, [d["wasted"] for d in data[s]], st, color=c, label=s, linewidth=2)
        a2.plot(Ns, [d["committed"] for d in data[s]], st, color=c, label=s, linewidth=2)
    # 标注 DELTA 超卖点
    for i, N in enumerate(Ns):
        if data["DELTA"][i]["oversold"]:
            a1.annotate("oversold", (N, 0), textcoords="offset points", xytext=(0, 8), color="tab:red", fontsize=7)
    a1.set_xlabel("concurrent demand N"); a1.set_ylabel("wasted compute (c_gen)"); a1.set_title("(a) cost — escrow≈0 & correct"); a1.legend(); a1.grid(True, alpha=0.3)
    a2.axhline(C, ls=":", color="gray", label=f"capacity={C}")
    a2.set_xlabel("concurrent demand N"); a2.set_ylabel("committed (deductions)"); a2.set_title("(b) committed — escrow/CAS cap at C, DELTA oversells"); a2.legend(fontsize=8); a2.grid(True, alpha=0.3)
    fig.suptitle("escrow turns the over-sell boundary into a mergeable gain:\n"
                 "correct (no oversell, caps at capacity) AND cheap (0 re-run), unlike DELTA (oversells) or strict-CAS (expensive re-runs)", fontsize=9.5, y=1.06)
    fig.tight_layout()
    out = os.path.join(RESULTS, "escrow.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
