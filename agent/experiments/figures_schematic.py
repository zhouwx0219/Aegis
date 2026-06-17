"""科研风示意图（publication-style schematics）用于汇报/论文。
英文标签（论文标准）。生成到 figures/：架构 / 提交协议 / 成本不对称 / 隔离级别定位。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, GREEN, RED, PURPLE, GOLD, GRAY = "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#7f7f7f"


def box(ax, x, y, w, h, text, fc, ec="#333", fs=10, tc="white", style="round,pad=0.02", lw=1.4, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, alpha=alpha, mutation_scale=14)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=lw, color=color, ls=ls))


# ---------- Fig 1: 系统架构 ----------
def fig_arch():
    fig, ax = plt.subplots(figsize=(8.4, 6.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 1, 8.6, 8, 1.0, "LLM Agents / Operators\n(generate k candidates · declare write intents)", BLUE, fs=10)
    ax.text(5, 8.25, "pybind11 (in-process)", ha="center", fontsize=8, color=GRAY, style="italic")
    arrow(ax, 5, 8.55, 5, 7.95)
    # kernel
    box(ax, 0.6, 3.7, 8.8, 4.2, "", "#eef2f8", ec=GREEN, tc="#333", lw=2)
    ax.text(5, 7.55, "CAST Transaction Kernel (C++)", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    box(ax, 1.0, 6.2, 8.0, 1.0, "Unified Object + Write Intents\nREAD · OVERWRITE(strict) · DELTA/APPEND(commutative) · CAS · ESCROW", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    box(ax, 1.0, 5.0, 3.8, 1.0, "Semantic-aware Validation\n(reads & commutative writes PASS)", GREEN, fs=8.5)
    box(ax, 5.2, 5.0, 3.8, 1.0, "Cost-Asymmetric Commit\ndirect→merge→reselect→regenerate", GREEN, fs=8.5)
    box(ax, 1.0, 3.9, 8.0, 0.8, "Cost Model (c_gen >> c_merge) · CandidateScheduler", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    arrow(ax, 5, 3.65, 5, 3.05)
    box(ax, 1, 2.0, 8, 1.0, "Versioned-KV abstraction (5 primitives)\nGet · GetVersion · PutIfVersion · BatchPutIfVersion · DeleteIfVersion", GRAY, fs=9)
    arrow(ax, 5, 1.95, 5, 1.35)
    box(ax, 1, 0.3, 8, 1.0, "Pluggable backends:  in-memory (reference)   |   RocksDB · TiKV · Redis  (future)", "#ffffff", tc="#333", ec=GRAY, lw=1.4, style="round,pad=0.02")
    ax.text(5, 9.75, "Figure 1.  CAST system architecture", ha="center", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------- Fig 2: 两阶段提交协议 ----------
def fig_protocol():
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(6, 6.6, "Figure 2.  CAST two-phase commit: semantic-aware validation + cost-asymmetric resolution", ha="center", fontsize=10.5, fontweight="bold")
    box(ax, 0.3, 3.0, 1.7, 1.0, "winner\ncandidate", BLUE, fs=9)
    # Phase 1 validation
    ax.text(4.3, 5.9, "Phase 1: validation (by write intent)", ha="center", fontsize=9, color=GREEN, fontweight="bold")
    box(ax, 2.6, 4.7, 3.4, 0.7, "READ / commutative → PASS", GREEN, fs=8.5)
    box(ax, 2.6, 3.75, 3.4, 0.7, "CAS → check condition", GOLD, tc="#222", fs=8.5)
    box(ax, 2.6, 2.8, 3.4, 0.7, "strict(OVERWRITE) → version check", BLUE, fs=8.5)
    arrow(ax, 2.0, 3.5, 2.6, 5.05); arrow(ax, 2.0, 3.5, 2.6, 4.1); arrow(ax, 2.0, 3.5, 2.6, 3.15)
    # Phase 2 resolution state machine
    ax.text(9.3, 5.9, "Phase 2: cost-asymmetric resolution", ha="center", fontsize=9, color=RED, fontweight="bold")
    box(ax, 7.0, 4.9, 2.0, 0.7, "direct\n(no conflict)", GREEN, fs=8)
    box(ax, 7.0, 3.95, 2.0, 0.7, "merge  (c_merge)", GREEN, fs=8)
    box(ax, 7.0, 3.0, 2.0, 0.7, "reselect  (~0)", GOLD, tc="#222", fs=8)
    box(ax, 7.0, 2.05, 2.0, 0.7, "regenerate  (c_gen)", RED, fs=8)
    arrow(ax, 6.0, 4.0, 7.0, 5.25); arrow(ax, 6.0, 4.0, 7.0, 4.3)
    arrow(ax, 6.0, 3.2, 7.0, 3.35); arrow(ax, 6.0, 3.1, 7.0, 2.4)
    # priority arrows down the state machine
    arrow(ax, 8.0, 4.85, 8.0, 4.68, color=GRAY, lw=1.2); arrow(ax, 8.0, 3.9, 8.0, 3.73, color=GRAY, lw=1.2); arrow(ax, 8.0, 2.95, 8.0, 2.78, color=GRAY, lw=1.2)
    box(ax, 10.1, 3.4, 1.6, 0.9, "commit\n(atomic)", "#333", fs=9)
    for yy in (5.25, 4.3, 3.35, 2.4):
        arrow(ax, 9.0, yy, 10.1, 3.85, color=GRAY, lw=1.0, ls=(0, (3, 2)))
    ax.text(8.0, 1.6, "cheaper ↑   prefer merge/reselect, push expensive regenerate to last", ha="center", fontsize=8, color=RED, style="italic")
    fig.savefig(os.path.join(OUT, "fig_protocol.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------- Fig 3: 成本不对称 motivation ----------
def fig_cost():
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    items = [("traditional txn abort\n(replay read-set)", 1e-6, GRAY),
             ("CAST semantic merge\n(KV arithmetic)", 1e-5, GREEN),
             ("agent abort = re-run LLM\n(inference + tools + $)", 3.0, RED)]
    ys = [2, 1, 0]
    for (lab, val, c), y in zip(items, ys):
        ax.barh(y, val, color=c, height=0.5, log=True)
        ax.text(val * 1.5, y, f"{lab}", va="center", fontsize=9)
    ax.set_yticks([]); ax.set_xscale("log")
    ax.set_xlabel("cost per retry / merge  (seconds, log scale)")
    ax.set_xlim(1e-7, 1e3)
    ax.set_title("Figure 3.  Cost asymmetry: agent abort (re-run LLM) is 10^4–10^6 × a semantic merge\n"
                 "→ classic CC's 'abort is cheap' assumption breaks; minimize wasted LLM compute", fontsize=9.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cost_asymmetry.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------- Fig 4: 隔离级别定位 ----------
def fig_isolation():
    fig, ax = plt.subplots(figsize=(9.2, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(5, 5.05, "Figure 4.  CAST = CSI-SS: a per-write-intent mixed isolation level", ha="center", fontsize=10.5, fontweight="bold")
    # spectrum line
    arrow(ax, 1, 1.0, 9, 1.0, color="#333", lw=1.6)
    ax.text(1, 0.6, "weaker", fontsize=8, color=GRAY); ax.text(8.4, 0.6, "serializable", fontsize=8, color=GRAY)
    for x, lab in [(1.6, "Read\nCommitted"), (3.6, "Snapshot\nIsolation (MVCC)"), (7.8, "Serializable\n(OCC/2PL/Silo/TicToc)")]:
        ax.plot([x], [1.0], "o", color=GRAY, ms=7); ax.text(x, 0.2, lab, ha="center", fontsize=8, color="#333")
    # CAST bracket spanning per-intent
    box(ax, 1.2, 2.4, 7.6, 2.1, "", "#eef7ef", ec=GREEN, tc="#222", lw=2)
    ax.text(5, 4.2, "CAST (CSI-SS): different guarantee per write intent", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    box(ax, 1.5, 2.7, 1.9, 1.0, "reads\n→ SI", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    box(ax, 3.6, 2.7, 2.2, 1.0, "commutative\n→ convergent (SEC)", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    box(ax, 6.0, 2.7, 1.4, 1.0, "CAS\n→ cond-safe", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    box(ax, 7.6, 2.7, 1.1, 1.0, "strict\n→ serial.", "#dbe7d4", tc="#222", ec=GREEN, fs=8.5)
    # dashed links to spectrum
    arrow(ax, 2.45, 2.7, 3.6, 1.05, color=GRAY, lw=1.0, ls=(0, (3, 2)))
    arrow(ax, 8.1, 2.7, 7.8, 1.05, color=GRAY, lw=1.0, ls=(0, (3, 2)))
    ax.text(5, 1.7, "knob: escrow lifts constrained commutative writes; downgrade to strict for serializable reads", ha="center", fontsize=8, color=GRAY, style="italic")
    fig.savefig(os.path.join(OUT, "fig_isolation.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


def main():
    fig_arch(); fig_protocol(); fig_cost(); fig_isolation()
    print("saved 4 schematic figures to", OUT)
    for f in ["fig_architecture", "fig_protocol", "fig_cost_asymmetry", "fig_isolation"]:
        print("  -", f + ".png")


if __name__ == "__main__":
    main()
