"""Plot the three-way contention sweeps (OCC vs SCC-2S vs CAST), paper-ready.
English labels only. Run after sweep_contention.py. Reads results/sweep3_*.csv.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STYLE = {"OCC": ("o-", "tab:blue"), "SCC_2S": ("^--", "tab:orange"), "CAST": ("s-", "tab:green")}
LABEL = {"OCC": "OCC (= SCC-1S)", "SCC_2S": "SCC-2S (speculation on)", "CAST": "CAST (ours)"}


def load(name):
    with open(os.path.join(RESULTS, f"sweep3_{name}.csv")) as f:
        return list(csv.DictReader(f))


def series(rows, strat):
    pts = sorted((float(r["x_value"]), float(r["waste_per_task"]))
                 for r in rows if r["strategy"] == strat)
    return [p[0] for p in pts], [p[1] for p in pts]


SPECS = [
    ("A_concurrency", "batch size (concurrency)", False,
     "(a) vs concurrency\n(all writes mergeable)"),
    ("B_mergeable", "mergeable-write fraction", False,
     "(b) vs mergeable fraction\n(high contention)"),
    ("C_asymmetry", "c_merge  (log; smaller = more asymmetric)", True,
     "(c) vs cost asymmetry\n(all writes mergeable)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, (name, xlabel, logx, title) in zip(axes, SPECS):
    rows = load(name)
    for strat in ("OCC", "SCC_2S", "CAST"):
        xs, ys = series(rows, strat)
        style, color = STYLE[strat]
        ax.plot(xs, ys, style, color=color, label=LABEL[strat], linewidth=2, markersize=6)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("wasted compute per task (units of c_gen)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("CAST vs strict OCC vs SCC-kS under the agent cost structure (lower is better)\n"
             "SCC-best collapses to k*=1 (=OCC): speculation never pays when every shadow is an expensive generation",
             fontsize=11, y=1.06)
fig.tight_layout()
out = os.path.join(RESULTS, "sweeps3.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
