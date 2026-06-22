"""Generate paper-facing SVG figures from existing ASTRA experiment outputs.

This script intentionally uses only the Python standard library. The legacy
experiment scripts still produce PNGs through matplotlib when that dependency is
available, but the CCFA artifact path should be reproducible in a minimal WSL
environment after building cast_core.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
from collections import defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "paper_figures")

COLORS = {
    "OCC": "#1f77b4",
    "SCC_2S": "#7f7f7f",
    "SCC_best": "#9467bd",
    "CAST": "#2ca02c",
    "HYBRID": "#2ca02c",
    "HYBRID-K1": "#98df8a",
    "OCC-K1": "#1f77b4",
    "OCC+K": "#aec7e8",
    "Silo": "#17becf",
    "TicToc": "#9467bd",
    "2PL": "#d62728",
    "MVCC": "#bcbd22",
    "DELTA": "#d62728",
    "strict-CAS": "#1f77b4",
    "escrow": "#2ca02c",
    "merge-all": "#ff7f0e",
}


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>'
            'text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
            '.title{font-size:18px;font-weight:700}'
            '.subtitle{font-size:12px;fill:#555}'
            '.axis{stroke:#222;stroke-width:1}'
            '.grid{stroke:#ddd;stroke-width:1}'
            '.tick{font-size:11px;fill:#444}'
            '.label{font-size:12px;fill:#222}'
            '.legend{font-size:12px;fill:#222}'
            '.panel{font-size:14px;font-weight:700}'
            '</style>',
            text(width / 2, 26, title, "title", anchor="middle"),
        ]

    def add(self, item: str) -> None:
        self.items.append(item)

    def save(self, path: str) -> None:
        self.items.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.items))


def text(x, y, s, cls="label", anchor="start", rotate=None):
    rot = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{rot}>{esc(s)}</text>'


def line(x1, y1, x2, y2, color="#222", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def rect(x, y, w, h, fill, stroke="none", width=1):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def circle(x, y, r, fill, stroke="white"):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def polyline(points, color, width=2.4):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'


def read_csv(name):
    with open(os.path.join(RESULTS, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def fnum(v):
    if v in ("", None):
        return 0.0
    return float(v)


def pretty(v):
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def compact_count(v):
    v = int(round(float(v)))
    if v >= 1000 and v % 1000 == 0:
        return f"{v // 1000}k"
    return str(v)


def scale(v, vmin, vmax, a, b):
    if vmax == vmin:
        return (a + b) / 2
    return a + (v - vmin) * (b - a) / (vmax - vmin)


def nice_ticks(vmin, vmax, n=5):
    if vmax <= vmin:
        return [vmin]
    return [vmin + i * (vmax - vmin) / (n - 1) for i in range(n)]


def line_panel(svg, x, y, w, h, title, xlabel, ylabel, series, x_labels=None, y_max=None):
    left, top, right, bottom = x + 54, y + 34, x + w - 18, y + h - 42
    svg.add(text(x, y + 14, title, "panel"))
    all_y = [v for s in series for v in s["ys"]]
    ymin = 0.0
    ymax = y_max if y_max is not None else max(all_y) * 1.12 if all_y else 1.0
    ymax = max(ymax, 1e-9)

    if x_labels is None:
        all_x = [v for s in series for v in s["xs"]]
        xmin, xmax = min(all_x), max(all_x)
        x_ticks = sorted(set(all_x))
        x_label_values = [str(int(t)) if float(t).is_integer() else pretty(t) for t in x_ticks]
    else:
        xmin, xmax = 0, len(x_labels) - 1
        x_ticks = list(range(len(x_labels)))
        x_label_values = x_labels

    for t in nice_ticks(ymin, ymax, 5):
        yy = scale(t, ymin, ymax, bottom, top)
        svg.add(line(left, yy, right, yy, "#e6e6e6"))
        svg.add(text(left - 8, yy + 4, pretty(t), "tick", anchor="end"))
    svg.add(line(left, top, left, bottom, "#222", 1.2))
    svg.add(line(left, bottom, right, bottom, "#222", 1.2))

    for t, lab in zip(x_ticks, x_label_values):
        xx = scale(t, xmin, xmax, left, right)
        svg.add(line(xx, bottom, xx, bottom + 4, "#222"))
        svg.add(text(xx, bottom + 18, lab, "tick", anchor="middle"))

    for s in series:
        pts = []
        for xv, yv in zip(s["xs"], s["ys"]):
            xx = scale(xv, xmin, xmax, left, right)
            yy = scale(yv, ymin, ymax, bottom, top)
            pts.append((xx, yy))
        svg.add(polyline(pts, s["color"]))
        for xx, yy in pts:
            svg.add(circle(xx, yy, 3.8, s["color"]))

    svg.add(text((left + right) / 2, y + h - 8, xlabel, "label", anchor="middle"))
    svg.add(text(x + 13, (top + bottom) / 2, ylabel, "label", anchor="middle", rotate=-90))

    lx, ly = right - 118, top + 8
    for i, s in enumerate(series):
        yy = ly + i * 17
        svg.add(line(lx, yy - 4, lx + 20, yy - 4, s["color"], 2.4))
        svg.add(circle(lx + 10, yy - 4, 3.5, s["color"]))
        svg.add(text(lx + 26, yy, s["label"], "legend"))


def bar_panel(svg, x, y, w, h, title, ylabel, labels, values, colors, y_max=None, yerrs=None, value_labels=True):
    left, top, right, bottom = x + 58, y + 34, x + w - 18, y + h - 44
    svg.add(text(x, y + 14, title, "panel"))
    ymax = y_max if y_max is not None else max(values) * 1.18 if values else 1.0
    ymax = max(ymax, 1e-9)
    for t in nice_ticks(0.0, ymax, 5):
        yy = scale(t, 0.0, ymax, bottom, top)
        svg.add(line(left, yy, right, yy, "#e6e6e6"))
        svg.add(text(left - 8, yy + 4, pretty(t), "tick", anchor="end"))
    svg.add(line(left, top, left, bottom, "#222", 1.2))
    svg.add(line(left, bottom, right, bottom, "#222", 1.2))

    n = len(labels)
    gap = 10
    bw = max(12, (right - left - gap * (n + 1)) / n)
    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        bx = left + gap + i * (bw + gap)
        by = scale(val, 0.0, ymax, bottom, top)
        svg.add(rect(bx, by, bw, bottom - by, color))
        if yerrs:
            err = yerrs[i]
            yhi = scale(val + err, 0.0, ymax, bottom, top)
            svg.add(line(bx + bw / 2, yhi, bx + bw / 2, by, "#333", 1))
            svg.add(line(bx + bw / 2 - 5, yhi, bx + bw / 2 + 5, yhi, "#333", 1))
        if value_labels:
            svg.add(text(bx + bw / 2, by - 5, pretty(val), "tick", anchor="middle"))
        svg.add(text(bx + bw / 2, bottom + 17, lab, "tick", anchor="middle", rotate=-12 if len(lab) > 6 else None))
    svg.add(text(x + 13, (top + bottom) / 2, ylabel, "label", anchor="middle", rotate=-90))


def group(rows, key, x_field, y_field):
    out = defaultdict(lambda: ([], []))
    for row in rows:
        out[row[key]][0].append(fnum(row[x_field]))
        out[row[key]][1].append(fnum(row[y_field]))
    return out


def fig_cost_asymmetry():
    svg = SVG(1160, 410, "Figure 1. Cost asymmetry and semantic merge boundaries")
    files = [
        ("sweep3_A_concurrency.csv", "(a) More concurrency increases OCC waste", "batch size", "wasted compute / task"),
        ("sweep3_B_mergeable.csv", "(b) Benefit tracks mergeable-write fraction", "mergeable fraction", "wasted compute / task"),
        ("sweep3_C_asymmetry.csv", "(c) CAST wins only when merge is cheap", "c_merge / c_gen", "wasted compute / task"),
    ]
    for i, (name, title, xlabel, ylabel) in enumerate(files):
        rows = read_csv(name)
        g = group(rows, "strategy", "x_value", "waste_per_task")
        keep = ["OCC", "SCC_2S", "CAST"]
        series = []
        labels = None
        if name.endswith("C_asymmetry.csv"):
            vals = sorted(set(fnum(r["x_value"]) for r in rows), reverse=True)
            labels = [pretty(v) for v in vals]
            pos = {v: j for j, v in enumerate(vals)}
        for k in keep:
            xs, ys = g[k]
            pairs = sorted(zip(xs, ys), key=lambda p: p[0], reverse=name.endswith("C_asymmetry.csv"))
            if labels:
                series.append({"label": k, "xs": [pos[p[0]] for p in pairs], "ys": [p[1] for p in pairs], "color": COLORS[k]})
            else:
                series.append({"label": k, "xs": [p[0] for p in pairs], "ys": [p[1] for p in pairs], "color": COLORS[k]})
        line_panel(svg, 24 + i * 378, 55, 350, 300, title, xlabel, ylabel, series, labels)
    return svg


def fig_true_concurrency():
    rows = read_csv("concurrent.csv")
    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)
    svg = SVG(900, 410, "Figure 2. Measured true-concurrency performance")
    series_tp, series_lat = [], []
    for s in ["OCC", "2PL", "HYBRID"]:
        rs = sorted(by[s], key=lambda r: fnum(r["threads"]))
        series_tp.append({"label": s, "xs": [fnum(r["threads"]) for r in rs], "ys": [fnum(r["throughput"]) for r in rs], "color": COLORS[s]})
        series_lat.append({"label": s, "xs": [fnum(r["threads"]) for r in rs], "ys": [fnum(r["latency_ms"]) for r in rs], "color": COLORS[s]})
    line_panel(svg, 36, 58, 390, 300, "(a) Throughput", "worker threads", "committed/s", series_tp)
    line_panel(svg, 478, 58, 390, 300, "(b) Latency", "worker threads", "ms / task", series_lat)
    svg.add(text(450, 390, "True Python threads + wall-clock timing; candidate generation is represented by sleep(c_gen).", "subtitle", anchor="middle"))
    return svg


def fig_semantic_reselect():
    sem = read_csv("semantic_validation.csv")
    exp = read_csv("explore.csv")
    svg = SVG(960, 430, "Figure 3. Semantic validation and multi-candidate reselect")

    sem_g = group(sem, "strategy", "read_size", "abort_rate")
    series = []
    for s in ["OCC", "MVCC", "HYBRID"]:
        xs, ys = sem_g[s]
        pairs = sorted(zip(xs, ys))
        series.append({"label": s, "xs": [p[0] for p in pairs], "ys": [p[1] for p in pairs], "color": COLORS[s]})
    line_panel(svg, 32, 58, 420, 305, "(a) Semantic validation reduces false conflicts", "read-set size", "abort rate", series, y_max=0.75)

    exp_g = group(exp, "strategy", "batch_size", "throughput")
    series = []
    for s in ["OCC", "HYBRID"]:
        xs, ys = exp_g[s]
        pairs = sorted(zip(xs, ys))
        series.append({"label": s, "xs": [p[0] for p in pairs], "ys": [p[1] for p in pairs], "color": COLORS[s]})
    line_panel(svg, 510, 58, 420, 305, "(b) Reselect helps even when merge is disabled", "batch size", "throughput", series)
    svg.add(text(480, 395, "Panel (b): strict/CAS workload, n_merge=0; HYBRID gains from reusing generated alternatives.", "subtitle", anchor="middle"))
    return svg


def fig_escrow():
    rows = read_csv("escrow.csv")
    svg = SVG(940, 410, "Figure 4. Escrow turns constrained commutative writes into safe concurrency")
    g_w = group(rows, "strategy", "N", "wasted")
    series = []
    for s in ["DELTA", "strict-CAS", "escrow"]:
        pairs = sorted(zip(*g_w[s]))
        series.append({"label": s, "xs": [p[0] for p in pairs], "ys": [p[1] for p in pairs], "color": COLORS[s]})
    line_panel(svg, 36, 58, 410, 300, "(a) Wasted compute", "concurrent demand N", "wasted c_gen", series)

    labels, vals, colors = [], [], []
    for r in rows:
        if r["N"] == "24":
            labels.append(r["strategy"])
            vals.append(max(0.0, -fnum(r["final"])))
            colors.append(COLORS[r["strategy"]])
    bar_panel(svg, 505, 58, 390, 300, "(b) Oversell at N=24", "seats below zero", labels, vals, colors, y_max=5)
    svg.add(text(470, 390, "DELTA is cheap but unsafe; strict-CAS is safe but wastes regeneration; escrow is safe and cheap.", "subtitle", anchor="middle"))
    return svg


def fig_llm():
    data = read_json("llm_analysis.json")
    replay = data["replay"]
    labels = ["OCC", "2PL", "merge-all", "HYBRID-K1", "HYBRID"]
    tp = [replay[l]["throughput"][0] for l in labels]
    tpe = [replay[l]["throughput"][1] for l in labels]
    ov = [replay[l]["oversell"][0] for l in labels]
    colors = [COLORS[l] for l in labels]
    svg = SVG(980, 450, "Figure 5. Real LLM-in-the-loop evidence")
    bar_panel(svg, 36, 72, 460, 310, "(a) Replay throughput on real DeepSeek traces", "booked/s", labels, tp, colors, yerrs=tpe)
    bar_panel(svg, 555, 72, 370, 310, "(b) Correctness: oversell events", "oversell count", labels, ov, colors, y_max=max(20, max(ov) * 1.15))
    lat = data["generation_latency_s"]
    cand = data["candidates"]
    derived = data["derived"]
    note = (
        f"48 real calls: mean c_gen={lat['mean']:.2f}s, p95={lat['p95']:.2f}s; "
        f"{cand['reselectable_fraction']*100:.0f}% tasks have >=2 alternatives; "
        f"HYBRID +{derived['hybrid_vs_occ_improvement_pct']:.1f}% vs OCC and 0 oversell."
    )
    svg.add(text(490, 424, note, "subtitle", anchor="middle"))
    return svg


def fig_baseline_family():
    rows = read_csv("ccfa_baseline_family.csv")
    svg = SVG(1040, 440, "Figure 6. Baseline family comparison at larger scale")
    n_tasks = compact_count(fnum(rows[0]["n_tasks"])) if rows else "?"
    threads = int(fnum(rows[0]["threads"])) if rows else 0
    k_value = int(fnum(rows[0]["k"])) if rows else 0

    policies = ["OCC", "Silo", "TicToc", "MVCC", "2PL", "HYBRID"]
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    series_tp = []
    series_regen = []
    for p in policies:
        rs = sorted(by_policy[p], key=lambda r: fnum(r["n_obj"]))
        series_tp.append({
            "label": p,
            "xs": [fnum(r["n_obj"]) for r in rs],
            "ys": [fnum(r["throughput"]) for r in rs],
            "color": COLORS[p],
        })
        if p != "2PL":
            series_regen.append({
                "label": p,
                "xs": [fnum(r["n_obj"]) for r in rs],
                "ys": [fnum(r["regen_per_task"]) for r in rs],
                "color": COLORS[p],
            })

    line_panel(
        svg, 34, 64, 470, 310,
        "(a) Throughput vs contention",
        "object pool size (larger = lower contention)",
        "committed/s",
        series_tp,
    )
    line_panel(
        svg, 548, 64, 450, 310,
        "(b) Regeneration pressure",
        "object pool size",
        "regen / task",
        series_regen,
        y_max=max(max(s["ys"]) for s in series_regen) * 1.15,
    )
    svg.add(text(
        520, 410,
        f"Synthetic CC family, {n_tasks} tasks, {threads} threads, k={k_value}, 3 seeds. Silo/TicToc/MVCC remain syntactic; HYBRID adds semantic merge.",
        "subtitle",
        anchor="middle",
    ))
    return svg


def fig_scale_out():
    threads = read_csv("ccfa_scale_threads.csv")
    tasks = read_csv("ccfa_scale_tasks.csv")
    svg = SVG(980, 440, "Figure 7. Workload scale-out")
    thread_tasks = compact_count(fnum(threads[0]["n_tasks"])) if threads else "?"
    max_threads = max(int(fnum(r["threads"])) for r in threads) if threads else 0
    max_tasks = max(fnum(r["n_tasks"]) for r in tasks) if tasks else 0

    policies = ["OCC", "MVCC", "2PL", "HYBRID"]
    by_policy = defaultdict(list)
    for r in threads:
        by_policy[r["policy"]].append(r)
    series_threads = []
    for p in policies:
        rs = sorted(by_policy[p], key=lambda r: fnum(r["threads"]))
        series_threads.append({
            "label": p,
            "xs": [fnum(r["threads"]) for r in rs],
            "ys": [fnum(r["throughput"]) for r in rs],
            "color": COLORS[p],
        })

    by_policy = defaultdict(list)
    for r in tasks:
        by_policy[r["policy"]].append(r)
    series_tasks = []
    for p in ["OCC", "HYBRID"]:
        rs = sorted(by_policy[p], key=lambda r: fnum(r["n_tasks"]))
        series_tasks.append({
            "label": p,
            "xs": [i for i, _ in enumerate(rs)],
            "ys": [fnum(r["throughput"]) for r in rs],
            "color": COLORS[p],
        })
    labels = [f"{int(fnum(r['n_tasks'])/1000)}k" for r in sorted(by_policy["OCC"], key=lambda r: fnum(r["n_tasks"]))]

    line_panel(svg, 36, 64, 430, 310, f"(a) Up to {max_threads} worker threads", "threads", "committed/s", series_threads)
    line_panel(svg, 522, 64, 410, 310, "(b) Stable under larger task counts", "tasks", "committed/s", series_tasks, x_labels=labels)
    svg.add(text(490, 410, f"Current CSV profile: {thread_tasks}-task thread sweep up to {max_threads} threads; task-count sweep up to {compact_count(max_tasks)} tasks.", "subtitle", anchor="middle"))
    return svg


def fig_agent_aware():
    rows = read_csv("ccfa_agent_aware.csv")
    svg = SVG(1050, 440, "Figure 8. Agent-aware fair baselines")
    n_tasks = compact_count(fnum(rows[0]["n_tasks"])) if rows else "?"
    n_flights = int(fnum(rows[0]["n_flights"])) if rows else 0
    threads = int(fnum(rows[0]["threads"])) if rows else 0
    k_value = int(fnum(rows[0]["k"])) if rows else 0
    order = ["OCC", "OCC+K", "HYBRID-K1", "HYBRID", "2PL", "merge-all"]
    row_by = {r["policy"]: r for r in rows}
    labels = order
    colors = [COLORS[p] for p in labels]
    tp = [fnum(row_by[p]["throughput"]) for p in labels]
    tp_ci = [fnum(row_by[p].get("throughput_ci", 0)) for p in labels]
    oversell = [fnum(row_by[p]["oversell"]) for p in labels]
    regen = [fnum(row_by[p]["regen"]) for p in labels]

    bar_panel(svg, 34, 70, 500, 300, "(a) Throughput with fair candidate access", "booked/s", labels, tp, colors, yerrs=tp_ci)
    bar_panel(svg, 590, 70, 390, 300, "(b) Unsafe upper bound exposes oversell", "oversell events", labels, oversell, colors, y_max=max(1, max(oversell) * 1.15))

    # Annotate regeneration counts under panel (a). This isolates OCC+K from HYBRID.
    x0 = 92
    for i, p in enumerate(labels):
        svg.add(text(x0 + i * 72, 394, f"regen={pretty(regen[i])}", "tick", anchor="middle", rotate=-18))
    svg.add(text(
        525, 425,
        f"{n_tasks} booking tasks, {n_flights} flights, {threads} threads, k={k_value}. OCC+K gets the same alternatives; HYBRID-K1 removes candidate reuse; merge-all is fast but unsafe.",
        "subtitle",
        anchor="middle",
    ))
    return svg


def fig_hotspot_mixed():
    rows = read_csv("ccfa_hotspot_mixed.csv")
    hot_rows = [r for r in rows if r["workload"] == "hotspot-mixed"] or rows
    svg = SVG(1060, 450, "Figure 9. Hotspot mixed-object workload")
    order = ["OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL"]
    short = {
        "OCC-K1": "OCC",
        "OCC+K": "OCC+K",
        "MVCC": "MVCC",
        "HYBRID-K1": "HY-K1",
        "HYBRID": "HYBRID",
        "2PL": "2PL",
    }
    row_by = {r["policy"]: r for r in hot_rows}
    labels = [short[p] for p in order]
    colors = [COLORS[p] for p in order]
    tp = [fnum(row_by[p]["throughput"]) for p in order]
    tp_ci = [fnum(row_by[p].get("throughput_ci", 0)) for p in order]
    gen_calls = [fnum(row_by[p]["generation_calls_per_task"]) for p in order]
    regen = [fnum(row_by[p]["regen_per_task"]) for p in order]

    bar_panel(svg, 34, 72, 500, 305, "(a) Throughput under hotspot conflicts", "committed/s", labels, tp, colors, yerrs=tp_ci)
    bar_panel(svg, 590, 72, 400, 305, "(b) Generated calls per task", "calls / task", labels, gen_calls, colors, y_max=max(gen_calls) * 1.2)

    x0 = 92
    for i, val in enumerate(regen):
        svg.add(text(x0 + i * 72, 403, f"regen={pretty(val)}", "tick", anchor="middle", rotate=-18))

    hy = row_by["HYBRID"]
    n_tasks = compact_count(fnum(hy["n_tasks"]))
    n_obj = int(fnum(hy["n_obj"]))
    threads = int(fnum(hy["threads"]))
    hot_bias = fnum(hy["hot_bias"])
    p_merge = fnum(hy["p_merge"])
    speedup_occ = fnum(hy["speedup_vs_occ_k1"])
    speedup_fair = fnum(hy["speedup_vs_occ_k"])
    note = (
        f"{n_tasks} tasks, {n_obj} objects, {threads} threads, hot-bias={hot_bias:.2f}, p_merge={p_merge:.2f}; "
        f"HYBRID is {speedup_occ:.2f}x vs OCC and {speedup_fair:.2f}x vs OCC+K."
    )
    svg.add(text(530, 432, note, "subtitle", anchor="middle"))
    return svg


def fig_vitabench_authoritative():
    rows = read_csv("vitabench_authoritative.csv")
    manifest = read_json("vitabench_authoritative_manifest.json")
    svg = SVG(1080, 460, "Figure 10. VitaBench environment-derived write workload")
    order = ["OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL", "merge-all"]
    short = {
        "OCC-K1": "OCC",
        "OCC+K": "OCC+K",
        "MVCC": "MVCC",
        "HYBRID-K1": "HY-K1",
        "HYBRID": "HYBRID",
        "2PL": "2PL",
        "merge-all": "unsafe",
    }
    row_by = {r["policy"]: r for r in rows}
    labels = [short[p] for p in order]
    colors = [COLORS[p] for p in order]
    tp = [fnum(row_by[p]["throughput"]) for p in order]
    tp_ci = [fnum(row_by[p].get("throughput_ci", 0)) for p in order]
    gen = [fnum(row_by[p]["generation_calls_per_task"]) for p in order]
    oversell = [fnum(row_by[p]["oversell"]) for p in order]

    bar_panel(svg, 34, 72, 455, 305, "(a) Throughput on real OTA resources", "booked/s", labels, tp, colors, yerrs=tp_ci)
    bar_panel(svg, 548, 72, 270, 305, "(b) Generation calls", "calls / task", labels, gen, colors, y_max=max(gen) * 1.2)
    bar_panel(svg, 850, 72, 190, 305, "(c) Safety", "oversell", labels, oversell, colors, y_max=max(1, max(oversell) * 1.15), value_labels=False)

    hy = row_by["HYBRID"]
    res = manifest.get("resources_by_category", {})
    res_text = ", ".join(f"{k}:{v}" for k, v in res.items())
    verified = manifest.get("quantity_decrement_verification", {}).get("verified", False)
    note = (
        f"Real VitaBench OTA resources ({res_text}); quantity decrement verified={verified}; "
        f"HYBRID {fnum(hy['speedup_vs_occ_k1']):.2f}x vs OCC and {fnum(hy['speedup_vs_occ_k']):.2f}x vs OCC+K."
    )
    svg.add(text(540, 438, note, "subtitle", anchor="middle"))
    return svg


def fig_rigorous_vitabench():
    rows = read_csv("rigorous_vitabench_summary.csv")
    max_threads = max(int(fnum(r["threads"])) for r in rows)
    use_rows = [r for r in rows if int(fnum(r["threads"])) == max_threads]
    row_by = {r["policy"]: r for r in use_rows}
    order = ["OCC-K1", "OCC+K", "MVCC", "HYBRID-K1", "HYBRID", "2PL", "merge-all"]
    short = {
        "OCC-K1": "OCC",
        "OCC+K": "OCC+K",
        "MVCC": "MVCC",
        "HYBRID-K1": "HY-K1",
        "HYBRID": "HYBRID",
        "2PL": "2PL",
        "merge-all": "unsafe",
    }
    labels = [short[p] for p in order]
    colors = [COLORS[p] for p in order]
    tp = [fnum(row_by[p]["throughput"]) for p in order]
    tp_ci = [fnum(row_by[p].get("throughput_ci", 0)) for p in order]
    p95 = [fnum(row_by[p]["p95_latency_ms"]) for p in order]
    p95_ci = [fnum(row_by[p].get("p95_latency_ms_ci", 0)) for p in order]
    sla = [100 * fnum(row_by[p]["sla_success_rate"]) for p in order]
    sla_ci = [100 * fnum(row_by[p].get("sla_success_rate_ci", 0)) for p in order]

    svg = SVG(1180, 460, "Figure 11. Large-scale rigorous VitaBench-derived benchmark")
    bar_panel(svg, 32, 72, 360, 300, "(a) Throughput", "safe commits/s", labels, tp, colors, yerrs=tp_ci)
    bar_panel(svg, 430, 72, 330, 300, "(b) P95 latency", "ms", labels, p95, colors, yerrs=p95_ci)
    bar_panel(svg, 800, 72, 330, 300, "(c) SLA success", "% tasks", labels, sla, colors, y_max=100, yerrs=sla_ci)

    hy = row_by["HYBRID"]
    occk = row_by["OCC+K"]
    note = (
        f"{compact_count(fnum(hy['n_tasks']))} tasks/seed, {max_threads} threads, 5 seeds, SLA={fnum(hy['sla_ms']):.1f}ms; "
        f"HYBRID +{fnum(hy['throughput_gain_vs_occ_k_pct']):.1f}% throughput vs OCC+K, "
        f"P95 {fnum(hy['p95_latency_ms']):.2f}ms vs {fnum(occk['p95_latency_ms']):.2f}ms, "
        f"SLA {100*fnum(hy['sla_success_rate']):.1f}% vs {100*fnum(occk['sla_success_rate']):.1f}%."
    )
    svg.add(text(590, 430, note, "subtitle", anchor="middle"))
    return svg


def fig_semantic_workloads():
    rows = read_csv("semantic_workloads_summary.csv")
    svg = SVG(1160, 460, "Figure 12. Semantic workload coverage beyond DELTA")
    workloads = ["append_log", "cas_claim", "mixed_checkout", "private_strict"]
    short = {
        "append_log": "APPEND",
        "cas_claim": "CAS",
        "mixed_checkout": "mixed",
        "private_strict": "control",
    }
    by = {(r["workload"], r["policy"]): r for r in rows}
    hy = [by[(w, "HYBRID")] for w in workloads]

    labels = [short[w] for w in workloads]
    colors = [COLORS["HYBRID"], COLORS["HYBRID"], COLORS["HYBRID"], "#999999"]
    speed_occ_k = [fnum(r["speedup_vs_occ_k"]) for r in hy]
    speed_branch = [fnum(r["speedup_vs_branch_txn"]) for r in hy]
    gen_calls = [fnum(r["generation_calls_per_task"]) for r in hy]
    merge = [fnum(r["merge_per_task"]) for r in hy]
    reselect = [fnum(r["reselect_per_task"]) for r in hy]

    bar_panel(
        svg, 32, 72, 330, 300,
        "(a) HYBRID speedup vs OCC+K",
        "x speedup",
        labels,
        speed_occ_k,
        colors,
        y_max=max(2.1, max(speed_occ_k) * 1.15),
    )
    bar_panel(
        svg, 410, 72, 330, 300,
        "(b) HYBRID speedup vs branch-txn",
        "x speedup",
        labels,
        speed_branch,
        colors,
        y_max=max(2.3, max(speed_branch) * 1.15),
    )
    bar_panel(
        svg, 790, 72, 310, 300,
        "(c) Generation calls per task",
        "calls / task",
        labels,
        gen_calls,
        colors,
        y_max=1.2,
    )

    note = (
        "APPEND: non-numeric semantic rebase; CAS: conditional validation + reselect; "
        f"mixed: merge/task={merge[2]:.2f}, reselect/task={reselect[2]:.2f}; "
        "control: private strict workload ties, showing the boundary."
    )
    svg.add(text(580, 430, note, "subtitle", anchor="middle"))
    return svg


def write_index(outputs):
    path = os.path.join(OUT, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Paper Figures\n\n")
        f.write("Generated by `python3 agent/experiments/paper_figures.py` from existing CSV/JSON results.\n\n")
        for name in outputs:
            f.write(f"- `{name}`\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    figures = [
        ("fig1_cost_asymmetry.svg", fig_cost_asymmetry()),
        ("fig2_true_concurrency.svg", fig_true_concurrency()),
        ("fig3_semantic_reselect.svg", fig_semantic_reselect()),
        ("fig4_escrow_correctness.svg", fig_escrow()),
        ("fig5_llm_in_loop.svg", fig_llm()),
    ]
    optional = [
        ("ccfa_baseline_family.csv", "fig6_baseline_family.svg", fig_baseline_family),
        ("ccfa_scale_threads.csv", "fig7_scale_out.svg", fig_scale_out),
        ("ccfa_agent_aware.csv", "fig8_agent_aware_baselines.svg", fig_agent_aware),
        ("ccfa_hotspot_mixed.csv", "fig9_hotspot_mixed.svg", fig_hotspot_mixed),
        ("vitabench_authoritative.csv", "fig10_vitabench_authoritative.svg", fig_vitabench_authoritative),
        ("rigorous_vitabench_summary.csv", "fig11_rigorous_vitabench.svg", fig_rigorous_vitabench),
        ("semantic_workloads_summary.csv", "fig12_semantic_workloads.svg", fig_semantic_workloads),
    ]
    for required, name, fn in optional:
        if os.path.exists(os.path.join(RESULTS, required)):
            figures.append((name, fn()))
        else:
            print(f"skip {name}: missing {required}; run ccfa_extended_experiments.py")
    outputs = []
    for name, svg in figures:
        path = os.path.join(OUT, name)
        svg.save(path)
        outputs.append(name)
        print("saved", path)
    write_index(outputs)


if __name__ == "__main__":
    main()
