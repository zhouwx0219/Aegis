"""统一评估口径：所有真并发实验共用的样式 + 置信区间助手。

目的（"统一评估坐标"）：
  - 每个并发协议在所有图里用**固定颜色 + marker**（OCC=蓝、MVCC=橄榄、2PL=红、
    Silo=青、TicToc=紫、CAST/HYBRID/ours=绿、CAST-all=橙…），跨图一致可比；
  - 吞吐统一单位 committed/s、延迟 ms、线程轴一致；
  - mean_ci() 统一报 **95% 置信区间**（t 分布；多 seed 的均值 ± 半宽）。
用法：
  from eval_common import POLICY_STYLE, mean_ci, fmt, color_of
  m, half = mean_ci(values)              # values=各 seed 的同一指标
  ax.errorbar(x, ys, yerr=halfs, **fmt("CAST"))
"""
import math
import statistics

# 95% 双侧 t 临界值表（df -> t）；df>=∞ 用正态 1.96
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 18: 2.101, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t_crit(df, conf=0.95):
    if conf != 0.95:
        # 仅支持 95%；其它置信度退化为正态近似
        return 1.96
    if df <= 0:
        return 0.0
    if df in _T95:
        return _T95[df]
    if df > 120:
        return 1.96
    # 取表中不小于 df 的最近键（保守，偏大）
    keys = sorted(_T95)
    for k in keys:
        if k >= df:
            return _T95[k]
    return 1.96


def mean_ci(values, conf=0.95):
    """返回 (均值, 95%CI 半宽)。半宽 = t_{conf,n-1} * s / sqrt(n)。n<=1 时半宽=0。"""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = statistics.mean(vals)
    if n == 1:
        return m, 0.0
    s = statistics.stdev(vals)
    half = _t_crit(n - 1, conf) * s / math.sqrt(n)
    return m, half


# 统一协议样式：name -> (color, marker, linestyle)
POLICY_STYLE = {
    "OCC":       ("tab:blue",   "o", "-"),
    "MVCC":      ("tab:olive",  "D", "-"),
    "MVCC-SI":   ("tab:olive",  "D", "-"),
    "2PL":       ("tab:red",    "d", "-."),
    "Silo":      ("tab:cyan",   "v", "-"),
    "TicToc":    ("tab:purple", "^", "-"),
    "SCC":       ("tab:gray",   "P", ":"),
    "SCC-2S":    ("tab:gray",   "P", ":"),
    "CAST":      ("tab:green",  "s", "-"),
    "CAST-all":  ("tab:orange", "^", "--"),
    "merge-all": ("tab:orange", "^", "--"),
    "HYBRID":    ("tab:green",  "s", "-"),
    # hybrid_cc 的提交结构对照
    "global":      ("tab:red",   "o", "-"),
    "per_object":  ("tab:green", "s", "-"),
}

# 消融变体调色（V0->V3 递增叠加机制）
VARIANT_COLORS = ["tab:blue", "tab:olive", "tab:purple", "tab:green"]

_FALLBACK = ("tab:gray", "o", "-")


def _style(name):
    return POLICY_STYLE.get(name, _FALLBACK)


def color_of(name):
    return _style(name)[0]


def fmt(name, linewidth=2, markersize=6, capsize=3):
    """给 ax.errorbar/plot 用的统一 kwargs（颜色+marker+线型+label）。"""
    c, mk, ls = _style(name)
    return dict(color=c, marker=mk, linestyle=ls, label=name,
                linewidth=linewidth, markersize=markersize, capsize=capsize)


def fmt_plot(name, linewidth=2, markersize=6):
    """ax.plot 用（无 capsize）。"""
    c, mk, ls = _style(name)
    return dict(color=c, marker=mk, linestyle=ls, label=name,
                linewidth=linewidth, markersize=markersize)


# 统一文案
CI_NOTE = "5 seeds, mean ± 95% CI"
