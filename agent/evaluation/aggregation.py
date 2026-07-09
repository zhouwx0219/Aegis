"""Pure aggregation helpers shared by evaluation runners."""

from __future__ import annotations

import math
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(percentile_value))) / 100.0
    position = rank * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_stddev(values: Sequence[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) <= 1:
        return 0.0
    avg = sum(rows) / len(rows)
    return math.sqrt(sum((value - avg) ** 2 for value in rows) / (len(rows) - 1))
