"""Evaluation helpers for comparing agent-side and DBx1000-native CC strategies."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Dbx1000NativeConfig",
    "Dbx1000NativeResult",
    "NativeCCStrategy",
    "StrategyAggregateSummary",
    "StrategyRunSummary",
    "list_native_strategies",
    "parse_summary",
    "run_native_matrix",
    "run_native_strategy",
    "run_strategy_matrix",
    "run_strategy_matrix_repeated",
]


def __getattr__(name: str) -> Any:
    if name in {
        "StrategyAggregateSummary",
        "StrategyRunSummary",
        "run_strategy_matrix",
        "run_strategy_matrix_repeated",
    }:
        cc_matrix = importlib.import_module(".cc_matrix", __name__)
        return getattr(cc_matrix, name)
    if name in {
        "Dbx1000NativeConfig",
        "Dbx1000NativeResult",
        "NativeCCStrategy",
        "list_native_strategies",
        "parse_summary",
        "run_native_matrix",
        "run_native_strategy",
    }:
        dbx1000_native = importlib.import_module(".dbx1000_native", __name__)
        return getattr(dbx1000_native, name)
    raise AttributeError(name)
