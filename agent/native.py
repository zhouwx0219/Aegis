"""Helpers for loading CAST-DAS native extensions with actionable errors."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable


class NativeExtensionError(ImportError):
    """Raised when a required native extension is missing or incompatible."""


def load_cast_core(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Load the pybind11 cast_core extension with a build-focused error."""

    try:
        return importer("cast_core")
    except ModuleNotFoundError as exc:
        if exc.name != "cast_core":
            raise
        raise NativeExtensionError(_cast_core_help()) from exc
    except ImportError as exc:
        raise NativeExtensionError(_cast_core_help()) from exc


def _cast_core_help() -> str:
    return (
        "Native extension 'cast_core' is not available for this Python runtime. "
        "From the repository root, run `python3 -m pip install -e .` and "
        "`bash build.sh`, then run tests with the same Linux/WSL Python that "
        "built the extension."
    )
