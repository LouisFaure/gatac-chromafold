"""
ChromaFold contact-map prediction from GATAC fragments and AnnData.

Reached as ``gatac.chromafold`` once installed, and as ``gatac chromafold``.
"""
import importlib
from typing import TYPE_CHECKING

# `gatac` imports gatac_chromafold.cli, and so this package, on every command:
# the modules, which load gatac.pp and RAPIDS, are imported on first use.
_LAZY = {
    "PARAMS": "inputs",
    "ChromaFoldInputs": "inputs",
    "chromosomes": "inputs",
    "prepare": "inputs",
    "ContactPrediction": "prediction",
    "predict": "prediction",
}

if TYPE_CHECKING:
    from .inputs import PARAMS, ChromaFoldInputs, chromosomes, prepare  # noqa: F401
    from .prediction import ContactPrediction, predict  # noqa: F401

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
