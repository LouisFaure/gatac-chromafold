"""
ChromaFold contact-map prediction from GATAC fragments and AnnData.

Reached as ``gatac.chromafold`` once installed, and as ``gatac chromafold``.
"""
from .inputs import PARAMS, ChromaFoldInputs, chromosomes, prepare
from .predict import ContactPrediction, predict

__all__ = ["PARAMS", "ChromaFoldInputs", "chromosomes", "prepare", "ContactPrediction", "predict"]
