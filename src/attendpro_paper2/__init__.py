"""Readable research interface for AttendPro Paper II."""

from .simulation import FEATURES, REGIMES, simulate_attendpro
from .modeling import evaluate, fit_models, pooled, predict, rule, tune_rule, usable

__all__ = [
    "FEATURES",
    "REGIMES",
    "simulate_attendpro",
    "evaluate",
    "fit_models",
    "pooled",
    "predict",
    "rule",
    "tune_rule",
    "usable",
]
