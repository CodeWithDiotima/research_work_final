"""
fusion.py
=========
Decision prediction and label determination for the DSS/DSG System.

In the single-classifier architecture (f_CB):
    p_t = p_catboost
    y_hat_t = argmax_y p_t(y)  ( -> 1 if p_t >= threshold else 0 )

Legacy multi-model fusion signatures are retained for backward compatibility.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from config import CFG
except ImportError:
    from dss_system.config import CFG


@dataclass
class FusionWeights:
    w_xgb: float = 1.0
    w_tr: float = 0.0


def fuse(
    p_primary: np.ndarray,
    p_secondary: Optional[np.ndarray] = None,
    weights: Optional[FusionWeights] = None,
) -> np.ndarray:
    """
    In the single CatBoost architecture:
        p_t = p_catboost
    Passes through p_primary directly when used with a single model.
    Retains weighted linear combination if legacy two-model predictions and weights are provided.
    """
    if p_secondary is None or weights is None:
        return p_primary
    return weights.w_xgb * p_primary + weights.w_tr * p_secondary


def predict_label(p_t: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """y_hat_t = argmax_y p_t(y) -> 1 if p_t >= threshold else 0."""
    return (p_t >= threshold).astype(int)


def fit_fusion_weights_on_val(
    p_xgb_val: np.ndarray,
    p_tr_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    cfg=None,
    n_grid: int = 101,
) -> FusionWeights:
    """
    Returns default weights. Retained for backward compatibility.
    """
    if p_tr_val is None or y_val is None:
        return FusionWeights(w_xgb=1.0, w_tr=0.0)

    cfg = cfg or CFG.fusion
    if not getattr(cfg, "learn_weights_on_val", False):
        return FusionWeights(cfg.w_xgb, cfg.w_tr)

    eps = 1e-12
    best_w, best_loss = cfg.w_xgb, float("inf")
    for w in np.linspace(0.0, 1.0, n_grid):
        p = w * p_xgb_val + (1 - w) * p_tr_val
        p = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
        if loss < best_loss:
            best_loss, best_w = loss, w
    return FusionWeights(w_xgb=float(best_w), w_tr=float(1 - best_w))

