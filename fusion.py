"""
fusion.py
=========
Step 6 — Fusion of the two classifiers.

    p_t = w_XGB * p_XGB,t + w_TR * p_TR,t     with  w_XGB + w_TR = 1
    y_hat_t = argmax_y p_t(y)                 ( -> 1 if p_t >= 0.5 else 0 )

Also supports learning w_XGB on the validation set (simple 1-D grid
search maximising validation log-loss / accuracy), per the project's
"determine the weights using the validation set" note.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from config import CFG


@dataclass
class FusionWeights:
    w_xgb: float
    w_tr: float


def fuse(p_xgb: np.ndarray, p_tr: np.ndarray, weights: FusionWeights) -> np.ndarray:
    """p_t = w_XGB * p_XGB + w_TR * p_TR."""
    return weights.w_xgb * p_xgb + weights.w_tr * p_tr


def predict_label(p_t: np.ndarray) -> np.ndarray:
    """y_hat_t = argmax_y p_t(y)."""
    return (p_t >= 0.5).astype(int)


def fit_fusion_weights_on_val(p_xgb_val: np.ndarray, p_tr_val: np.ndarray,
                               y_val: np.ndarray, cfg=CFG.fusion,
                               n_grid: int = 101) -> FusionWeights:
    """
    1-D grid search over w_XGB in [0,1] (w_TR = 1 - w_XGB) that minimises
    validation log-loss. Falls back to the configured default weights if
    `learn_weights_on_val` is False.
    """
    if not cfg.learn_weights_on_val:
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
