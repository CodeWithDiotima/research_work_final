"""
stability.py
============
Steps 12-15 — Class Stability, Probability Stability, Decision
Sufficiency Score, and the stopping rule.

    CS_t   = (1/M) sum_m I[y~^(m) == y_hat_t]                       (Step 12)
    D_JS^(m) = 1/2 D_KL(p_t || r^(m)) + 1/2 D_KL(p~_t^(m) || r^(m))  (Step 13)
    PD_t   = (1/M) sum_m D_JS^(m);   PD~_t = PD_t / log(2)          (Step 13)
    DSS_t  = alpha * CS_t + (1 - alpha) * (1 - PD~_t)               (Step 14)
    DSS_t >= tau_DSS  ->  STOP, return final diagnosis              (Step 15)
"""

from __future__ import annotations

import numpy as np

from config import CFG


def kl_divergence_bernoulli(p: np.ndarray, q: np.ndarray, eps: float = CFG.dss.epsilon) -> np.ndarray:
    """
    D_KL(p || q) for Bernoulli distributions over {0,1}, summed over
    both outcomes:  sum_c p(c) log( (p(c)+eps) / (q(c)+eps) ).
    p, q are P(Y=1) probabilities (P(Y=0) = 1-p implied).
    """
    p1, p0 = p, 1 - p
    q1, q0 = q, 1 - q
    return (p1 * np.log((p1 + eps) / (q1 + eps))
            + p0 * np.log((p0 + eps) / (q0 + eps)))


def class_stability(y_plausible: np.ndarray, y_hat_t: np.ndarray) -> np.ndarray:
    """
    CS_t = (1/M) sum_m I[y~^(m) == y_hat_t]

    y_plausible: shape (M, batch) — predicted labels for each of the M
                 plausible profiles.
    y_hat_t:     shape (batch,)   — the current fused-prediction label.
    Returns:     shape (batch,)
    """
    match = (y_plausible == y_hat_t[None, :]).astype(np.float32)
    return match.mean(axis=0)


def probability_stability(p_t: np.ndarray, p_plausible: np.ndarray,
                           eps: float = CFG.dss.epsilon) -> np.ndarray:
    """
    Jensen-Shannon-divergence-based probability stability.

    p_t:          shape (batch,)    — current fused probability p_t.
    p_plausible:  shape (M, batch)  — fused probability p~_t^(m) for each
                  of the M plausible profiles.
    Returns PD~_t (normalised to [0,1]), shape (batch,).
    """
    r = (p_t[None, :] + p_plausible) / 2.0                       # r^(m)
    d1 = kl_divergence_bernoulli(np.broadcast_to(p_t, p_plausible.shape), r, eps)
    d2 = kl_divergence_bernoulli(p_plausible, r, eps)
    d_js = 0.5 * d1 + 0.5 * d2                                    # D_JS^(m)
    pd_t = d_js.mean(axis=0)                                      # PD_t
    pd_t_norm = pd_t / np.log(2)                                  # PD~_t
    return pd_t_norm


def decision_sufficiency_score(cs_t: np.ndarray, pd_t_norm: np.ndarray,
                                alpha: float = CFG.dss.alpha) -> np.ndarray:
    """DSS_t = alpha * CS_t + (1 - alpha) * (1 - PD~_t)."""
    return alpha * cs_t + (1 - alpha) * (1 - pd_t_norm)


def is_sufficient(dss_t: np.ndarray, tau_dss: float = CFG.dss.tau_dss) -> np.ndarray:
    """DSS_t >= tau_DSS -> STOP (boolean mask)."""
    return dss_t >= tau_dss


def evaluate_stability_and_dss(p_t: np.ndarray, y_hat_t: np.ndarray,
                                p_plausible: np.ndarray, y_plausible: np.ndarray,
                                alpha: float = CFG.dss.alpha):
    """
    Convenience wrapper running Steps 12-14 in one call.

    Returns (CS_t, PD_t_norm, DSS_t), each shape (batch,).
    """
    cs_t = class_stability(y_plausible, y_hat_t)
    pd_t_norm = probability_stability(p_t, p_plausible)
    dss_t = decision_sufficiency_score(cs_t, pd_t_norm, alpha)
    return cs_t, pd_t_norm, dss_t
