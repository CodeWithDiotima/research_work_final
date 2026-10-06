"""
acquisition.py
===============
Steps 18-29 — Candidate variable evaluation and adaptive feature
selection.

  Step 18  Candidate values v_j^(k), k = 1..K_j, sampled from
           P_psi_j(v_j | x_obs); temporarily set x_j = v_j^(k).
  Step 19  Run f_XGB and f_TR on every (candidate value, plausible
           completion) pair, then fuse.
  Step 20  Future class stability   CS_j^(k)
  Step 21  Future probability stability   PD~_j^(k)
  Step 22  Future DSS   DSS_j^(k) = alpha*CS_j^(k) + (1-alpha)(1-PD~_j^(k))
  Step 23  Expected future DSS   E[DSS_{t+1} | v_j] = sum_k P(v_j^(k)|x_obs) DSS_j^(k)
  Step 24  Decision Stabilization Gain   DSG_j = E[DSS_{t+1}|v_j] - DSS_t
  Step 25  Graph connectivity Conn_j            (graph.py)
  Step 26  Acquisition cost   C_j = eta_a*C_acq + eta_c*C_comm + eta_e*C_comp
  Step 27  Acquisition score  S_j = lam1*R_j + lam2*Conn_j + lam3*DSG_j - lam4*C_j
  Step 28  Select next variable  j* = argmax_{j in U_t} S_j
  Step 29  Acquire actual value x*_{j*}  (external — real measurement)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from config import CFG
from fusion import FusionWeights, fuse, predict_label
from stability import evaluate_stability_and_dss
from graph import ClinicalGraph


# ---------------------------------------------------------------------------
# Step 26 — Acquisition cost model (hardcode / override per feature as needed)
# ---------------------------------------------------------------------------
# C_acq: intrinsic acquisition burden (e.g. invasiveness, patient time)
# C_comm: communication/logistics cost (e.g. lab turnaround)
# C_comp: computational cost of incorporating the feature
# All three are on an arbitrary 1-10 raw scale; only their normalised
# combination C~_j is compared across features, so absolute units don't
# matter as long as they're used consistently.
#
# Placeholder values reflect how each BRFSS-derived feature is actually
# collected: most are single self-report survey questions (cheap), while
# HighBP/HighChol require an in-clinic measurement or blood draw
# (expensive), and BMI needs a height/weight measurement (moderate).
# Replace with real per-feature costs (time, $, invasiveness) once
# available — nothing else in acquisition.py needs to change.
DEFAULT_RAW_COSTS: Dict[str, Dict[str, float]] = {
    "HighBP":               {"acq": 4, "comm": 2, "comp": 1},  # BP cuff reading
    "HighChol":             {"acq": 6, "comm": 5, "comp": 1},  # blood draw + lab turnaround
    "CholCheck":            {"acq": 1, "comm": 1, "comp": 1},  # self-report yes/no
    "BMI":                  {"acq": 3, "comm": 1, "comp": 1},  # height/weight measurement
    "Smoker":               {"acq": 1, "comm": 1, "comp": 1},
    "Stroke":               {"acq": 1, "comm": 1, "comp": 1},
    "HeartDiseaseorAttack": {"acq": 1, "comm": 1, "comp": 1},
    "PhysActivity":         {"acq": 1, "comm": 1, "comp": 1},
    "Fruits":                {"acq": 1, "comm": 1, "comp": 1},
    "Veggies":               {"acq": 1, "comm": 1, "comp": 1},
    "HvyAlcoholConsump":    {"acq": 1, "comm": 1, "comp": 1},
    "AnyHealthcare":        {"acq": 1, "comm": 1, "comp": 1},
    "NoDocbcCost":          {"acq": 1, "comm": 1, "comp": 1},
    "GenHlth":              {"acq": 1, "comm": 1, "comp": 1},
    "MentHlth":             {"acq": 1, "comm": 1, "comp": 1},
    "PhysHlth":             {"acq": 1, "comm": 1, "comp": 1},
    "DiffWalk":             {"acq": 1, "comm": 1, "comp": 1},
    "Sex":                   {"acq": 1, "comm": 1, "comp": 1},
    "Age":                   {"acq": 1, "comm": 1, "comp": 1},
    "Education":            {"acq": 1, "comm": 1, "comp": 1},
    "Income":               {"acq": 2, "comm": 2, "comp": 1},  # sensitive question, slight friction
}


def raw_feature_cost(feature_name: str, cfg=CFG.acquisition) -> float:
    c = DEFAULT_RAW_COSTS.get(feature_name, {"acq": 1.0, "comm": 1.0, "comp": 1.0})
    return cfg.eta_a * c["acq"] + cfg.eta_c * c["comm"] + cfg.eta_e * c["comp"]


def normalized_costs(feature_names: List[str], cfg=CFG.acquisition,
                      eps: float = CFG.dss.epsilon) -> np.ndarray:
    """C~_j for every feature j (Step 26)."""
    raw = np.array([raw_feature_cost(f, cfg) for f in feature_names])
    return (raw - raw.min()) / (raw.max() - raw.min() + eps)


# ---------------------------------------------------------------------------
# Steps 18-24 — per-candidate-feature future-DSS evaluation
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """The two diagnostic classifiers + generative model, bundled together
    so acquisition.py doesn't need to import model internals directly."""
    xgb_predict: callable   # (X_obs, m) -> p_XGB
    tr_predict: callable    # (X_obs, m) -> p_TR
    sample_candidate_values: callable  # (X_obs, m, feature_idx, K) -> (K, batch) values
    sample_plausible: callable         # (X_obs, m, n_samples) -> (M, batch, F) profiles
    fusion_weights: FusionWeights


def expected_future_dss_for_feature(
    feature_idx: int,
    X_obs: np.ndarray,      # (batch, F) current observed-filled features
    m: np.ndarray,          # (batch, F) current mask
    dss_t: np.ndarray,      # (batch,) current DSS
    models: ModelBundle,
    cfg=CFG.acquisition,
    dss_cfg=CFG.dss,
) -> np.ndarray:
    """
    Steps 18-24 for a single candidate feature j, vectorised over the
    batch of patients still under evaluation.

    Returns DSG_j = E[DSS_{t+1} | v_j] - DSS_t, shape (batch,).
    """
    batch, F = X_obs.shape
    K = cfg.n_candidate_values_K

    # Step 18: candidate values v_j^(k) and their probabilities P(v_j^(k)|x_obs)
    candidate_values, candidate_probs = models.sample_candidate_values(
        X_obs, m, feature_idx, K
    )  # each shape (K, batch)

    dss_per_candidate = np.zeros((K, batch), dtype=np.float64)

    for k in range(K):
        # Temporarily set x_j = v_j^(k) and mark it observed.
        X_obs_k = X_obs.copy()
        X_obs_k[:, feature_idx] = candidate_values[k]
        m_k = m.copy()
        m_k[:, feature_idx] = 1

        # Step 8 (re-run for the new mask): plausible completions of the
        # remaining still-missing features.
        plausible = models.sample_plausible(X_obs_k, m_k, dss_cfg.max_acquisitions
                                             and CFG.cvae.n_samples_M)
        M_samples, _, _ = plausible.shape

        # Step 19: run both classifiers on every plausible profile.
        p_xgb_now = models.xgb_predict(X_obs_k, m_k)
        p_tr_now = models.tr_predict(X_obs_k, m_k)
        p_t_k = fuse(p_xgb_now, p_tr_now, models.fusion_weights)
        y_hat_k = predict_label(p_t_k)

        p_plaus = np.zeros((M_samples, batch))
        y_plaus = np.zeros((M_samples, batch), dtype=np.int64)
        for mi in range(M_samples):
            profile = plausible[mi]  # (batch, F), fully observed
            full_mask = np.ones_like(m_k)
            p_xgb_m = models.xgb_predict(profile, full_mask)
            p_tr_m = models.tr_predict(profile, full_mask)
            p_plaus[mi] = fuse(p_xgb_m, p_tr_m, models.fusion_weights)
            y_plaus[mi] = predict_label(p_plaus[mi])

        # Steps 20-22: future class/probability stability -> future DSS.
        cs_k, pd_k, dss_k = evaluate_stability_and_dss(
            p_t_k, y_hat_k, p_plaus, y_plaus, alpha=dss_cfg.alpha
        )
        dss_per_candidate[k] = dss_k

    # Step 23: expected future DSS, probability-weighted over candidate values.
    weights = candidate_probs / (candidate_probs.sum(axis=0, keepdims=True) + dss_cfg.epsilon)
    expected_dss_next = (weights * dss_per_candidate).sum(axis=0)  # (batch,)

    # Step 24: Decision Stabilization Gain.
    dsg_j = expected_dss_next - dss_t
    return dsg_j


# ---------------------------------------------------------------------------
# Steps 25-28 — combine relevance, connectivity, DSG and cost into S_j,
# then pick j*
# ---------------------------------------------------------------------------
def acquisition_scores(
    unavailable_features: List[int],
    dsg: Dict[int, np.ndarray],          # feature_idx -> (batch,) DSG_j
    graph: ClinicalGraph,
    observed_mask_bool: np.ndarray,      # (F,) which features are currently observed
    cfg=CFG.acquisition,
) -> Dict[int, np.ndarray]:
    """
    Step 27: S_j = lambda1*R_j + lambda2*Conn_j + lambda3*DSG_j - lambda4*C_j
    for every candidate j in U_t. Returns {feature_idx: (batch,) scores}.
    """
    conn = graph.connectivity(observed_mask_bool)         # (F,)
    costs = normalized_costs(graph.feature_names, cfg)    # (F,)

    scores = {}
    for j in unavailable_features:
        R_j = graph.R_norm[j]
        Conn_j = conn[j]
        C_j = costs[j]
        DSG_j = dsg[j]
        scores[j] = (cfg.lambda_1 * R_j
                     + cfg.lambda_2 * Conn_j
                     + cfg.lambda_3 * DSG_j
                     - cfg.lambda_4 * C_j)
    return scores


def select_next_variable(scores: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Step 28: j*_t = argmax_{j in U_t} S_j, computed per-patient.

    Returns an array of shape (batch,) with the chosen feature index for
    each patient still requiring acquisition.
    """
    feature_ids = list(scores.keys())
    stacked = np.stack([scores[j] for j in feature_ids], axis=0)  # (n_candidates, batch)
    best = np.argmax(stacked, axis=0)
    return np.array([feature_ids[b] for b in best])


def acquire_actual_value(X_true: np.ndarray, X_obs: np.ndarray, m: np.ndarray,
                          patient_idx: int, feature_idx: int) -> None:
    """
    Step 29: acquire the actual patient value x*_{j*} and update the
    live record in place.

        O_{t+1} = O_t U {j*} ; U_{t+1} = U_t \\ {j*}

    In production this is where a real measurement/lab result/EHR
    lookup would be fetched; here it pulls from the ground-truth array
    (X_true) as a stand-in until that integration exists.
    """
    X_obs[patient_idx, feature_idx] = X_true[patient_idx, feature_idx]
    m[patient_idx, feature_idx] = 1
