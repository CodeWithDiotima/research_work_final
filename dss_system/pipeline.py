"""
pipeline.py
===========
End-to-end orchestration of the Decision-Sufficiency-Guided Adaptive
Clinical Information Acquisition System.

    D -> Z -> (x_obs, m) -> {f_XGB, f_TR} -> p_t -> y_hat_t
      -> P_Psi(x_miss|x_obs) -> {x~^(1..M)}
      -> {f_XGB, f_TR} on each -> CS_t, PD_t -> DSS_t
      -> DSS_t >= tau_DSS ? STOP
                           : {R_j, Conn_j, DSG_j, C_j} -> S_j -> j*
                             -> acquire x*_j* -> repeat

This module wires config.py, data_processing.py, models/*, fusion.py,
stability.py, graph.py and acquisition.py into the single per-patient
loop described in "The Complete Mathematical Flow" of the spec.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from config import CFG, FEATURE_COLUMNS
from fusion import FusionWeights, fuse, predict_label, fit_fusion_weights_on_val
from stability import evaluate_stability_and_dss, is_sufficient
from graph import ClinicalGraph, build_clinical_graph
from acquisition import (
    ModelBundle,
    expected_future_dss_for_feature,
    acquisition_scores,
    select_next_variable,
    acquire_actual_value,
)
from models.xgboost_model import XGBClassifier
from models.tabresnet import TabResNetClassifier
from models.cvae import CVAETrainer


@dataclass
class TrainedSystem:
    """Everything the adaptive-acquisition loop needs at inference time."""
    xgb: XGBClassifier
    tabresnet: TabResNetClassifier
    cvae: CVAETrainer
    fusion_weights: FusionWeights
    graph: ClinicalGraph
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def _candidate_value_sampler(system: TrainedSystem, cfg=CFG.acquisition):
    """
    Builds the Step-18 candidate-value sampler used by acquisition.py:
    draws K plausible values for one feature, conditioned on x_obs, using
    the trained CVAE decoder, with uniform Monte-Carlo weights.
    """

    def sample_candidate_values(X_obs, m, feature_idx, K):
        # Reuse the CVAE's plausible-profile sampler and read off the
        # column for `feature_idx` — this is exactly g_j(x_{-j}) restricted
        # to feature j.
        profiles = system.cvae.generate_plausible_profiles(X_obs, m, n_samples=K)
        values = profiles[:, :, feature_idx]         # (K, batch)
        probs = np.full_like(values, 1.0 / K)         # uniform MC weights
        return values, probs

    return sample_candidate_values


def build_model_bundle(system: TrainedSystem) -> ModelBundle:
    return ModelBundle(
        xgb_predict=lambda X_obs, m: system.xgb.predict_proba_y1(X_obs, m),
        tr_predict=lambda X_obs, m: system.tabresnet.predict_proba_y1(X_obs, m),
        sample_candidate_values=_candidate_value_sampler(system),
        sample_plausible=lambda X_obs, m, n: system.cvae.generate_plausible_profiles(X_obs, m, n),
        fusion_weights=system.fusion_weights,
    )


@dataclass
class PatientOutcome:
    y_hat: int
    p_t: float
    dss: float
    acquired_features: List[str]
    n_acquisitions: int
    total_cost: float


def run_adaptive_acquisition(
    X_true: np.ndarray,          # (batch, F) ground-truth features (Step 29 source)
    X_obs_init: np.ndarray,      # (batch, F) initially observed-filled features
    m_init: np.ndarray,          # (batch, F) initial observation mask
    system: TrainedSystem,
    cfg=CFG,
) -> List[PatientOutcome]:
    """
    The full "Complete Mathematical Flow" loop, run for a batch of
    patients until every patient either reaches DSS_t >= tau_DSS or hits
    the acquisition cap (Steps 1-29, repeated).
    """
    batch, F = X_obs_init.shape
    X_obs = X_obs_init.copy()
    m = m_init.copy()
    models = build_model_bundle(system)

    active = np.ones(batch, dtype=bool)            # patients still acquiring
    acquired_features: List[List[int]] = [[] for _ in range(batch)]
    total_cost = np.zeros(batch)
    from acquisition import normalized_costs
    per_feature_cost = normalized_costs(system.feature_names, cfg.acquisition)

    final_y_hat = np.zeros(batch, dtype=int)
    final_p = np.zeros(batch)
    final_dss = np.zeros(batch)

    for t in range(cfg.dss.max_acquisitions):
        if not active.any():
            break

        idx = np.where(active)[0]
        Xo, mm = X_obs[idx], m[idx]

        # p_t, y_hat_t (Step 4-6)
        p_xgb = models.xgb_predict(Xo, mm)
        p_tr = models.tr_predict(Xo, mm)
        p_t = fuse(p_xgb, p_tr, models.fusion_weights)
        y_hat_t = predict_label(p_t)

        # Plausible completions + CS_t, PD_t, DSS_t (Steps 7-14)
        plausible = models.sample_plausible(Xo, mm, cfg.cvae.n_samples_M)
        M = plausible.shape[0]
        p_plaus = np.zeros((M, len(idx)))
        y_plaus = np.zeros((M, len(idx)), dtype=int)
        full_mask = np.ones_like(mm)
        for mi in range(M):
            profile = plausible[mi]
            pxm = models.xgb_predict(profile, full_mask)
            ptm = models.tr_predict(profile, full_mask)
            p_plaus[mi] = fuse(pxm, ptm, models.fusion_weights)
            y_plaus[mi] = predict_label(p_plaus[mi])

        cs_t, pd_t, dss_t = evaluate_stability_and_dss(
            p_t, y_hat_t, p_plaus, y_plaus, alpha=cfg.dss.alpha
        )

        # Step 15: stopping decision.
        sufficient = is_sufficient(dss_t, cfg.dss.tau_dss)
        for local_i, global_i in enumerate(idx):
            if sufficient[local_i]:
                final_y_hat[global_i] = y_hat_t[local_i]
                final_p[global_i] = p_t[local_i]
                final_dss[global_i] = dss_t[local_i]
                active[global_i] = False

        remaining_local = np.where(~sufficient)[0]
        if len(remaining_local) == 0:
            continue

        rem_global = idx[remaining_local]
        Xo_r, mm_r = X_obs[rem_global], m[rem_global]
        dss_r = dss_t[remaining_local]

        # Steps 16-27: score every unavailable feature for the remaining
        # patients. (Boilerplate simplification: evaluated per-patient in
        # Python loop for clarity — vectorise across the batch for scale.)
        unavailable = [j for j in range(F)]  # scored per-patient via mask below
        dsg: dict = {}
        for j in range(F):
            still_missing = (mm_r[:, j] == 0)
            if not still_missing.any():
                continue
            dsg_j = expected_future_dss_for_feature(
                j, Xo_r, mm_r, dss_r, models, cfg=cfg.acquisition, dss_cfg=cfg.dss
            )
            dsg_j = np.where(still_missing, dsg_j, -np.inf)  # can't re-acquire an observed feature
            dsg[j] = dsg_j

        if not dsg:
            # nothing left to acquire for anyone in this remainder — stop
            for local_i, global_i in enumerate(rem_global):
                final_y_hat[global_i] = y_hat_t[remaining_local[local_i]]
                final_p[global_i] = p_t[remaining_local[local_i]]
                final_dss[global_i] = dss_r[local_i]
                active[global_i] = False
            continue

        avg_observed_mask = (m[active].mean(axis=0) > 0) if active.any() else np.zeros(F, dtype=bool)
        scores = acquisition_scores(
            unavailable_features=list(dsg.keys()),
            dsg=dsg,
            graph=system.graph,
            observed_mask_bool=avg_observed_mask,
            cfg=cfg.acquisition,
        )
        j_star = select_next_variable(scores)  # (len(rem_global),)

        for local_i, global_i in enumerate(rem_global):
            j = int(j_star[local_i])
            acquire_actual_value(X_true, X_obs, m, global_i, j)
            acquired_features[global_i].append(j)
            total_cost[global_i] += per_feature_cost[j]

    # Any patient that never reached sufficiency by max_acquisitions:
    # finalise with their last computed prediction.
    still_active = np.where(active)[0]
    if len(still_active) > 0:
        Xo, mm = X_obs[still_active], m[still_active]
        p_xgb = models.xgb_predict(Xo, mm)
        p_tr = models.tr_predict(Xo, mm)
        p_t = fuse(p_xgb, p_tr, models.fusion_weights)
        y_hat_t = predict_label(p_t)
        for local_i, global_i in enumerate(still_active):
            final_y_hat[global_i] = y_hat_t[local_i]
            final_p[global_i] = p_t[local_i]
            final_dss[global_i] = final_dss[global_i]  # last computed / 0 if never scored

    outcomes = []
    for i in range(batch):
        outcomes.append(PatientOutcome(
            y_hat=int(final_y_hat[i]),
            p_t=float(final_p[i]),
            dss=float(final_dss[i]),
            acquired_features=[system.feature_names[j] for j in acquired_features[i]],
            n_acquisitions=len(acquired_features[i]),
            total_cost=float(total_cost[i]),
        ))
    return outcomes


def train_full_system(prepared, cfg=CFG) -> TrainedSystem:
    """
    Trains f_XGB, f_TR, the CVAE generator, fits fusion weights on the
    validation set, and builds the clinical information graph — i.e.
    everything needed before the adaptive-acquisition loop can run.
    """
    xgb = XGBClassifier(cfg.xgb).fit(
        prepared.X_train * prepared.m_train, prepared.m_train, prepared.y_train,
        prepared.X_val * prepared.m_val, prepared.m_val, prepared.y_val,
    )

    input_dim = prepared.X_train.shape[1] * 2
    tabresnet = TabResNetClassifier(input_dim, cfg.tabresnet).fit(
        prepared.X_train * prepared.m_train, prepared.m_train, prepared.y_train,
        prepared.X_val * prepared.m_val, prepared.m_val, prepared.y_val,
    )

    cvae = CVAETrainer(prepared.X_train.shape[1], cfg.cvae).fit(
        prepared.X_train, prepared.m_train
    )

    p_xgb_val = xgb.predict_proba_y1(prepared.X_val * prepared.m_val, prepared.m_val)
    p_tr_val = tabresnet.predict_proba_y1(prepared.X_val * prepared.m_val, prepared.m_val)
    fusion_weights = fit_fusion_weights_on_val(p_xgb_val, p_tr_val, prepared.y_val, cfg.fusion)

    graph = build_clinical_graph(prepared.X_train, prepared.y_train)

    return TrainedSystem(
        xgb=xgb, tabresnet=tabresnet, cvae=cvae,
        fusion_weights=fusion_weights, graph=graph,
    )
