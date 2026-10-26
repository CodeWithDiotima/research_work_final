"""
pipeline.py
===========
End-to-end orchestration of the Decision-Sufficiency-Guided Adaptive
Clinical Information Acquisition System (DSS/DSG).

Mathematical Pipeline Flow:
---------------------------
1. Training Phase:
   - Features X and masks m -> Train {f_XGB, f_TR} classifiers.
   - Model missingness generative distribution P(x_miss | x_obs) using CVAE.
   - Learn ensemble fusion weights w on validation set.
   - Build Clinical Information Graph G = (V, E, W).

2. Adaptive Acquisition Loop (per patient case):
   - Given initial partial observation (x_obs, m):
     a. Compute fused prediction: p_t = w_XGB * p_XGB + w_TR * p_TR -> y_hat_t.
     b. Generate M plausible complete profiles via CVAE: x~^(1..M) ~ P_psi(x_miss | x_obs).
     c. Measure classification stability (CS_t) and prediction divergence (PD_t) across M completions.
     d. Calculate Decision Sufficiency Score: DSS_t = alpha * CS_t + (1 - alpha) * (1 - PD_t).
     e. If DSS_t >= tau_DSS (or max acquisitions reached): STOP and output diagnosis y_hat_t.
     f. Else: For each unobserved feature j, estimate expected future DSS gain (DSG_j).
     g. Score each candidate feature: S_j = lambda_1*R_j + lambda_2*Conn_j + lambda_3*DSG_j - lambda_4*C_j.
     h. Select highest scoring test: j* = argmax(S_j).
     i. Acquire actual ground-truth value for feature j*, update (x_obs, m), and repeat.
"""

# Enable modern Python 3.7+ typing capabilities
from __future__ import annotations

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

# Import NumPy for array math, vectorization, and logical indexing
import numpy as np
# Import dataclass decorators for clean data structuring
from dataclasses import dataclass, field
# Import typing aids for static analysis
from typing import List, Optional

# Import global config parameters and feature definitions
from config import CFG, FEATURE_COLUMNS
# Import Bayesian fusion utilities: weight container, fusion formula, argmax classifier, and optimizer
from fusion import FusionWeights, fuse, predict_label, fit_fusion_weights_on_val
# Import decision sufficiency calculation and threshold-checking utilities
from stability import evaluate_stability_and_dss, is_sufficient
# Import clinical graph structures and correlation/mutual-information graph builder
from graph import ClinicalGraph, build_clinical_graph
# Import feature acquisition scoring components
from acquisition import (
    ModelBundle,
    expected_future_dss_for_feature,
    acquisition_scores,
    select_next_variable,
    acquire_actual_value,
)
# Import individual model wrappers
from models.lightgbm_model import LightGBMClassifierWrapper
from models.cvae import CVAETrainer


@dataclass
class TrainedSystem:
    """
    Container packaging all trained models and structures needed at inference time:
    - lightgbm: Fitted LightGBM gradient boosted tree classifier (f_LGBM)
    - cvae: Fitted Conditional Variational Autoencoder missing-value generator
    - graph: Clinical Information Graph capturing feature-feature correlations
    - feature_names: Ordered list of all 21 clinical feature names
    - fusion_weights: Optional[FusionWeights] (retained for backward compatibility)
    """
    lightgbm: LightGBMClassifierWrapper
    cvae: CVAETrainer
    graph: ClinicalGraph
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    fusion_weights: Optional[FusionWeights] = None

    @property
    def catboost(self):
        """Backward-compatibility alias."""
        return self.lightgbm

    @property
    def xgb(self):
        """Backward-compatibility alias."""
        return self.lightgbm

    @property
    def tabresnet(self):
        """Backward-compatibility alias."""
        return self.lightgbm


def _candidate_value_sampler(system: TrainedSystem, cfg=CFG.acquisition):
    """
    Builds the Step-18 candidate-value sampler used during feature acquisition:
    Draws K plausible counterfactual values for feature j given currently observed features x_obs,
    using the trained CVAE decoder, assigning equal Monte-Carlo probability weights (1/K).
    """

    def sample_candidate_values(X_obs, m, feature_idx, K):
        # Generate K plausible complete patient profiles using the generative CVAE
        profiles = system.cvae.generate_plausible_profiles(X_obs, m, n_samples=K)
        # Extract the specific column corresponding to candidate feature `feature_idx`
        # Shape: (K, batch_size)
        values = profiles[:, :, feature_idx]
        # Assign uniform Monte Carlo probabilities (1/K) to each candidate sample
        probs = np.full_like(values, 1.0 / K)
        # Return candidate values and their associated probabilities
        return values, probs

    # Return the sampling closure
    return sample_candidate_values


def build_model_bundle(system: TrainedSystem) -> ModelBundle:
    """
    Packages model inference functions into a clean ModelBundle interface
    expected by the feature acquisition module.
    """
    predict_fn = lambda X_obs, m: system.lightgbm.predict_proba_y1(X_obs, m)
    return ModelBundle(
        classifier_predict=predict_fn,
        sample_candidate_values=_candidate_value_sampler(system),
        sample_plausible=lambda X_obs, m, n: system.cvae.generate_plausible_profiles(X_obs, m, n),
        fusion_weights=system.fusion_weights,
        xgb_predict=predict_fn,
        tr_predict=predict_fn,
    )


@dataclass
class PatientOutcome:
    """
    Data record summarizing the final clinical decision and acquisition history for one patient:
    - y_hat: Final predicted diagnosis (0 = healthy, 1 = diabetic)
    - p_t: Final fused disease probability
    - dss: Final Decision Sufficiency Score achieved
    - acquired_features: List of feature names acquired during the session
    - n_acquisitions: Total number of features ordered
    - total_cost: Cumulative clinical cost incurred
    """
    y_hat: int
    p_t: float
    dss: float
    acquired_features: List[str]
    n_acquisitions: int
    total_cost: float


def run_adaptive_acquisition(
    X_true: np.ndarray,          # (batch, F) Ground-truth feature values (lab oracle)
    X_obs_init: np.ndarray,      # (batch, F) Initially observed feature values
    m_init: np.ndarray,          # (batch, F) Initial observation masks (1=known, 0=unknown)
    system: TrainedSystem,       # Trained model bundle
    cfg=CFG,                     # Pipeline hyperparameter configuration
) -> List[PatientOutcome]:
    """
    Simulates the complete adaptive clinical acquisition loop for a batch of patients.
    Iteratively acquires high-value tests until DSS >= tau_DSS or budget cap is reached.
    """
    # Extract batch size (number of patients) and number of features (F = 21)
    batch, F = X_obs_init.shape
    
    # Make local working copies of observed features and masks to update iteratively
    X_obs = X_obs_init.copy()
    m = m_init.copy()
    
    # Assemble model inference bundle for predictions and generative completions
    models = build_model_bundle(system)

    # Boolean array tracking which patients are still actively acquiring information
    active = np.ones(batch, dtype=bool)
    
    # History list recording ordered feature indices for each patient
    acquired_features: List[List[int]] = [[] for _ in range(batch)]
    
    # Cumulative clinical cost accumulator for each patient
    total_cost = np.zeros(batch)
    
    # Import cost normalizer and precalculate per-feature normalized costs C_j
    from acquisition import normalized_costs
    per_feature_cost = normalized_costs(system.feature_names, cfg.acquisition)

    # Arrays to store final outputs for each patient upon termination
    final_y_hat = np.zeros(batch, dtype=int)
    final_p = np.zeros(batch)
    final_dss = np.zeros(batch)

    # Main acquisition loop: iterates up to max_acquisitions steps (e.g., 15)
    for t in range(cfg.dss.max_acquisitions):
        # If all patients have satisfied the stopping rule, terminate early
        if not active.any():
            break

        # Get global indices of patients who still need more information
        idx = np.where(active)[0]
        # Slice current observed feature matrix and mask for active patients
        Xo, mm = X_obs[idx], m[idx]

        # -------------------------------------------------------------------
        # Step 4-6: Current Single-Classifier Prediction
        # -------------------------------------------------------------------
        # Predict risk probability from LightGBM using currently observed features
        p_t = models.classifier_predict(Xo, mm)
        # Convert continuous probabilities into binary labels: 1 if p_t >= threshold else 0
        th = getattr(system.lightgbm, "best_threshold_", 0.5)
        y_hat_t = predict_label(p_t, threshold=th)

        # -------------------------------------------------------------------
        # Step 7-14: Decision Sufficiency Evaluation under CVAE Completions
        # -------------------------------------------------------------------
        # Sample M plausible complete profiles from CVAE conditioned on observed features
        plausible = models.sample_plausible(Xo, mm, cfg.cvae.n_samples_M)
        M = plausible.shape[0]
        
        # Arrays to store predictions across all M counterfactual plausible completions
        p_plaus = np.zeros((M, len(idx)))
        y_plaus = np.zeros((M, len(idx)), dtype=int)
        
        # Since completions are fully imputed, the mask is all ones
        full_mask = np.ones_like(mm)
        
        # Evaluate model predictions across each generated plausible completion
        for mi in range(M):
            profile = plausible[mi]
            p_plaus[mi] = models.classifier_predict(profile, full_mask)
            y_plaus[mi] = predict_label(p_plaus[mi], threshold=th)

        # Calculate Classification Stability (CS_t), Predictive Divergence (PD_t), and DSS_t
        cs_t, pd_t, dss_t = evaluate_stability_and_dss(
            p_t, y_hat_t, p_plaus, y_plaus, alpha=cfg.dss.alpha
        )

        # -------------------------------------------------------------------
        # Step 15: Stopping Rule Check (DSS_t >= tau_DSS)
        # -------------------------------------------------------------------
        sufficient = is_sufficient(dss_t, cfg.dss.tau_dss)
        
        # Freeze and stop acquisition for patients whose evidence is already sufficient
        for local_i, global_i in enumerate(idx):
            if sufficient[local_i]:
                final_y_hat[global_i] = y_hat_t[local_i]
                final_p[global_i] = p_t[local_i]
                final_dss[global_i] = dss_t[local_i]
                active[global_i] = False  # Mark patient as finished

        # Find patients in the current batch who are still insufficient
        remaining_local = np.where(~sufficient)[0]
        if len(remaining_local) == 0:
            continue

        # Map back to global patient indices
        rem_global = idx[remaining_local]
        Xo_r, mm_r = X_obs[rem_global], m[rem_global]
        dss_r = dss_t[remaining_local]

        # -------------------------------------------------------------------
        # Steps 16-27: Score Every Missing Feature & Select Best Test
        # -------------------------------------------------------------------
        dsg: dict = {}
        for j in range(F):
            # Check which remaining patients are actually missing feature j
            still_missing = (mm_r[:, j] == 0)
            if not still_missing.any():
                continue
            
            # Step 18-24: Compute expected Decision Sufficiency Gain DSG_j if feature j is acquired
            dsg_j = expected_future_dss_for_feature(
                j, Xo_r, mm_r, dss_r, models, cfg=cfg.acquisition, dss_cfg=cfg.dss
            )
            # Mask out features that are already known so they cannot be acquired again
            dsg_j = np.where(still_missing, dsg_j, -np.inf)
            dsg[j] = dsg_j

        # If no features remain unobserved for these patients, finalize and exit
        if not dsg:
            for local_i, global_i in enumerate(rem_global):
                final_y_hat[global_i] = y_hat_t[remaining_local[local_i]]
                final_p[global_i] = p_t[remaining_local[local_i]]
                final_dss[global_i] = dss_r[local_i]
                active[global_i] = False
            continue

        # Compute average observation pattern across active patients to evaluate graph connectivity
        avg_observed_mask = (m[active].mean(axis=0) > 0) if active.any() else np.zeros(F, dtype=bool)
        
        # Step 25-27: Calculate total multi-factor acquisition score S_j for each candidate feature
        scores = acquisition_scores(
            unavailable_features=list(dsg.keys()),
            dsg=dsg,
            graph=system.graph,
            observed_mask_bool=avg_observed_mask,
            cfg=cfg.acquisition,
        )
        
        # Step 28: Greedily pick the highest-scoring clinical test j* = argmax(S_j)
        j_star = select_next_variable(scores)

        # -------------------------------------------------------------------
        # Step 29: Feature Acquisition & State Update
        # -------------------------------------------------------------------
        for local_i, global_i in enumerate(rem_global):
            j = int(j_star[local_i])
            # Reveal actual laboratory value from X_true, update X_obs and set mask m[j] = 1
            acquire_actual_value(X_true, X_obs, m, global_i, j)
            # Record feature in the patient's acquired test history
            acquired_features[global_i].append(j)
            # Add cost of the acquired test to the patient's running total
            total_cost[global_i] += per_feature_cost[j]

    # -----------------------------------------------------------------------
    # Finalize any patient who exhausted max_acquisitions without reaching tau_DSS
    # -----------------------------------------------------------------------
    still_active = np.where(active)[0]
    if len(still_active) > 0:
        Xo, mm = X_obs[still_active], m[still_active]
        p_t = models.classifier_predict(Xo, mm)
        y_hat_t = predict_label(p_t)
        for local_i, global_i in enumerate(still_active):
            final_y_hat[global_i] = y_hat_t[local_i]
            final_p[global_i] = p_t[local_i]

    # Package each patient's summary into a PatientOutcome object
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
    Orchestrates the entire system training process before running inference:
    1. Trains LightGBM classifier on partially observed data with masks.
    2. Trains CVAE to model the distribution of missing features given observed features.
    3. Constructs the Clinical Information Graph capturing mutual information relationships.
    """
    # -----------------------------------------------------------------------
    # Step 1: Train the LightGBM Classifier (f_LGBM)
    # -----------------------------------------------------------------------
    print("[1/3] Training LightGBM classifier (f_LGBM)...")
    lightgbm = LightGBMClassifierWrapper(cfg.lightgbm).fit(
        prepared.X_train * prepared.m_train, prepared.m_train, prepared.y_train,
        prepared.X_val * prepared.m_val, prepared.m_val, prepared.y_val,
    )

    # -----------------------------------------------------------------------
    # Step 2: Train the Generative CVAE Model
    # -----------------------------------------------------------------------
    print("[2/3] Training CVAE generative profile completion model...")
    # CVAE learns P(x_missing | x_observed) to hallucinate plausible missing measurements
    cvae = CVAETrainer(prepared.X_train.shape[1], cfg.cvae).fit(
        prepared.X_train, prepared.m_train
    )

    # -----------------------------------------------------------------------
    # Step 3: Construct the Clinical Information Graph G = (V, E, W)
    # -----------------------------------------------------------------------
    print("[3/3] Constructing Clinical Information Graph...")
    # Nodes V = 21 features; Edges E weighted by mutual information and clinical association
    X_graph = getattr(prepared, "X_train_orig", prepared.X_train)
    y_graph = getattr(prepared, "y_train_orig", prepared.y_train)
    graph = build_clinical_graph(X_graph, y_graph)
    
    # Return the assembled trained system bundle
    return TrainedSystem(
        lightgbm=lightgbm,
        cvae=cvae,
        graph=graph,
        fusion_weights=FusionWeights(w_xgb=1.0, w_tr=0.0),
    )

