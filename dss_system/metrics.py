"""
metrics.py
==========
Comprehensive evaluation metrics and reporting suite for the
Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition System (DSS/DSG).

Computes and reports values across five core dimensions:

1. Prediction Performance:
   - Balanced Accuracy
   - Macro-F1
   - Precision
   - Recall / Sensitivity
   - Specificity
   - ROC-AUC
   - PR-AUC
   - Brier Score

2. Adaptive Acquisition Dynamics:
   - Final DSS
   - \\Delta DSS (Sufficiency Gain)
   - Mean / Median Number of Acquired Variables
   - Acquisition Ratio (w.r.t total features & initially missing features)
   - Total Acquisition Cost (mean, median, cohort total)
   - Early Stop Rate (fraction achieving \\tau_DSS before budget limit)

3. Decision Efficiency:
   - DSS Gain per Acquisition
   - DSS Gain per Unit Cost
   - Accuracy vs. Cost / Acquisition Pareto Performance

4. Counterfactual Stability:
   - Class Stability (CS)
   - Mean JS Divergence (PD / D_JS)

5. Robustness & Ablation Studies:
   - Performance across different missing-information rates
   - Performance across different \\tau_DSS thresholds
   - Ablation of scoring weights (\\lambda_1, \\lambda_2, \\lambda_3, \\lambda_4)
"""

from __future__ import annotations

import os
import sys
import json
import time
import copy
import pickle
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

# Ensure dss_system directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from config import CFG, FEATURE_COLUMNS, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, PipelineConfig
from data_processing import prepare_dataset, sample_observation_masks, apply_mask, PreparedData
from fusion import FusionWeights, fuse, predict_label
from stability import evaluate_stability_and_dss, class_stability, probability_stability, decision_sufficiency_score
from graph import ClinicalGraph, build_clinical_graph
from acquisition import normalized_costs, raw_feature_cost, DEFAULT_RAW_COSTS
from pipeline import (
    TrainedSystem,
    PatientOutcome,
    build_model_bundle,
    run_adaptive_acquisition,
    train_full_system,
)
from models.xgboost_model import XGBClassifier
from models.tabresnet import TabResNetClassifier
from models.cvae import CVAETrainer

DEFAULT_CHECKPOINT_PATH = os.path.join(_current_dir, "dss_checkpoint.pt")
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(_current_dir), "evaluation_plots")


# ===========================================================================
# 1. PREDICTION METRICS
# ===========================================================================
def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Computes all standard and clinical diagnostic prediction metrics:
    - Balanced Accuracy = (Sensitivity + Specificity) / 2
    - Macro-F1
    - Precision (PPV)
    - Recall / Sensitivity (TPR)
    - Specificity (TNR)
    - ROC-AUC
    - PR-AUC
    - Brier Score
    - Accuracy (overall correct fraction)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Diagnostic sensitivities and specificities
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    balanced_acc = float((sensitivity + specificity) / 2.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = float(auc(fpr, tpr)) if len(fpr) > 1 else 0.5

    # PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(auc(r_curve, p_curve)) if len(r_curve) > 1 else 0.5

    # Brier Score (Mean Squared Error on probabilities)
    brier = float(brier_score_loss(y_true, y_proba))

    return {
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "accuracy": accuracy,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ===========================================================================
# 2. ADAPTIVE ACQUISITION METRICS
# ===========================================================================
def compute_adaptive_acquisition_metrics(
    initial_dss: np.ndarray,
    outcomes: List[PatientOutcome],
    initial_masks: np.ndarray,
    n_features: int = len(FEATURE_COLUMNS),
    max_acquisitions: int = CFG.dss.max_acquisitions,
    tau_dss: float = CFG.dss.tau_dss,
) -> Dict[str, Any]:
    """
    Computes acquisition dynamics metrics:
    - Final DSS (mean, std, median)
    - \\Delta DSS (mean, std, median, min, max)
    - Mean and Median Number of Acquired Variables
    - Acquisition Ratio:
        * w.r.t total available feature pool F
        * w.r.t initially unobserved/missing features
    - Total Acquisition Cost (mean, median, cohort sum)
    - Early Stop Rate: fraction stopping with DSS >= tau_DSS prior to max_acquisitions
    """
    final_dss = np.array([o.dss for o in outcomes], dtype=float)
    delta_dss = final_dss - initial_dss
    n_acqs = np.array([o.n_acquisitions for o in outcomes], dtype=int)
    costs = np.array([o.total_cost for o in outcomes], dtype=float)

    # Initial unobserved feature counts
    init_missing = n_features - initial_masks.sum(axis=1)

    # Acquisition ratios
    acq_ratio_total = float(np.mean(n_acqs) / n_features) if n_features > 0 else 0.0
    safe_missing = np.maximum(init_missing, 1)
    acq_ratio_missing = float(np.mean(n_acqs / safe_missing))

    # Early stop rate: patients who satisfied DSS >= tau_DSS strictly before reaching max acquisitions
    early_stopped = (final_dss >= tau_dss) & (n_acqs < max_acquisitions)
    early_stop_rate = float(np.mean(early_stopped))
    
    # Reached sufficiency rate (any step)
    sufficiency_rate = float(np.mean(final_dss >= tau_dss))

    return {
        "final_dss_mean": float(np.mean(final_dss)),
        "final_dss_std": float(np.std(final_dss)),
        "final_dss_median": float(np.median(final_dss)),
        "delta_dss_mean": float(np.mean(delta_dss)),
        "delta_dss_std": float(np.std(delta_dss)),
        "delta_dss_median": float(np.median(delta_dss)),
        "delta_dss_min": float(np.min(delta_dss)),
        "delta_dss_max": float(np.max(delta_dss)),
        "mean_acquisitions": float(np.mean(n_acqs)),
        "median_acquisitions": float(np.median(n_acqs)),
        "min_acquisitions": int(np.min(n_acqs)),
        "max_acquisitions_observed": int(np.max(n_acqs)),
        "acquisition_ratio_total": acq_ratio_total,
        "acquisition_ratio_missing": acq_ratio_missing,
        "mean_total_cost": float(np.mean(costs)),
        "median_total_cost": float(np.median(costs)),
        "std_total_cost": float(np.std(costs)),
        "cohort_total_cost": float(np.sum(costs)),
        "early_stop_rate": early_stop_rate,
        "sufficiency_rate": sufficiency_rate,
    }


# ===========================================================================
# 3. DECISION EFFICIENCY METRICS & PARETO PERFORMANCE
# ===========================================================================
def compute_decision_efficiency_metrics(
    initial_dss: np.ndarray,
    outcomes: List[PatientOutcome],
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes decision efficiency metrics:
    - DSS Gain per Acquisition:
        * Patient-level average: mean(Delta DSS / n_acquisitions)
        * Cohort-level ratio: mean(Delta DSS) / mean(n_acquisitions)
    - DSS Gain per Unit Cost:
        * Patient-level average: mean(Delta DSS / Cost)
        * Cohort-level ratio: mean(Delta DSS) / mean(Cost)
    - Pareto trade-off performance:
        * Accuracy vs. Cost
        * Accuracy vs. Acquisitions
        * Pareto-optimal operating points
    """
    final_dss = np.array([o.dss for o in outcomes], dtype=float)
    delta_dss = final_dss - initial_dss
    n_acqs = np.array([o.n_acquisitions for o in outcomes], dtype=int)
    costs = np.array([o.total_cost for o in outcomes], dtype=float)

    # DSS gain per acquisition
    valid_acq = n_acqs > 0
    if valid_acq.any():
        gain_per_acq_patient = float(np.mean(delta_dss[valid_acq] / n_acqs[valid_acq]))
    else:
        gain_per_acq_patient = 0.0
    gain_per_acq_cohort = float(np.mean(delta_dss) / max(np.mean(n_acqs), 1e-8))

    # DSS gain per unit cost
    valid_cost = costs > 0
    if valid_cost.any():
        gain_per_cost_patient = float(np.mean(delta_dss[valid_cost] / costs[valid_cost]))
    else:
        gain_per_cost_patient = 0.0
    gain_per_cost_cohort = float(np.mean(delta_dss) / max(np.mean(costs), 1e-8))

    return {
        "dss_gain_per_acquisition_patient_mean": gain_per_acq_patient,
        "dss_gain_per_acquisition_cohort_ratio": gain_per_acq_cohort,
        "dss_gain_per_unit_cost_patient_mean": gain_per_cost_patient,
        "dss_gain_per_unit_cost_cohort_ratio": gain_per_cost_cohort,
    }


def compute_pareto_performance(
    pareto_points: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Calculates the Pareto frontier and trade-off efficiency between
    cost/acquisitions and diagnostic accuracy.

    A point (Cost_i, Acc_i) dominates (Cost_j, Acc_j) if:
        Cost_i <= Cost_j AND Acc_i >= Acc_j (with at least one strict inequality).
    The Pareto frontier comprises all non-dominated configurations.
    """
    if not pareto_points:
        return {"pareto_frontier": [], "au_bac": 0.0}

    # Sort points by cost ascending
    sorted_points = sorted(pareto_points, key=lambda p: (p.get("cost", 0.0), -p.get("accuracy", 0.0)))

    # Find non-dominated points
    frontier = []
    max_acc_so_far = -1.0
    for pt in sorted_points:
        acc = pt.get("accuracy", 0.0)
        if acc > max_acc_so_far:
            frontier.append(pt)
            max_acc_so_far = acc

    # Area Under Budget-Accuracy Curve (AU-BAC) via trapezoidal integration
    au_bac = 0.0
    if len(frontier) >= 2:
        costs = [p.get("cost", 0.0) for p in frontier]
        accs = [p.get("accuracy", 0.0) for p in frontier]
        cost_range = max(costs) - min(costs)
        if cost_range > 1e-6:
            # Normalized trapezoidal area
            raw_area = np.trapz(accs, costs)
            au_bac = float(raw_area / cost_range)
        else:
            au_bac = float(accs[0])
    elif len(frontier) == 1:
        au_bac = float(frontier[0].get("accuracy", 0.0))

    return {
        "pareto_frontier": frontier,
        "au_bac": au_bac,
    }


# ===========================================================================
# 4. COUNTERFACTUAL STABILITY METRICS
# ===========================================================================
def compute_stability_metrics(
    system: TrainedSystem,
    X_obs: np.ndarray,
    m: np.ndarray,
    n_samples_M: int = CFG.cvae.n_samples_M,
    alpha: float = CFG.dss.alpha,
) -> Dict[str, float]:
    """
    Assesses counterfactual stability using generative CVAE completions (Steps 7-14):
    - Class Stability (CS_t): fraction of plausible completions where diagnosis matches y_hat
    - Predictive Divergence (PD_t): Jensen-Shannon divergence D_JS between p_t and completions
    - Mean JS Divergence: raw JS divergence in nats / bits
    - Normalized Predictive Divergence: PD~_t = PD_t / ln(2)
    - Resulting Decision Sufficiency Score (DSS_t)
    """
    batch = X_obs.shape[0]

    # Predict observed state
    p_xgb = system.xgb.predict_proba_y1(X_obs, m)
    p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
    p_t = fuse(p_xgb, p_tr, system.fusion_weights)
    y_hat_t = predict_label(p_t)

    # Sample M plausible profiles via CVAE
    plausible = system.cvae.generate_plausible_profiles(X_obs, m, n_samples=n_samples_M)
    M = plausible.shape[0]

    full_mask = np.ones_like(m)
    p_plaus = np.zeros((M, batch), dtype=float)
    y_plaus = np.zeros((M, batch), dtype=int)

    for i in range(M):
        profile = plausible[i]
        px = system.xgb.predict_proba_y1(profile, full_mask)
        pt = system.tabresnet.predict_proba_y1(profile, full_mask)
        p_plaus[i] = fuse(px, pt, system.fusion_weights)
        y_plaus[i] = predict_label(p_plaus[i])

    # Steps 12-14: Class stability, probability stability, and DSS
    cs_t, pd_t_norm, dss_t = evaluate_stability_and_dss(
        p_t, y_hat_t, p_plaus, y_plaus, alpha=alpha
    )

    # Raw JS Divergence (PD_t before dividing by ln(2))
    raw_js_div = pd_t_norm * np.log(2.0)

    return {
        "class_stability_mean": float(np.mean(cs_t)),
        "class_stability_median": float(np.median(cs_t)),
        "class_stability_std": float(np.std(cs_t)),
        "mean_js_divergence_nats": float(np.mean(raw_js_div)),
        "mean_js_divergence_bits": float(np.mean(raw_js_div / np.log(2.0))),
        "predictive_divergence_norm_mean": float(np.mean(pd_t_norm)),
        "probability_stability_mean": float(np.mean(1.0 - pd_t_norm)),
        "mean_dss": float(np.mean(dss_t)),
        "cs_array": cs_t,
        "pd_norm_array": pd_t_norm,
        "dss_array": dss_t,
    }


# ===========================================================================
# 5. ROBUSTNESS & ABLATION EXPERIMENTS
# ===========================================================================
def evaluate_robustness_missingness(
    system: TrainedSystem,
    X_test: np.ndarray,
    y_test: np.ndarray,
    missing_rates: List[float] = [0.20, 0.40, 0.60, 0.80],
    n_patients: int = 50,
    cfg: PipelineConfig = CFG,
) -> List[Dict[str, Any]]:
    """
    Evaluates system robustness across varying missing-information rates:
    Simulates patients with missing rate mu in {20%, 40%, 60%, 80%}
    (i.e. observed fraction = 1 - mu).
    """
    n = min(n_patients, len(X_test))
    X_sub = X_test[:n]
    y_sub = y_test[:n]
    F = X_test.shape[1]

    results = []
    for miss_rate in missing_rates:
        obs_rate = max(0.05, min(0.95, 1.0 - miss_rate))
        
        # Build masks with exactly / approximately this observation fraction
        rng = np.random.default_rng(42)
        k_obs = max(1, min(F - 1, int(round(obs_rate * F))))
        m = np.zeros((n, F), dtype=int)
        for i in range(n):
            chosen = rng.choice(F, size=k_obs, replace=False)
            m[i, chosen] = 1

        X_obs = apply_mask(X_sub, m)

        # Baseline masked prediction
        p_xgb = system.xgb.predict_proba_y1(X_obs, m)
        p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
        p_init = fuse(p_xgb, p_tr, system.fusion_weights)
        y_init = predict_label(p_init)
        
        # Stability and initial DSS
        stab_init = compute_stability_metrics(system, X_obs, m, n_samples_M=cfg.cvae.n_samples_M)
        dss_init = stab_init["dss_array"]

        # Adaptive acquisition run
        outcomes = run_adaptive_acquisition(X_sub, X_obs, m, system, cfg=cfg)
        y_final = np.array([o.y_hat for o in outcomes])
        p_final = np.array([o.p_t for o in outcomes])

        pred_metrics = compute_prediction_metrics(y_sub, y_final, p_final)
        acq_metrics = compute_adaptive_acquisition_metrics(
            dss_init, outcomes, m, n_features=F, max_acquisitions=cfg.dss.max_acquisitions, tau_dss=cfg.dss.tau_dss
        )
        eff_metrics = compute_decision_efficiency_metrics(dss_init, outcomes, y_sub)

        row = {
            "missing_rate": miss_rate,
            "observed_rate": obs_rate,
            "balanced_accuracy": pred_metrics["balanced_accuracy"],
            "macro_f1": pred_metrics["macro_f1"],
            "roc_auc": pred_metrics["roc_auc"],
            "brier_score": pred_metrics["brier_score"],
            "final_dss": acq_metrics["final_dss_mean"],
            "delta_dss": acq_metrics["delta_dss_mean"],
            "mean_acquisitions": acq_metrics["mean_acquisitions"],
            "mean_cost": acq_metrics["mean_total_cost"],
            "early_stop_rate": acq_metrics["early_stop_rate"],
            "dss_gain_per_cost": eff_metrics["dss_gain_per_unit_cost_cohort_ratio"],
        }
        results.append(row)

    return results


def evaluate_robustness_tau(
    system: TrainedSystem,
    X_test: np.ndarray,
    y_test: np.ndarray,
    m_init: np.ndarray,
    tau_values: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    n_patients: int = 50,
    cfg: PipelineConfig = CFG,
) -> List[Dict[str, Any]]:
    """
    Evaluates performance sensitivity across different decision sufficiency
    stopping thresholds tau_DSS in [0.70, 0.95].
    """
    n = min(n_patients, len(X_test))
    X_sub = X_test[:n]
    y_sub = y_test[:n]
    m_sub = m_init[:n]
    F = X_test.shape[1]

    X_obs_init = apply_mask(X_sub, m_sub)
    stab_init = compute_stability_metrics(system, X_obs_init, m_sub, n_samples_M=cfg.cvae.n_samples_M)
    dss_init = stab_init["dss_array"]

    results = []
    for tau in tau_values:
        tau_cfg = copy.deepcopy(cfg)
        tau_cfg.dss.tau_dss = float(tau)

        outcomes = run_adaptive_acquisition(X_sub, X_obs_init, m_sub, system, cfg=tau_cfg)
        y_final = np.array([o.y_hat for o in outcomes])
        p_final = np.array([o.p_t for o in outcomes])

        pred_metrics = compute_prediction_metrics(y_sub, y_final, p_final)
        acq_metrics = compute_adaptive_acquisition_metrics(
            dss_init, outcomes, m_sub, n_features=F, max_acquisitions=tau_cfg.dss.max_acquisitions, tau_dss=tau
        )
        eff_metrics = compute_decision_efficiency_metrics(dss_init, outcomes, y_sub)

        row = {
            "tau_dss": tau,
            "final_dss": acq_metrics["final_dss_mean"],
            "delta_dss": acq_metrics["delta_dss_mean"],
            "early_stop_rate": acq_metrics["early_stop_rate"],
            "mean_acquisitions": acq_metrics["mean_acquisitions"],
            "mean_cost": acq_metrics["mean_total_cost"],
            "balanced_accuracy": pred_metrics["balanced_accuracy"],
            "macro_f1": pred_metrics["macro_f1"],
            "roc_auc": pred_metrics["roc_auc"],
            "dss_gain_per_cost": eff_metrics["dss_gain_per_unit_cost_cohort_ratio"],
        }
        results.append(row)

    return results


def evaluate_ablation_lambdas(
    system: TrainedSystem,
    X_test: np.ndarray,
    y_test: np.ndarray,
    m_init: np.ndarray,
    n_patients: int = 50,
    cfg: PipelineConfig = CFG,
) -> List[Dict[str, Any]]:
    """
    Ablation study on next-feature acquisition scoring weights:
    S_j = lambda_1 * R_j + lambda_2 * Conn_j + lambda_3 * DSG_j - lambda_4 * C_j

    Configurations tested:
    1. Full System:               (0.35, 0.20, 0.35, 0.10)
    2. No Relevance (l1=0):       (0.00, 0.20, 0.35, 0.10)
    3. No Connectivity (l2=0):    (0.35, 0.00, 0.35, 0.10)
    4. No DSG (l3=0):             (0.35, 0.20, 0.00, 0.10)
    5. Cost-blind (l4=0):         (0.35, 0.20, 0.35, 0.00)
    6. DSG Only:                  (0.00, 0.00, 1.00, 0.00)
    7. Relevance Only:            (1.00, 0.00, 0.00, 0.00)
    8. Random Selection Baseline
    """
    n = min(n_patients, len(X_test))
    X_sub = X_test[:n]
    y_sub = y_test[:n]
    m_sub = m_init[:n]
    F = X_test.shape[1]

    X_obs_init = apply_mask(X_sub, m_sub)
    stab_init = compute_stability_metrics(system, X_obs_init, m_sub, n_samples_M=cfg.cvae.n_samples_M)
    dss_init = stab_init["dss_array"]

    variants = [
        ("Full Model (Default)", (0.35, 0.20, 0.35, 0.10)),
        ("Ablate Relevance (lambda_1=0)", (0.00, 0.20, 0.35, 0.10)),
        ("Ablate Connectivity (lambda_2=0)", (0.35, 0.00, 0.35, 0.10)),
        ("Ablate DSG (lambda_3=0)", (0.35, 0.20, 0.00, 0.10)),
        ("Ablate Cost (lambda_4=0)", (0.35, 0.20, 0.35, 0.00)),
        ("DSG Only (Greedy Info-Gain)", (0.00, 0.00, 1.00, 0.00)),
        ("Relevance Only (Static Correlation)", (1.00, 0.00, 0.00, 0.00)),
    ]

    results = []
    for name, (l1, l2, l3, l4) in variants:
        v_cfg = copy.deepcopy(cfg)
        v_cfg.acquisition.lambda_1 = l1
        v_cfg.acquisition.lambda_2 = l2
        v_cfg.acquisition.lambda_3 = l3
        v_cfg.acquisition.lambda_4 = l4

        outcomes = run_adaptive_acquisition(X_sub, X_obs_init, m_sub, system, cfg=v_cfg)
        y_final = np.array([o.y_hat for o in outcomes])
        p_final = np.array([o.p_t for o in outcomes])

        pred_metrics = compute_prediction_metrics(y_sub, y_final, p_final)
        acq_metrics = compute_adaptive_acquisition_metrics(
            dss_init, outcomes, m_sub, n_features=F, max_acquisitions=v_cfg.dss.max_acquisitions, tau_dss=v_cfg.dss.tau_dss
        )
        eff_metrics = compute_decision_efficiency_metrics(dss_init, outcomes, y_sub)

        row = {
            "variant": name,
            "lambda_weights": f"({l1:.2f}, {l2:.2f}, {l3:.2f}, {l4:.2f})",
            "balanced_accuracy": pred_metrics["balanced_accuracy"],
            "macro_f1": pred_metrics["macro_f1"],
            "final_dss": acq_metrics["final_dss_mean"],
            "delta_dss": acq_metrics["delta_dss_mean"],
            "mean_acquisitions": acq_metrics["mean_acquisitions"],
            "mean_cost": acq_metrics["mean_total_cost"],
            "early_stop_rate": acq_metrics["early_stop_rate"],
            "dss_gain_per_cost": eff_metrics["dss_gain_per_unit_cost_cohort_ratio"],
        }
        results.append(row)

    # Random selection baseline simulation
    random_cfg = copy.deepcopy(cfg)
    # With all lambdas=0, random noise drives argmax selection
    random_cfg.acquisition.lambda_1 = 0.0
    random_cfg.acquisition.lambda_2 = 0.0
    random_cfg.acquisition.lambda_3 = 0.0
    random_cfg.acquisition.lambda_4 = 0.0
    # Add minor perturbation in acquisition loop or test
    outcomes_rand = run_adaptive_acquisition(X_sub, X_obs_init, m_sub, system, cfg=random_cfg)
    y_final_r = np.array([o.y_hat for o in outcomes_rand])
    p_final_r = np.array([o.p_t for o in outcomes_rand])
    pred_r = compute_prediction_metrics(y_sub, y_final_r, p_final_r)
    acq_r = compute_adaptive_acquisition_metrics(dss_init, outcomes_rand, m_sub, n_features=F)
    eff_r = compute_decision_efficiency_metrics(dss_init, outcomes_rand, y_sub)
    results.append({
        "variant": "Random Selection Baseline",
        "lambda_weights": "(0.00, 0.00, 0.00, 0.00)",
        "balanced_accuracy": pred_r["balanced_accuracy"],
        "macro_f1": pred_r["macro_f1"],
        "final_dss": acq_r["final_dss_mean"],
        "delta_dss": acq_r["delta_dss_mean"],
        "mean_acquisitions": acq_r["mean_acquisitions"],
        "mean_cost": acq_r["mean_total_cost"],
        "early_stop_rate": acq_r["early_stop_rate"],
        "dss_gain_per_cost": eff_r["dss_gain_per_unit_cost_cohort_ratio"],
    })

    return results


# ===========================================================================
# 6. MODEL CHECKPOINTING (SAVE & LOAD)
# ===========================================================================
def save_trained_system(system: TrainedSystem, path: str = DEFAULT_CHECKPOINT_PATH) -> None:
    """Serializes the TrainedSystem bundle to disk for instant subsequent evaluation."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "xgb": system.xgb,
        "tabresnet_state": system.tabresnet.net.state_dict(),
        "tabresnet_cfg": system.tabresnet.cfg,
        "tabresnet_dim": system.tabresnet.net.stem.in_features,
        "cvae_state": system.cvae.model.state_dict(),
        "cvae_cfg": system.cvae.cfg,
        "cvae_dim": system.cvae.model.feature_dim,
        "fusion_weights": (system.fusion_weights.w_xgb, system.fusion_weights.w_tr),
        "graph": system.graph,
        "feature_names": system.feature_names,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"System checkpoint successfully saved to: {path}")


def load_trained_system(path: str = DEFAULT_CHECKPOINT_PATH) -> Optional[TrainedSystem]:
    """Loads a previously fitted TrainedSystem from disk if present."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)

        tabresnet = TabResNetClassifier(payload["tabresnet_dim"], payload["tabresnet_cfg"])
        tabresnet.net.load_state_dict(payload["tabresnet_state"])

        cvae = CVAETrainer(payload["cvae_dim"], payload["cvae_cfg"])
        cvae.model.load_state_dict(payload["cvae_state"])

        w_xgb, w_tr = payload["fusion_weights"]
        fusion_weights = FusionWeights(w_xgb=w_xgb, w_tr=w_tr)

        system = TrainedSystem(
            xgb=payload["xgb"],
            tabresnet=tabresnet,
            cvae=cvae,
            fusion_weights=fusion_weights,
            graph=payload["graph"],
            feature_names=payload["feature_names"],
        )
        print(f"Loaded trained system from checkpoint: {path}")
        return system
    except Exception as e:
        print(f"Could not load checkpoint ({e}). Will train fresh system.")
        return None


# ===========================================================================
# 7. FORMATTED REPORT PRINTER & ARTIFACT EXPORTER
# ===========================================================================
def print_full_metrics_report(results: Dict[str, Any]) -> str:
    """
    Constructs and prints an academic-grade formatted markdown report
    covering all five requested dimensions.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(" DECISION-SUFFICIENCY-GUIDED ADAPTIVE CLINICAL ACQUISITION SYSTEM")
    lines.append(" COMPREHENSIVE PERFORMANCE & DIAGNOSTIC METRICS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Section 1: Prediction Performance
    lines.append("+" + "-" * 78 + "+")
    lines.append("| 1. PREDICTION PERFORMANCE METRICS                                            |")
    lines.append("+" + "-" * 78 + "+")
    lines.append("| Metric                     | Initial (Partially Observed) | Post-Adaptive Acquisition |")
    lines.append("+----------------------------+------------------------------+---------------------------+")
    
    p_init = results.get("prediction_initial", {})
    p_final = results.get("prediction_final", {})

    metrics_rows = [
        ("Balanced Accuracy", "balanced_accuracy", "{:.4f}"),
        ("Macro-F1", "macro_f1", "{:.4f}"),
        ("Precision (PPV)", "precision", "{:.4f}"),
        ("Recall / Sensitivity", "recall_sensitivity", "{:.4f}"),
        ("Specificity (TNR)", "specificity", "{:.4f}"),
        ("ROC-AUC", "roc_auc", "{:.4f}"),
        ("PR-AUC", "pr_auc", "{:.4f}"),
        ("Brier Score (MSE)", "brier_score", "{:.4f}"),
        ("Raw Accuracy", "accuracy", "{:.4f}"),
    ]

    for label, key, fmt in metrics_rows:
        val_i = fmt.format(p_init.get(key, 0.0)) if key in p_init else "N/A"
        val_f = fmt.format(p_final.get(key, 0.0)) if key in p_final else "N/A"
        lines.append(f"| {label:<26} | {val_i:<28} | {val_f:<25} |")
    lines.append("+----------------------------+------------------------------+---------------------------+")
    lines.append("")

    # Section 2: Adaptive Acquisition Dynamics
    lines.append("+" + "-" * 78 + "+")
    lines.append("| 2. ADAPTIVE ACQUISITION DYNAMICS                                             |")
    lines.append("+" + "-" * 78 + "+")
    acq = results.get("adaptive_acquisition", {})
    lines.append(f"| * Final DSS (Mean +/- Std):               {acq.get('final_dss_mean', 0.0):.4f} +/- {acq.get('final_dss_std', 0.0):.4f} (Median: {acq.get('final_dss_median', 0.0):.4f}){' ':>11}|")
    lines.append(f"| * Delta DSS Gain (Mean +/- Std):          {acq.get('delta_dss_mean', 0.0):.4f} +/- {acq.get('delta_dss_std', 0.0):.4f} (Median: {acq.get('delta_dss_median', 0.0):.4f}){' ':>11}|")
    lines.append(f"| * Delta DSS Range:                        [{acq.get('delta_dss_min', 0.0):.4f}, {acq.get('delta_dss_max', 0.0):.4f}]{' ':>42}|")
    lines.append(f"| * Acquired Variables (Mean / Median):     {acq.get('mean_acquisitions', 0.0):.2f} / {acq.get('median_acquisitions', 0.0):.1f} features{' ':>24}|")
    lines.append(f"| * Acquisition Ratio (Total Features F):   {acq.get('acquisition_ratio_total', 0.0)*100:.2f}% ({acq.get('mean_acquisitions', 0.0):.1f}/21 tests){' ':>23}|")
    lines.append(f"| * Acquisition Ratio (Missing Variables):  {acq.get('acquisition_ratio_missing', 0.0)*100:.2f}% of unobserved features{' ':>19}|")
    lines.append(f"| * Total Acquisition Cost per Patient:     Mean: {acq.get('mean_total_cost', 0.0):.3f} | Median: {acq.get('median_total_cost', 0.0):.3f}{' ':>16}|")
    lines.append(f"| * Cohort Total Cost:                      {acq.get('cohort_total_cost', 0.0):.2f} units{' ':>42}|")
    lines.append(f"| * Early Stop Rate (DSS >= tau_DSS):       {acq.get('early_stop_rate', 0.0)*100:.2f}%{' ':>47}|")
    lines.append("+" + "-" * 78 + "+")
    lines.append("")

    # Section 3: Decision Efficiency & Pareto Performance
    lines.append("+" + "-" * 78 + "+")
    lines.append("| 3. DECISION EFFICIENCY & PARETO PERFORMANCE                                  |")
    lines.append("+" + "-" * 78 + "+")
    eff = results.get("decision_efficiency", {})
    lines.append(f"| * DSS Gain per Acquisition (Patient Avg):  {eff.get('dss_gain_per_acquisition_patient_mean', 0.0):.4f} / test{' ':>24}|")
    lines.append(f"| * DSS Gain per Acquisition (Cohort Ratio): {eff.get('dss_gain_per_acquisition_cohort_ratio', 0.0):.4f} / test{' ':>24}|")
    lines.append(f"| * DSS Gain per Unit Cost (Patient Avg):    {eff.get('dss_gain_per_unit_cost_patient_mean', 0.0):.4f} / cost unit{' ':>19}|")
    lines.append(f"| * DSS Gain per Unit Cost (Cohort Ratio):   {eff.get('dss_gain_per_unit_cost_cohort_ratio', 0.0):.4f} / cost unit{' ':>19}|")
    
    pareto = results.get("pareto_performance", {})
    lines.append(f"| * Area Under Budget-Accuracy Curve (AU-BAC): {pareto.get('au_bac', 0.0):.4f}{' ':>27}|")
    lines.append("|                                                                              |")
    lines.append("| Pareto Tradeoff Frontier (Acquisitions/Cost vs. Diagnostic Accuracy):        |")
    lines.append("|   Step/Budget | Mean Acq | Mean Cost | Balanced Acc | Macro-F1 | Mean DSS    |")
    lines.append("|   ------------+----------+-----------+--------------+----------+---------    |")
    for pt in pareto.get("pareto_frontier", []):
        name = pt.get("label", "Operating Pt")
        acq_pt = pt.get("acquisitions", 0.0)
        c_pt = pt.get("cost", 0.0)
        b_acc = pt.get("balanced_accuracy", pt.get("accuracy", 0.0))
        f1_pt = pt.get("macro_f1", 0.0)
        dss_pt = pt.get("dss", 0.0)
        lines.append(f"|   {name:<11} | {acq_pt:>8.2f} | {c_pt:>9.3f} | {b_acc:>12.4f} | {f1_pt:>8.4f} | {dss_pt:>7.4f}     |")
    lines.append("+" + "-" * 78 + "+")
    lines.append("")

    # Section 4: Counterfactual Stability
    lines.append("+" + "-" * 78 + "+")
    lines.append("| 4. COUNTERFACTUAL STABILITY (Steps 12-14)                                    |")
    lines.append("+" + "-" * 78 + "+")
    stab_i = results.get("stability_initial", {})
    stab_f = results.get("stability_final", {})
    lines.append(f"| * Initial Class Stability (CS_0):       {stab_i.get('class_stability_mean', 0.0):.4f} +/- {stab_i.get('class_stability_std', 0.0):.4f}{' ':>22}|")
    lines.append(f"| * Final Class Stability (CS_final):     {stab_f.get('class_stability_mean', 0.0):.4f} +/- {stab_f.get('class_stability_std', 0.0):.4f}{' ':>22}|")
    lines.append(f"| * Mean JS Divergence Initial (nats):    {stab_i.get('mean_js_divergence_nats', 0.0):.4f} (bits: {stab_i.get('mean_js_divergence_bits', 0.0):.4f}){' ':>17}|")
    lines.append(f"| * Mean JS Divergence Final (nats):      {stab_f.get('mean_js_divergence_nats', 0.0):.4f} (bits: {stab_f.get('mean_js_divergence_bits', 0.0):.4f}){' ':>17}|")
    lines.append(f"| * Normalized Pred Divergence (PD~_0):   {stab_i.get('predictive_divergence_norm_mean', 0.0):.4f}{' ':>37}|")
    lines.append(f"| * Normalized Pred Divergence (PD~_fin): {stab_f.get('predictive_divergence_norm_mean', 0.0):.4f}{' ':>37}|")
    lines.append("+" + "-" * 78 + "+")
    lines.append("")

    # Section 5: Robustness Studies
    lines.append("+" + "-" * 78 + "+")
    lines.append("| 5. ROBUSTNESS & SENSITIVITY ANALYSES                                         |")
    lines.append("+" + "-" * 78 + "+")
    lines.append("| [5.1] Performance Across Different Missing-Information Rates:                |")
    lines.append("|   Missing Rate | Bal Acc | Macro-F1 | ROC-AUC | Final DSS | Mean Acq | Mean Cost |")
    lines.append("|   -------------+---------+----------+---------+-----------+----------+---------- |")
    for r in results.get("robustness_missingness", []):
        mr = f"{r.get('missing_rate', 0.0)*100:.0f}%"
        ba = r.get("balanced_accuracy", 0.0)
        f1 = r.get("macro_f1", 0.0)
        auc_v = r.get("roc_auc", 0.0)
        fd = r.get("final_dss", 0.0)
        ma = r.get("mean_acquisitions", 0.0)
        mc = r.get("mean_cost", 0.0)
        lines.append(f"|   {mr:<12} | {ba:>7.4f} | {f1:>8.4f} | {auc_v:>7.4f} | {fd:>9.4f} | {ma:>8.2f} | {mc:>9.3f} |")

    lines.append("|                                                                              |")
    lines.append("| [5.2] Performance Across Stopping Thresholds (tau_DSS):                      |")
    lines.append("|   tau_DSS | Final DSS | Early Stop | Mean Acq | Mean Cost | Bal Acc | Macro-F1   |")
    lines.append("|   --------+-----------+------------+----------+-----------+---------+----------  |")
    for r in results.get("robustness_tau", []):
        tau = f"{r.get('tau_dss', 0.0):.2f}"
        fd = r.get("final_dss", 0.0)
        es = f"{r.get('early_stop_rate', 0.0)*100:.1f}%"
        ma = r.get("mean_acquisitions", 0.0)
        mc = r.get("mean_cost", 0.0)
        ba = r.get("balanced_accuracy", 0.0)
        f1 = r.get("macro_f1", 0.0)
        lines.append(f"|   {tau:<7} | {fd:>9.4f} | {es:>10} | {ma:>8.2f} | {mc:>9.3f} | {ba:>7.4f} | {f1:>8.4f}   |")

    lines.append("|                                                                              |")
    lines.append("| [5.3] Ablation Study on Scoring Weights (lambda_1, lambda_2, lambda_3, lambda_4): |")
    lines.append("|   Variant                     | Weights             | Bal Acc | Final DSS | Cost |")
    lines.append("|   ----------------------------+---------------------+---------+-----------+------|")
    for r in results.get("ablation_lambdas", []):
        v_name = r.get("variant", "")[:26]
        weights = r.get("lambda_weights", "")[:19]
        ba = r.get("balanced_accuracy", 0.0)
        fd = r.get("final_dss", 0.0)
        mc = r.get("mean_cost", 0.0)
        lines.append(f"|   {v_name:<26} | {weights:<19} | {ba:>7.4f} | {fd:>9.4f} |{mc:>5.2f} |")
    lines.append("+" + "-" * 78 + "+")
    lines.append("=" * 80)

    report_str = "\n".join(lines)
    try:
        print(report_str)
    except UnicodeEncodeError:
        print(report_str.encode("ascii", errors="replace").decode("ascii"))
    return report_str


def save_report_artifacts(results: Dict[str, Any], output_dir: str = REPORT_OUTPUT_DIR) -> Tuple[str, str]:
    """Saves both formatted Markdown and machine-readable JSON reports."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics_report.json")
    md_path = os.path.join(output_dir, "metrics_report.md")

    # Serialize JSON (stripping numpy arrays)
    def clean_obj(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, dict):
            return {k: clean_obj(v) for k, v in o.items() if not k.endswith("_array")}
        if isinstance(o, list):
            return [clean_obj(v) for v in o]
        return o

    clean_dict = clean_obj(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)

    # Formatted Markdown report
    md_content = f"# DSS Clinical System: Comprehensive Metrics Report\n\n```text\n{print_full_metrics_report(results)}\n```\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved metrics artifacts:\n  - JSON: {json_path}\n  - Markdown: {md_path}")
    return json_path, md_path


# ===========================================================================
# 8. FULL END-TO-END EVALUATION SUITE
# ===========================================================================
def run_full_metrics_evaluation(
    data_path: Optional[str] = None,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    n_eval_patients: int = 50,
    quick: bool = False,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Executes the comprehensive evaluation pipeline:
    1. Loads or trains the clinical DSS system.
    2. Runs initial masked prediction evaluation.
    3. Runs counterfactual stability evaluation under CVAE.
    4. Simulates adaptive information acquisition loop.
    5. Measures decision efficiency and Pareto trade-off performance.
    6. Conducts robustness sweeps over missingness, tau_DSS, and lambda weights.
    7. Displays and saves all requested metrics.
    """
    print("\n[Step 1/6] Preparing CDC Diabetes Health Indicators Dataset...")
    prepared = prepare_dataset(path=data_path)
    print(f"Dataset ready. Train: {prepared.X_train.shape}, Test: {prepared.X_test.shape}")

    # Adjust config for quick verification if requested
    eval_cfg = copy.deepcopy(CFG)
    if quick:
        print("Running in fast-eval mode (optimized mini-batch iterations)...")
        eval_cfg.xgb.n_estimators = 80
        eval_cfg.tabresnet.epochs = 12
        eval_cfg.tabresnet.patience = 5
        eval_cfg.cvae.epochs = 10
        eval_cfg.cvae.n_samples_M = 15
        eval_cfg.acquisition.n_candidate_values_K = 3
    else:
        eval_cfg.cvae.n_samples_M = 25
        eval_cfg.acquisition.n_candidate_values_K = 3

    # Load or train system
    system = None
    if not force_retrain:
        system = load_trained_system(checkpoint_path)

    if system is None:
        print("[Step 2/6] Training full DSS system bundle (XGBoost, TabResNet, CVAE, Graph)...")
        system = train_full_system(prepared, cfg=eval_cfg)
        save_trained_system(system, checkpoint_path)
    else:
        print("[Step 2/6] Using loaded trained system bundle.")

    # Cohort extraction for adaptive acquisition
    n = min(n_eval_patients, len(prepared.X_test))
    X_test_eval = prepared.X_test[:n]
    y_test_eval = prepared.y_test[:n]
    m_test_eval = prepared.m_test[:n]
    X_obs_init = apply_mask(X_test_eval, m_test_eval)

    # -----------------------------------------------------------------------
    # Step 3: Initial Partially Observed Prediction & Stability
    # -----------------------------------------------------------------------
    print(f"\n[Step 3/6] Evaluating Initial (Partially Observed) state for {n} patients...")
    p_xgb_0 = system.xgb.predict_proba_y1(X_obs_init, m_test_eval)
    p_tr_0 = system.tabresnet.predict_proba_y1(X_obs_init, m_test_eval)
    p_fused_0 = fuse(p_xgb_0, p_tr_0, system.fusion_weights)
    y_hat_0 = predict_label(p_fused_0)

    pred_initial = compute_prediction_metrics(y_test_eval, y_hat_0, p_fused_0)
    stability_initial = compute_stability_metrics(
        system, X_obs_init, m_test_eval, n_samples_M=eval_cfg.cvae.n_samples_M, alpha=eval_cfg.dss.alpha
    )
    initial_dss = stability_initial["dss_array"]

    # -----------------------------------------------------------------------
    # Step 4: Adaptive Acquisition Run
    # -----------------------------------------------------------------------
    print(f"\n[Step 4/6] Executing Adaptive Information Acquisition Loop...")
    outcomes = run_adaptive_acquisition(
        X_test_eval, X_obs_init, m_test_eval, system, cfg=eval_cfg
    )
    y_hat_final = np.array([o.y_hat for o in outcomes])
    p_final = np.array([o.p_t for o in outcomes])

    # Final post-acquisition prediction & stability
    pred_final = compute_prediction_metrics(y_test_eval, y_hat_final, p_final)
    
    # Reconstruct final observation mask after acquisition
    m_final = m_test_eval.copy()
    X_final = X_obs_init.copy()
    for idx_p, o in enumerate(outcomes):
        for feat_name in o.acquired_features:
            j = system.feature_names.index(feat_name)
            m_final[idx_p, j] = 1
            X_final[idx_p, j] = X_test_eval[idx_p, j]

    stability_final = compute_stability_metrics(
        system, X_final, m_final, n_samples_M=eval_cfg.cvae.n_samples_M, alpha=eval_cfg.dss.alpha
    )

    # Acquisition & Decision Efficiency
    acq_metrics = compute_adaptive_acquisition_metrics(
        initial_dss, outcomes, m_test_eval, n_features=len(FEATURE_COLUMNS),
        max_acquisitions=eval_cfg.dss.max_acquisitions, tau_dss=eval_cfg.dss.tau_dss
    )
    eff_metrics = compute_decision_efficiency_metrics(initial_dss, outcomes, y_test_eval)

    # -----------------------------------------------------------------------
    # Step 5: Multi-Budget Pareto Curve Construction
    # -----------------------------------------------------------------------
    print("[Step 5/6] Computing Accuracy vs. Cost/Acquisition Pareto Frontier...")
    pareto_pts = []
    # Point 0: Initial state (0 acquisitions, 0 cost)
    pareto_pts.append({
        "label": "Budget 0 (Init)",
        "acquisitions": 0.0,
        "cost": 0.0,
        "balanced_accuracy": pred_initial["balanced_accuracy"],
        "accuracy": pred_initial["accuracy"],
        "macro_f1": pred_initial["macro_f1"],
        "dss": float(np.mean(initial_dss)),
    })

    # Test acquisition budget checkpoints: B in {1, 3, 5, 8, 15}
    budgets = [1, 3, 5, 8, eval_cfg.dss.max_acquisitions]
    for b in budgets:
        b_cfg = copy.deepcopy(eval_cfg)
        b_cfg.dss.max_acquisitions = b
        b_out = run_adaptive_acquisition(X_test_eval, X_obs_init, m_test_eval, system, cfg=b_cfg)
        b_y = np.array([o.y_hat for o in b_out])
        b_p = np.array([o.p_t for o in b_out])
        b_pred = compute_prediction_metrics(y_test_eval, b_y, b_p)
        b_acq = compute_adaptive_acquisition_metrics(initial_dss, b_out, m_test_eval, len(FEATURE_COLUMNS))
        pareto_pts.append({
            "label": f"Budget max={b}",
            "acquisitions": b_acq["mean_acquisitions"],
            "cost": b_acq["mean_total_cost"],
            "balanced_accuracy": b_pred["balanced_accuracy"],
            "accuracy": b_pred["accuracy"],
            "macro_f1": b_pred["macro_f1"],
            "dss": b_acq["final_dss_mean"],
        })

    pareto_perf = compute_pareto_performance(pareto_pts)

    # -----------------------------------------------------------------------
    # Step 6: Robustness & Ablation Sweeps
    # -----------------------------------------------------------------------
    print("[Step 6/6] Conducting Robustness Sweeps and Weight Ablations...")
    n_robust = min(30 if quick else 50, n)
    
    rob_missing = evaluate_robustness_missingness(
        system, X_test_eval, y_test_eval,
        missing_rates=[0.20, 0.40, 0.60, 0.80],
        n_patients=n_robust,
        cfg=eval_cfg,
    )

    rob_tau = evaluate_robustness_tau(
        system, X_test_eval, y_test_eval, m_test_eval,
        tau_values=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        n_patients=n_robust,
        cfg=eval_cfg,
    )

    ablations = evaluate_ablation_lambdas(
        system, X_test_eval, y_test_eval, m_test_eval,
        n_patients=n_robust,
        cfg=eval_cfg,
    )

    # Assemble comprehensive results dictionary
    full_results = {
        "prediction_initial": pred_initial,
        "prediction_final": pred_final,
        "adaptive_acquisition": acq_metrics,
        "decision_efficiency": eff_metrics,
        "pareto_performance": pareto_perf,
        "stability_initial": stability_initial,
        "stability_final": stability_final,
        "robustness_missingness": rob_missing,
        "robustness_tau": rob_tau,
        "ablation_lambdas": ablations,
    }

    # Print and export artifacts
    print_full_metrics_report(full_results)
    save_report_artifacts(full_results)

    return full_results


# ===========================================================================
# 9. CLI ENTRY POINT
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate and Report Comprehensive Clinical DSS Evaluation Metrics"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional local path to CDC Diabetes dataset CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to load/save trained system model bundle",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Instantly display the latest generated metrics report without re-running simulation",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=30,
        help="Number of test patients in evaluation cohort (default: 30)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Enable fast evaluation mode (optimized Monte Carlo samples, default: True)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full extensive Monte Carlo evaluation (slower, higher precision)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training even if a model checkpoint exists",
    )

    args = parser.parse_args()

    if args.show:
        json_path = os.path.join(REPORT_OUTPUT_DIR, "metrics_report.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print_full_metrics_report(data)
            return
        else:
            print("No existing report found. Running metrics evaluation now...")

    run_full_metrics_evaluation(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        n_eval_patients=args.n_patients,
        quick=not args.full,
        force_retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
