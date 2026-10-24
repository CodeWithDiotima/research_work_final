"""
evaluate.py
============
Diagnostic evaluation for the single LightGBM diagnostic classifier (f_LGBM)
and the Decision-Sufficiency-Guided (DSS/DSG) System.

IMPORTANT: this project evaluates the classifier under partial observability —
f_LGBM is called as f(x_obs, m), matching how it was trained.

Three checkpoints, in pipeline order:
  1. evaluate_masked_accuracy()     -> accuracy, balanced accuracy, precision, recall, F1, AUC
     on train/val/test, with features masked exactly as in training.
  2. evaluate_experiments_a_and_b() -> formal evaluation of:
     - Experiment A: LightGBM, all 21 features available (m = 1).
     - Experiment B: LightGBM, masked features + mask indicators (2 * F = 42).
  3. evaluate_system()              -> confusion matrix heatmap + ROC/PR
     curves + performance metrics bar chart on the masked TEST split.
  4. evaluate_plausible_stability() -> for each masked test patient,
     generates M CVAE plausible completions, re-diagnoses each one with
     LightGBM, and reports Class Stability (CS), Predictive Divergence (PD),
     and Decision Sufficiency Score (DSS).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from config import CFG
    from fusion import predict_label
    from stability import evaluate_stability_and_dss
    from extra_plots import plot_confusion_heatmap, plot_training_history
except ImportError:
    from dss_system.config import CFG
    from dss_system.fusion import predict_label
    from dss_system.stability import evaluate_stability_and_dss
    from dss_system.extra_plots import plot_confusion_heatmap, plot_training_history

OUT_DIR = "evaluation_plots"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _metrics_dict(y_true, y_pred, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(specificity),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc(fpr, tpr),
    }


def _splits(prepared):
    return {
        "Train": (prepared.X_train, prepared.y_train, prepared.m_train),
        "Val":   (prepared.X_val,   prepared.y_val,   prepared.m_val),
        "Test":  (prepared.X_test,  prepared.y_test,  prepared.m_test),
    }


# ---------------------------------------------------------------------------
# Checkpoint 1 — masked accuracy across train/val/test
# ---------------------------------------------------------------------------
def evaluate_masked_accuracy(system, prepared):
    """
    f_LGBM called as f(x_obs, m) — the SAME masked input used during training —
    on train/val/test. This evaluates diagnostic performance under partial observability.
    """
    th = getattr(system.lightgbm, "best_threshold_", 0.5)
    all_results = {}
    for split_name, (X, y, m) in _splits(prepared).items():
        X_obs = X * m
        p_lgb = system.lightgbm.predict_proba_y1(X_obs, m)
        y_pred = predict_label(p_lgb, threshold=th)
        all_results[("LightGBM", split_name)] = _metrics_dict(y, y_pred, p_lgb)

    print(f"\n{'Model':10s} {'Split':6s} {'Acc':>7s} {'BalAcc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    print("-" * 65)
    for (name, split_name), m in all_results.items():
        print(f"{name:10s} {split_name:6s} {m['accuracy']:7.4f} {m['balanced_accuracy']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['f1']:7.4f} {m['auc']:7.4f}")

    # Bar chart: accuracy by split
    split_names = ["Train", "Val", "Test"]
    x = np.arange(len(split_names))
    width = 0.5

    fig, ax = plt.subplots(figsize=(6, 4.5))
    values = [all_results[("LightGBM", s)]["balanced_accuracy"] for s in split_names]
    bars = ax.bar(x, values, width, label="LightGBM (BalAcc)", color="#0284c7")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"LightGBM Balanced Accuracy Across Splits (Threshold = {th:.2f})")
    ax.legend()
    fig.tight_layout()

    _ensure_out_dir()
    path = os.path.join(OUT_DIR, "train_val_test_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    return all_results


# ---------------------------------------------------------------------------
# Checkpoint 1b — Formal Evaluation: Experiment A vs Experiment B
# ---------------------------------------------------------------------------
def evaluate_experiments_a_and_b(system, prepared):
    """
    Evaluates the two formal experimental settings on the held-out test split:
      - Experiment A: LightGBM, all 21 features available (m = 1).
      - Experiment B: LightGBM, masked features + mask indicators (2 * F = 42).
    """
    th = getattr(system.lightgbm, "best_threshold_", 0.5)
    X_test, y_test, m_test = prepared.X_test, prepared.y_test, prepared.m_test
    
    # Experiment A: All 21 features available (m = 1)
    m_full = np.ones_like(m_test)
    p_full = system.lightgbm.predict_proba_y1(X_test, m_full)
    y_full = predict_label(p_full, threshold=th)
    metrics_a = _metrics_dict(y_test, y_full, p_full)

    # Experiment B: Masked features (20-60% observed) + mask indicators (2 * F = 42)
    X_obs = X_test * m_test
    p_masked = system.lightgbm.predict_proba_y1(X_obs, m_test)
    y_masked = predict_label(p_masked, threshold=th)
    metrics_b = _metrics_dict(y_test, y_masked, p_masked)

    print("\n" + "=" * 70)
    print("FORMAL EXPERIMENTAL EVALUATION ON HELD-OUT TEST SPLIT")
    print("=" * 70)
    print(f"{'Experiment':30s} {'Acc':>7s} {'BalAcc':>7s} {'Sens':>7s} {'Spec':>7s} {'F1':>7s} {'ROC-AUC':>8s}")
    print("-" * 70)
    print(f"{'Exp A: Full 21 Features (m=1)':30s} {metrics_a['accuracy']:7.4f} {metrics_a['balanced_accuracy']:7.4f} "
          f"{metrics_a['recall']:7.4f} {metrics_a['specificity']:7.4f} {metrics_a['f1']:7.4f} {metrics_a['auc']:8.4f}")
    print(f"{'Exp B: Masked (2*F=42 Inputs)':30s} {metrics_b['accuracy']:7.4f} {metrics_b['balanced_accuracy']:7.4f} "
          f"{metrics_b['recall']:7.4f} {metrics_b['specificity']:7.4f} {metrics_b['f1']:7.4f} {metrics_b['auc']:8.4f}")
    print("=" * 70)

    # Comparison bar chart
    _ensure_out_dir()
    labels = ["Acc", "BalAcc", "Sens", "Spec", "F1", "AUC"]
    vals_a = [metrics_a["accuracy"], metrics_a["balanced_accuracy"], metrics_a["recall"], metrics_a["specificity"], metrics_a["f1"], metrics_a["auc"]]
    vals_b = [metrics_b["accuracy"], metrics_b["balanced_accuracy"], metrics_b["recall"], metrics_b["specificity"], metrics_b["f1"], metrics_b["auc"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, vals_a, width, label="Exp A: Full Features (m=1)", color="#0284c7")
    ax.bar(x + width/2, vals_b, width, label="Exp B: Masked Features (2*F=42)", color="#0d9488")

    ax.set_ylabel("Score")
    ax.set_title(f"LightGBM Diagnostic Comparison: Full vs Masked Inputs (Threshold={th:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "experiment_a_vs_b_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    return {"exp_a": metrics_a, "exp_b": metrics_b}


# ---------------------------------------------------------------------------
# Checkpoint 2 — confusion matrices, ROC/PR curves on the masked TEST split
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title: str = "LightGBM", filename: str = "confusion_matrix_lightgbm.png",
                           class_names=("No Diabetes", "Diabetes")):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title} (masked input)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_out_dir()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return cm


def plot_roc_curve(y_true, y_proba, title: str = "LightGBM", filename: str = "roc_lightgbm.png", ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(fpr, tpr, label=f"{label or title} (AUC={roc_auc:.3f})", color="#0284c7", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")

    if standalone:
        fig.tight_layout()
        _ensure_out_dir()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")
    return roc_auc


def plot_pr_curve(y_true, y_proba, title: str = "LightGBM", filename: str = "pr_lightgbm.png", ax=None, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(recall, precision, label=f"{label or title} (AP={ap:.3f})", color="#f97316", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {title}")
    ax.legend(loc="lower left")

    if standalone:
        fig.tight_layout()
        _ensure_out_dir()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")
    return ap


def plot_metric_comparison(results: dict, filename: str = "model_comparison.png"):
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.6 / len(model_names)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name, color="#0284c7")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m[:7].upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("LightGBM Performance Metrics (Masked Test Split)")
    ax.legend()
    fig.tight_layout()

    _ensure_out_dir()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(history: dict, title_prefix: str = "LightGBM"):
    """
    Loss-vs-round and accuracy-vs-round (train vs val) for LightGBM.
    """
    _ensure_out_dir()
    if not history or "loss" not in history or len(history["loss"]) == 0:
        return
    rounds = np.arange(1, len(history["loss"]) + 1)
    unit = "Boosting round"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rounds, history["loss"], label="Train loss", color="#0284c7")
    if "val_loss" in history and len(history["val_loss"]) == len(rounds):
        ax.plot(rounds, history["val_loss"], label="Val loss", color="#f97316")
    ax.set_xlabel(unit)
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} — Loss vs {unit}")
    ax.legend()
    fig.tight_layout()
    path_loss = os.path.join(OUT_DIR, f"loss_vs_epoch_{title_prefix.lower()}.png")
    fig.savefig(path_loss, dpi=150)
    plt.close(fig)
    print(f"Saved: {path_loss}")

    if "accuracy" in history and len(history["accuracy"]) == len(rounds):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(rounds, history["accuracy"], label="Train accuracy", color="#0284c7")
        if "val_accuracy" in history and len(history["val_accuracy"]) == len(rounds):
            ax.plot(rounds, history["val_accuracy"], label="Val accuracy", color="#f97316")
        ax.set_xlabel(unit)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_prefix} — Accuracy vs {unit}")
        ax.legend()
        fig.tight_layout()
        path_acc = os.path.join(OUT_DIR, f"accuracy_vs_epoch_{title_prefix.lower()}.png")
        fig.savefig(path_acc, dpi=150)
        plt.close(fig)
        print(f"Saved: {path_acc}")


def evaluate_system(system, prepared):
    """
    Confusion matrix heatmap + ROC + PR + metrics bar chart for LightGBM
    on the masked TEST split (x_obs = X_test * m_test).
    """
    _ensure_out_dir()
    X_test, y_test, m_test = prepared.X_test, prepared.y_test, prepared.m_test
    X_obs = X_test * m_test
    th = getattr(system.lightgbm, "best_threshold_", 0.5)

    p_lgb = system.lightgbm.predict_proba_y1(X_obs, m_test)
    y_pred = predict_label(p_lgb, threshold=th)

    print(f"\n===== LightGBM — classification report (masked test, threshold={th:.2f}) =====")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # Pass 1: Heatmap
    plot_confusion_heatmap(y_test, y_pred, title="LightGBM",
                           filename="confusion_matrix_lightgbm.png")

    # Pass 2: ROC & PR Curves
    plot_roc_curve(y_test, p_lgb, title="LightGBM", filename="roc_lightgbm.png")
    plot_pr_curve(y_test, p_lgb, title="LightGBM", filename="pr_lightgbm.png")

    # Pass 3: Metrics & Comparison plot
    results = {"LightGBM": _metrics_dict(y_test, y_pred, p_lgb)}
    plot_metric_comparison(results, filename="model_comparison.png")

    print(f"\n===== Summary (masked test, threshold={th:.2f}) =====")
    for name, m in results.items():
        print(f"{name:10s} | acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Checkpoint 3 — does the diagnosis stay the same under plausible completions?
# ---------------------------------------------------------------------------
def evaluate_plausible_stability(system, prepared, n_samples: int = None, split: str = "Test"):
    """
    Steps 7-14: for each masked patient in `split`, generate M CVAE plausible completions,
    re-diagnose every one with LightGBM, and report Class Stability (CS_t),
    Probability Stability (1 - PD_t), and DSS_t.
    """
    n_samples = n_samples or CFG.cvae.n_samples_M
    X, y, m = _splits(prepared)[split]
    X_obs = X * m
    th = getattr(system.lightgbm, "best_threshold_", 0.5)

    # Masked-input diagnosis
    p_t = system.lightgbm.predict_proba_y1(X_obs, m)
    y_hat_t = predict_label(p_t, threshold=th)

    # Step 7-8: M plausible completions
    plausible = system.cvae.generate_plausible_profiles(X_obs, m, n_samples)  # (M, N, F)
    full_mask = np.ones_like(m)

    p_plaus = np.zeros((n_samples, len(X)))
    y_plaus = np.zeros((n_samples, len(X)), dtype=int)
    for i in range(n_samples):
        profile = plausible[i]
        p_plaus[i] = system.lightgbm.predict_proba_y1(profile, full_mask)
        y_plaus[i] = predict_label(p_plaus[i], threshold=th)

    # Steps 12-14
    cs_t, pd_t, dss_t = evaluate_stability_and_dss(p_t, y_hat_t, p_plaus, y_plaus, alpha=CFG.dss.alpha)

    masked_acc = (y_hat_t == y).mean()
    bal_acc = balanced_accuracy_score(y, y_hat_t)

    print(f"\n===== Plausible-completion stability ({split}) =====")
    print(f"Masked-input diagnosis accuracy:          {masked_acc:.4f} (balanced acc: {bal_acc:.4f})")
    print(f"Mean Class Stability (CS_t):              {cs_t.mean():.4f}  "
          f"(fraction whose diagnosis didn't flip across {n_samples} plausible profiles)")
    print(f"Mean Probability Stability (1-PD_t):      {(1 - pd_t).mean():.4f}")
    print(f"Mean DSS_t:                               {dss_t.mean():.4f}")

    return {"CS_t": cs_t, "PD_t": pd_t, "DSS_t": dss_t, "y_hat_t": y_hat_t, "p_t": p_t}
