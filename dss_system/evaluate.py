"""
evaluate.py
============
Diagnostic evaluation of the trained classifiers on the held-out test
set (Steps 4-6: f_XGB, f_TR, and the fused p_t), with a fully-observed
mask (m = all ones) so these numbers reflect model quality on complete
records, separate from the adaptive-acquisition loop in main.py.

Produces, for f_XGB, f_TR and the fused model:
  - Confusion matrix (plot + raw counts)
  - ROC curve + AUC
  - Precision-Recall curve + average precision
  - A text classification report (precision/recall/F1/support)
  - A grouped bar chart comparing accuracy/precision/recall/F1/AUC
    across all three models

All figures are saved as PNGs under ./evaluation_plots/ and also shown
inline if run in a notebook (e.g. Colab).

Usage:
    python evaluate.py                 # trains fresh, then evaluates
    python evaluate.py --data path.csv # trains on a local CSV instead
                                        # of the live UCI fetch
"""

from __future__ import annotations

import argparse
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
    precision_score,
    recall_score,
    f1_score,
)

from data_processing import prepare_dataset
from pipeline import train_full_system
from fusion import fuse, predict_label

OUT_DIR = "evaluation_plots"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str,
                           class_names=("No Diabetes", "Diabetes")):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")
    return cm


def plot_roc_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(fpr, tpr, label=f"{label or title} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve" if not standalone else f"ROC Curve — {title}")
    ax.legend(loc="lower right")

    if standalone:
        fig.tight_layout()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"Saved: {path}")
    return roc_auc


def plot_pr_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(recall, precision, label=f"{label or title} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve" if not standalone else f"PR Curve — {title}")
    ax.legend(loc="lower left")

    if standalone:
        fig.tight_layout()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"Saved: {path}")
    return ap


def plot_metric_comparison(results: dict, filename: str = "model_comparison.png"):
    """
    results: {model_name: {"accuracy":.., "precision":.., "recall":..,
                            "f1":.., "auc":..}}
    Grouped bar chart comparing every model across all five metrics.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Diagnostic Model Comparison")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def _metrics_dict(y_true, y_pred, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc(fpr, tpr),
    }


def evaluate_system(system, prepared):
    """
    Runs f_XGB, f_TR and the fused model on the fully-observed test set
    and produces every plot + a printed classification report for each.
    """
    _ensure_out_dir()

    X_test = prepared.X_test
    y_test = prepared.y_test
    full_mask = np.ones_like(X_test)  # fully-observed evaluation (no acquisition)

    p_xgb = system.xgb.predict_proba_y1(X_test, full_mask)
    p_tr = system.tabresnet.predict_proba_y1(X_test, full_mask)
    p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

    y_xgb = predict_label(p_xgb)
    y_tr = predict_label(p_tr)
    y_fused = predict_label(p_fused)

    results = {}
    model_outputs = {
        "XGBoost": (y_xgb, p_xgb),
        "TabResNet": (y_tr, p_tr),
        "Fused": (y_fused, p_fused),
    }

    # Side-by-side ROC and PR plots across all three models
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5.5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5.5))

    for name, (y_pred, y_proba) in model_outputs.items():
        print(f"\n===== {name} — classification report =====")
        print(classification_report(y_test, y_pred,
                                      target_names=["No Diabetes", "Diabetes"]))

        plot_confusion_matrix(y_test, y_pred, title=name,
                               filename=f"confusion_matrix_{name.lower()}.png")

        plot_roc_curve(y_test, y_proba, title=name,
                        filename=f"roc_{name.lower()}.png", ax=ax_roc, label=name)
        plot_pr_curve(y_test, y_proba, title=name,
                      filename=f"pr_{name.lower()}.png", ax=ax_pr, label=name)

        results[name] = _metrics_dict(y_test, y_pred, y_proba)

    ax_roc.set_title("ROC Curve — All Models")
    ax_pr.set_title("Precision-Recall Curve — All Models")
    fig_roc.tight_layout(); fig_pr.tight_layout()
    fig_roc.savefig(os.path.join(OUT_DIR, "roc_all_models.png"), dpi=150)
    fig_pr.savefig(os.path.join(OUT_DIR, "pr_all_models.png"), dpi=150)
    plt.show()
    print(f"Saved: {OUT_DIR}/roc_all_models.png")
    print(f"Saved: {OUT_DIR}/pr_all_models.png")

    plot_metric_comparison(results)

    print("\n===== Summary =====")
    for name, m in results.items():
        print(f"{name:10s} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to a local CDC Diabetes CSV; omit to fetch live from UCI")
    args = parser.parse_args()

    print("Preparing dataset...")
    prepared = prepare_dataset(path=args.data)

    print("Training f_XGB, f_TR, CVAE, fitting fusion weights, building graph...")
    system = train_full_system(prepared)

    print("Evaluating on held-out test set...")
    evaluate_system(system, prepared)


if __name__ == "__main__":
    main()
