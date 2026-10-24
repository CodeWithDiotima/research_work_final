"""
extra_plots.py
==============
Drop-in plotting helpers for evaluation and diagnostic visualization:

  * plot_confusion_heatmap()  -> seaborn/matplotlib heatmap (counts + row-normalised %)
  * plot_training_history()   -> loss-vs-round and train/val accuracy-vs-round
  * history_from_catboost()   -> converts CatBoost evals_result_ into a history dict

A `history` dict format:
    {
        "train_loss": [...], "val_loss": [...],
        "train_acc":  [...], "val_acc":  [...],
    }
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.metrics import confusion_matrix

OUT_DIR = "evaluation_plots"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "LightGBM",
    filename: str = "confusion_matrix_lightgbm.png",
    class_names=("No Diabetes", "Diabetes"),
):
    """Heatmap with the raw count and the row-normalised percentage in each cell."""
    _ensure_out_dir()
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    annot = np.array(
        [[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    annot[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    ax.set_title(f"Confusion Matrix — {title} (masked input)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return cm


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_history(history: dict, model_name: str = "LightGBM", filename: str | None = None):
    """Left: loss vs round (train & val). Right: accuracy vs round (train & val)."""
    _ensure_out_dir()
    filename = filename or f"training_history_{model_name.lower()}.png"

    # accept both history key styles (train_loss/train_acc OR loss/accuracy)
    alias = {"loss": "train_loss", "accuracy": "train_acc", "val_accuracy": "val_acc"}
    history = {alias.get(k, k): v for k, v in history.items()}

    unit = "Boosting round" if any(k in model_name.lower() for k in ["lightgbm", "catboost", "xgb"]) else "Epoch"

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss
    has_loss = False
    for key, label in [("train_loss", "Train loss"), ("val_loss", "Val loss")]:
        if key in history and len(history[key]):
            has_loss = True
            ax_loss.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_loss.set_xlabel(unit)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_name} — Loss vs {unit}")
    ax_loss.grid(alpha=0.3)
    if has_loss:
        ax_loss.legend()

    # Accuracy
    has_acc = False
    for key, label in [("train_acc", "Train accuracy"), ("val_acc", "Val accuracy")]:
        if key in history and len(history[key]):
            has_acc = True
            ax_acc.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_acc.set_xlabel(unit)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"{model_name} — Train vs Validation Accuracy")
    ax_acc.grid(alpha=0.3)
    if has_acc:
        ax_acc.legend()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def history_from_lightgbm(evals_result: dict) -> dict:
    """
    Convert LightGBM evals_result dictionary into a standard history dict.
    """
    learn = evals_result.get("learn", {})
    val = evals_result.get("val", {})
    loss = learn.get("binary_logloss", [])
    val_loss = val.get("binary_logloss", [])
    learn_err = learn.get("binary_error", [])
    val_err = val.get("binary_error", [])
    return {
        "train_loss": [float(v) for v in loss],
        "val_loss": [float(v) for v in val_loss],
        "train_acc": [1.0 - float(e) for e in learn_err] if learn_err else [],
        "val_acc": [1.0 - float(e) for e in val_err] if val_err else [],
    }


def history_from_catboost(evals_result: dict, loss_metric: str = "Logloss") -> dict:
    """
    Convert `catboost_model.evals_result_` into a standard history dict.
    """
    learn = evals_result.get("learn", {})
    val = evals_result.get("validation", {})
    loss = learn.get(loss_metric, [])
    val_loss = val.get(loss_metric, [])

    acc = (
        learn.get("Accuracy:use_weights=false")
        or learn.get("Accuracy:use_weights=true")
        or learn.get("Accuracy", [])
    )
    val_acc = (
        val.get("Accuracy:use_weights=false")
        or val.get("Accuracy:use_weights=true")
        or val.get("Accuracy", [])
    )

    return {
        "train_loss": [float(v) for v in loss],
        "val_loss": [float(v) for v in val_loss],
        "train_acc": [float(v) for v in acc] if acc else [],
        "val_acc": [float(v) for v in val_acc] if val_acc else [],
    }


def history_from_xgb(evals_result: dict, loss_metric: str = "logloss", err_metric: str = "error") -> dict:
    """Legacy helper for XGBoost evals_result."""
    tr = evals_result.get("validation_0", {})
    va = evals_result.get("validation_1", {})
    out = {}
    if loss_metric in tr:
        out["train_loss"] = tr[loss_metric]
    if loss_metric in va:
        out["val_loss"] = va[loss_metric]
    if err_metric in tr:
        out["train_acc"] = [1.0 - e for e in tr[err_metric]]
    if err_metric in va:
        out["val_acc"] = [1.0 - e for e in va[err_metric]]
    return out