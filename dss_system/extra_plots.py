# """
# extra_plots.py
# ==============
# Drop-in plotting helpers for evaluate.py:

#   * plot_confusion_heatmap()  -> seaborn heatmap (counts + row-normalised %)
#   * plot_training_history()   -> loss-vs-epoch and train/val accuracy-vs-epoch
#   * history_from_xgb()        -> converts XGBoost evals_result_ into a history dict

# A `history` dict looks like:
#     {
#         "train_loss": [...], "val_loss": [...],
#         "train_acc":  [...], "val_acc":  [...],
#     }
# (one value per epoch / boosting round). Missing keys are skipped.

# Requires: pip install seaborn
# """

# from __future__ import annotations

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# OUT_DIR = "evaluation_plots"


# def _ensure_out_dir():
#     os.makedirs(OUT_DIR, exist_ok=True)


# # ---------------------------------------------------------------------------
# # Confusion matrix heatmap
# # ---------------------------------------------------------------------------
# def plot_confusion_heatmap(y_true, y_pred, title: str, filename: str,
#                            class_names=("No Diabetes", "Diabetes")):
#     """Heatmap with the raw count and the row-normalised percentage in each cell."""
#     _ensure_out_dir()
#     cm = confusion_matrix(y_true, y_pred)
#     row_sums = cm.sum(axis=1, keepdims=True)
#     cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

#     annot = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
#                       for i in range(cm.shape[0])])

#     fig, ax = plt.subplots(figsize=(5.5, 4.8))
#     sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
#                 xticklabels=class_names, yticklabels=class_names,
#                 linewidths=0.5, linecolor="white", ax=ax)
#     ax.set_title(f"Confusion Matrix — {title} (masked input)")
#     ax.set_xlabel("Predicted label")
#     ax.set_ylabel("True label")
#     fig.tight_layout()

#     path = os.path.join(OUT_DIR, filename)
#     fig.savefig(path, dpi=150)
#     plt.show()
#     print(f"Saved: {path}")
#     return cm


# # ---------------------------------------------------------------------------
# # Training curves
# # ---------------------------------------------------------------------------
# def plot_training_history(history: dict, model_name: str, filename: str | None = None):
#     """Left: loss vs epoch (train & val). Right: accuracy vs epoch (train & val)."""
#     _ensure_out_dir()
#     filename = filename or f"training_history_{model_name.lower()}.png"

#     fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

#     # Loss
#     for key, label in [("train_loss", "Train loss"), ("val_loss", "Val loss")]:
#         if key in history and len(history[key]):
#             ax_loss.plot(range(1, len(history[key]) + 1), history[key], label=label)
#     ax_loss.set_xlabel("Epoch")
#     ax_loss.set_ylabel("Loss")
#     ax_loss.set_title(f"{model_name} — Loss vs Epoch")
#     ax_loss.grid(alpha=0.3)
#     ax_loss.legend()

#     # Accuracy
#     for key, label in [("train_acc", "Train accuracy"), ("val_acc", "Val accuracy")]:
#         if key in history and len(history[key]):
#             ax_acc.plot(range(1, len(history[key]) + 1), history[key], label=label)
#     ax_acc.set_xlabel("Epoch")
#     ax_acc.set_ylabel("Accuracy")
#     ax_acc.set_ylim(0, 1.05)
#     ax_acc.set_title(f"{model_name} — Train vs Validation Accuracy")
#     ax_acc.grid(alpha=0.3)
#     ax_acc.legend()

#     fig.tight_layout()
#     path = os.path.join(OUT_DIR, filename)
#     fig.savefig(path, dpi=150)
#     plt.show()
#     print(f"Saved: {path}")


# def history_from_xgb(evals_result: dict, loss_metric: str = "logloss",
#                      err_metric: str = "error") -> dict:
#     """
#     Convert `model.evals_result()` (sklearn API) to a history dict.
#     Assumes eval_set=[(X_train, y_train), (X_val, y_val)] so keys are
#     'validation_0' (train) and 'validation_1' (val), and that eval_metric
#     included both "logloss" and "error" (accuracy = 1 - error).
#     """
#     tr, va = evals_result["validation_0"], evals_result["validation_1"]
#     hist = {"train_loss": tr.get(loss_metric, []), "val_loss": va.get(loss_metric, [])}
#     if err_metric in tr:
#         hist["train_acc"] = [1 - e for e in tr[err_metric]]
#         hist["val_acc"] = [1 - e for e in va[err_metric]]
#     return hist



"""
extra_plots.py
==============
Drop-in plotting helpers for evaluate.py:

  * plot_confusion_heatmap()  -> seaborn heatmap (counts + row-normalised %)
  * plot_training_history()   -> loss-vs-epoch and train/val accuracy-vs-epoch
  * history_from_xgb()        -> converts XGBoost evals_result_ into a history dict

A `history` dict looks like:
    {
        "train_loss": [...], "val_loss": [...],
        "train_acc":  [...], "val_acc":  [...],
    }
(one value per epoch / boosting round). Missing keys are skipped.

Requires: pip install seaborn
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:          # fall back to pure matplotlib if seaborn isn't installed
    sns = None
from sklearn.metrics import confusion_matrix

OUT_DIR = "evaluation_plots"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(y_true, y_pred, title: str, filename: str,
                           class_names=("No Diabetes", "Diabetes")):
    """Heatmap with the raw count and the row-normalised percentage in each cell."""
    _ensure_out_dir()
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    annot = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    if sns is not None:
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white", ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix — {title} (masked input)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
    return cm


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_history(history: dict, model_name: str, filename: str | None = None):
    """Left: loss vs epoch (train & val). Right: accuracy vs epoch (train & val)."""
    _ensure_out_dir()
    filename = filename or f"training_history_{model_name.lower()}.png"

    # accept both history key styles (train_loss/train_acc  OR  loss/accuracy)
    alias = {"loss": "train_loss", "accuracy": "train_acc", "val_accuracy": "val_acc"}
    history = {alias.get(k, k): v for k, v in history.items()}

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss
    for key, label in [("train_loss", "Train loss"), ("val_loss", "Val loss")]:
        if key in history and len(history[key]):
            ax_loss.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_name} — Loss vs Epoch")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # Accuracy
    for key, label in [("train_acc", "Train accuracy"), ("val_acc", "Val accuracy")]:
        if key in history and len(history[key]):
            ax_acc.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"{model_name} — Train vs Validation Accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def history_from_xgb(evals_result: dict, loss_metric: str = "logloss",
                     err_metric: str = "error") -> dict:
    """
    Convert `model.evals_result()` (sklearn API) to a history dict.
    Assumes eval_set=[(X_train, y_train), (X_val, y_val)] so keys are
    'validation_0' (train) and 'validation_1' (val), and that eval_metric
    included both "logloss" and "error" (accuracy = 1 - error).
    """
    tr, va = evals_result["validation_0"], evals_result["validation_1"]
    hist = {"train_loss": tr.get(loss_metric, []), "val_loss": va.get(loss_metric, [])}
    if err_metric in tr:
        hist["train_acc"] = [1 - e for e in tr[err_metric]]
        hist["val_acc"] = [1 - e for e in va[err_metric]]
    return hist