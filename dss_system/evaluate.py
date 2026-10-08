# # """
# # evaluate.py
# # ============
# # Diagnostic evaluation for f_XGB, f_TR and the fused classifier.

# # IMPORTANT: this project never evaluates the classifiers on fully-
# # observed data — f_XGB/f_TR are always called as f(x_obs, m), matching
# # how they were trained (Step 4-6). Every function here respects that.

# # Three checkpoints, in pipeline order:
# #   1. evaluate_masked_accuracy()     -> accuracy/precision/recall/F1/AUC
# #      on train/val/test, with features masked exactly as in training.
# #   2. evaluate_system()              -> confusion matrices + ROC/PR
# #      curves + comparison bar chart, on the masked TEST split.
# #   3. evaluate_plausible_stability() -> for each masked test patient,
# #      generate M CVAE plausible completions (Steps 7-8), re-diagnose
# #      each one (Steps 9-11), and report Class Stability / Probability
# #      Stability / DSS (Steps 12-14) — i.e. "does the diagnosis change
# #      once we guess the missing features?"

# # Call all three from main.py right after train_full_system().
# # """

# # from __future__ import annotations
# # from extra_plots import plot_confusion_heatmap, plot_training_history
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import (
# #     confusion_matrix,
# #     classification_report,
# #     roc_curve,
# #     auc,
# #     precision_recall_curve,
# #     average_precision_score,
# #     accuracy_score,
# #     precision_score,
# #     recall_score,
# #     f1_score,
# # )

# # from config import CFG
# # from fusion import fuse, predict_label
# # from stability import evaluate_stability_and_dss

# # OUT_DIR = "evaluation_plots"


# # def _ensure_out_dir():
# #     os.makedirs(OUT_DIR, exist_ok=True)


# # def _metrics_dict(y_true, y_pred, y_proba) -> dict:
# #     fpr, tpr, _ = roc_curve(y_true, y_proba)
# #     return {
# #         "accuracy": accuracy_score(y_true, y_pred),
# #         "precision": precision_score(y_true, y_pred, zero_division=0),
# #         "recall": recall_score(y_true, y_pred, zero_division=0),
# #         "f1": f1_score(y_true, y_pred, zero_division=0),
# #         "auc": auc(fpr, tpr),
# #     }


# # def _splits(prepared):
# #     return {
# #         "Train": (prepared.X_train, prepared.y_train, prepared.m_train),
# #         "Val":   (prepared.X_val,   prepared.y_val,   prepared.m_val),
# #         "Test":  (prepared.X_test,  prepared.y_test,  prepared.m_test),
# #     }


# # # ---------------------------------------------------------------------------
# # # Checkpoint 1 — masked accuracy across train/val/test
# # # ---------------------------------------------------------------------------
# # def evaluate_masked_accuracy(system, prepared):
# #     """
# #     f_XGB, f_TR and the fused model, each called as f(x_obs, m) — the
# #     SAME masked input used during training — on train/val/test. This is
# #     the only valid "static" accuracy check for this system; there is no
# #     fully-observed evaluation anywhere in the pipeline.
# #     """
# #     all_results = {}
# #     for split_name, (X, y, m) in _splits(prepared).items():
# #         X_obs = X * m
# #         p_xgb = system.xgb.predict_proba_y1(X_obs, m)
# #         p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
# #         p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

# #         for name, p in [("XGBoost", p_xgb), ("TabResNet", p_tr), ("Fused", p_fused)]:
# #             y_pred = predict_label(p)
# #             all_results[(name, split_name)] = _metrics_dict(y, y_pred, p)

# #     print(f"\n{'Model':10s} {'Split':6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
# #     print("-" * 56)
# #     for (name, split_name), m in all_results.items():
# #         print(f"{name:10s} {split_name:6s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
# #               f"{m['recall']:7.4f} {m['f1']:7.4f} {m['auc']:7.4f}")

# #     # Bar chart: accuracy by split, one bar group per model
# #     model_names = ["XGBoost", "TabResNet", "Fused"]
# #     split_names = ["Train", "Val", "Test"]
# #     x = np.arange(len(split_names))
# #     width = 0.8 / len(model_names)

# #     fig, ax = plt.subplots(figsize=(7, 5))
# #     for i, name in enumerate(model_names):
# #         values = [all_results[(name, s)]["accuracy"] for s in split_names]
# #         bars = ax.bar(x + i * width, values, width, label=name)
# #         for b, v in zip(bars, values):
# #             ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
# #                      ha="center", va="bottom", fontsize=8)

# #     ax.set_xticks(x + width * (len(model_names) - 1) / 2)
# #     ax.set_xticklabels(split_names)
# #     ax.set_ylim(0, 1.05)
# #     ax.set_ylabel("Accuracy")
# #     ax.set_title("Train vs Validation vs Test Accuracy (masked input)")
# #     ax.legend()
# #     fig.tight_layout()

# #     _ensure_out_dir()
# #     path = os.path.join(OUT_DIR, "train_val_test_accuracy.png")
# #     fig.savefig(path, dpi=150)
# #     plt.show()
# #     print(f"Saved: {path}")

# #     return all_results


# # # ---------------------------------------------------------------------------
# # # Checkpoint 2 — confusion matrices, ROC/PR curves on the masked TEST split
# # # ---------------------------------------------------------------------------
# # def plot_confusion_matrix(y_true, y_pred, title: str, filename: str,
# #                            class_names=("No Diabetes", "Diabetes")):
# #     cm = confusion_matrix(y_true, y_pred)

# #     fig, ax = plt.subplots(figsize=(5, 4.5))
# #     im = ax.imshow(cm, cmap="Blues")
# #     ax.set_title(f"Confusion Matrix — {title} (masked input)")
# #     ax.set_xlabel("Predicted label")
# #     ax.set_ylabel("True label")
# #     ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
# #     ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)

# #     thresh = cm.max() / 2.0
# #     for i in range(cm.shape[0]):
# #         for j in range(cm.shape[1]):
# #             ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
# #                      color="white" if cm[i, j] > thresh else "black", fontsize=12)

# #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# #     fig.tight_layout()
# #     path = os.path.join(OUT_DIR, filename)
# #     fig.savefig(path, dpi=150)
# #     plt.show()
# #     print(f"Saved: {path}")
# #     return cm


# # def plot_roc_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
# #     fpr, tpr, _ = roc_curve(y_true, y_proba)
# #     roc_auc = auc(fpr, tpr)

# #     standalone = ax is None
# #     if standalone:
# #         fig, ax = plt.subplots(figsize=(5.5, 5))

# #     ax.plot(fpr, tpr, label=f"{label or title} (AUC={roc_auc:.3f})")
# #     ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
# #     ax.set_xlabel("False Positive Rate")
# #     ax.set_ylabel("True Positive Rate")
# #     ax.set_title("ROC Curve" if not standalone else f"ROC Curve — {title}")
# #     ax.legend(loc="lower right")

# #     if standalone:
# #         fig.tight_layout()
# #         path = os.path.join(OUT_DIR, filename)
# #         fig.savefig(path, dpi=150)
# #         plt.show()
# #         print(f"Saved: {path}")
# #     return roc_auc


# # def plot_pr_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
# #     precision, recall, _ = precision_recall_curve(y_true, y_proba)
# #     ap = average_precision_score(y_true, y_proba)

# #     standalone = ax is None
# #     if standalone:
# #         fig, ax = plt.subplots(figsize=(5.5, 5))

# #     ax.plot(recall, precision, label=f"{label or title} (AP={ap:.3f})")
# #     ax.set_xlabel("Recall")
# #     ax.set_ylabel("Precision")
# #     ax.set_title("Precision-Recall Curve" if not standalone else f"PR Curve — {title}")
# #     ax.legend(loc="lower left")

# #     if standalone:
# #         fig.tight_layout()
# #         path = os.path.join(OUT_DIR, filename)
# #         fig.savefig(path, dpi=150)
# #         plt.show()
# #         print(f"Saved: {path}")
# #     return ap

# # def plot_metric_comparison(results: dict, filename: str = "model_comparison.png"):
# #     metrics = ["accuracy", "precision", "recall", "f1", "auc"]
# #     model_names = list(results.keys())
# #     x = np.arange(len(metrics))
# #     width = 0.8 / len(model_names)

# #     fig, ax = plt.subplots(figsize=(8, 5))
# #     for i, name in enumerate(model_names):
# #         values = [results[name][m] for m in metrics]
# #         bars = ax.bar(x + i * width, values, width, label=name)
# #         for b, v in zip(bars, values):
# #             ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
# #                      ha="center", va="bottom", fontsize=8)

# #     ax.set_xticks(x + width * (len(model_names) - 1) / 2)
# #     ax.set_xticklabels([m.upper() for m in metrics])
# #     ax.set_ylim(0, 1.05)
# #     ax.set_title("Diagnostic Model Comparison (masked input, test split)")
# #     ax.legend()
# #     fig.tight_layout()

# #     path = os.path.join(OUT_DIR, filename)
# #     fig.savefig(path, dpi=150)
# #     plt.show()
# #     print(f"Saved: {path}")


# # def evaluate_system(system, prepared):
# #     """
# #     Confusion matrix + ROC + PR + comparison bar chart for f_XGB, f_TR,
# #     and Fused, on the masked TEST split (x_obs = X_test * m_test).
# #     """
# #     _ensure_out_dir()
# #     X_test, y_test, m_test = prepared.X_test, prepared.y_test, prepared.m_test
# #     X_obs = X_test * m_test

# #     p_xgb = system.xgb.predict_proba_y1(X_obs, m_test)
# #     p_tr = system.tabresnet.predict_proba_y1(X_obs, m_test)
# #     p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

# #     model_outputs = {
# #         "XGBoost": (predict_label(p_xgb), p_xgb),
# #         "TabResNet": (predict_label(p_tr), p_tr),
# #         "Fused": (predict_label(p_fused), p_fused),
# #     }

# #     results = {}
# #     fig_roc, ax_roc = plt.subplots(figsize=(6, 5.5))
# #     fig_pr, ax_pr = plt.subplots(figsize=(6, 5.5))

# #     for name, (y_pred, y_proba) in model_outputs.items():
# #         print(f"\n===== {name} — classification report (masked test) =====")
# #         print(classification_report(y_test, y_pred,
# #                                       target_names=["No Diabetes", "Diabetes"]))

# #         # plot_confusion_matrix(y_test, y_pred, title=name,
# #         #                        filename=f"confusion_matrix_{name.lower()}.png")
# #         plot_confusion_heatmap(y_test, y_pred, title=name,
# #                        filename=f"confusion_matrix_{name.lower()}.png")
# #         plot_roc_curve(y_test, y_proba, title=name,
# #                         filename=f"roc_{name.lower()}.png", ax=ax_roc, label=name)
# #         plot_pr_curve(y_test, y_proba, title=name,
# #                       filename=f"pr_{name.lower()}.png", ax=ax_pr, label=name)

# #         results[name] = _metrics_dict(y_test, y_pred, y_proba)

# #     ax_roc.set_title("ROC Curve — All Models (masked test)")
# #     ax_pr.set_title("Precision-Recall Curve — All Models (masked test)")
# #     fig_roc.tight_layout(); fig_pr.tight_layout()
# #     fig_roc.savefig(os.path.join(OUT_DIR, "roc_all_models.png"), dpi=150)
# #     fig_pr.savefig(os.path.join(OUT_DIR, "pr_all_models.png"), dpi=150)
# #     plt.show()
# #     print(f"Saved: {OUT_DIR}/roc_all_models.png")
# #     print(f"Saved: {OUT_DIR}/pr_all_models.png")

# #     plot_metric_comparison(results)

# #     print("\n===== Summary (masked test) =====")
# #     for name, m in results.items():
# #         print(f"{name:10s} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
# #               f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

# #     return results


# # # ---------------------------------------------------------------------------
# # # Checkpoint 3 — does the diagnosis stay the same under plausible completions?
# # # ---------------------------------------------------------------------------
# # def evaluate_plausible_stability(system, prepared, n_samples: int = None, split: str = "Test"):
# #     """
# #     Steps 7-14, as a standalone check: for each masked patient in
# #     `split`, generate M CVAE plausible completions, re-diagnose every
# #     one, and report Class Stability (CS_t), Probability Stability
# #     (1 - PD_t), and DSS_t — i.e. whether filling in the unknowns with
# #     plausible guesses changes the diagnosis.
# #     """
# #     n_samples = n_samples or CFG.cvae.n_samples_M
# #     X, y, m = _splits(prepared)[split]
# #     X_obs = X * m

# #     # Step 4-6: the masked-only diagnosis (same as checkpoint 1's Fused/{split} row)
# #     p_xgb = system.xgb.predict_proba_y1(X_obs, m)
# #     p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
# #     p_t = fuse(p_xgb, p_tr, system.fusion_weights)
# #     y_hat_t = predict_label(p_t)

# #     # Step 7-8: M plausible completions
# #     plausible = system.cvae.generate_plausible_profiles(X_obs, m, n_samples)  # (M, N, F)
# #     full_mask = np.ones_like(m)

# #     p_plaus = np.zeros((n_samples, len(X)))
# #     y_plaus = np.zeros((n_samples, len(X)), dtype=int)
# #     for i in range(n_samples):
# #         profile = plausible[i]
# #         p_xgb_i = system.xgb.predict_proba_y1(profile, full_mask)
# #         p_tr_i = system.tabresnet.predict_proba_y1(profile, full_mask)
# #         p_plaus[i] = fuse(p_xgb_i, p_tr_i, system.fusion_weights)
# #         y_plaus[i] = predict_label(p_plaus[i])

# #     # Steps 12-14
# #     cs_t, pd_t, dss_t = evaluate_stability_and_dss(p_t, y_hat_t, p_plaus, y_plaus)

# #     masked_acc = (y_hat_t == y).mean()

# #     print(f"\n===== Plausible-completion stability ({split}) =====")
# #     print(f"Masked-input diagnosis accuracy:      {masked_acc:.4f}")
# #     print(f"Mean Class Stability (CS_t):          {cs_t.mean():.4f}  "
# #           f"(fraction whose diagnosis didn't flip across {n_samples} plausible profiles)")
# #     print(f"Mean Probability Stability (1-PD_t):  {(1 - pd_t).mean():.4f}")
# #     print(f"Mean DSS_t:                           {dss_t.mean():.4f}")

# #     return {"CS_t": cs_t, "PD_t": pd_t, "DSS_t": dss_t, "y_hat_t": y_hat_t, "p_t": p_t}





# """
# evaluate.py
# ============
# Diagnostic evaluation for f_XGB, f_TR and the fused classifier.

# IMPORTANT: this project never evaluates the classifiers on fully-
# observed data — f_XGB/f_TR are always called as f(x_obs, m), matching
# how they were trained (Step 4-6). Every function here respects that.

# Three checkpoints, in pipeline order:
#   1. evaluate_masked_accuracy()     -> accuracy/precision/recall/F1/AUC
#      on train/val/test, with features masked exactly as in training.
#   2. evaluate_system()              -> confusion matrices + ROC/PR
#      curves + comparison bar chart, on the masked TEST split.
#   3. evaluate_plausible_stability() -> for each masked test patient,
#      generate M CVAE plausible completions (Steps 7-8), re-diagnose
#      each one (Steps 9-11), and report Class Stability / Probability
#      Stability / DSS (Steps 12-14) — i.e. "does the diagnosis change
#      once we guess the missing features?"

# Call all three from main.py right after train_full_system().
# """

# from __future__ import annotations
# from extra_plots import plot_confusion_heatmap, plot_training_history
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     roc_curve,
#     auc,
#     precision_recall_curve,
#     average_precision_score,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# from config import CFG
# from fusion import fuse, predict_label
# from stability import evaluate_stability_and_dss

# OUT_DIR = "evaluation_plots"


# def _ensure_out_dir():
#     os.makedirs(OUT_DIR, exist_ok=True)


# def _metrics_dict(y_true, y_pred, y_proba) -> dict:
#     fpr, tpr, _ = roc_curve(y_true, y_proba)
#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall": recall_score(y_true, y_pred, zero_division=0),
#         "f1": f1_score(y_true, y_pred, zero_division=0),
#         "auc": auc(fpr, tpr),
#     }


# def _splits(prepared):
#     return {
#         "Train": (prepared.X_train, prepared.y_train, prepared.m_train),
#         "Val":   (prepared.X_val,   prepared.y_val,   prepared.m_val),
#         "Test":  (prepared.X_test,  prepared.y_test,  prepared.m_test),
#     }


# # ---------------------------------------------------------------------------
# # Checkpoint 1 — masked accuracy across train/val/test
# # ---------------------------------------------------------------------------
# def evaluate_masked_accuracy(system, prepared):
#     """
#     f_XGB, f_TR and the fused model, each called as f(x_obs, m) — the
#     SAME masked input used during training — on train/val/test. This is
#     the only valid "static" accuracy check for this system; there is no
#     fully-observed evaluation anywhere in the pipeline.
#     """
#     all_results = {}
#     for split_name, (X, y, m) in _splits(prepared).items():
#         X_obs = X * m
#         p_xgb = system.xgb.predict_proba_y1(X_obs, m)
#         p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
#         p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

#         for name, p in [("XGBoost", p_xgb), ("TabResNet", p_tr), ("Fused", p_fused)]:
#             y_pred = predict_label(p)
#             all_results[(name, split_name)] = _metrics_dict(y, y_pred, p)

#     print(f"\n{'Model':10s} {'Split':6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
#     print("-" * 56)
#     for (name, split_name), m in all_results.items():
#         print(f"{name:10s} {split_name:6s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
#               f"{m['recall']:7.4f} {m['f1']:7.4f} {m['auc']:7.4f}")

#     # Bar chart: accuracy by split, one bar group per model
#     model_names = ["XGBoost", "TabResNet", "Fused"]
#     split_names = ["Train", "Val", "Test"]
#     x = np.arange(len(split_names))
#     width = 0.8 / len(model_names)

#     fig, ax = plt.subplots(figsize=(7, 5))
#     for i, name in enumerate(model_names):
#         values = [all_results[(name, s)]["accuracy"] for s in split_names]
#         bars = ax.bar(x + i * width, values, width, label=name)
#         for b, v in zip(bars, values):
#             ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
#                      ha="center", va="bottom", fontsize=8)

#     ax.set_xticks(x + width * (len(model_names) - 1) / 2)
#     ax.set_xticklabels(split_names)
#     ax.set_ylim(0, 1.05)
#     ax.set_ylabel("Accuracy")
#     ax.set_title("Train vs Validation vs Test Accuracy (masked input)")
#     ax.legend()
#     fig.tight_layout()

#     _ensure_out_dir()
#     path = os.path.join(OUT_DIR, "train_val_test_accuracy.png")
#     fig.savefig(path, dpi=150)
#     plt.show()
#     print(f"Saved: {path}")

#     return all_results


# # ---------------------------------------------------------------------------
# # Checkpoint 2 — confusion matrices, ROC/PR curves on the masked TEST split
# # ---------------------------------------------------------------------------
# def plot_confusion_matrix(y_true, y_pred, title: str, filename: str,
#                            class_names=("No Diabetes", "Diabetes")):
#     cm = confusion_matrix(y_true, y_pred)

#     fig, ax = plt.subplots(figsize=(5, 4.5))
#     im = ax.imshow(cm, cmap="Blues")
#     ax.set_title(f"Confusion Matrix — {title} (masked input)")
#     ax.set_xlabel("Predicted label")
#     ax.set_ylabel("True label")
#     ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
#     ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)

#     thresh = cm.max() / 2.0
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
#                      color="white" if cm[i, j] > thresh else "black", fontsize=12)

#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     fig.tight_layout()
#     path = os.path.join(OUT_DIR, filename)
#     fig.savefig(path, dpi=150)
#     plt.show()
#     print(f"Saved: {path}")
#     return cm


# def plot_roc_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
#     fpr, tpr, _ = roc_curve(y_true, y_proba)
#     roc_auc = auc(fpr, tpr)

#     standalone = ax is None
#     if standalone:
#         fig, ax = plt.subplots(figsize=(5.5, 5))

#     ax.plot(fpr, tpr, label=f"{label or title} (AUC={roc_auc:.3f})")
#     ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curve" if not standalone else f"ROC Curve — {title}")
#     ax.legend(loc="lower right")

#     if standalone:
#         fig.tight_layout()
#         path = os.path.join(OUT_DIR, filename)
#         fig.savefig(path, dpi=150)
#         plt.show()
#         print(f"Saved: {path}")
#     return roc_auc


# def plot_pr_curve(y_true, y_proba, title: str, filename: str, ax=None, label=None):
#     precision, recall, _ = precision_recall_curve(y_true, y_proba)
#     ap = average_precision_score(y_true, y_proba)

#     standalone = ax is None
#     if standalone:
#         fig, ax = plt.subplots(figsize=(5.5, 5))

#     ax.plot(recall, precision, label=f"{label or title} (AP={ap:.3f})")
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.set_title("Precision-Recall Curve" if not standalone else f"PR Curve — {title}")
#     ax.legend(loc="lower left")

#     if standalone:
#         fig.tight_layout()
#         path = os.path.join(OUT_DIR, filename)
#         fig.savefig(path, dpi=150)
#         plt.show()
#         print(f"Saved: {path}")
#     return ap

# def plot_metric_comparison(results: dict, filename: str = "model_comparison.png"):
#     metrics = ["accuracy", "precision", "recall", "f1", "auc"]
#     model_names = list(results.keys())
#     x = np.arange(len(metrics))
#     width = 0.8 / len(model_names)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     for i, name in enumerate(model_names):
#         values = [results[name][m] for m in metrics]
#         bars = ax.bar(x + i * width, values, width, label=name)
#         for b, v in zip(bars, values):
#             ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
#                      ha="center", va="bottom", fontsize=8)

#     ax.set_xticks(x + width * (len(model_names) - 1) / 2)
#     ax.set_xticklabels([m.upper() for m in metrics])
#     ax.set_ylim(0, 1.05)
#     ax.set_title("Diagnostic Model Comparison (masked input, test split)")
#     ax.legend()
#     fig.tight_layout()

#     path = os.path.join(OUT_DIR, filename)
#     fig.savefig(path, dpi=150)
#     plt.show()
#     print(f"Saved: {path}")


# def plot_training_curves(history: dict, title_prefix: str = "Model"):
#     """
#     Loss-vs-epoch and accuracy-vs-epoch (train vs val) for any model whose
#     history has keys: loss, val_loss, accuracy, val_accuracy.
#     Works for both TabResNet (per epoch) and XGBoost (per boosting round).
#     """
#     _ensure_out_dir()
#     rounds = np.arange(1, len(history["loss"]) + 1)
#     unit = "Boosting round" if title_prefix.lower() == "xgboost" else "Epoch"

#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     ax.plot(rounds, history["loss"], label="Train loss")
#     ax.plot(rounds, history["val_loss"], label="Val loss")
#     ax.set_xlabel(unit); ax.set_ylabel("Loss")
#     ax.set_title(f"{title_prefix} — Loss vs {unit}")
#     ax.legend(); fig.tight_layout()
#     path_loss = os.path.join(OUT_DIR, f"loss_vs_epoch_{title_prefix.lower()}.png")
#     fig.savefig(path_loss, dpi=150); plt.show()
#     print(f"Saved: {path_loss}")

#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     ax.plot(rounds, history["accuracy"], label="Train accuracy")
#     ax.plot(rounds, history["val_accuracy"], label="Val accuracy")
#     ax.set_xlabel(unit); ax.set_ylabel("Accuracy")
#     ax.set_title(f"{title_prefix} — Accuracy vs {unit}")
#     ax.legend(); fig.tight_layout()
#     path_acc = os.path.join(OUT_DIR, f"accuracy_vs_epoch_{title_prefix.lower()}.png")
#     fig.savefig(path_acc, dpi=150); plt.show()
#     print(f"Saved: {path_acc}")


# def evaluate_system(system, prepared):
#     """
#     Confusion matrix + ROC + PR + comparison bar chart for f_XGB, f_TR,
#     and Fused, on the masked TEST split (x_obs = X_test * m_test).
#     """
#     _ensure_out_dir()
#     X_test, y_test, m_test = prepared.X_test, prepared.y_test, prepared.m_test
#     X_obs = X_test * m_test

#     p_xgb = system.xgb.predict_proba_y1(X_obs, m_test)
#     p_tr = system.tabresnet.predict_proba_y1(X_obs, m_test)
#     p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

#     model_outputs = {
#         "XGBoost": (predict_label(p_xgb), p_xgb),
#         "TabResNet": (predict_label(p_tr), p_tr),
#         "Fused": (predict_label(p_fused), p_fused),
#     }

#     results = {}
#     fig_roc, ax_roc = plt.subplots(figsize=(6, 5.5))
#     fig_pr, ax_pr = plt.subplots(figsize=(6, 5.5))

#     for name, (y_pred, y_proba) in model_outputs.items():
#         print(f"\n===== {name} — classification report (masked test) =====")
#         print(classification_report(y_test, y_pred,
#                                       target_names=["No Diabetes", "Diabetes"]))

#         # plot_confusion_matrix(y_test, y_pred, title=name,
#         #                        filename=f"confusion_matrix_{name.lower()}.png")
#         plot_confusion_heatmap(y_test, y_pred, title=name,
#                        filename=f"confusion_matrix_{name.lower()}.png")
#         plot_roc_curve(y_test, y_proba, title=name,
#                         filename=f"roc_{name.lower()}.png", ax=ax_roc, label=name)
#         plot_pr_curve(y_test, y_proba, title=name,
#                       filename=f"pr_{name.lower()}.png", ax=ax_pr, label=name)

#         results[name] = _metrics_dict(y_test, y_pred, y_proba)

#     ax_roc.set_title("ROC Curve — All Models (masked test)")
#     ax_pr.set_title("Precision-Recall Curve — All Models (masked test)")
#     fig_roc.tight_layout(); fig_pr.tight_layout()
#     fig_roc.savefig(os.path.join(OUT_DIR, "roc_all_models.png"), dpi=150)
#     fig_pr.savefig(os.path.join(OUT_DIR, "pr_all_models.png"), dpi=150)
#     plt.show()
#     print(f"Saved: {OUT_DIR}/roc_all_models.png")
#     print(f"Saved: {OUT_DIR}/pr_all_models.png")

#     plot_metric_comparison(results)

#     print("\n===== Summary (masked test) =====")
#     for name, m in results.items():
#         print(f"{name:10s} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
#               f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

#     return results


# # ---------------------------------------------------------------------------
# # Checkpoint 3 — does the diagnosis stay the same under plausible completions?
# # ---------------------------------------------------------------------------
# def evaluate_plausible_stability(system, prepared, n_samples: int = None, split: str = "Test"):
#     """
#     Steps 7-14, as a standalone check: for each masked patient in
#     `split`, generate M CVAE plausible completions, re-diagnose every
#     one, and report Class Stability (CS_t), Probability Stability
#     (1 - PD_t), and DSS_t — i.e. whether filling in the unknowns with
#     plausible guesses changes the diagnosis.
#     """
#     n_samples = n_samples or CFG.cvae.n_samples_M
#     X, y, m = _splits(prepared)[split]
#     X_obs = X * m

#     # Step 4-6: the masked-only diagnosis (same as checkpoint 1's Fused/{split} row)
#     p_xgb = system.xgb.predict_proba_y1(X_obs, m)
#     p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
#     p_t = fuse(p_xgb, p_tr, system.fusion_weights)
#     y_hat_t = predict_label(p_t)

#     # Step 7-8: M plausible completions
#     plausible = system.cvae.generate_plausible_profiles(X_obs, m, n_samples)  # (M, N, F)
#     full_mask = np.ones_like(m)

#     p_plaus = np.zeros((n_samples, len(X)))
#     y_plaus = np.zeros((n_samples, len(X)), dtype=int)
#     for i in range(n_samples):
#         profile = plausible[i]
#         p_xgb_i = system.xgb.predict_proba_y1(profile, full_mask)
#         p_tr_i = system.tabresnet.predict_proba_y1(profile, full_mask)
#         p_plaus[i] = fuse(p_xgb_i, p_tr_i, system.fusion_weights)
#         y_plaus[i] = predict_label(p_plaus[i])

#     # Steps 12-14
#     cs_t, pd_t, dss_t = evaluate_stability_and_dss(p_t, y_hat_t, p_plaus, y_plaus)

#     masked_acc = (y_hat_t == y).mean()

#     print(f"\n===== Plausible-completion stability ({split}) =====")
#     print(f"Masked-input diagnosis accuracy:      {masked_acc:.4f}")
#     print(f"Mean Class Stability (CS_t):          {cs_t.mean():.4f}  "
#           f"(fraction whose diagnosis didn't flip across {n_samples} plausible profiles)")
#     print(f"Mean Probability Stability (1-PD_t):  {(1 - pd_t).mean():.4f}")
#     print(f"Mean DSS_t:                           {dss_t.mean():.4f}")

#     return {"CS_t": cs_t, "PD_t": pd_t, "DSS_t": dss_t, "y_hat_t": y_hat_t, "p_t": p_t}







"""
evaluate.py
============
Diagnostic evaluation for f_XGB, f_TR and the fused classifier.

IMPORTANT: this project never evaluates the classifiers on fully-
observed data — f_XGB/f_TR are always called as f(x_obs, m), matching
how they were trained (Step 4-6). Every function here respects that.

Three checkpoints, in pipeline order:
  1. evaluate_masked_accuracy()     -> accuracy/precision/recall/F1/AUC
     on train/val/test, with features masked exactly as in training.
  2. evaluate_system()              -> confusion matrices + ROC/PR
     curves + comparison bar chart, on the masked TEST split.
  3. evaluate_plausible_stability() -> for each masked test patient,
     generate M CVAE plausible completions (Steps 7-8), re-diagnose
     each one (Steps 9-11), and report Class Stability / Probability
     Stability / DSS (Steps 12-14) — i.e. "does the diagnosis change
     once we guess the missing features?"

Call all three from main.py right after train_full_system().
"""

from __future__ import annotations
from extra_plots import plot_confusion_heatmap, plot_training_history
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

from config import CFG
from fusion import fuse, predict_label
from stability import evaluate_stability_and_dss

OUT_DIR = "evaluation_plots"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _metrics_dict(y_true, y_pred, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
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
    f_XGB, f_TR and the fused model, each called as f(x_obs, m) — the
    SAME masked input used during training — on train/val/test. This is
    the only valid "static" accuracy check for this system; there is no
    fully-observed evaluation anywhere in the pipeline.
    """
    all_results = {}
    for split_name, (X, y, m) in _splits(prepared).items():
        X_obs = X * m
        p_xgb = system.xgb.predict_proba_y1(X_obs, m)
        p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
        p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

        for name, p in [("XGBoost", p_xgb), ("TabResNet", p_tr), ("Fused", p_fused)]:
            y_pred = predict_label(p)
            all_results[(name, split_name)] = _metrics_dict(y, y_pred, p)

    print(f"\n{'Model':10s} {'Split':6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    print("-" * 56)
    for (name, split_name), m in all_results.items():
        print(f"{name:10s} {split_name:6s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['f1']:7.4f} {m['auc']:7.4f}")

    # Bar chart: accuracy by split, one bar group per model
    model_names = ["XGBoost", "TabResNet", "Fused"]
    split_names = ["Train", "Val", "Test"]
    x = np.arange(len(split_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(model_names):
        values = [all_results[(name, s)]["accuracy"] for s in split_names]
        bars = ax.bar(x + i * width, values, width, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Validation vs Test Accuracy (masked input)")
    ax.legend()
    fig.tight_layout()

    _ensure_out_dir()
    path = os.path.join(OUT_DIR, "train_val_test_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

    return all_results


# ---------------------------------------------------------------------------
# Checkpoint 2 — confusion matrices, ROC/PR curves on the masked TEST split
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title: str, filename: str,
                           class_names=("No Diabetes", "Diabetes")):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title} (masked input)")
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
    ax.set_title("Diagnostic Model Comparison (masked input, test split)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_training_curves(history: dict, title_prefix: str = "Model"):
    """
    Loss-vs-epoch and accuracy-vs-epoch (train vs val) for any model whose
    history has keys: loss, val_loss, accuracy, val_accuracy.
    Works for both TabResNet (per epoch) and XGBoost (per boosting round).
    """
    _ensure_out_dir()
    rounds = np.arange(1, len(history["loss"]) + 1)
    unit = "Boosting round" if title_prefix.lower() == "xgboost" else "Epoch"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rounds, history["loss"], label="Train loss")
    ax.plot(rounds, history["val_loss"], label="Val loss")
    ax.set_xlabel(unit); ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} — Loss vs {unit}")
    ax.legend(); fig.tight_layout()
    path_loss = os.path.join(OUT_DIR, f"loss_vs_epoch_{title_prefix.lower()}.png")
    fig.savefig(path_loss, dpi=150); plt.show()
    print(f"Saved: {path_loss}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rounds, history["accuracy"], label="Train accuracy")
    ax.plot(rounds, history["val_accuracy"], label="Val accuracy")
    ax.set_xlabel(unit); ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix} — Accuracy vs {unit}")
    ax.legend(); fig.tight_layout()
    path_acc = os.path.join(OUT_DIR, f"accuracy_vs_epoch_{title_prefix.lower()}.png")
    fig.savefig(path_acc, dpi=150); plt.show()
    print(f"Saved: {path_acc}")


def evaluate_system(system, prepared):
    """
    Confusion matrix + ROC + PR + comparison bar chart for f_XGB, f_TR,
    and Fused, on the masked TEST split (x_obs = X_test * m_test).
    """
    _ensure_out_dir()
    X_test, y_test, m_test = prepared.X_test, prepared.y_test, prepared.m_test
    X_obs = X_test * m_test

    p_xgb = system.xgb.predict_proba_y1(X_obs, m_test)
    p_tr = system.tabresnet.predict_proba_y1(X_obs, m_test)
    p_fused = fuse(p_xgb, p_tr, system.fusion_weights)

    model_outputs = {
        "XGBoost": (predict_label(p_xgb), p_xgb),
        "TabResNet": (predict_label(p_tr), p_tr),
        "Fused": (predict_label(p_fused), p_fused),
    }

    results = {}

    # Pass 1: classification report + confusion-matrix HEATMAP per model.
    # Done BEFORE the shared ROC/PR figures are created, so no other figure
    # is open when each heatmap is drawn / shown.
    for name, (y_pred, y_proba) in model_outputs.items():
        print(f"\n===== {name} — classification report (masked test) =====")
        print(classification_report(y_test, y_pred,
                                      target_names=["No Diabetes", "Diabetes"]))
        plot_confusion_heatmap(y_test, y_pred, title=name,
                               filename=f"confusion_matrix_{name.lower()}.png")

    # Pass 2: shared ROC / PR figures + metrics
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5.5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5.5))
    for name, (y_pred, y_proba) in model_outputs.items():
        plot_roc_curve(y_test, y_proba, title=name,
                        filename=f"roc_{name.lower()}.png", ax=ax_roc, label=name)
        plot_pr_curve(y_test, y_proba, title=name,
                      filename=f"pr_{name.lower()}.png", ax=ax_pr, label=name)

        results[name] = _metrics_dict(y_test, y_pred, y_proba)

    ax_roc.set_title("ROC Curve — All Models (masked test)")
    ax_pr.set_title("Precision-Recall Curve — All Models (masked test)")
    fig_roc.tight_layout(); fig_pr.tight_layout()
    fig_roc.savefig(os.path.join(OUT_DIR, "roc_all_models.png"), dpi=150)
    fig_pr.savefig(os.path.join(OUT_DIR, "pr_all_models.png"), dpi=150)
    plt.show()
    print(f"Saved: {OUT_DIR}/roc_all_models.png")
    print(f"Saved: {OUT_DIR}/pr_all_models.png")

    plot_metric_comparison(results)

    print("\n===== Summary (masked test) =====")
    for name, m in results.items():
        print(f"{name:10s} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Checkpoint 3 — does the diagnosis stay the same under plausible completions?
# ---------------------------------------------------------------------------
def evaluate_plausible_stability(system, prepared, n_samples: int = None, split: str = "Test"):
    """
    Steps 7-14, as a standalone check: for each masked patient in
    `split`, generate M CVAE plausible completions, re-diagnose every
    one, and report Class Stability (CS_t), Probability Stability
    (1 - PD_t), and DSS_t — i.e. whether filling in the unknowns with
    plausible guesses changes the diagnosis.
    """
    n_samples = n_samples or CFG.cvae.n_samples_M
    X, y, m = _splits(prepared)[split]
    X_obs = X * m

    # Step 4-6: the masked-only diagnosis (same as checkpoint 1's Fused/{split} row)
    p_xgb = system.xgb.predict_proba_y1(X_obs, m)
    p_tr = system.tabresnet.predict_proba_y1(X_obs, m)
    p_t = fuse(p_xgb, p_tr, system.fusion_weights)
    y_hat_t = predict_label(p_t)

    # Step 7-8: M plausible completions
    plausible = system.cvae.generate_plausible_profiles(X_obs, m, n_samples)  # (M, N, F)
    full_mask = np.ones_like(m)

    p_plaus = np.zeros((n_samples, len(X)))
    y_plaus = np.zeros((n_samples, len(X)), dtype=int)
    for i in range(n_samples):
        profile = plausible[i]
        p_xgb_i = system.xgb.predict_proba_y1(profile, full_mask)
        p_tr_i = system.tabresnet.predict_proba_y1(profile, full_mask)
        p_plaus[i] = fuse(p_xgb_i, p_tr_i, system.fusion_weights)
        y_plaus[i] = predict_label(p_plaus[i])

    # Steps 12-14
    cs_t, pd_t, dss_t = evaluate_stability_and_dss(p_t, y_hat_t, p_plaus, y_plaus)

    masked_acc = (y_hat_t == y).mean()

    print(f"\n===== Plausible-completion stability ({split}) =====")
    print(f"Masked-input diagnosis accuracy:      {masked_acc:.4f}")
    print(f"Mean Class Stability (CS_t):          {cs_t.mean():.4f}  "
          f"(fraction whose diagnosis didn't flip across {n_samples} plausible profiles)")
    print(f"Mean Probability Stability (1-PD_t):  {(1 - pd_t).mean():.4f}")
    print(f"Mean DSS_t:                           {dss_t.mean():.4f}")

    return {"CS_t": cs_t, "PD_t": pd_t, "DSS_t": dss_t, "y_hat_t": y_hat_t, "p_t": p_t}
