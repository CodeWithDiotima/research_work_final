import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit(
        "LightGBM is missing. Run: python -m pip install lightgbm"
    )

# ------------------------------------------------------------
# 1. Locate the project and import its dataset configuration
# ------------------------------------------------------------
cwd = Path.cwd()

if (cwd / "dss_system" / "config.py").exists():
    module_dir = cwd / "dss_system"
elif (cwd / "config.py").exists():
    module_dir = cwd
else:
    raise SystemExit(
        "Run this script from the project root containing dss_system, "
        "or from inside the dss_system directory."
    )

sys.path.insert(0, str(module_dir))

from config import FEATURE_COLUMNS, TARGET_COLUMN
from data_processing import load_raw_dataframe

# ------------------------------------------------------------
# 2. Load the original, unbalanced dataset
# ------------------------------------------------------------
csv_path = sys.argv[1] if len(sys.argv) > 1 else None

try:
    df = load_raw_dataframe(csv_path)
except Exception as exc:
    if not csv_path:
        raise SystemExit(
            "Could not load the dataset from UCI. Check your internet "
            "connection and install ucimlrepo if necessary:\n"
            "python -m pip install ucimlrepo\n"
            f"Original error: {exc}"
        ) from exc

    # Fallback for local CSV files without the configured ID column.
    try:
        df = pd.read_csv(csv_path)
    except Exception as csv_exc:
        raise SystemExit(
            f"Could not read CSV: {csv_exc}"
        ) from csv_exc

df = df.reset_index(drop=True)

required_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise SystemExit(
        f"Dataset is missing required columns: {missing_columns}"
    )

y = pd.to_numeric(
    df[TARGET_COLUMN], errors="raise"
).to_numpy().astype(int)

if set(np.unique(y)) != {0, 1}:
    raise SystemExit(
        f"Expected binary labels 0 and 1, got {np.unique(y)}. "
        "Check that the dataset uses Diabetes_binary."
    )

# ------------------------------------------------------------
# 3. Create a writable feature array
# ------------------------------------------------------------
# IMPORTANT: .copy() fixes the read-only NumPy array error.
# The original feature order is preserved.
X = (
    df[FEATURE_COLUMNS]
    .apply(pd.to_numeric, errors="raise")
    .to_numpy(dtype=np.float32)
    .copy()
)

# Convert infinite values to NaN for safe preprocessing.
X[~np.isfinite(X)] = np.nan

if len(X) != len(y):
    raise ValueError("Feature and target row counts do not match.")

# ------------------------------------------------------------
# 4. Stratified 70/15/15 split BEFORE any class weighting
# ------------------------------------------------------------
all_idx = np.arange(len(y))

train_idx, remainder_idx = train_test_split(
    all_idx,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

val_idx, test_idx = train_test_split(
    remainder_idx,
    test_size=0.50,
    random_state=43,
    stratify=y[remainder_idx],
)

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

# ------------------------------------------------------------
# 5. Impute using training data statistics only
# ------------------------------------------------------------
medians = np.nanmedian(X[train_idx], axis=0)
medians = np.where(np.isfinite(medians), medians, 0.0)

# X is writable because of the copy() above.
rows, cols = np.where(~np.isfinite(X))
X[rows, cols] = medians[cols]

Xtr, ytr = X[train_idx], y[train_idx]
Xv, yv = X[val_idx], y[val_idx]
Xte, yte = X[test_idx], y[test_idx]


def report_split(name, labels):
    counts = np.bincount(labels, minlength=2)
    proportions = counts / len(labels)

    print(
        f"{name:10} n={len(labels):7d} | "
        f"class 0={counts[0]:7d} | "
        f"class 1={counts[1]:7d} | "
        f"proportions={proportions}"
    )


print("\n" + "=" * 65)
print("ORIGINAL DATA: STRATIFIED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 65)

report_split("Original", y)
report_split("Train", ytr)
report_split("Validation", yv)
report_split("Test", yte)

print(f"\nNumber of original features: {X.shape[1]}")

test_counts = np.bincount(yte, minlength=2)
majority_baseline = test_counts.max() / len(yte)

print(
    "Majority-class baseline accuracy: "
    f"{majority_baseline:.4f}"
)

# ------------------------------------------------------------
# 6. Generate independent observation masks
# ------------------------------------------------------------
# Each record has between 20% and 60% of its features observed.
# A separate seed is used for each split.
def make_masks(n_samples, n_features, seed):
    rng = np.random.default_rng(seed)

    masks = np.zeros(
        (n_samples, n_features),
        dtype=np.float32,
    )

    for i in range(n_samples):
        observed_fraction = rng.uniform(0.20, 0.60)

        n_observed = max(
            1,
            int(round(observed_fraction * n_features)),
        )

        chosen = rng.choice(
            n_features,
            size=n_observed,
            replace=False,
        )

        masks[i, chosen] = 1.0

    return masks


mtr = make_masks(len(Xtr), X.shape[1], seed=101)
mv = make_masks(len(Xv), X.shape[1], seed=202)
mte = make_masks(len(Xte), X.shape[1], seed=303)


def masked_input(values, mask):
    observed_values = values * mask

    return np.concatenate(
        [observed_values, mask],
        axis=1,
    ).astype(np.float32)


Xtr_masked = masked_input(Xtr, mtr)
Xv_masked = masked_input(Xv, mv)
Xte_masked = masked_input(Xte, mte)

print("\nMean observed-feature proportions (single mask):")
print(f"Train      : {mtr.mean():.3f}")
print(f"Validation : {mv.mean():.3f}")
print(f"Test       : {mte.mean():.3f}")

# ------------------------------------------------------------
# 6b. Mask augmentation: K random masks per training patient
# ------------------------------------------------------------
MASK_AUG_K = 5  # number of different masks per training record

# Tile training features and labels K times
Xtr_aug = np.tile(Xtr, (MASK_AUG_K, 1))          # (K*N_train, F)
ytr_aug = np.tile(ytr, MASK_AUG_K)                # (K*N_train,)

# Generate K*N_train independent masks (each patient sees K different patterns)
mtr_aug = make_masks(len(Xtr_aug), X.shape[1], seed=999)
Xtr_masked_aug = masked_input(Xtr_aug, mtr_aug)

print(f"\nMask augmentation (K={MASK_AUG_K}):")
print(f"  Augmented training rows : {len(Xtr_aug):,} "
      f"(original {len(Xtr):,} x {MASK_AUG_K})")
print(f"  Augmented mean observed : {mtr_aug.mean():.3f}")

# ------------------------------------------------------------
# 7. Helpers for probabilities and metrics
# ------------------------------------------------------------
def get_positive_probability(model, values):
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Positive class 1 is missing from model classes: {classes}"
        )

    positive_col = classes.index(1)

    return model.predict_proba(values)[:, positive_col]


def calculate_metrics(labels, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "sensitivity": recall_score(
            labels, predictions, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            labels, probabilities
        ),
        "average_precision": average_precision_score(
            labels, probabilities
        ),
        "brier_score": brier_score_loss(
            labels, probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ------------------------------------------------------------
# 8. Train and evaluate one experiment
# ------------------------------------------------------------
def run_experiment(
    name,
    train_x,
    val_x,
    test_x,
    class_weight,
    train_y=None,
):
    print("\n\n" + "=" * 65)
    print(f"TRAINING: {name}")
    print("=" * 65)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    # Use provided train_y if given (needed for augmented experiments),
    # otherwise fall back to the global ytr.
    fit_y = train_y if train_y is not None else ytr

    model.fit(
        train_x,
        fit_y,
        eval_set=[(val_x, yv)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=80,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    p_val = get_positive_probability(model, val_x)
    p_test = get_positive_probability(model, test_x)

    # Threshold is selected from validation data only.
    thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = max(
        thresholds,
        key=lambda threshold: (
            balanced_accuracy_score(
                yv,
                (p_val >= threshold).astype(int),
            ),
            -abs(threshold - 0.5),
        ),
    )

    rows_out = []

    for threshold_name, threshold in [
        ("default_0.50", 0.50),
        ("validation_tuned", best_threshold),
    ]:
        metrics = calculate_metrics(
            yte,
            p_test,
            threshold,
        )

        rows_out.append({
            "experiment": name,
            "threshold_method": threshold_name,
            **metrics,
        })

        predictions = (
            p_test >= threshold
        ).astype(int)

        print(f"\n{name} | {threshold_name}")
        print(f"Best boosting iteration: {model.best_iteration_}")
        print(f"Decision threshold: {threshold:.2f}")

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "macro_f1",
            "roc_auc",
            "average_precision",
            "brier_score",
        ]:
            print(f"{metric:22}: {metrics[metric]:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(
            yte,
            predictions,
            labels=[0, 1],
        ))

        print("\nClassification report:")
        print(classification_report(
            yte,
            predictions,
            labels=[0, 1],
            target_names=["No Diabetes", "Diabetes"],
            zero_division=0,
        ))

    return rows_out


# ------------------------------------------------------------
# 9. Run the three controlled experiments
# ------------------------------------------------------------
results = []

# Experiment A: all 21 features, no class weighting.
results += run_experiment(
    "Full features - unweighted",
    Xtr,
    Xv,
    Xte,
    None,
)

# Experiment B: all 21 features, balanced class weights.
results += run_experiment(
    "Full features - balanced weights",
    Xtr,
    Xv,
    Xte,
    "balanced",
)

# Experiment C: partial records plus mask indicators (1 mask per patient).
results += run_experiment(
    "Masked (1 mask) - balanced weights",
    Xtr_masked,
    Xv_masked,
    Xte_masked,
    "balanced",
)

# Experiment D: mask augmentation — K=5 masks per training patient.
results += run_experiment(
    f"Masked (K={MASK_AUG_K} aug) - balanced weights",
    Xtr_masked_aug,
    Xv_masked,     # val/test stay single-mask (fair evaluation)
    Xte_masked,
    "balanced",
    train_y=ytr_aug,
)

# ------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------
result_df = pd.DataFrame(results)

out_path = cwd / "dss_diagnostic_results.csv"

result_df.to_csv(
    out_path,
    index=False,
)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(
    result_df[
        [
            "experiment",
            "threshold_method",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "macro_f1",
            "roc_auc",
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {out_path}")
print("\nYour existing project source files were not modified.")
