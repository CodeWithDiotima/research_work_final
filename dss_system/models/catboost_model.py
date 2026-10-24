"""
models/catboost_model.py
========================
Single Primary Diagnostic Classifier for DSS/DSG System:

    f_CB(x_obs, m) -> p_t = P_CB(Y=1 | x_obs, m)

CatBoost operates on tabular clinical data with partial observability.
Features are prepared by:
1. Encoding unobserved entries (m <= 0.5) as np.nan (derived strictly from
   the observation mask, preserving actual 0.0 values as true observed readings).
2. Appending the explicit binary observation mask m as auxiliary feature indicators,
   giving an augmented representation of width 2 * F = 42 features.
3. Leveraging CatBoost's native NaN handling (nan_mode="Min") and class balancing
   (auto_class_weights="Balanced") to handle the natural prevalence distribution.
"""

from __future__ import annotations

import os
import numpy as np
from catboost import CatBoostClassifier as CatBoostCore

try:
    from config import CFG, CatBoostConfig
except ImportError:
    from dss_system.config import CFG, CatBoostConfig


def _prepare_features(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Constructs the 2*F feature matrix [x_nan | m].
    
    - x_nan: entries where m <= 0.5 are set to np.nan; observed entries keep their value.
    - m: explicit binary indicators so trees can split on feature presence directly.
    """
    X_obs = np.asarray(X_obs, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    
    is_1d = False
    if X_obs.ndim == 1:
        is_1d = True
        X_obs = X_obs[np.newaxis, :]
    if m.ndim == 1:
        m = m[np.newaxis, :]

    X_nan = X_obs.copy()
    X_nan[m <= 0.5] = np.nan
    augmented = np.concatenate([X_nan, m], axis=1)
    return augmented


class CatBoostClassifier:
    """
    Single primary diagnostic classifier wrapping CatBoostCore.
    Serves as f_CB for the entire DSS/DSG clinical acquisition pipeline.
    """

    def __init__(self, cfg: CatBoostConfig | None = None):
        self.cfg = cfg if cfg is not None else CFG.catboost
        self.model: CatBoostCore | None = None
        self.history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X_obs_train: np.ndarray,
        m_train: np.ndarray,
        y_train: np.ndarray,
        X_obs_val: np.ndarray,
        m_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "CatBoostClassifier":
        """
        Fits CatBoost on masked training data with validation monitoring and early stopping.
        """
        Xtr = _prepare_features(X_obs_train, m_train)
        Xva = _prepare_features(X_obs_val, m_val)
        y_tr = np.asarray(y_train, dtype=np.int64)
        y_va = np.asarray(y_val, dtype=np.int64)

        self.model = CatBoostCore(
            iterations=self.cfg.iterations,
            learning_rate=self.cfg.learning_rate,
            depth=self.cfg.depth,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            random_seed=self.cfg.random_seed,
            thread_count=self.cfg.thread_count,
            auto_class_weights=self.cfg.auto_class_weights,
            loss_function=self.cfg.loss_function,
            eval_metric=self.cfg.eval_metric,
            nan_mode=self.cfg.nan_mode,
            custom_metric=["Accuracy"],
            verbose=self.cfg.verbose,
        )

        self.model.fit(
            Xtr,
            y_tr,
            eval_set=[(Xtr, y_tr), (Xva, y_va)],
            use_best_model=True,
            verbose=self.cfg.verbose,
        )

        self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))

        # Extract training / validation history for learning curves
        evals = getattr(self.model, "evals_result_", {}) or {}
        learn_evals = evals.get("learn", {}) or evals.get("validation_0", {})
        val_evals = evals.get("validation", {}) or evals.get("validation_1", {})

        loss = learn_evals.get("Logloss", [])
        val_loss = val_evals.get("Logloss", [])

        acc = (
            learn_evals.get("Accuracy:use_weights=false")
            or learn_evals.get("Accuracy:use_weights=true")
            or learn_evals.get("Accuracy", [])
        )
        val_acc = (
            val_evals.get("Accuracy:use_weights=false")
            or val_evals.get("Accuracy:use_weights=true")
            or val_evals.get("Accuracy", [])
        )

        self.history = {
            "loss": [float(v) for v in loss],
            "val_loss": [float(v) for v in val_loss],
            "accuracy": [float(v) for v in acc] if acc else [0.0] * len(loss),
            "val_accuracy": [float(v) for v in val_acc] if val_acc else [0.0] * len(val_loss),
        }
        return self

    def predict_proba_y1(self, X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Returns p_CB = P(Y=1 | x_obs, m) for each row as a 1D float array of shape (N,).
        Performs batching to handle large datasets efficiently without memory spikes.
        """
        assert self.model is not None, "CatBoostClassifier must be fit() before calling predict_proba_y1"
        X = _prepare_features(X_obs, m)
        n_samples = X.shape[0]

        # Determine class 1 index
        pos_idx = 1
        if self.classes_ is not None and len(self.classes_) > 1:
            matches = np.where(self.classes_ == 1)[0]
            if len(matches) > 0:
                pos_idx = int(matches[0])

        batch_size = 2048
        if n_samples <= batch_size:
            probs = self.model.predict_proba(X)
            return probs[:, pos_idx].astype(np.float64)

        probs_list = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            chunk = X[start_idx:end_idx]
            p_chunk = self.model.predict_proba(chunk)[:, pos_idx]
            probs_list.append(p_chunk)

        return np.concatenate(probs_list, axis=0).astype(np.float64)

    def predict(self, X_obs: np.ndarray, m: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary classification predictions based on threshold."""
        return (self.predict_proba_y1(X_obs, m) >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Saves CatBoost model to disk."""
        assert self.model is not None, "No fitted model to save"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> "CatBoostClassifier":
        """Loads CatBoost model from disk."""
        self.model = CatBoostCore()
        self.model.load_model(path)
        self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))
        return self
