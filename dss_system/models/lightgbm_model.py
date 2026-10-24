"""
models/lightgbm_model.py
========================
Single Primary Diagnostic Classifier for DSS/DSG System:

    f_LGBM(x_obs, m) -> p_t = P_LGBM(Y=1 | x_obs, m)

LightGBM operates on tabular clinical data with partial observability.
Features are prepared by:
1. Encoding unobserved entries (m <= 0.5) as np.nan (derived strictly from
   the observation mask, preserving actual 0.0 values as true observed readings).
2. Appending the explicit binary observation mask m as auxiliary feature indicators,
   giving an augmented representation of width 2 * F = 42 features.
3. Leveraging LightGBM's native NaN handling and class balancing
   (class_weight="balanced") to handle the natural CDC prevalence distribution.
4. Validation threshold tuning to optimize balanced classification accuracy.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from lightgbm import (
    LGBMClassifier as LGBMCore,
    early_stopping,
    record_evaluation,
    log_evaluation,
)

try:
    from config import CFG, LightGBMConfig
except ImportError:
    from dss_system.config import CFG, LightGBMConfig


def _prepare_features(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Constructs the 2*F feature matrix [x_nan | m].
    
    - x_nan: entries where m <= 0.5 are set to np.nan; observed entries keep their value.
    - m: explicit binary indicators so trees can split on feature presence directly.
    """
    X_obs = np.asarray(X_obs, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    
    if X_obs.ndim == 1:
        X_obs = X_obs[np.newaxis, :]
    if m.ndim == 1:
        m = m[np.newaxis, :]

    X_nan = X_obs.copy()
    X_nan[m <= 0.5] = np.nan
    augmented = np.concatenate([X_nan, m], axis=1)
    return augmented


class LightGBMClassifierWrapper:
    """
    Single primary diagnostic classifier wrapping LightGBM LGBMClassifier.
    Serves as f_LGBM for the entire DSS/DSG clinical acquisition pipeline.
    """

    def __init__(self, cfg: LightGBMConfig | None = None):
        self.cfg = cfg if cfg is not None else CFG.lightgbm
        self.model: LGBMCore | None = None
        self.history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        self.classes_: np.ndarray = np.array([0, 1])
        self.best_threshold_: float = 0.5

    def fit(
        self,
        X_obs_train: np.ndarray,
        m_train: np.ndarray,
        y_train: np.ndarray,
        X_obs_val: np.ndarray,
        m_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "LightGBMClassifierWrapper":
        """
        Fits LightGBM on masked training data with validation monitoring and early stopping.
        """
        Xtr = _prepare_features(X_obs_train, m_train)
        Xva = _prepare_features(X_obs_val, m_val)
        y_tr = np.asarray(y_train, dtype=np.int64)
        y_va = np.asarray(y_val, dtype=np.int64)

        self.model = LGBMCore(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            max_depth=self.cfg.max_depth,
            min_child_samples=self.cfg.min_child_samples,
            subsample=self.cfg.subsample,
            subsample_freq=self.cfg.subsample_freq,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            class_weight=self.cfg.class_weight,
            verbosity=self.cfg.verbosity,
            objective="binary",
        )

        evals_result: dict = {}
        callbacks = [
            early_stopping(stopping_rounds=self.cfg.early_stopping_rounds, verbose=False),
            record_evaluation(evals_result),
        ]
        if self.cfg.verbosity > 0:
            callbacks.append(log_evaluation(period=50))
        else:
            callbacks.append(log_evaluation(period=0))

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*The argument 'eval_set' is deprecated.*")
            self.model.fit(
                Xtr,
                y_tr,
                eval_set=[(Xtr, y_tr), (Xva, y_va)],
                eval_names=["learn", "val"],
                eval_metric=["binary_logloss", "binary_error"],
                callbacks=callbacks,
            )

        self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))

        # Extract training / validation history for learning curves
        learn_evals = evals_result.get("learn", {})
        val_evals = evals_result.get("val", {})

        loss = learn_evals.get("binary_logloss", [])
        val_loss = val_evals.get("binary_logloss", [])

        learn_err = learn_evals.get("binary_error", [])
        val_err = val_evals.get("binary_error", [])

        acc = [1.0 - float(e) for e in learn_err]
        val_acc = [1.0 - float(e) for e in val_err]

        self.history = {
            "loss": [float(v) for v in loss],
            "val_loss": [float(v) for v in val_loss],
            "accuracy": acc if acc else [0.0] * len(loss),
            "val_accuracy": val_acc if val_acc else [0.0] * len(val_loss),
        }

        # Tune decision threshold on validation set to maximize balanced accuracy
        self.tune_threshold(X_obs_val, m_val, y_val, metric="balanced_accuracy")
        return self

    def tune_threshold(
        self,
        X_obs_val: np.ndarray,
        m_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "balanced_accuracy",
    ) -> float:
        """
        Sweeps decision threshold tau in [0.05, 0.95] on validation set to optimize metric.
        """
        probs = self.predict_proba_y1(X_obs_val, m_val)
        y_true = np.asarray(y_val, dtype=np.int64)

        thresholds = np.linspace(0.05, 0.95, 91)
        best_score = -1.0
        best_th = 0.5

        for th in thresholds:
            preds = (probs >= th).astype(int)
            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, preds)
            elif metric == "f1":
                score = f1_score(y_true, preds, zero_division=0)
            else:
                score = np.mean(preds == y_true)

            if score > best_score:
                best_score = score
                best_th = float(th)

        self.best_threshold_ = best_th
        return best_th

    def predict_proba_y1(self, X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Returns p_LGBM = P(Y=1 | x_obs, m) for each row as a 1D float array of shape (N,).
        Performs batching to handle large datasets efficiently without memory spikes.
        """
        assert self.model is not None, "LightGBMClassifierWrapper must be fit() before calling predict_proba_y1"
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

    def predict(
        self,
        X_obs: np.ndarray,
        m: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Binary classification predictions based on threshold (defaults to tuned threshold)."""
        th = threshold if threshold is not None else self.best_threshold_
        return (self.predict_proba_y1(X_obs, m) >= th).astype(int)

    def save(self, path: str) -> None:
        """Saves LightGBM model and wrapper state to disk using joblib."""
        assert self.model is not None, "No fitted model to save"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model,
            "classes_": self.classes_,
            "history": self.history,
            "best_threshold_": self.best_threshold_,
            "cfg": self.cfg,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> "LightGBMClassifierWrapper":
        """Loads LightGBM model and wrapper state from disk."""
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.classes_ = payload.get("classes_", np.array([0, 1]))
            self.history = payload.get("history", {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []})
            self.best_threshold_ = float(payload.get("best_threshold_", 0.5))
            if "cfg" in payload:
                self.cfg = payload["cfg"]
        else:
            # Fallback if raw LGBMClassifier was saved directly
            self.model = payload
            self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))
            self.best_threshold_ = 0.5
        return self


# Backward-compatible alias
LightGBMClassifier = LightGBMClassifierWrapper
