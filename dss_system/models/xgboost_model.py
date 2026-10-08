"""
models/xgboost_model.py
========================
Step 4 — XGBoost classifier.

    f_XGB(x_obs, m) -> p_XGB = P_XGB(Y=1 | x_obs, m)

XGBoost sees the masked feature vector x_obs (unobserved entries are
filled, e.g. with 0 after z-scoring) concatenated with the mask m
itself, so the model can distinguish "feature is 0" from "feature is
missing".
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from config import CFG


def _with_mask(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Concatenate [x_obs | m] so missingness is an explicit input signal."""
    return np.concatenate([X_obs, m.astype(np.float32)], axis=1)


class XGBClassifier:
    def __init__(self, cfg=CFG.xgb):
        self.cfg = cfg
        self.model: xgb.XGBClassifier | None = None
        self.history: dict[str, list[float]] = {}

    def fit(self, X_obs_train, m_train, y_train, X_obs_val, m_val, y_val):
        Xtr = _with_mask(X_obs_train, m_train)
        Xva = _with_mask(X_obs_val, m_val)

        self.model = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            objective=self.cfg.objective,
            eval_metric=["error", "logloss"],
            random_state=self.cfg.random_state,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            n_jobs=-1,
        )
        self.model.fit(
            Xtr, y_train,
            eval_set=[(Xtr, y_train), (Xva, y_val)],
            verbose=False,
        )
        evals = self.model.evals_result()
        tr = evals["validation_0"]
        va = evals["validation_1"]
        self.history = {
            "loss": tr.get("logloss", []),
            "val_loss": va.get("logloss", []),
            "accuracy": [1.0 - e for e in tr.get("error", [])],
            "val_accuracy": [1.0 - e for e in va.get("error", [])],
        }
        return self

    def predict_proba_y1(self, X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Returns p_XGB = P(Y=1 | x_obs, m) for each row."""
        assert self.model is not None, "call .fit() first"
        X = _with_mask(X_obs, m)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        assert self.model is not None
        self.model.save_model(path)

    def load(self, path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        return self
