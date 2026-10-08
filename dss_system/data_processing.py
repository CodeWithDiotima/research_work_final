"""
data_processing.py
===================
Implements Steps 1-3 of the uploaded specification:

  Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N
  Step 2 — Preprocessing               z_ij = (x_ij - mu_j) / (sigma_j + eps)
                                        x_ij -> phi_j(x_ij)   (categorical)
  Step 3 — Partial information         m_j in {0,1}, x_obs = x ⊙ m

Also implements:
  * Undersampling to EXACT class balance before the 70/15/15 split.
  * The observation-mask sampler used to simulate partial patient records.

This file is dataset-agnostic boilerplate: point `load_raw_dataframe`
at the real CDC Diabetes Health Indicators CSV when it is available.
Everything downstream (splits, encoders, masks) works off the schema
declared in config.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import (
    CFG,
    ID_COLUMN,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
)


# ---------------------------------------------------------------------------
# Step 1 — Load raw dataset  D = {(x_i, y_i)}
# ---------------------------------------------------------------------------
UCI_DATASET_ID = 891  # CDC Diabetes Health Indicators
# https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators


def load_from_ucimlrepo() -> pd.DataFrame:
    """
    Fetch the CDC Diabetes Health Indicators dataset directly from the
    UCI ML Repository (id=891) via the `ucimlrepo` package, and combine
    it into a single dataframe with ID_COLUMN, TARGET_COLUMN and
    FEATURE_COLUMNS — matching what the rest of the pipeline expects.

    Requires: pip install ucimlrepo
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError(
            "ucimlrepo is required to auto-fetch the dataset. "
            "Install it with: pip install ucimlrepo"
        ) from e

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.reset_index(drop=True)   # 21 feature columns
    y = dataset.data.targets.reset_index(drop=True)     # Diabetes_binary

    df = X.copy()
    # y comes back as a 1-column dataframe; normalise its column name.
    target_col_in_y = y.columns[0]
    df[TARGET_COLUMN] = y[target_col_in_y].to_numpy()

    # ucimlrepo's `ids` block (patient ID) isn't always populated; fall
    # back to a positional ID if it's missing.
    ids = getattr(dataset.data, "ids", None)
    if ids is not None and len(ids) == len(df):
        id_col_in_ids = ids.columns[0]
        df[ID_COLUMN] = ids[id_col_in_ids].to_numpy()
    else:
        df[ID_COLUMN] = np.arange(len(df))

    missing = set([ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]) - set(df.columns)
    if missing:
        raise ValueError(f"ucimlrepo dataset is missing expected columns: {missing}")

    return df[[ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]]


def load_raw_dataframe(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw CDC Diabetes Health Indicators dataframe.

    Expected columns: ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS.

    - If `path` is given, loads that CSV directly (e.g. a local export
      of the dataset, such as Kaggle's
      diabetes_binary_health_indicators_BRFSS2015.csv).
    - If `path` is None, fetches the dataset live from the UCI ML
      Repository (id=891) via `ucimlrepo` — this matches
      https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
    """
    if path is None:
        return load_from_ucimlrepo()

    df = pd.read_csv(path)
    missing = set([ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Exact-balance undersampling (performed BEFORE the 70/15/15 split)
# ---------------------------------------------------------------------------
def undersample_exact_balance(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly undersample the majority class so that the positive and
    negative class counts are EXACTLY equal.
    """
    pos = df[df[TARGET_COLUMN] == 1]
    neg = df[df[TARGET_COLUMN] == 0]
    n = min(len(pos), len(neg))
    pos_bal = pos.sample(n=n, random_state=random_state)
    neg_bal = neg.sample(n=n, random_state=random_state)
    balanced = pd.concat([pos_bal, neg_bal], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    assert (balanced[TARGET_COLUMN] == 1).sum() == (balanced[TARGET_COLUMN] == 0).sum()
    return balanced


# ---------------------------------------------------------------------------
# 70 / 15 / 15 split
# ---------------------------------------------------------------------------
@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def train_val_test_split(df: pd.DataFrame, cfg=CFG.split) -> Splits:
    assert abs(cfg.train_frac + cfg.val_frac + cfg.test_frac - 1.0) < 1e-9

    df = df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * cfg.train_frac))
    n_val = int(round(n * cfg.val_frac))

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return Splits(train=train_df.reset_index(drop=True),
                   val=val_df.reset_index(drop=True),
                   test=test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Step 2 — Preprocessing: encoding + normalization
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Fits on TRAIN ONLY, then transforms train/val/test consistently.

    Continuous j:   z_ij = (x_ij - mu_j) / (sigma_j + eps)
    Categorical j:  x_ij -> phi_j(x_ij)   (here: integer / one-hot codebook)
    """

    def __init__(self,
                 continuous_features=CONTINUOUS_FEATURES,
                 categorical_features=CATEGORICAL_FEATURES,
                 eps: float = 1e-8):
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.eps = eps
        self.mu_: Dict[str, float] = {}
        self.sigma_: Dict[str, float] = {}
        self.phi_: Dict[str, Dict] = {}  # categorical value -> code, per feature

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for j in self.continuous_features:
            self.mu_[j] = float(df[j].mean())
            self.sigma_[j] = float(df[j].std())
        for j in self.categorical_features:
            uniques = sorted(df[j].dropna().unique().tolist())
            self.phi_[j] = {val: code for code, val in enumerate(uniques)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for j in self.continuous_features:
            out[j] = (out[j] - self.mu_[j]) / (self.sigma_[j] + self.eps)
        for j in self.categorical_features:
            codebook = self.phi_[j]
            unseen = max(codebook.values(), default=-1) + 1
            out[j] = out[j].map(codebook).fillna(unseen).astype(int)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def build_feature_matrix(df: pd.DataFrame,
                          feature_columns=FEATURE_COLUMNS,
                          target_column=TARGET_COLUMN) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy(dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Step 3 — Partial information / observation mask
# ---------------------------------------------------------------------------
def sample_observation_masks(n_samples: int,
                              n_features: int = len(FEATURE_COLUMNS),
                              cfg=CFG.mask) -> np.ndarray:
    """
    Sample observation masks m in {0,1}^F, one row per patient.

    m_j = 1 -> feature j available (observed)
    m_j = 0 -> feature j unavailable (missing / not yet acquired)

    Each patient's observed fraction is drawn uniformly from
    [min_observed_frac, max_observed_frac], simulating the state of a
    partially-acquired record at some point in the acquisition loop.
    """
    rng = np.random.default_rng(cfg.random_state)
    masks = np.zeros((n_samples, n_features), dtype=np.int64)
    for i in range(n_samples):
        frac = rng.uniform(cfg.min_observed_frac, cfg.max_observed_frac)
        k = max(1, int(round(frac * n_features)))
        idx = rng.choice(n_features, size=k, replace=False)
        masks[i, idx] = 1
    return masks


def apply_mask(X: np.ndarray, m: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    x_obs = x ⊙ m  (unobserved entries filled with `fill_value`, since raw
    NaNs are inconvenient for most downstream models; the mask array `m`
    itself is what actually encodes "missing" and should always be passed
    alongside x_obs to any model/consumer).
    """
    X_obs = X.copy()
    X_obs[m == 0] = fill_value
    return X_obs


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    preprocessor: Preprocessor
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    m_train: np.ndarray
    m_val: np.ndarray
    m_test: np.ndarray


def prepare_dataset(path: Optional[str] = None,
                     df: Optional[pd.DataFrame] = None) -> PreparedData:
    """
    Full Step 1-3 pipeline:
      load -> exact-balance undersample -> 70/15/15 split -> fit/transform
      preprocessing on train only -> sample observation masks per split.
    """
    if df is None:
        df = load_raw_dataframe(path)

    balanced = undersample_exact_balance(df, random_state=CFG.split.random_state)
    splits = train_val_test_split(balanced, cfg=CFG.split)

    pre = Preprocessor().fit(splits.train)
    train_p = pre.transform(splits.train)
    val_p = pre.transform(splits.val)
    test_p = pre.transform(splits.test)

    X_train, y_train = build_feature_matrix(train_p)
    X_val, y_val = build_feature_matrix(val_p)
    X_test, y_test = build_feature_matrix(test_p)

    m_train = sample_observation_masks(len(X_train), cfg=CFG.mask)
    m_val = sample_observation_masks(len(X_val), cfg=CFG.mask)
    m_test = sample_observation_masks(len(X_test), cfg=CFG.mask)

    return PreparedData(
        preprocessor=pre,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        m_train=m_train, m_val=m_val, m_test=m_test,
    )


if __name__ == "__main__":
    # Smoke test with synthetic data matching the schema, since the real
    # dataset is not wired in yet.
    rng = np.random.default_rng(0)
    n = 2000
    synth = pd.DataFrame({ID_COLUMN: np.arange(n)})
    for j in CONTINUOUS_FEATURES:
        synth[j] = rng.normal(25, 5, size=n)
    for j in CATEGORICAL_FEATURES:
        synth[j] = rng.integers(0, 5, size=n)
    synth[TARGET_COLUMN] = rng.integers(0, 2, size=n)

    prepared = prepare_dataset(df=synth)
    print("X_train:", prepared.X_train.shape, "y_train balance:",
          np.bincount(prepared.y_train))
    print("mask example:", prepared.m_train[0])
    final_dataset= prepare_dataset()
    # print("X_train",final_dataset.X_train.shape)
    # print("y_train",final_dataset.y_train.shape)
    # print("m_train",final_dataset.m_train[0][0])
    # print("X_val",final_dataset.X_val.shape)
    # print("X_val",final_dataset.y_val.shape)
    # print("m_val",final_dataset.m_val.shape)
    # print("X_test",final_dataset.X_test.shape)
    # print("y_test",final_dataset.y_test.shape)
    # print("m_test",final_dataset.m_test.shape)