"""
graph.py
========
Step 16 — Clinical information graph  G = (V, E, W)
Step 17 — Feature relevance  R_j
Step 25 — Graph connectivity  Conn_j

    A_ij = Assoc(X_i, X_j)                       (association measure)
    W_ij = (A_ij - W_min) / (W_max - W_min + eps) (normalised edge weight)

    R_j = Relevance(X_j, Y)
    R~_j = (R_j - R_min) / (R_max - R_min + eps)  (normalised relevance)

    Conn_j = (1/|O_t|) sum_{i in O_t} W_ij
    Conn~_j = (Conn_j - Conn_min) / (Conn_max - Conn_min + eps)

Association/relevance are computed from the TRAINING data only (fit
once, reused throughout the adaptive-acquisition loop).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from config import CFG, FEATURE_COLUMNS


def _mutual_info_association(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    A_ij = Assoc(X_i, X_j), estimated as normalised mutual information
    between (binned) feature pairs. Returns an (F, F) symmetric matrix.
    """
    n, F = X.shape
    # Discretise each feature into bins for a stable MI estimate.
    binned = np.zeros_like(X, dtype=np.int64)
    for j in range(F):
        col = X[:, j]
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-6
        binned[:, j] = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)

    A = np.zeros((F, F), dtype=np.float64)
    for i in range(F):
        for j in range(i, F):
            mi = _mutual_information(binned[:, i], binned[:, j], n_bins)
            A[i, j] = A[j, i] = mi
    return A


def _mutual_information(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    joint = np.zeros((n_bins, n_bins))
    for x, y in zip(a, b):
        joint[x, y] += 1
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (pa * pb + 1e-12)
        log_ratio = np.where(joint > 0, np.log(ratio + 1e-12), 0.0)
    return float(np.sum(joint * log_ratio))


def _normalize(values: np.ndarray, eps: float = CFG.dss.epsilon) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    return (values - vmin) / (vmax - vmin + eps)


@dataclass
class ClinicalGraph:
    feature_names: List[str]
    A: np.ndarray          # raw association matrix, shape (F, F)
    W: np.ndarray          # normalised edge weights, shape (F, F)
    R: np.ndarray          # raw relevance, shape (F,)
    R_norm: np.ndarray     # normalised relevance R~_j, shape (F,)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def connectivity(self, observed_mask: np.ndarray) -> np.ndarray:
        """
        Step 25: Conn_j = (1/|O_t|) sum_{i in O_t} W_ij, normalised to Conn~_j.

        observed_mask: boolean array of shape (F,), True where feature i
        is currently observed (i.e. i in O_t).
        """
        O_t = np.where(observed_mask)[0]
        if len(O_t) == 0:
            conn = np.zeros(self.n_features)
        else:
            conn = self.W[:, O_t].mean(axis=1)
        return _normalize(conn)


def build_clinical_graph(X_train: np.ndarray, y_train: np.ndarray,
                          feature_names: List[str] = FEATURE_COLUMNS,
                          n_bins: int = 10) -> ClinicalGraph:
    """
    Fits the clinical information graph G = (V, E, W) and feature
    relevance R_j from the training split.

    R_j uses mutual information between X_j and the label Y (same
    estimator family as the edge-association measure, for consistency).
    """
    A = _mutual_info_association(X_train, n_bins=n_bins)
    W = _normalize(A)  # normalised over the whole matrix; diagonal excluded below
    np.fill_diagonal(W, 0.0)

    F = X_train.shape[1]
    R = np.zeros(F)
    y_binned = y_train.astype(np.int64)
    for j in range(F):
        col = X_train[:, j]
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-6
        col_binned = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)
        R[j] = _mutual_information(col_binned, y_binned, max(n_bins, 2))
    R_norm = _normalize(R)

    return ClinicalGraph(feature_names=feature_names, A=A, W=W, R=R, R_norm=R_norm)
