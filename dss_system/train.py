"""
train.py
========
Entry point: prepare the dataset (Steps 1-3), train f_XGB, f_TR, and the
CVAE generator, fit fusion weights, build the clinical graph, and report
static test-set accuracy for the fused classifier (independent of the
adaptive-acquisition loop, which is exercised separately in main.py).

Usage:

    # Fetches live from the UCI ML Repository (id=891) — requires
    # `pip install ucimlrepo` (already in requirements.txt):
    python train.py

    # Or point at a local CSV export of the same dataset instead:
    python train.py --data path/to/cdc_diabetes.csv
"""

from __future__ import annotations

import argparse
import numpy as np

from data_processing import prepare_dataset
from pipeline import train_full_system
from fusion import fuse, predict_label


def evaluate_fused_test_accuracy(system, prepared) -> float:
    X_obs_test = prepared.X_test * prepared.m_test
    p_xgb = system.xgb.predict_proba_y1(X_obs_test, prepared.m_test)
    p_tr = system.tabresnet.predict_proba_y1(X_obs_test, prepared.m_test)
    p_t = fuse(p_xgb, p_tr, system.fusion_weights)
    y_hat = predict_label(p_t)
    return float((y_hat == prepared.y_test).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to the CDC Diabetes Health Indicators file")
    args = parser.parse_args()

    print("[1/3] Preparing dataset (undersample -> exact balance -> 70/15/15 split)...")
    prepared = prepare_dataset(path=args.data)
    print(f"  train={prepared.X_train.shape}  val={prepared.X_val.shape}  "
          f"test={prepared.X_test.shape}")
    print(f"  train class balance: {np.bincount(prepared.y_train)}")

    print("[2/3] Training f_XGB, f_TR, CVAE, fitting fusion weights, building graph...")
    system = train_full_system(prepared)
    print(f"  fusion weights: w_XGB={system.fusion_weights.w_xgb:.3f} "
          f"w_TR={system.fusion_weights.w_tr:.3f}")

    print("[3/3] Evaluating fused classifier on the held-out test set...")
    acc = evaluate_fused_test_accuracy(system, prepared)
    print(f"  fused test accuracy: {acc:.4f}")

    return system, prepared


if __name__ == "__main__":
    main()
