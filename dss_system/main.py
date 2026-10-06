"""
main.py
=======
Demonstrates the complete Decision-Sufficiency-Guided adaptive
acquisition loop end to end:

    train the system (train.py) -> take a batch of test patients with
    only a few features initially observed -> run_adaptive_acquisition
    -> print each patient's final diagnosis, DSS, and which features
    were acquired along the way.
"""

from __future__ import annotations

import numpy as np

from config import CFG, FEATURE_COLUMNS
from data_processing import prepare_dataset, sample_observation_masks, apply_mask
from pipeline import train_full_system, run_adaptive_acquisition
from train import evaluate_fused_test_accuracy


def main(data_path: str = None, n_demo_patients: int = 20):
    prepared = prepare_dataset(path=data_path)
    system = train_full_system(prepared)

    acc = evaluate_fused_test_accuracy(system, prepared)
    print(f"Fused static test accuracy: {acc:.4f}\n")

    # Take a small batch of test patients, start each with a sparse
    # observation mask (Step 3), and let the loop decide what to acquire.
    n = min(n_demo_patients, prepared.X_test.shape[0])
    X_true = prepared.X_test[:n]
    m0 = sample_observation_masks(n, n_features=len(FEATURE_COLUMNS), cfg=CFG.mask)
    X_obs0 = apply_mask(X_true, m0)

    outcomes = run_adaptive_acquisition(X_true, X_obs0, m0, system, cfg=CFG)

    correct = 0
    for i, o in enumerate(outcomes):
        true_label = int(prepared.y_test[i])
        correct += int(o.y_hat == true_label)
        print(f"Patient {i:02d} | y_true={true_label} y_hat={o.y_hat} "
              f"p_t={o.p_t:.3f} DSS={o.dss:.3f} "
              f"acquisitions={o.n_acquisitions} cost={o.total_cost:.2f} "
              f"features_acquired={o.acquired_features}")

    print(f"\nAdaptive-loop accuracy on demo batch: {correct}/{n} = {correct/n:.3f}")


if __name__ == "__main__":
    main()
