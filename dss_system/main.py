"""
main.py
=======
Demonstrates the complete end-to-end clinical workflow:
1. Dataset Preparation: Loads CDC Diabetes dataset with stratified 70/15/15 split.
2. System Training: Trains LightGBM (f_LGBM), CVAE, and builds the Clinical Information Graph.
3. Evaluation Checkpoints:
   - Checkpoint 1: Evaluates LightGBM classification accuracy under partial observation masks.
   - Checkpoint 1b: Evaluates Experiment A (Full features) vs Experiment B (Masked features).
   - Checkpoint 2: Generates confusion matrices, ROC/PR curves, and performance plots.
   - Checkpoint 3: Tests decision stability under CVAE generative completions.
4. Adaptive Acquisition Simulation:
   - Takes a demo cohort of patients starting with incomplete medical profiles.
   - Iteratively queries next-best clinical features based on Decision Sufficiency (DSS).
   - Stops dynamically when confidence threshold is satisfied, reporting final cost and diagnosis.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Ensure module path resolution
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import CFG, FEATURE_COLUMNS
from data_processing import prepare_dataset, sample_observation_masks, apply_mask
from pipeline import train_full_system, run_adaptive_acquisition
from train import evaluate_test_accuracy
from evaluate import (
    evaluate_system,
    evaluate_plausible_stability,
    evaluate_masked_accuracy,
    evaluate_experiments_a_and_b,
    plot_training_curves,
)


def main(data_path: str = None, n_demo_patients: int = 20):
    """
    Executes the full training, evaluation, and adaptive clinical acquisition demo.
    
    Parameters:
        data_path: Optional path to a local CSV file. If None, fetches from UCI ML repository.
        n_demo_patients: Number of patient cases to simulate in the adaptive acquisition loop.
    """
    # -----------------------------------------------------------------------
    # Step 1: Prepare and balance the clinical dataset
    # -----------------------------------------------------------------------
    print("Preparing dataset (loading, stratified 70/15/15 split, standardizing continuous features)...")
    prepared = prepare_dataset(path=data_path)
    print(f"Dataset ready: train={prepared.X_train.shape}, val={prepared.X_val.shape}, test={prepared.X_test.shape}\n")

    # -----------------------------------------------------------------------
    # Step 2: Train all components of the clinical DSS system
    # -----------------------------------------------------------------------
    print("Beginning system training...")
    system = train_full_system(prepared)
    print("System training complete.\n")

    # Plot LightGBM training history (loss and accuracy curves)
    if hasattr(system.lightgbm, "history") and system.lightgbm.history:
        plot_training_curves(system.lightgbm.history, title_prefix="LightGBM")

    # Calculate baseline static accuracy on the test set using initially available features
    acc = evaluate_test_accuracy(system, prepared)
    th = getattr(system.lightgbm, "best_threshold_", 0.5)
    print(f"LightGBM static test accuracy (masked, threshold={th:.2f}): {acc:.4f}\n")

    # -----------------------------------------------------------------------
    # Checkpoint 1: Masked Accuracy Analysis
    # -----------------------------------------------------------------------
    print("Checkpoint 1: Evaluating masked accuracy across splits...")
    evaluate_masked_accuracy(system, prepared)

    # -----------------------------------------------------------------------
    # Checkpoint 1b: Formal Experiments A and B (Full Features vs Masked Features)
    # -----------------------------------------------------------------------
    print("\nCheckpoint 1b: Evaluating Experiment A (Full) vs Experiment B (Masked)...")
    evaluate_experiments_a_and_b(system, prepared)

    # -----------------------------------------------------------------------
    # Checkpoint 2: Detailed Performance Metrics & Visualizations
    # -----------------------------------------------------------------------
    print("\nCheckpoint 2: Evaluating system metrics & generating test-set plots...")
    evaluate_system(system, prepared)

    # -----------------------------------------------------------------------
    # Checkpoint 3: Counterfactual Stability under CVAE Completions
    # -----------------------------------------------------------------------
    print("\nCheckpoint 3: Evaluating plausible stability under CVAE...")
    evaluate_plausible_stability(system, prepared, split="Test")

    # -----------------------------------------------------------------------
    # Step 4: Decision-Sufficiency-Guided Adaptive Acquisition Demo
    # -----------------------------------------------------------------------
    n = min(n_demo_patients, prepared.X_test.shape[0])
    X_true = prepared.X_test[:n]
    m0 = sample_observation_masks(n, n_features=len(FEATURE_COLUMNS), cfg=CFG.mask)
    X_obs0 = apply_mask(X_true, m0)

    print(f"\nSimulating adaptive acquisition loop on {n} demo patients...")
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