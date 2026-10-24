"""
train.py
========
Entry point for training the static components of the Clinical DSS System:
1. Loads and prepares the CDC Diabetes dataset (balancing, splitting, standardizing).
2. Trains the XGBoost classifier (f_XGB).
3. Trains the TabResNet neural network (f_TR).
4. Trains the Conditional Variational Autoencoder (CVAE) missing-value generator.
5. Optimizes Bayesian fusion weights (w_XGB, w_TR) on validation data.
6. Constructs the Clinical Information Graph G = (V, E, W).
7. Evaluates and prints static fused classification accuracy on the test set.

Usage:
------
    # Download dataset from UCI ML Repository automatically and train:
    python train.py

    # Or provide a local CSV file path:
    python train.py --data path/to/cdc_diabetes.csv
"""

# Enable modern Python 3.7+ type annotation behaviors
from __future__ import annotations

# Import argparse to handle command-line flags (e.g., --data)
import argparse
# Import NumPy for numerical array operations and class distribution counts
import numpy as np

# Import dataset preparation function (handles loading, balancing, splitting, scaling)
from data_processing import prepare_dataset
# Import pipeline orchestrator that trains all models and builds the graph
from pipeline import train_full_system
# Import ensemble fusion math: p_t = w1*p1 + w2*p2, and decision thresholding
from fusion import fuse, predict_label


def evaluate_fused_test_accuracy(system, prepared) -> float:
    """
    Evaluates the accuracy of the combined (fused) ensemble model on the held-out test set.
    
    Parameters:
        system: TrainedSystem container containing trained models and fusion weights.
        prepared: PreparedData container holding test features, masks, and ground-truth labels.
        
    Returns:
        float: Accuracy score between 0.0 and 1.0 (e.g., 0.85 = 85% correct).
    """
    # Simulate partial observability: element-wise multiply test features by the observation mask
    # Features where m == 0 are zeroed out (treated as missing/unobserved)
    X_obs_test = prepared.X_test * prepared.m_test

    # Step 4: Predict diabetes risk probability P(y=1) using the trained XGBoost model
    p_xgb = system.xgb.predict_proba_y1(X_obs_test, prepared.m_test)
    # print(prepared.m_test)
    # Step 5: Predict diabetes risk probability P(y=1) using the trained TabResNet neural network
    p_tr = system.tabresnet.predict_proba_y1(X_obs_test, prepared.m_test)

    # Step 6: Fuse the two probability predictions using optimal validation-learned weights
    # Formula: p_t = (w_XGB * p_XGB) + (w_TR * p_TR)
    p_t = fuse(p_xgb, p_tr, system.fusion_weights)

    # Convert continuous probabilities into discrete binary class predictions (0 or 1)
    # y_hat = 1 if p_t >= 0.5, else 0
    y_hat = predict_label(p_t)

    # Compare predictions against actual ground-truth labels (prepared.y_test)
    # (y_hat == prepared.y_test) creates a boolean array of True/False
    # .mean() computes the fraction of True predictions (accuracy)
    return float((y_hat == prepared.y_test).mean())


def main():
    """Main CLI execution routine."""
    # Initialize the argument parser for command line execution
    parser = argparse.ArgumentParser(description="Train DSS Clinical Acquisition Models")
    
    # Optional argument allowing user to specify a local CSV dataset file path
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="Optional path to local CDC Diabetes Health Indicators CSV file"
    )
    
    # Parse command line arguments provided by the user
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Data Preparation & Preprocessing
    # -----------------------------------------------------------------------
    print("[1/3] Preparing dataset (undersample -> exact balance -> 70/15/15 split)...")
    # Loads dataset, performs exact 50/50 majority undersampling, splits 70/15/15,
    # standardizes continuous columns, and samples initial observation masks
    prepared = prepare_dataset(path=args.data)
    
    # Print the shape/dimensions of each split (number of patients, number of features)
    print(f"  train={prepared.X_train.shape}  val={prepared.X_val.shape}  "
          f"test={prepared.X_test.shape}")
    
    # Print the exact count of 0s (no diabetes) and 1s (diabetes) to verify exact balance
    print(f"  train class balance: {np.bincount(prepared.y_train)}")

    # -----------------------------------------------------------------------
    # Step 2: System Training
    # -----------------------------------------------------------------------
    print("[2/3] Training f_XGB, f_TR, CVAE, fitting fusion weights, building graph...")
    # Fits XGBoost, TabResNet, CVAE, calculates optimal fusion weights, and builds clinical graph
    system = train_full_system(prepared)
    
    # Display the learned ensemble weights showing relative contribution of each model
    print(f"  fusion weights: w_XGB={system.fusion_weights.w_xgb:.3f} "
          f"w_TR={system.fusion_weights.w_tr:.3f}")
          
    from extra_plots import plot_training_history

    # 2-panel figure: Loss (left) + Accuracy (right)
    plot_training_history(system.tabresnet.history, model_name="TabResNet")
    plot_training_history(system.xgb.history, model_name="XGBoost")

    # -----------------------------------------------------------------------
    # Step 3: Evaluation on Unseen Test Patients
    # -----------------------------------------------------------------------
    print("[3/3] Evaluating fused classifier on the held-out test set...")
    # Compute accuracy of the fused system under partially observed test data
    acc = evaluate_fused_test_accuracy(system, prepared)
    print(f"  fused test accuracy: {acc:.4f}")

    # Return trained system bundle and preprocessed data objects
    return system, prepared


# Standard Python convention: execute main() only if script is run directly
if __name__ == "__main__":
    main()

