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


def evaluate_test_accuracy(system, prepared) -> float:
    """
    Evaluates the accuracy of the primary CatBoost diagnostic classifier on the held-out test set.
    
    Parameters:
        system: TrainedSystem container containing trained CatBoost model.
        prepared: PreparedData container holding test features, masks, and ground-truth labels.
        
    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    X_obs_test = prepared.X_test * prepared.m_test
    # Predict diabetes risk probability P(y=1) using the trained CatBoost model
    p_t = system.catboost.predict_proba_y1(X_obs_test, prepared.m_test)
    y_hat = predict_label(p_t)
    return float((y_hat == prepared.y_test).mean())


# Backward-compatible alias
evaluate_fused_test_accuracy = evaluate_test_accuracy


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
    print("[2/3] Training CatBoost classifier (f_CB), CVAE, building graph...")
    system = train_full_system(prepared)
          
    from extra_plots import plot_training_history

    # 2-panel figure: Loss (left) + Accuracy (right) for CatBoost
    plot_training_history(system.catboost.history, model_name="CatBoost")

    # -----------------------------------------------------------------------
    # Step 3: Evaluation on Unseen Test Patients
    # -----------------------------------------------------------------------
    print("[3/3] Evaluating CatBoost classifier on the held-out test set...")
    acc = evaluate_test_accuracy(system, prepared)
    print(f"  CatBoost test accuracy: {acc:.4f}")

    # Return trained system bundle and preprocessed data objects
    return system, prepared


# Standard Python convention: execute main() only if script is run directly
if __name__ == "__main__":
    main()

