# """
# main.py
# =======
# Demonstrates the complete end-to-end clinical workflow:
# 1. Dataset Preparation: Loads and balances the CDC Diabetes dataset.
# 2. System Training: Trains XGBoost, TabResNet, CVAE, fusion weights, and clinical graph.
# 3. Evaluation Checkpoints:
#    - Checkpoint 1: Evaluates classification accuracy under partial observation masks.
#    - Checkpoint 2: Generates confusion matrices, ROC/PR curves, and comparison plots.
#    - Checkpoint 3: Tests decision stability under CVAE generative completions.
# 4. Adaptive Acquisition Simulation:
#    - Takes a demo cohort of patients starting with incomplete medical profiles.
#    - Iteratively queries next-best clinical features based on Decision Sufficiency (DSS).
#    - Stops dynamically when confidence threshold is satisfied, reporting final cost and diagnosis.
# """

# # Enable modern Python 3.7+ type annotations
# from __future__ import annotations

# # Import NumPy for array manipulation and mathematical operations
# import numpy as np

# # Import configuration constants (hyperparameters and feature column names)
# from config import CFG, FEATURE_COLUMNS
# # Import data processing helpers: dataset split, mask generator, and masking operator
# from data_processing import prepare_dataset, sample_observation_masks, apply_mask
# # Import core pipeline functions: full model trainer and the adaptive acquisition loop
# from pipeline import train_full_system, run_adaptive_acquisition
# # Import baseline static test evaluation function
# from train import evaluate_fused_test_accuracy
# # Import evaluation checkpoints and plotting routines
# from evaluate import evaluate_system, evaluate_plausible_stability, evaluate_masked_accuracy


# def main(data_path: str = None, n_demo_patients: int = 20):
#     """
#     Executes the full training, evaluation, and adaptive clinical acquisition demo.
    
#     Parameters:
#         data_path: Optional path to a local CSV file. If None, fetches from UCI ML repository.
#         n_demo_patients: Number of patient cases to simulate in the adaptive acquisition loop.
#     """
#     # -----------------------------------------------------------------------
#     # Step 1: Prepare and balance the clinical dataset
#     # -----------------------------------------------------------------------
#     print("Preparing dataset (loading, undersampling to 50/50 balance, 70/15/15 split)...")
#     # Loads raw records, enforces 50/50 balance, standardizes columns, and splits into train/val/test
#     prepared = prepare_dataset(path=data_path)
#     # Output split dimensions (number of patients x number of features)
#     print(f"Dataset ready: train={prepared.X_train.shape}, val={prepared.X_val.shape}, test={prepared.X_test.shape}\n")

#     # -----------------------------------------------------------------------
#     # Step 2: Train all components of the clinical DSS system
#     # -----------------------------------------------------------------------
#     print("Beginning system training...")
#     # Fits f_XGB, f_TR, CVAE generator, fits ensemble fusion weights, and constructs the clinical graph
#     system = train_full_system(prepared)
#     print("System training complete.\n")
    
#     # Print underlying XGBoost model representation
#     print(system.xgb)
    
#     # Calculate baseline static accuracy on the test set using initially available features
#     acc = evaluate_fused_test_accuracy(system, prepared)
#     print(f"Fused static test accuracy: {acc:.4f}\n")

#     # -----------------------------------------------------------------------
#     # Checkpoint 1: Masked Accuracy Analysis
#     # -----------------------------------------------------------------------
#     print("Checkpoint 1: Evaluating masked accuracy across splits...")
#     # Tests how well XGBoost, TabResNet, and the Fused model perform when features are partially missing
#     evaluate_masked_accuracy(system, prepared)

#     # -----------------------------------------------------------------------
#     # Checkpoint 2: Detailed Performance Metrics & Visualizations
#     # -----------------------------------------------------------------------
#     print("\nCheckpoint 2: Evaluating system metrics & generating test-set plots...")
#     # Computes Precision, Recall, F1, ROC-AUC, PR-AUC, and exports diagnostic plots to evaluation_plots/
#     evaluate_system(system, prepared)

#     # -----------------------------------------------------------------------
#     # Checkpoint 3: Counterfactual Stability under CVAE Completions
#     # -----------------------------------------------------------------------
#     print("\nCheckpoint 3: Evaluating plausible stability under CVAE...")
#     # Assesses whether completing missing patient values via CVAE keeps the diagnosis stable
#     evaluate_plausible_stability(system, prepared, split="Test")

#     # -----------------------------------------------------------------------
#     # Step 4: Decision-Sufficiency-Guided Adaptive Acquisition Demo
#     # -----------------------------------------------------------------------
#     # Select a cohort of test patients (e.g., 20 patients)
#     n = min(n_demo_patients, prepared.X_test.shape[0])
    
#     # Extract their true, complete clinical profiles (used as ground-truth when acquiring new tests)
#     X_true = prepared.X_test[:n]
    
#     # Generate initial sparse observation masks (e.g., simulating patients with only 20-60% known features)
#     m0 = sample_observation_masks(n, n_features=len(FEATURE_COLUMNS), cfg=CFG.mask)
    
#     # Apply initial mask: known features retain real values; unobserved features are set to 0.0
#     X_obs0 = apply_mask(X_true, m0)

#     # Run the adaptive acquisition loop: dynamically selects which tests to order for each patient
#     outcomes = run_adaptive_acquisition(X_true, X_obs0, m0, system, cfg=CFG)

#     # Track how many patients receive an accurate final diagnosis
#     correct = 0
#     for i, o in enumerate(outcomes):
#         # Extract ground truth label (0 = healthy, 1 = diabetic)
#         true_label = int(prepared.y_test[i])
        
#         # Check if model prediction matches actual diagnosis
#         correct += int(o.y_hat == true_label)
        
#         # Display clinical decision trajectory for this patient:
#         # - y_true: actual disease status
#         # - y_hat: model's final diagnosis
#         # - p_t: predicted risk probability
#         # - DSS: final Decision Sufficiency Score (confidence level)
#         # - acquisitions: number of new tests performed
#         # - cost: total clinical cost accumulated
#         # - features_acquired: specific tests chosen dynamically
#         print(f"Patient {i:02d} | y_true={true_label} y_hat={o.y_hat} "
#               f"p_t={o.p_t:.3f} DSS={o.dss:.3f} "
#               f"acquisitions={o.n_acquisitions} cost={o.total_cost:.2f} "
#               f"features_acquired={o.acquired_features}")

#     # Calculate and display overall diagnostic accuracy achieved by the adaptive acquisition process
#     print(f"\nAdaptive-loop accuracy on demo batch: {correct}/{n} = {correct/n:.3f}")


# # Standard entry point: run main() when script is executed directly from terminal
# if __name__ == "__main__":
#     main()

"""
main.py
=======
Demonstrates the complete end-to-end clinical workflow:
1. Dataset Preparation: Loads and balances the CDC Diabetes dataset.
2. System Training: Trains XGBoost, TabResNet, CVAE, fusion weights, and clinical graph.
3. Evaluation Checkpoints:
   - Checkpoint 1: Evaluates classification accuracy under partial observation masks.
   - Checkpoint 2: Generates confusion matrices, ROC/PR curves, and comparison plots.
   - Checkpoint 3: Tests decision stability under CVAE generative completions.
4. Adaptive Acquisition Simulation:
   - Takes a demo cohort of patients starting with incomplete medical profiles.
   - Iteratively queries next-best clinical features based on Decision Sufficiency (DSS).
   - Stops dynamically when confidence threshold is satisfied, reporting final cost and diagnosis.
"""

# Enable modern Python 3.7+ type annotations
from __future__ import annotations

# Import NumPy for array manipulation and mathematical operations
import numpy as np

# Import configuration constants (hyperparameters and feature column names)
from config import CFG, FEATURE_COLUMNS
# Import data processing helpers: dataset split, mask generator, and masking operator
from data_processing import prepare_dataset, sample_observation_masks, apply_mask
# Import core pipeline functions: full model trainer and the adaptive acquisition loop
from pipeline import train_full_system, run_adaptive_acquisition
# Import baseline static test evaluation function
from train import evaluate_fused_test_accuracy
# Import evaluation checkpoints and plotting routines
from evaluate import evaluate_system, evaluate_plausible_stability, evaluate_masked_accuracy, plot_training_curves


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
    print("Preparing dataset (loading, undersampling to 50/50 balance, 70/15/15 split)...")
    # Loads raw records, enforces 50/50 balance, standardizes columns, and splits into train/val/test
    prepared = prepare_dataset(path=data_path)
    # Output split dimensions (number of patients x number of features)
    print(f"Dataset ready: train={prepared.X_train.shape}, val={prepared.X_val.shape}, test={prepared.X_test.shape}\n")

    # -----------------------------------------------------------------------
    # Step 2: Train all components of the clinical DSS system
    # -----------------------------------------------------------------------
    print("Beginning system training...")
    # Fits f_XGB, f_TR, CVAE generator, fits ensemble fusion weights, and constructs the clinical graph
    system = train_full_system(prepared)
    print("System training complete.\n")
    
    # Print underlying XGBoost model representation
    print(system.xgb)

    # Train-vs-validation loss/accuracy curves for both models (same history shape)
    plot_training_curves(system.xgb.history, title_prefix="XGBoost")
    plot_training_curves(system.tabresnet.history, title_prefix="TabResNet")
    
    # Calculate baseline static accuracy on the test set using initially available features
    acc = evaluate_fused_test_accuracy(system, prepared)
    print(f"Fused static test accuracy: {acc:.4f}\n")

    # -----------------------------------------------------------------------
    # Checkpoint 1: Masked Accuracy Analysis
    # -----------------------------------------------------------------------
    print("Checkpoint 1: Evaluating masked accuracy across splits...")
    # Tests how well XGBoost, TabResNet, and the Fused model perform when features are partially missing
    evaluate_masked_accuracy(system, prepared)

    # -----------------------------------------------------------------------
    # Checkpoint 2: Detailed Performance Metrics & Visualizations
    # -----------------------------------------------------------------------
    print("\nCheckpoint 2: Evaluating system metrics & generating test-set plots...")
    # Computes Precision, Recall, F1, ROC-AUC, PR-AUC, and exports diagnostic plots to evaluation_plots/
    evaluate_system(system, prepared)

    # -----------------------------------------------------------------------
    # Checkpoint 3: Counterfactual Stability under CVAE Completions
    # -----------------------------------------------------------------------
    print("\nCheckpoint 3: Evaluating plausible stability under CVAE...")
    # Assesses whether completing missing patient values via CVAE keeps the diagnosis stable
    evaluate_plausible_stability(system, prepared, split="Test")

    # -----------------------------------------------------------------------
    # Step 4: Decision-Sufficiency-Guided Adaptive Acquisition Demo
    # -----------------------------------------------------------------------
    # Select a cohort of test patients (e.g., 20 patients)
    n = min(n_demo_patients, prepared.X_test.shape[0])
    
    # Extract their true, complete clinical profiles (used as ground-truth when acquiring new tests)
    X_true = prepared.X_test[:n]
    
    # Generate initial sparse observation masks (e.g., simulating patients with only 20-60% known features)
    m0 = sample_observation_masks(n, n_features=len(FEATURE_COLUMNS), cfg=CFG.mask)
    
    # Apply initial mask: known features retain real values; unobserved features are set to 0.0
    X_obs0 = apply_mask(X_true, m0)

    # Run the adaptive acquisition loop: dynamically selects which tests to order for each patient
    outcomes = run_adaptive_acquisition(X_true, X_obs0, m0, system, cfg=CFG)

    # Track how many patients receive an accurate final diagnosis
    correct = 0
    for i, o in enumerate(outcomes):
        # Extract ground truth label (0 = healthy, 1 = diabetic)
        true_label = int(prepared.y_test[i])
        
        # Check if model prediction matches actual diagnosis
        correct += int(o.y_hat == true_label)
        
        # Display clinical decision trajectory for this patient:
        # - y_true: actual disease status
        # - y_hat: model's final diagnosis
        # - p_t: predicted risk probability
        # - DSS: final Decision Sufficiency Score (confidence level)
        # - acquisitions: number of new tests performed
        # - cost: total clinical cost accumulated
        # - features_acquired: specific tests chosen dynamically
        print(f"Patient {i:02d} | y_true={true_label} y_hat={o.y_hat} "
              f"p_t={o.p_t:.3f} DSS={o.dss:.3f} "
              f"acquisitions={o.n_acquisitions} cost={o.total_cost:.2f} "
              f"features_acquired={o.acquired_features}")

    # Calculate and display overall diagnostic accuracy achieved by the adaptive acquisition process
    print(f"\nAdaptive-loop accuracy on demo batch: {correct}/{n} = {correct/n:.3f}")


# Standard entry point: run main() when script is executed directly from terminal
if __name__ == "__main__":
    main()