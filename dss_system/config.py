"""
config.py
=========
Central configuration for the Decision-Sufficiency-Guided Adaptive
Clinical Information Acquisition System (DSS/DSG pipeline).

Every symbol here maps directly onto a symbol in the mathematical
specification (e.g., "Step 6: Fusion", "Step 14: DSS calculation")
so the code and the mathematical theory stay easily traceable.
"""

# Import dataclass utilities for clean, structured, type-annotated configuration classes
from dataclasses import dataclass, field
# Import List type hint for type annotations on feature lists
from typing import List


# ===========================================================================
# 1. DATASET SCHEMA (CDC Diabetes Health Indicators Dataset)
# ===========================================================================
# Unique patient identifier column name
ID_COLUMN = "ID"

# Ground truth binary target column name: 0 = No diabetes, 1 = Pre-diabetes / Diabetes
TARGET_COLUMN = "Diabetes_binary"

# List of all 21 clinical feature names (F = 21, corresponding to Step 1 of the spec)
# Each feature represents a clinical test, survey question, or physical measurement
FEATURE_COLUMNS: List[str] = [
    "HighBP",               # High Blood Pressure (0 = No, 1 = Yes)
    "HighChol",             # High Cholesterol (0 = No, 1 = Yes)
    "CholCheck",            # Cholesterol check in past 5 years (0 = No, 1 = Yes)
    "BMI",                  # Body Mass Index (numerical continuous value)
    "Smoker",               # Smoked at least 100 cigarettes in lifetime (0 = No, 1 = Yes)
    "Stroke",               # Ever told had a stroke (0 = No, 1 = Yes)
    "HeartDiseaseorAttack", # Coronary heart disease or myocardial infarction (0 = No, 1 = Yes)
    "PhysActivity",         # Physical activity in past 30 days (0 = No, 1 = Yes)
    "Fruits",               # Consume fruit 1 or more times per day (0 = No, 1 = Yes)
    "Veggies",              # Consume vegetables 1 or more times per day (0 = No, 1 = Yes)
    "HvyAlcoholConsump",    # Heavy alcohol consumption (0 = No, 1 = Yes)
    "AnyHealthcare",        # Has any kind of healthcare coverage (0 = No, 1 = Yes)
    "NoDocbcCost",          # Needed to see doctor in past 12 months but could not due to cost (0 = No, 1 = Yes)
    "GenHlth",              # General health rating (scale 1: excellent to 5: poor)
    "MentHlth",             # Days of poor mental health in past 30 days (0 to 30)
    "PhysHlth",             # Days of physical illness/injury in past 30 days (0 to 30)
    "DiffWalk",             # Serious difficulty walking or climbing stairs (0 = No, 1 = Yes)
    "Sex",                  # Sex of the patient (0 = Female, 1 = Male)
    "Age",                  # 13-level age category (1 = 18-24, ..., 13 = 80+)
    "Education",            # Education level scale (1 to 6)
    "Income",               # Income scale (1: <$10k to 8: >=$75k)
]

# Continuous features normalized via z-score standardization (Step 2)
# Formula: z = (x - mean) / std
CONTINUOUS_FEATURES: List[str] = ["BMI", "MentHlth", "PhysHlth"]

# Categorical and ordinal features encoded as normalized numbers or one-hot vectors (Step 2)
CATEGORICAL_FEATURES: List[str] = [c for c in FEATURE_COLUMNS if c not in CONTINUOUS_FEATURES]

# Total number of features available in the dataset (F = 21)
N_FEATURES = len(FEATURE_COLUMNS)


# ===========================================================================
# 2. DATA SPLIT & SAMPLING CONFIGURATION
# ===========================================================================
@dataclass
class SplitConfig:
    """Configures dataset splitting and class balancing (Steps 1-3)."""
    # 70% of data allocated for training models (f_XGB, f_TR, CVAE)
    train_frac: float = 0.70
    # 15% of data for validation (hyperparameter tuning & Bayesian fusion weights)
    val_frac: float = 0.15
    # 15% of data held-out for final model evaluation & adaptive testing
    test_frac: float = 0.15
    random_state: int = 42
    # Stratified split on natural prevalence by default; class balance handled by CatBoost auto_class_weights
    exact_balance: bool = False


# ===========================================================================
# 3. OBSERVATION MASK CONFIGURATION
# ===========================================================================
@dataclass
class MaskConfig:
    """Configures observation mask generation m in {0,1}^F (Step 3)."""
    # Minimum fraction of features that are initially observed/known for a patient (20%)
    min_observed_frac: float = 0.20
    # Maximum fraction of features that are initially observed/known for a patient (60%)
    max_observed_frac: float = 0.60
    # Random seed for reproducible mask generation
    random_state: int = 42


# ===========================================================================
# 4. CONDITIONAL VARIATIONAL AUTOENCODER (CVAE) CONFIGURATION
# ===========================================================================
@dataclass
class CVAEConfig:
    """
    Hyperparameters for the generative models g_1..g_F (Steps 7-8).
    The CVAE models P(x_missing | x_observed) to generate M plausible complete profiles.
    """
    # Dimensionality of the latent Gaussian bottleneck space z ~ N(mu, sigma^2)
    latent_dim: int = 16
    # Number of neurons in each hidden layer of the encoder and decoder
    hidden_dim: int = 128
    # Number of hidden layers in both encoder and decoder neural networks
    n_hidden_layers: int = 2
    # Dropout rate to prevent overfitting in the neural network
    dropout: float = 0.1
    # Learning rate for the Adam optimizer during CVAE training
    lr: float = 1e-3
    # Number of full training epochs over the dataset
    epochs: int = 100
    # Number of patient profiles processed in one gradient update step
    batch_size: int = 256
    # Weight of KL divergence in ELBO loss (Loss = Reconstruction + beta * KL)
    beta: float = 1.0
    # Number of plausible complete profiles M sampled per patient to assess stability (Step 8)
    n_samples_M: int = 100


# ===========================================================================
# 5. CATBOOST CLASSIFIER CONFIGURATION (f_CB - Primary Diagnostic Classifier)
# ===========================================================================
@dataclass
class CatBoostConfig:
    """
    Hyperparameters for the CatBoost gradient boosted decision trees classifier.
    Serves as the single primary diagnostic classifier predicting diabetes probability
    p_t = p_catboost given observed features and observation mask.
    """
    # Maximum number of boosting trees / iterations to build
    iterations: int = 500
    # Learning rate (shrinkage factor) applied to each tree's contribution
    learning_rate: float = 0.05
    # Maximum depth of each decision tree
    depth: int = 6
    # L2 regularization term on leaf weights to prevent overfitting
    l2_leaf_reg: float = 3.0
    # Stops training early if validation loss does not improve for 30 consecutive rounds
    early_stopping_rounds: int = 30
    # Random seed for reproducible tree construction
    random_seed: int = 42
    # Number of CPU threads used (-1 uses all available threads)
    thread_count: int = -1
    # Automatically compute class weights to handle natural CDC class imbalance (~13.9% positive)
    auto_class_weights: str = "Balanced"
    # Loss metric used for optimization
    loss_function: str = "Logloss"
    # Evaluation metric used to monitor validation performance
    eval_metric: str = "Logloss"
    # How numerical NaNs are treated by the split search ('Min' puts NaNs in the smallest bucket)
    nan_mode: str = "Min"
    # Logging verbosity
    verbose: bool = False


# ===========================================================================
# 5b. XGBOOST CLASSIFIER CONFIGURATION (Legacy / Reference)
# ===========================================================================
@dataclass
class XGBConfig:
    """
    Hyperparameters for the gradient boosted decision trees classifier (Step 4).
    Predicts diabetes probability p_XGB given observed features and observation mask.
    """
    # Maximum number of boosting trees to build
    n_estimators: int = 500
    # Maximum depth of each decision tree (controls interaction complexity)
    max_depth: int = 6
    # Learning rate (shrinkage factor) applied to each tree's contribution
    learning_rate: float = 0.05
    # Fraction of training samples randomly selected to train each tree
    subsample: float = 0.8
    # Fraction of features randomly sampled for building each tree
    colsample_bytree: float = 0.8
    # L2 regularization term on leaf weights to combat overfitting
    reg_lambda: float = 1.0
    # Binary classification objective function outputting probabilities
    objective: str = "binary:logistic"
    # Evaluation metric used to monitor validation performance
    eval_metric: str = "logloss"
    # Random seed for reproducible tree construction
    random_state: int = 42
    # Stops training early if validation loss does not improve for 30 consecutive rounds
    early_stopping_rounds: int = 30


# ===========================================================================
# 6. TABRESNET DEEP LEARNING CLASSIFIER CONFIGURATION (f_TR)
# ===========================================================================
@dataclass
class TabResNetConfig:
    """
    Hyperparameters for the Residual Multi-Layer Perceptron classifier (Step 5).
    Predicts diabetes probability p_TR using residual skip-connections over tabular data.
    """
    # Number of hidden units in each linear projection layer
    hidden_dim: int = 128
    # Number of residual blocks in the network architecture
    n_blocks: int = 4
    # Dropout probability inside each residual block
    dropout: float = 0.2
    # Learning rate for AdamW optimizer
    lr: float = 1e-3
    # Weight decay (L2 penalty) to prevent weight explosion
    weight_decay: float = 1e-5
    # Maximum number of training epochs
    epochs: int = 100
    # Mini-batch size during training
    batch_size: int = 256
    # Number of epochs without validation loss improvement before early stopping triggers
    patience: int = 15


# ===========================================================================
# 7. MODEL FUSION CONFIGURATION (Step 6)
# ===========================================================================
@dataclass
class FusionConfig:
    """
    Configures Bayesian ensemble fusion: p_t = w_XGB * p_XGB + w_TR * p_TR (Step 6).
    Combines the complementary strengths of gradient boosting and deep learning.
    """
    # Initial prior weight for the XGBoost model
    w_xgb: float = 0.5
    # Initial prior weight for the TabResNet neural network
    w_tr: float = 0.5
    # When True, searches for optimal weights (w_XGB, w_TR) on the validation set
    learn_weights_on_val: bool = True


# ===========================================================================
# 8. DECISION SUFFICIENCY SCORE (DSS) CONFIGURATION (Steps 12-15)
# ===========================================================================
@dataclass
class DSSConfig:
    """
    Controls decision stability and the acquisition stopping rule (Steps 12-15).
    DSS_t = alpha * CS_t + (1 - alpha) * (1 - PD_t)
    """
    # Weight balancing Classification Stability (CS) and Predictive Divergence (PD)
    # alpha = 0.5 weights both metrics equally
    alpha: float = 0.5
    # Stopping threshold tau_DSS: if DSS_t >= tau_DSS, the current evidence is deemed sufficient
    tau_dss: float = 0.85
    # Tiny epsilon constant to prevent division-by-zero or log(0) in calculations
    epsilon: float = 1e-8
    # Maximum number of clinical features allowed to be acquired per patient
    max_acquisitions: int = 15


# ===========================================================================
# 9. FEATURE ACQUISITION SCORING CONFIGURATION (Steps 16-27)
# ===========================================================================
@dataclass
class AcquisitionConfig:
    """
    Hyperparameters for ranking candidate features via acquisition score S_j:
    S_j = lambda_1 * R_j + lambda_2 * Conn_j + lambda_3 * DSG_j - lambda_4 * C_j
    """
    # lambda_1: Weight for Feature Relevance R_j (Mutual Information with diagnosis)
    lambda_1: float = 0.35
    # lambda_2: Weight for Graph Connectivity Conn_j (Degree in Clinical Graph)
    lambda_2: float = 0.20
    # lambda_3: Weight for Decision Sufficiency Gain DSG_j (Expected improvement in DSS)
    lambda_3: float = 0.35
    # lambda_4: Penalty weight for Acquisition Cost C_j (Invasiveness/Financial cost)
    lambda_4: float = 0.10

    # Multi-component cost weights: C_j = eta_a * C_acq + eta_c * C_comm + eta_e * C_comp
    # eta_a: Weight on physical/clinical acquisition burden (e.g., blood draw invasiveness)
    eta_a: float = 0.50
    # eta_c: Weight on communication latency / transmission overhead
    eta_c: float = 0.25
    # eta_e: Weight on computational complexity to evaluate the feature
    eta_e: float = 0.25

    # K_j: Number of candidate values sampled per unobserved feature to estimate expected DSS (Step 18)
    n_candidate_values_K: int = 5


# ===========================================================================
# 10. ROOT PIPELINE CONFIGURATION
# ===========================================================================
@dataclass
class PipelineConfig:
    """Aggregates all individual configuration modules into a single unified object."""
    # Data splitting and balancing configuration
    split: SplitConfig = field(default_factory=SplitConfig)
    # Feature masking configuration
    mask: MaskConfig = field(default_factory=MaskConfig)
    # Generative CVAE model configuration
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    # Primary diagnostic classifier configuration (CatBoost f_CB)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    # XGBoost classifier configuration (legacy / reference)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    # TabResNet neural network configuration
    tabresnet: TabResNetConfig = field(default_factory=TabResNetConfig)
    # Model ensemble fusion configuration
    fusion: FusionConfig = field(default_factory=FusionConfig)
    # Decision sufficiency and stopping criteria configuration
    dss: DSSConfig = field(default_factory=DSSConfig)
    # Next-best-feature acquisition scoring configuration
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    # Computation hardware device: 'cpu' or 'cuda' (for GPU acceleration)
    device: str = "cpu"


# Instantiate the global singleton configuration object used throughout the codebase
CFG = PipelineConfig()

