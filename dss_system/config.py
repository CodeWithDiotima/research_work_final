"""
config.py
=========
Central configuration for the Decision-Sufficiency-Guided Adaptive
Clinical Information Acquisition System (DSS/DSG pipeline).

Every symbol here maps directly onto a symbol in the uploaded
mathematical specification (see docstring references, e.g. "Step 6",
"Step 14") so the code and the spec stay traceable to each other.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Dataset schema (CDC Diabetes Health Indicators — ID + target + 21 features)
# ---------------------------------------------------------------------------
ID_COLUMN = "ID"
TARGET_COLUMN = "Diabetes_binary"

# All 21 candidate features, x_i = [x_i1, ..., x_iF], F = 21 (Step 1)
FEATURE_COLUMNS: List[str] = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Which of the 21 features are treated as continuous (z-score, Step 2) vs
# categorical (encoded via phi_j, Step 2). Adjust freely — this is the only
# place that needs to change if the schema changes.
CONTINUOUS_FEATURES: List[str] = ["BMI", "MentHlth", "PhysHlth"]
CATEGORICAL_FEATURES: List[str] = [c for c in FEATURE_COLUMNS if c not in CONTINUOUS_FEATURES]

N_FEATURES = len(FEATURE_COLUMNS)  # F


@dataclass
class SplitConfig:
    """Undersampling + 70/15/15 split (per project notes)."""
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_state: int = 42
    # Undersampling must give EXACTLY equal positive/negative counts
    # before the split is taken.
    exact_balance: bool = True


@dataclass
class MaskConfig:
    """Partial-information / observation mask m_j in {0,1} (Step 3)."""
    # Fraction of features hidden per patient when simulating partial
    # observability for training/evaluation of the acquisition loop.
    min_observed_frac: float = 0.20
    max_observed_frac: float = 0.60
    random_state: int = 42


@dataclass
class CVAEConfig:
    """Conditional Generation Models g_1..g_F (Step 7-8)."""
    latent_dim: int = 16
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    beta: float = 1.0          # KL weight in the ELBO
    n_samples_M: int = 100     # M plausible profiles per Step 8


@dataclass
class XGBConfig:
    """XGBoost classifier f_XGB (Step 4)."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    random_state: int = 42
    early_stopping_rounds: int = 30


@dataclass
class TabResNetConfig:
    """TabResNet classifier f_TR (Step 5)."""
    hidden_dim: int = 128
    n_blocks: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 256
    patience: int = 15   # early-stopping patience on validation loss


@dataclass
class FusionConfig:
    """Prediction fusion p_t = w_XGB * p_XGB + w_TR * p_TR (Step 6)."""
    w_xgb: float = 0.5
    w_tr: float = 0.5
    learn_weights_on_val: bool = True  # if True, fit w_XGB via val-set search


@dataclass
class DSSConfig:
    """Decision Sufficiency Score (Steps 12-15)."""
    alpha: float = 0.5          # DSS_t = alpha*CS_t + (1-alpha)*(1-PD_t)
    tau_dss: float = 0.85       # stopping threshold
    epsilon: float = 1e-8       # numerical stabiliser used throughout (eps)
    max_acquisitions: int = 15  # hard cap on the acquisition loop length


@dataclass
class AcquisitionConfig:
    """Acquisition score S_j (Steps 16-27)."""
    lambda_1: float = 0.35   # weight on feature relevance R_j
    lambda_2: float = 0.20   # weight on graph connectivity Conn_j
    lambda_3: float = 0.35   # weight on expected DSS gain DSG_j
    lambda_4: float = 0.10   # weight (penalty) on acquisition cost C_j

    # Cost model weights: C_j = eta_a*C_acq + eta_c*C_comm + eta_e*C_comp
    eta_a: float = 0.5
    eta_c: float = 0.25
    eta_e: float = 0.25

    n_candidate_values_K: int = 5  # K_j: number of candidate values sampled
    # per unavailable feature when evaluating S_j (Step 18)


@dataclass
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    tabresnet: TabResNetConfig = field(default_factory=TabResNetConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    dss: DSSConfig = field(default_factory=DSSConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    device: str = "cpu"   # set to "cuda" if available


CFG = PipelineConfig()
