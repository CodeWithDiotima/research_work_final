# Codebase & Methodology Audit Notes

This document provides a strict, evidence-based audit of the **Decision-Sufficiency-Guided Adaptive Clinical Acquisition System (DSS/DSG)** codebase. It explicitly distinguishes between fully implemented modules, reference/legacy architectures, deliberate engineering modifications, and clinical assumptions.

---

## 1. Verified & Fully Implemented Modules

| Module / Stage | Source File | Implementation Status | Repository Evidence |
| :--- | :--- | :--- | :--- |
| **Data Ingestion & Split** | `data_processing.py` | Complete | Stratified 70/15/15 split on `Diabetes_binary` preserving natural ~13.9% prevalence. Fetches UCI ID 891 or loads local CSV. |
| **Preprocessing** | `data_processing.py` | Complete | `Preprocessor` fits z-score on training continuous features (`BMI`, `MentHlth`, `PhysHlth`) and integer codebook on 18 categorical features. |
| **Masking & Augmentation** | `data_processing.py`, `config.py` | Complete | Random masks $m \in \{0,1\}^F$ (20%–60% observed). Mask augmentation tiles training records $K=3$ or $K=5$ times with independent mask streams. |
| **Primary Classifier** | `models/lightgbm_model.py` | Complete | LightGBM wrapper $f_{\text{LGBM}}$ receiving $2F=42$ inputs ($[x_{\text{nan}}, m]$), `class_weight='balanced'`, early stopping, and validation threshold tuning. |
| **Plausible Profile Generator** | `models/cvae.py` | Complete | PyTorch Conditional VAE modeling $P_\Psi(x_{\text{miss}} \mid x_{\text{obs}})$ with ELBO loss. Generates $M$ completions per patient: $\tilde{x} = x_{\text{obs}} \odot m + \hat{x} \odot (1-m)$. |
| **Clinical Graph** | `graph.py` | Complete | Clinical graph $G=(V,E,W)$ with mutual information edge associations $A_{ij}$, normalized edge weights $\tilde{W}_{ij}$, feature relevance $\tilde{R}_j$, and dynamic connectivity $\text{Conn}_j$. |
| **Decision Sufficiency** | `stability.py` | Complete | Exact implementation of Class Stability ($CS_t$), Bernoulli Jensen-Shannon Predictive Divergence ($\widetilde{PD}_t$), and Decision Sufficiency Score ($DSS_t = \alpha CS_t + (1-\alpha)(1-\widetilde{PD}_t)$). |
| **Adaptive Acquisition** | `acquisition.py`, `pipeline.py` | Complete | Evaluates counterfactuals $v_j^{(k)}$ to compute $DSG_j$, normalized costs $\tilde{C}_j$, multi-objective score $S_j$, greedy test selection $j^* = \arg\max S_j$, and live state updates. |
| **Benchmark & Reporting** | `metrics.py`, `evaluate.py`, `extra_plots.py` | Complete | Full suite generating confusion matrices, ROC/PR curves, Pareto frontiers across 5 budgets, missingness sweeps, $\tau_{\text{DSS}}$ sweeps, and $\lambda$ ablations. |

---

## 2. Reference, Legacy, and Alternative Implementations

1. **XGBoost Classifier (`models/xgboost_model.py`)**:
   - *Status*: Fully coded and functional.
   - *Audit Note*: Originally served as the tree classifier in early development. Retained in `models/` for reference and benchmarking.
2. **TabResNet Classifier (`models/tabresnet.py`)**:
   - *Status*: Fully coded and functional (Residual MLP with 4 blocks, BatchNorm, Dropout, AdamW).
   - *Audit Note*: Deep learning tabular model implemented to support hybrid neural-tree fusion. Currently optional; the active default pipeline utilizes LightGBM due to faster training and native NaN handling.
3. **CatBoost Classifier (`models/catboost_model.py`)**:
   - *Status*: Fully coded reference model with native NaN handling.
4. **Ensemble Fusion (`fusion.py`)**:
   - *Status*: Implements Bayesian probability weighting $p_t = w_1 p_1 + w_2 p_2$ and validation loss grid-search optimization.
   - *Audit Note*: Active pipeline config sets $w_{\text{LGBM}}=1.0, w_{\text{TR}}=0.0$ by default to reduce computational overhead during CVAE counterfactual generation.

---

## 3. Deliberate Engineering Improvements vs. Algorithm 1 Specification

| Specification Feature | Theoretical Algorithm 1 Spec | Actual Codebase Implementation | Rationale |
| :--- | :--- | :--- | :--- |
| **Missingness Generator** | $F$ individual autoregressive regressors $g_1 \dots g_F$ chained sequentially | Single unified Conditional Variational Autoencoder (CVAE) | Chaining 21 separate regressors is computationally prohibitive and accumulates compounding error. A single CVAE jointly models all missing features in one forward pass. |
| **Candidate Value Expectation** | Probability-weighted sum $\sum P(v_j^{(k)} \mid x_{\text{obs}}) DSS_j^{(k)}$ | Direct Monte Carlo averaging $\frac{1}{K}\sum DSS_j^{(k)}$ over CVAE samples | Sampling candidate values directly from the CVAE latent space implicitly weights candidates according to their posterior likelihood $P(v_j \mid x_{\text{obs}})$. |
| **Decision Thresholding** | Static $\arg\max$ cutoff ($0.50$) | Validation-tuned threshold optimizing Balanced Accuracy | CDC Diabetes has a natural ~13.9% positive class prevalence. A static 0.50 threshold biases decisions toward majority negative; tuning restores medical sensitivity. |

---

## 4. Key Assumptions & Evaluation Constraints

1. **Synthetic Feature Masking**: Initial missingness is simulated using uniformly sampled random masks (20%–60% observed) because the raw CDC BRFSS public dataset does not record which survey questions were answered first.
2. **Oracle Simulation**: During evaluation of adaptive acquisition, the acquired feature $x^*_{j^*}$ is retrieved from the patient's ground-truth row in the held-out test matrix ($X_{\text{true}}$). In hospital deployment, this corresponds to ordering an actual medical test or EHR laboratory query.
3. **Heuristic Cost Calibration**: Multi-component clinical costs ($C_j = \eta_a C_{\text{acq}} + \eta_c C_{\text{comm}} + \eta_e C_{\text{comp}}$) use standardized ordinal weights based on medical burden (e.g. self-report survey = 1, blood draw = 6). Real hospital dollar costs can be swapped into `DEFAULT_RAW_COSTS` without modifying pipeline logic.
4. **Clinical Safety Boundary**: This system is an algorithmic decision-support research pipeline and is **not** an FDA/CE-cleared automated diagnostic instrument. All recommendations require clinical physician oversight.
