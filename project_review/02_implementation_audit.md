# 02 — Implementation Audit

---

## 1. Repository Inventory

```
dss_system_final/
├── README.md                          # Project overview and quick-start guide
├── requirements.txt                   # Python dependencies
├── diagnose_dss.py                    # Standalone diagnostic benchmark script (4 experiments)
├── dss_diagnostic_results.csv         # Output of diagnose_dss.py (full dataset, ~38k test patients)
├── metrics.py                         # Root-level shim (6 lines, likely unused)
├── workspace.json                     # IDE metadata (not part of pipeline)
├── workspacememory.md                 # IDE notes (not part of pipeline)
│
├── dss_system/                        # MAIN PACKAGE
│   ├── config.py                      # All hyperparameters and feature schema
│   ├── data_processing.py             # Data loading, splitting, preprocessing, masking
│   ├── pipeline.py                    # Training orchestration + adaptive acquisition loop
│   ├── train.py                       # CLI training entry point
│   ├── main.py                        # Full end-to-end demo entry point
│   ├── fusion.py                      # Prediction and label determination
│   ├── stability.py                   # CS_t, PD_t (JS divergence), DSS_t, stopping rule
│   ├── graph.py                       # Clinical Information Graph (MI edges, R_j, Conn_j)
│   ├── acquisition.py                 # DSG_j, cost model C_j, score S_j, selection, oracle reveal
│   ├── evaluate.py                    # Evaluation checkpoints and plots
│   ├── metrics.py                     # Comprehensive benchmark suite
│   ├── extra_plots.py                 # Plotting utilities
│   ├── dss_checkpoint.pt              # Saved PyTorch checkpoint (CVAE weights)
│   └── models/
│       ├── __init__.py
│       ├── lightgbm_model.py          # PRIMARY CLASSIFIER (active pipeline)
│       ├── cvae.py                    # CVAE generative model (active pipeline)
│       ├── catboost_model.py          # Reference/legacy (not called by train_full_system)
│       ├── xgboost_model.py           # Reference/legacy (not called by train_full_system)
│       └── tabresnet.py               # Reference/legacy (not called by train_full_system)
│
├── evaluation_plots/                  # Generated plots and metrics
│   ├── metrics_report.json            # JSON benchmark results (30-patient demo cohort)
│   ├── metrics_report.md              # Human-readable benchmark report
│   └── *.png                          # ROC, PR, confusion matrix, training curves
│
└── overleaf_report/                   # LaTeX report source
    ├── main.tex                       # Full LaTeX paper
    ├── references.bib                 # Bibliography
    ├── AUDIT_NOTES.md                 # Internal codebase audit notes
    └── README.md                      # Report overview
```

---

## 2. Role of Every Important File

### `config.py` — Central Configuration
- Defines FEATURE_COLUMNS (21 features), TARGET_COLUMN, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES.
- Contains PipelineConfig dataclass aggregating all sub-configs.
- DSS parameters: alpha=0.5, tau_dss=0.85, max_acquisitions=15.
- Acquisition weights: lambda_1=0.35 (relevance), lambda_2=0.20 (connectivity), lambda_3=0.35 (DSG), lambda_4=0.10 (cost).
- Cost weights: eta_a=0.50 (acquisition), eta_c=0.25 (communication), eta_e=0.25 (computation).

### `data_processing.py` — Data Pipeline (Steps 1-3)
- `load_raw_dataframe()`: fetches UCI ID 891 or reads local CSV.
- `train_val_test_split()`: stratified 70/15/15 split on Diabetes_binary. Line 146-173.
- `Preprocessor.fit()`: z-score on BMI, MentHlth, PhysHlth using TRAINING stats only. Line 198-205.
- `Preprocessor.transform()`: applies fitted stats to val/test. Categorical → integer codebook. Line 207-218.
- `sample_observation_masks()`: random {0,1}^F masks with 20-60% observed fraction. Line 232-253.
- `apply_mask()`: x_obs = X * m (zeros for missing). Line 256-262.
- `prepare_dataset()`: end-to-end convenience wrapper. Returns PreparedData dataclass. Line 284-361.

### `models/lightgbm_model.py` — Primary Classifier
- `_prepare_features()`: builds 42-column matrix [x_nan | m]. Missing entries → np.nan. Line 38-56.
- `LightGBMClassifierWrapper.fit()`: trains LGBMClassifier with early stopping on val logloss. Line 77-158.
- `tune_threshold()`: sweeps threshold 0.05-0.95 to maximize balanced accuracy on validation. Line 160-191.
- `predict_proba_y1()`: returns P(Y=1 | x_obs, m) as 1-D float array. Line 193-221.

### `models/cvae.py` — Conditional VAE
- `CVAEEncoder`: q(z | x, m). Input dim = 3F (x, m, x_obs). Outputs (mu, logvar). Line 31-49.
- `CVAEDecoder`: p(x | z, x_obs, m). Input dim = latent_dim + 2F. Outputs reconstructed x. Line 52-68.
- `CVAE.elbo_loss()`: MSE reconstruction (observed + missing separately) + beta * KL divergence. Line 94-103.
- `CVAE.sample_plausible()`: draws M samples; applies identity x_plausible = x_obs*m + x_hat*(1-m). Line 106-124.
- `CVAETrainer.fit()`: trains CVAE with AdamW optimizer. Line 136-163.
- `CVAETrainer.generate_plausible_profiles()`: returns (M, batch, F) numpy array. Line 165-175.

### `stability.py` — DSS Calculations (Steps 12-15)
- `kl_divergence_bernoulli(p, q)`: D_KL for Bernoulli distributions. Line 21-30.
- `class_stability(y_plausible, y_hat_t)`: CS_t = (1/M) sum I[y~^(m) == y_hat_t]. Line 33-43.
- `probability_stability(p_t, p_plausible)`: JS divergence, normalized by log(2). Returns PD_t_norm. Line 46-62.
- `decision_sufficiency_score(cs_t, pd_t_norm, alpha)`: DSS_t = alpha*CS_t + (1-alpha)*(1-PD_t_norm). Line 65-68.
- `is_sufficient(dss_t, tau_dss)`: returns boolean mask where DSS_t >= tau_DSS. Line 71-73.

### `graph.py` — Clinical Information Graph (Steps 16-17, 25)
- `_mutual_info_association(X)`: computes (F,F) association matrix using binned MI. Line 30-49.
- `build_clinical_graph()`: builds ClinicalGraph with A (raw MI), W (normalized), R_j, R_norm. Line 97-122.
- `ClinicalGraph.connectivity(observed_mask)`: Conn_j = mean W[j, observed] over observed set. Line 82-94.

### `acquisition.py` — Feature Selection (Steps 18-29)
- `DEFAULT_RAW_COSTS`: hardcoded per-feature costs (acq, comm, comp on 1-10 scale). Line 57-79.
- `normalized_costs()`: C_j_norm = (C_j - min) / (max - min + eps). Line 87-91.
- `expected_future_dss_for_feature()`: for each candidate k of feature j, sets x_j = v_j^(k), runs CVAE + classifier, computes future DSS, averages over K=5 candidates. Returns DSG_j. Line 110-175.
- `acquisition_scores()`: S_j = lambda_1*R_j + lambda_2*Conn_j + lambda_3*DSG_j - lambda_4*C_j. Line 182-207.
- `select_next_variable()`: returns argmax S_j per patient. Line 210-220.
- `acquire_actual_value()`: X_obs[patient, j] = X_true[patient, j]; m[patient, j] = 1. Line 223-236.

### `pipeline.py` — Orchestration
- `train_full_system()`: trains LightGBM → trains CVAE → builds graph. Returns TrainedSystem. Line 342-382.
- `run_adaptive_acquisition()`: complete loop. Per iteration: predict → sample M plausible → compute CS/PD/DSS → check stopping → compute DSG for all missing features → score → select → acquire. Line 155-339.
- `TrainedSystem`: container for lightgbm, cvae, graph. `.xgb` and `.tabresnet` are aliases for `.lightgbm`. Line 66-95.

---

## 3. Pipeline-Compliance Matrix

| Stage | Expected Functionality | Actual Implementation | File:Function | Evidence (lines) | Status | Consequence | Recommendation |
|-------|------------------------|----------------------|---------------|-------------------|--------|-------------|----------------|
| 1. Data preprocessing | Z-score normalization; categorical encoding; train/val/test split; no leakage | Implemented: Preprocessor.fit on train only; z-score on 3 continuous features; integer codebook on 18 categorical; stratified 70/15/15 | `data_processing.py:179-326` | Lines 199-218 (fit/transform), 307 (split) | IMPLEMENTED | Clean — preprocessor is not re-fit on val/test | None |
| 2. Observation masks | m_j in {0,1}; x_obs = x*m; 20-60% observed | sample_observation_masks generates random binary masks; apply_mask fills zeros; classifier receives mask explicitly | `data_processing.py:232-262` | Lines 248-252 (mask logic) | IMPLEMENTED | Masking is synthetic, not real clinical missingness — must be stated | Clarify in paper |
| 3. Classifier — LightGBM | f_theta(x_obs, m) -> p | LightGBM with 42-input [x_nan, m]; NaN for missing; class_weight=balanced; val threshold tuning | `models/lightgbm_model.py:38-191` | Lines 38-56 (prepare), 94-132 (fit) | IMPLEMENTED | Active single-model pipeline; fusion is pass-through | None |
| 3b. Classifier — XGBoost | f_XGB(x_obs, m) -> p_XGB (reference design) | Code exists; NOT called by train_full_system | `models/xgboost_model.py` | pipeline.py:88-90 (alias returns LightGBM) | DIFFERENT FROM REFERENCE | XGBoost is not used; claiming XGBoost in paper would be inaccurate | State LightGBM is active; XGBoost is reference code only |
| 3c. Classifier — TabResNet | f_TR(x_obs, m) -> p_TR (reference design) | Code exists; NOT called by train_full_system | `models/tabresnet.py` | pipeline.py:92-95 (alias returns LightGBM) | DIFFERENT FROM REFERENCE | TabResNet is not used | Same as above |
| 3d. Model fusion | p_t = w_XGB*p_XGB + w_TR*p_TR | fusion.py fuse() called with w_xgb=1.0, w_tr=0.0 → pass-through | `fusion.py:31-44`, `pipeline.py:381` | fusion.py line 42-43 | DIFFERENT FROM REFERENCE | No actual fusion in active pipeline | Describe as single-model with fusion infrastructure retained |
| 4. CVAE — conditional generation | P(x_miss | x_obs); generative model; M plausible completions | Full CVAE: encoder q(z|x,m), decoder p(x|z,x_obs,m), ELBO loss, identity constraint x~=x_obs*m+x_hat*(1-m) | `models/cvae.py:31-183` | Lines 87-124 (forward + sample) | IMPLEMENTED | Values generated for missing features only; observed values are preserved exactly | None |
| 5. Clinical graph | G=(V,E,W); MI-weighted edges; feature relevance | build_clinical_graph: mutual information A_ij, normalized W, relevance R_j | `graph.py:97-122` | Lines 107 (MI association), 112-120 (relevance) | IMPLEMENTED | Graph captures statistical association, NOT causal relationships | Clearly state "statistical association" in paper |
| 6. Feature relevance | R_j = mutual information with target | graph.py R_j computed as MI(X_j, Y) with 10-bin discretization | `graph.py:112-120` | Line 119 (_mutual_information call) | IMPLEMENTED | MI estimation may be coarse due to 10-bin discretization | Consider checking sensitivity to bin count |
| 6b. Acquisition cost | C_j = eta_a*C_acq + eta_c*C_comm + eta_e*C_comp | DEFAULT_RAW_COSTS with hardcoded 1-10 ordinal scale per feature | `acquisition.py:57-91` | Lines 57-91 | IMPLEMENTED (heuristic) | Costs are heuristic estimates, not real hospital dollar figures | State clearly that costs are ordinal proxies |
| 7. Plausible completion + prediction | Sample M completions; run classifier; compare | pipeline.py: sample_plausible(M=100), predict on each completion via LightGBM | `pipeline.py:218-232` | Lines 218-232 | IMPLEMENTED | M=100 is computationally feasible | None |
| 8. Class stability CS_t | CS_t = (1/M) sum I[y~^m == y_hat_t] | stability.py class_stability: (y_plausible == y_hat_t).mean(axis=0) | `stability.py:33-43` | Line 42-43 | IMPLEMENTED | Exact formula from reference | None |
| 9. Predictive distribution stability PD_t | JS divergence; normalized by log(2) | stability.py probability_stability: Bernoulli JS divergence, pd_t / log(2) | `stability.py:46-62` | Lines 56-62 | IMPLEMENTED | Exact formula from reference | None |
| 10. DSS calculation | DSS_t = alpha*CS_t + (1-alpha)*(1-PD_norm) | stability.py decision_sufficiency_score: alpha=0.5 by default | `stability.py:65-68` | Line 67-68 | IMPLEMENTED | Exact formula from reference | None |
| 11. DSS stopping | DSS_t >= tau_DSS (tau=0.85) | is_sufficient: dss_t >= tau_dss | `stability.py:71-73` | Line 73 | IMPLEMENTED | Stop on threshold OR max_acquisitions=15 | None |
| 12. Candidate feature evaluation | For each j in U_t, estimate future DSS | expected_future_dss_for_feature: K=5 CVAE samples per candidate feature | `acquisition.py:110-175` | Lines 128-174 | IMPLEMENTED | Monte Carlo uniform weights (not posterior-weighted explicitly) | Note the approximation |
| 13. DSG_j | DSG_j = E[DSS_t+1 | acquiring j] - DSS_t | dsg_j = expected_dss_next - dss_t | `acquisition.py:173-175` | Line 174 | IMPLEMENTED | Exact from reference | None |
| 14. Acquisition score S_j | S_j = lambda_1*R_j + lambda_2*Conn_j + lambda_3*DSG_j - lambda_4*C_j | acquisition_scores: exact formula with lambda weights from config | `acquisition.py:182-207` | Lines 203-206 | IMPLEMENTED | All four terms present with correct signs | None |
| 15. Adaptive loop mechanics | Select j*, update x_obs and m, update cost, repeat | pipeline.py: select_next_variable, acquire_actual_value, active flag | `pipeline.py:301-314` | Lines 302-314 | IMPLEMENTED | Oracle: acquires from X_true, not real clinical source | State oracle simulation clearly |
| 16. Final evaluation | Accuracy, balanced acc, ROC-AUC, F1, confusion matrix | evaluate.py: full sklearn metric suite; plot generation | `evaluate.py:61-74`, `metrics.py` | Lines 61-74 | IMPLEMENTED | Evaluation framework is comprehensive | Clarify demo vs full-dataset cohort |

---

## 4. Missing or Partially Implemented Modules

### Genuinely Missing
- **PDF reference document** (`Research_work_Tamaghna(4).pdf`): not found in the workspace. Audit used the reference pipeline description from the master prompt.
- **Real clinical data acquisition**: the `acquire_actual_value()` function (acquisition.py:223-236) uses `X_true` (ground truth matrix) as an oracle. A real-world deployment would call a hospital API or EHR lookup.

### Partially Implemented
- **Ensemble fusion**: The infrastructure exists (FusionWeights, fuse(), fit_fusion_weights_on_val()), but is configured with w_xgb=1.0, w_tr=0.0, making it a pass-through. XGBoost and TabResNet models are NOT trained by the active pipeline.
- **Acquisition cost model**: Costs are hardcoded ordinal estimates in DEFAULT_RAW_COSTS. The multi-component formula C_j = eta_a*C_acq + eta_c*C_comm + eta_e*C_comp is correctly implemented, but values are proxies, not validated clinical cost figures.

### Legacy/Reference-Only (Coded but Not Used in Active Pipeline)
- `models/xgboost_model.py`: XGBClassifier wrapper — fully coded, not called.
- `models/tabresnet.py`: TabResNetClassifier wrapper — fully coded, not called.
- `models/catboost_model.py`: CatBoostClassifier wrapper — fully coded, not called.
- `fusion.py:fit_fusion_weights_on_val()`: grid-search fusion weight optimizer — coded, returns default weights in current pipeline.

---

## 5. Data Leakage Risks

| Risk | Assessment | Evidence |
|------|------------|----------|
| Preprocessor fit leakage | NONE | Preprocessor.fit() called on splits.train only (data_processing.py:323) |
| Target leakage into features | NONE | Assertion at data_processing.py:296-297 confirms TARGET not in FEATURE_COLUMNS |
| ID leakage into features | NONE | Same assertion at line 297 |
| Graph built on masked data | LOW RISK | build_clinical_graph called on X_train_orig (unmasked training features) — appropriate |
| Threshold tuning on test set | NONE | tune_threshold() uses X_obs_val (validation) only, line 170 |
| Val/test contamination | NONE | train_test_split with stratify ensures disjoint sets; asserted in diagnose_dss.py:131-133 |

**Overall leakage assessment: No data leakage detected.**

---

## 6. Evaluation Concerns

1. **Demo cohort size**: metrics_report.json results are from a 30-patient demo cohort. These numbers (e.g., early_stop_rate=1.0, mean_acquisitions=0.17) may not generalize to the full dataset due to very small sample size.

2. **Two different test sets**: diagnose_dss.py and the dss_system pipeline use slightly different split seeds. diagnose_dss.py uses random_state=42 for 70/30 then random_state=43 for 50/50 of the 30%, while data_processing.py uses random_state=42 for both splits. This means the test sets are NOT identical between the two files.

3. **Precision metric**: In the full-dataset results (dss_diagnostic_results.csv), precision for masked inputs is ~0.25-0.27, which is low but expected given the natural 13.9% class prevalence and sensitivity-optimizing threshold choice.

4. **Confusion matrix in metrics_report.json** uses only 30 test patients. TN=19, TP=1, FP=10, FN=0 is a valid result but not representative of full-scale performance.

---

## 7. Misleading Names or Documentation

1. **TrainedSystem.xgb and TrainedSystem.tabresnet properties** (pipeline.py:88-95): These return `self.lightgbm`. The property names suggest XGBoost and TabResNet are active, but they are aliases. This is a backward-compatibility shim that could mislead readers.

2. **train.py docstring** (lines 6-10): States "Trains the XGBoost classifier (f_XGB), Trains the TabResNet neural network (f_TR), Trains the Conditional Variational Autoencoder (CVAE), Optimizes Bayesian fusion weights." In the current pipeline, only LightGBM and CVAE are trained. This docstring reflects the intended architecture, not the current implementation.

3. **FusionConfig.learn_weights_on_val: bool = True** (config.py:278): This flag is True but the function is called with p_tr_val=None, so it falls back to default weights (fusion.py:62-63). The flag is effectively inactive.

---

## 8. Five Most Important Corrections Before Claiming Full Implementation

1. **Document that LightGBM is the active classifier**, not XGBoost/TabResNet. Remove or annotate the misleading .xgb and .tabresnet aliases in TrainedSystem.

2. **Update train.py docstring** to accurately describe what is trained (LightGBM + CVAE, not XGBoost + TabResNet + fusion).

3. **Clarify demo cohort vs full dataset** in all evaluation outputs. metrics_report.json should include a header stating n=30 demo patients.

4. **State clearly in paper** that feature acquisition is simulated via oracle (X_true) and does NOT represent a real clinical test ordering system.

5. **Harmonize the split seeds** between diagnose_dss.py and the dss_system pipeline, or clearly document that they produce different test sets and results should not be compared directly.
