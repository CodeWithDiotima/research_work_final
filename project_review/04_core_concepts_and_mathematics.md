# 04 — Core Concepts and Mathematics

---

## Concept 1: Binary Classification and Probability Prediction

**Formal definition:** A function f: X → [0,1] that maps an input vector x to P(Y=1 | x), the probability that the output Y equals the positive class (1=Diabetic).

**Beginner explanation:** The model outputs a number between 0 and 1. If the number is 0.85, it means "85% chance this patient is diabetic." If you want a yes/no answer, you pick a threshold (e.g., 0.5): predict "Diabetic" if probability >= threshold.

**Formula:** `y_hat = 1 if f(x) >= tau else 0`

**In this system:** LightGBM produces `p_t = P(Y=1 | x_obs, m)`. Threshold τ* is tuned on validation to maximize balanced accuracy.

**Viva Q:** What does "predict_proba_y1" return?  
**A:** It returns the probability P(Y=1), i.e., the probability that the patient has diabetes or pre-diabetes. It does NOT return a binary label — that's produced by applying a threshold.

---

## Concept 2: Features, Target Labels, and Dataset Representation

**Formal definition:** D = {(x_i, y_i)}_{i=1}^N where x_i ∈ ℝ^21 is the feature vector and y_i ∈ {0,1} is the binary label.

**Features (21 total):**
- 3 continuous: BMI, MentHlth (days of poor mental health), PhysHlth (days of physical illness).
- 18 binary/ordinal: HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex, CholCheck, GenHlth (1–5 scale), Age (1–13 levels), Education (1–6), Income (1–8).

**Target:** Diabetes_binary (0 = No diabetes, 1 = Pre-diabetes or Diabetes).

**Viva Q:** How many features does the LightGBM model actually receive?  
**A:** 42 — the 21 observed feature values (with missing entries encoded as NaN) plus 21 binary mask indicators. This is the 2×F representation.

---

## Concept 3: Training, Validation, and Test Sets

**Roles:**
- Training set (70%): used to fit model parameters (LightGBM trees, CVAE weights).
- Validation set (15%): used for hyperparameter selection, threshold tuning, and early stopping.
- Test set (15%): used ONLY for final evaluation — model never sees this during development.

**In this system:** 70/15/15 stratified split in `data_processing.py:154–173`. Random state=42 throughout.

**Viva Q:** Why is a separate validation set needed if we already have a test set?  
**A:** The test set must remain unseen until the very end. Using the test set to choose the decision threshold would introduce leakage — we would have optimized a hyperparameter on data we claim to be evaluating on. The validation set serves as a "practice test" for hyperparameter decisions.

---

## Concept 4: Data Preprocessing — Normalization and Encoding

**Z-score standardization:** `z = (x - μ_train) / (σ_train + ε)`
- Applied to BMI, MentHlth, PhysHlth.
- μ_train and σ_train computed from TRAINING DATA ONLY.

**Integer codebook (categorical encoding):**
- Unique values sorted and assigned sequential integers.
- Unseen values (not in training) mapped to max_code + 1.

**In this system:** `Preprocessor.fit(train_df)` at `data_processing.py:198–205`, `transform(val_df)` at `207–218`.

**Viva Q:** Why must the preprocessor be fit ONLY on the training set?  
**A:** If we compute mean and standard deviation from the full dataset (including val and test), the test statistics "contaminate" the model's knowledge of val/test distributions. This is data leakage — the model has indirectly seen information from the test set before evaluation.

---

## Concept 5: Data Leakage

**Definition:** Data leakage occurs when information from outside the training set inappropriately influences the model during training or evaluation, causing overly optimistic performance estimates.

**In this system — three leakage guards implemented:**
1. `Preprocessor.fit()` on train only (`data_processing.py:323`).
2. Assertion `TARGET_COLUMN not in FEATURE_COLUMNS` (`data_processing.py:296`).
3. `tune_threshold()` on validation only (`lightgbm_model.py:170`).

**Viva Q:** Could there be leakage from the graph?  
**A:** The graph is built on `X_train_orig` (unmasked training features), not on val or test features. So no — graph construction does not leak test information. This is explicitly handled at `pipeline.py:372–374`.

---

## Concept 6: Class Imbalance and Balanced Evaluation

**Problem:** 86% of CDC respondents are not diabetic. A model that always predicts "No Diabetes" would have 86% accuracy but 0% sensitivity.

**Solutions implemented:**
- `class_weight="balanced"` in LightGBM: automatically upweights minority class samples.
- Threshold tuning to maximize BALANCED ACCURACY = (sensitivity + specificity) / 2.

**Balanced accuracy formula:** `(TP/(TP+FN) + TN/(TN+FP)) / 2`

**Viva Q:** Why is balanced accuracy more appropriate than raw accuracy for this dataset?  
**A:** Raw accuracy rewards predicting the majority class. With 86% non-diabetic prevalence, a model that always predicts "no diabetes" achieves 86% accuracy but 0% sensitivity. Balanced accuracy penalizes this by equally weighting sensitivity and specificity.

---

## Concept 7: Observation Masks and Partial Information

**Definition:** A binary mask m ∈ {0,1}^F where m_j = 1 means feature j is observed, m_j = 0 means missing.

**Masked input:** `x_obs = x ⊙ m`. Unobserved positions filled with 0 (fill_value).

**Classifier input augmentation:** The model receives `[x_nan | m]` where unobserved entries in x_nan are set to NaN (not 0), so the model can distinguish true-zero observed values from missing values.

**In this system:** `apply_mask()` at `data_processing.py:256–262`. `_prepare_features()` at `lightgbm_model.py:38–56`.

**Viva Q:** Does the classifier receive the mask as input?  
**A:** Yes, explicitly. The 21 binary mask indicators are appended as columns 22–42 of the input to LightGBM. Without this, the model cannot distinguish "BMI was measured and is 0" from "BMI was not measured."

---

## Concept 8: Synthetic Masking vs. Naturally Missing Data

**Synthetic masking (used):** Random masks are applied to a complete dataset. We artificially hide some features.

**Natural missingness:** Real EHR data often has naturally missing values — often MNAR (Missing Not at Random), where the decision to test is itself informative.

**Limitation:** Our system is trained and evaluated on synthetically masked data (MCAR). Its behavior on real MNAR clinical data is unvalidated.

**Viva Q:** Is your masking mechanism realistic?  
**A:** Partially. Random uniform masking (MCAR) is a simplification. In real practice, the pattern of missing data carries information — which tests were ordered reflects the clinician's initial assessment. Our framework handles any binary mask, but training on MCAR may not match real-world patterns. This is a stated limitation.

---

## Concept 9: Missing-Value Imputation vs. Plausible-Completion Sampling

**Imputation:** Replace missing values with a fixed estimate (mean, median). Treats uncertainty as resolved.

**Plausible-completion sampling (this system):** Generate M different possible completions from P(x_miss | x_obs). Used NOT for the prediction, but for uncertainty quantification.

**Critical distinction:** The CVAE does NOT impute features for the final prediction. LightGBM predicts using only the actually observed features (with NaN for missing). CVAE completions are used ONLY for computing CS_t, PD_t, and DSS_t.

**Viva Q:** Does your system impute missing values to make its final prediction?  
**A:** No. The final prediction is always made by LightGBM on the actually observed features (NaN for missing). The CVAE generates plausible completions only to measure how stable the prediction would be if we knew more — for uncertainty quantification, not prediction.

---

## Concept 10: Conditional Probability P(x_miss | x_obs)

**Definition:** The probability distribution over missing features given observed features.

**In this system:** The CVAE models this distribution. The decoder `p(x | z, x_obs, m)` implicitly captures P(x_miss | x_obs) through the generative process.

**Viva Q:** How does the CVAE approximate P(x_miss | x_obs)?  
**A:** The CVAE learns an implicit distribution. The encoder maps (x, m) to a Gaussian in latent space. The decoder maps (z, x_obs, m) to a reconstruction. By sampling z from N(0,I) at inference and running only the decoder, we draw samples from the approximate posterior.

---

## Concept 11: CVAE Architecture (Verified in Code)

**Architecture (from `cvae.py`):**
- **Encoder:** Input = [x | m | x_obs] = 3F = 63 dimensions. Two hidden layers (128 units, ReLU, Dropout 0.1). Outputs μ, log σ² (latent_dim=16).
- **Reparameterization:** `z = μ + ε × exp(0.5 × log σ²)`, where ε ~ N(0,I).
- **Decoder:** Input = [z | x_obs | m] = 16 + 21 + 21 = 58 dimensions. Two hidden layers (128 units). Output = 21-dimensional x̂.
- **Loss:** `ELBO = MSE(x̂, x) + β × KL(q(z|x,m) || N(0,I))`

**KL divergence (closed form for Gaussian):**
`KL = -0.5 × Σ(1 + log σ² - μ² - σ²)`

**Identity constraint at inference:**
`x_plausible = x_obs × m + x̂ × (1-m)` — observed values are NEVER changed.

**Viva Q:** Why use a VAE instead of a regular autoencoder?  
**A:** A regular autoencoder learns a deterministic mapping and cannot generate diverse samples. The VAE forces the latent space to follow N(0,I) through the KL term, enabling us to sample new latent codes at inference and decode them into diverse plausible completions.

---

## Concept 12: Latent Variables and KL Regularization

**Latent variable z:** A compressed 16-dimensional representation of the patient's underlying health profile.

**KL regularization:** Forces the encoder's posterior q(z|x,m) to stay close to the prior N(0,I). Without this, the model could "cheat" by encoding all information into z without generalizing.

**β parameter:** β=1.0 in this system. Higher β gives stronger regularization (disentanglement) but weaker reconstruction.

**Viva Q:** What happens if β is too large?  
**A:** The model prioritizes compressing z over accurate reconstruction — it trades reconstruction quality for latent regularity. For our task, reconstruction quality is important, so β=1.0 is used.

---

## Concept 13: LightGBM

**What it is:** Gradient Boosted Decision Trees (GBDT). An ensemble of weak decision trees where each tree corrects the errors of the previous ensemble. LightGBM is Microsoft's implementation with leaf-wise tree growth.

**Key properties used:**
- Native NaN handling: LightGBM finds optimal splits for NaN values.
- `class_weight="balanced"`: automatically computes sample weights for class balance.
- Gradient-based feature selection: automatically identifies the most informative of 42 input features.

**Viva Q:** Why LightGBM instead of XGBoost or TabResNet?  
**A:** Three reasons: (1) Native NaN support for missing entries without imputation; (2) faster training via histogram-based algorithms; (3) computational efficiency during the CVAE-intensive acquisition loop. XGBoost and TabResNet code exists for benchmarking but are not in the active pipeline.

---

## Concept 14: TabResNet and XGBoost (Coded, Not Active)

**TabResNet:** A Residual MLP with 4 blocks (BatchNorm, Dropout, skip connections). Fully coded in `tabresnet.py` but NOT trained by `train_full_system()`. `TrainedSystem.tabresnet` returns LightGBM (backward-compat alias).

**XGBoost:** Traditional GBDT. Fully coded in `xgboost_model.py` but NOT trained by the active pipeline. `TrainedSystem.xgb` returns LightGBM.

**Viva Q:** Did you train TabResNet in your experiments?  
**A:** I coded TabResNet and it works correctly, but in the final active pipeline, LightGBM is the single classifier. The `.tabresnet` property is a backward-compatibility alias that returns the LightGBM model.

---

## Concept 15: Ensemble Fusion (Coded, Configured as Pass-Through)

**Reference formula:** `p_t = w_XGB × p_XGB + w_TR × p_TR`

**Actual config:** `w_xgb=1.0, w_tr=0.0` → p_t = p_LightGBM (pass-through).

**Viva Q:** Does your system use an ensemble?  
**A:** The fusion infrastructure is implemented in `fusion.py`, but the active pipeline uses LightGBM as the sole classifier (w=1.0). The XGBoost and TabResNet models would need training and non-trivial weights for true fusion.

---

## Concept 16: Feature Relevance R_j

**Formula:** `R_j = MI(X_j, Y)` — mutual information between feature j and the diabetes label, estimated from training data using 10-bin histogram discretization.

**Normalized:** `R̃_j = (R_j - R_min) / (R_max - R_min + ε)`

**Role:** R_j appears in the acquisition score S_j as the λ₁ term.

**Viva Q:** What is the difference between feature relevance R_j and LightGBM's feature importance?  
**A:** R_j uses mutual information (statistical association). LightGBM's feature_importances_ measures split-gain contribution within the trained model. They tend to agree on top features but differ in exact ordering.

---

## Concept 17: Clinical Information Graph

**Nodes:** 21 clinical features.  
**Edges:** Weighted by `A_ij = MI(X_i, X_j)`.  
**Normalized weights:** `W_ij = (A_ij - min) / (max - min + ε)`; diagonal = 0.

**Connectivity:** `Conn_j = (1/|O_t|) × Σ_{i ∈ O_t} W_ij`

**Role:** Helps identify candidate features that are informatively connected to already-observed features.

**STATISTICAL ASSOCIATION, NOT CAUSATION.** MI measures uncertainty reduction, not causal direction.

**Viva Q:** Why is a clinical graph needed instead of just using feature relevance?  
**A:** R_j tells us how informative a feature is for the diagnosis. Conn_j tells us how correlated a missing feature is with what we already know. The graph helps avoid redundant acquisitions — if HighBP is observed, acquiring HighChol (highly correlated) may provide less new information than acquiring GenHlth.

---

## Concept 18: Class Stability (CS_t)

**Formula:** `CS_t = (1/M) × Σ_{m=1}^{M} I[ỹ^(m) == ŷ_t]`

**Interpretation:** Fraction of plausible completions where the predicted class agrees with the current prediction.

- CS_t = 1.0: All 100 completions agree → robust to missing information.
- CS_t = 0.5: Half agree → highly uncertain.

**In this system:** `class_stability()` at `stability.py:33–43`.

**Viva Q:** Can the predicted class remain unchanged even when the probability changes significantly?  
**A:** Yes — this is exactly why we need both CS and PD. CS_t = 1.0 means all completions predict the same class, but some might predict with p=0.51 and others with p=0.95. CS captures class-level stability; PD captures probability-level stability.

---

## Concept 19: Jensen-Shannon Divergence and PD_t

**KL Divergence:** `D_KL(p || q) = Σ_c p(c) log(p(c)/q(c))`  
**JS Divergence:** `D_JS(p || q) = 0.5 × D_KL(p || r) + 0.5 × D_KL(q || r)` where `r = 0.5(p+q)`  
- Symmetric, bounded in [0, log 2].

**Normalization:** `PD̃_t = D_JS / log(2)` → maps to [0, 1].

**In this system (Bernoulli):** Distributions are over {0,1}. For each plausible completion m:
- `r^(m) = 0.5(p_t + p_plaus_m)`
- `D_JS^(m) = 0.5 D_KL(p_t || r) + 0.5 D_KL(p_plaus || r)`
- `PD_t = mean over M of D_JS^(m)`

**Implemented at:** `stability.py:46–62`

**Viva Q:** Why use JS divergence instead of variance of the probabilities?  
**A:** JS divergence has information-theoretic grounding — it measures distributional distance, is symmetric, bounded, and properly accounts for non-linear decision boundaries (a shift from 0.50 to 0.51 is much more consequential for the binary decision than a shift from 0.90 to 0.91).

---

## Concept 20: Decision Sufficiency Score (DSS)

**Formula:** `DSS_t = α × CS_t + (1 - α) × (1 - PD̃_t)`

**Parameters:** α=0.5 (equal weighting).

**Range:** [0, 1]. Higher is better.

**Intuition:**
- Term 1 (α × CS_t): how often does the predicted class stay the same across 100 plausible completions?
- Term 2 ((1-α) × (1-PD̃_t)): how small are the probability differences across completions?

**Threshold τ_DSS = 0.85:** If DSS_t ≥ 0.85, stop acquiring features.

**Example:**
- CS_t = 0.92, PD̃_t = 0.013 → DSS_t = 0.5×0.92 + 0.5×(1-0.013) = 0.9535 ≥ 0.85 → **STOP**
- CS_t = 0.65, PD̃_t = 0.15 → DSS_t = 0.5×0.65 + 0.5×(1-0.15) = 0.750 < 0.85 → **CONTINUE**

**DSS does NOT guarantee clinical correctness.** It measures prediction stability, not accuracy. A biased model could be stably wrong.

---

## Concept 21: Decision Stabilization Gain (DSG_j)

**Formula:** `DSG_j = E[DSS_{t+1} | acquiring feature j] - DSS_t`

**Positive DSG_j:** acquiring feature j is expected to increase decision sufficiency.  
**Negative DSG_j:** acquiring feature j might destabilize the current prediction.

**How computed:** For K=5 CVAE-sampled candidate values of feature j, set x_j = v_j^(k) and m_j=1, compute future DSS, average over K=5 candidates (`acquisition.py:110–175`).

**Viva Q:** What does it mean if DSG_j is negative?  
**A:** Observing feature j is expected to DECREASE decision sufficiency — it might reveal conflicting information. The system avoids such features by assigning them a low acquisition score S_j.

---

## Concept 22: Acquisition Score S_j

**Formula:** `S_j = λ₁ R̃_j + λ₂ Conñ_j + λ₃ DSG_j - λ₄ C̃_j`

**Weights:** λ₁=0.35, λ₂=0.20, λ₃=0.35, λ₄=0.10.

**Components:**
- λ₁ × R̃_j: feature relevance — prefers diagnostic features.
- λ₂ × Conñ_j: graph connectivity — prefers features correlated with observed ones.
- λ₃ × DSG_j: expected DSS gain — prefers features that stabilize prediction.
- λ₄ × C̃_j: cost penalty — discourages expensive features.

**Implemented at:** `acquisition.py:182–207`.

**Ablation results (from metrics_report.json):**
- Removing relevance (λ₁=0): balanced accuracy drops 0.8621 → 0.8448.
- Removing DSG (λ₃=0): balanced accuracy drops 0.8621 → 0.8448.
- Random baseline: same balanced accuracy (0.8621) but at higher cost (0.076 vs 0.033).

---

## Concept 23: Acquisition Cost Model

**Formula:** `C_j = η_a × C_acq + η_c × C_comm + η_e × C_comp`

**Weights:** η_a=0.50, η_c=0.25, η_e=0.25.

**Example costs (DEFAULT_RAW_COSTS in `acquisition.py:57–79`):**
- CholCheck (self-report): acq=1, comm=1, comp=1 → C = 1.0
- HighChol (blood draw): acq=6, comm=5, comp=1 → C = 4.5
- HighBP (BP cuff): acq=4, comm=2, comp=1 → C = 2.75

**Limitation:** These are ordinal proxy values, not real dollar costs.

---

## Concept 24: Sensitivity, Specificity, Precision, Recall, F1

**For binary Diabetic (1) vs. No Diabetes (0):**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Sensitivity (Recall) | TP / (TP+FN) | Fraction of diabetics correctly identified |
| Specificity | TN / (TN+FP) | Fraction of non-diabetics correctly dismissed |
| Precision | TP / (TP+FP) | Fraction of "Diabetic" predictions that are truly diabetic |
| F1 | 2×Prec×Rec / (Prec+Rec) | Harmonic mean of precision and recall |
| Balanced Accuracy | (Sens + Spec) / 2 | Class-imbalance-aware accuracy |

**Viva Q:** Why is precision so low (~0.25–0.27) in the masked experiments?  
**A:** The dataset has 13.9% positive prevalence. Even with 73% sensitivity and 67% specificity on ~38,000 test patients, the math gives precision ≈ 0.26 due to the base rate. Low precision with a sensitive classifier on a low-prevalence condition is expected.

---

## Concept 25: ROC Curve and AUC

**ROC curve:** Plots sensitivity vs. (1−specificity) across all thresholds.

**AUC:** Probability that a randomly chosen positive patient gets a higher score than a randomly chosen negative patient. AUC=0.5 is random; AUC=1.0 is perfect.

**From results:** Full-feature balanced model ROC-AUC = 0.8285 (full dataset). Masked K=5 aug: 0.7678.

**Viva Q:** What does ROC-AUC measure?  
**A:** AUC measures how well the model RANKS patients by risk — it evaluates the quality of probability estimates, not just the binary decision at any single threshold. It is threshold-independent.

---

## Concept 26: Confusion Matrix

```
                   Predicted No Diabetes | Predicted Diabetic
True No Diabetes:        TN              |        FP
True Diabetic:           FN              |        TP
```

**From metrics_report.json (30-patient demo):**
- Initial: TN=19, FP=10, FN=0, TP=1.
- Post-acquisition: TN=21, FP=8, FN=0, TP=1.

FN=0 means NO diabetic patients were missed (sensitivity=1.0). FP=10→8 means false alarms decreased after acquisition.

---

## Concept 27: Overfitting and Early Stopping

**Prevention in this system:**
- LightGBM early stopping: stops when val logloss stops improving for 30 rounds.
- LightGBM regularization: reg_lambda=1.0, min_child_samples=20.
- CVAE dropout=0.1, AdamW optimizer.

**Viva Q:** How do you know LightGBM is not overfitting?  
**A:** We monitor validation logloss during training and use early stopping. The gap between training and validation metrics is small. Full-dataset test accuracy is consistent with validation performance.

---

## Concept 28: Model Calibration

**Calibration:** A well-calibrated model's predictions of P(Y=1)=0.7 should correspond to ~70% true positives among patients with that predicted probability.

**Brier score:** `(1/N) × Σ(p_i - y_i)²` — lower is better. Full-dataset Brier: 0.0969–0.1979.

**Limitation:** LightGBM probabilities are not explicitly calibrated (no Platt scaling). This is a limitation for clinical use.

---

## Concept 29: External Validation vs. ML Risk Prediction vs. Medical Diagnosis

**ML risk prediction (this system):** Uses survey data to estimate statistical risk. Output is a probability.

**Medical diagnosis:** Made by a licensed clinician integrating tests, examination, history, judgment, and regulatory standards.

**External validation:** Testing on a completely independent dataset — NOT done in this project.

**Critical statement:** This system is a research prototype. It MUST NOT be used to make or influence actual medical decisions without clinical oversight and regulatory review.

---

## Concept 30: The Adaptive Acquisition Loop — Big Picture

**The central question:** Given partially observed patient data, is the current evidence sufficient for a reliable prediction? If not, which feature should we collect next?

**Answer architecture:**
1. Predict with what we have → p_t, ŷ_t
2. Generate 100 plausible completions → quantify uncertainty
3. Compute DSS_t from class stability and predictive divergence
4. If DSS_t ≥ τ_DSS → STOP, report diagnosis
5. Else: compute DSG_j for each missing feature, score S_j, select j* = argmax S_j
6. Acquire feature j* (from X_true oracle), update x_obs and m
7. Repeat

**Viva Q:** Can you summarize the entire system in three sentences?  
**A:** We train a LightGBM classifier to predict diabetes risk from partially observed patient features (21 health indicators with 40–80% missing). A Conditional Variational Autoencoder generates 100 plausible completions of missing features, and we combine class stability and probability stability into a Decision Sufficiency Score to assess whether the current evidence supports a reliable diagnosis. When evidence is insufficient, a multi-objective scoring function integrating feature relevance, graph connectivity, expected DSS gain, and acquisition cost selects the most valuable next feature to collect.

---

## Concept 31: Oracle Simulation

**Definition:** During evaluation, the "acquired" feature value is looked up from the ground-truth matrix X_true — not from a real clinical test order.

**Implemented at:** `acquisition.py:223–236`:
```python
X_obs[patient_idx, feature_idx] = X_true[patient_idx, feature_idx]
m[patient_idx, feature_idx] = 1
```

**Implication for paper:** We can claim the framework WORKS algorithmically but NOT that it has been deployed in a clinical setting.

**Viva Q:** How are feature values "acquired" during the adaptive loop?  
**A:** In our evaluation, we simulate acquisition by revealing the patient's ground-truth value from the held-out test matrix. In a real clinical deployment, this step would correspond to ordering a laboratory test, querying an EHR system, or asking the patient a specific survey question. The framework is agnostic to how the value arrives — it only needs the actual value to update x_obs and m.
