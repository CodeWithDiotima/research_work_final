# 05 — Viva Questionnaire with Model Answers

> **60+ questions organized by topic.** Questions marked ★★★ are highest priority — examiner favorites. Questions marked ⚡ are "gotcha" follow-ups that test deeper understanding.

---

## SECTION A: PROBLEM STATEMENT AND MOTIVATION (Questions 1–10)

---

### Q1. ★★★ What problem does your project solve?

**A:** Standard diabetes risk prediction models assume all patient features are available at prediction time. In practice, clinical data is often incomplete — laboratory tests may be pending, survey questions may be unanswered, and cost or privacy concerns may prevent collecting all 21 BRFSS health indicators. My project addresses two questions: (a) given an incomplete patient record, is the current information sufficient for a reliable prediction? And (b) if not, which additional information should be collected next to maximally improve decision reliability while minimizing acquisition cost?

---

### Q2. Why is this problem important?

**A:** In clinical settings, over-testing increases healthcare costs and patient burden, while under-testing risks missed diagnoses. An intelligent system that knows when to stop testing (because evidence is sufficient) and what to test next (when evidence is insufficient) can reduce unnecessary medical tests while maintaining diagnostic accuracy. With diabetes affecting over 37 million Americans, even small improvements in triage efficiency have large-scale impact.

---

### Q3. ★★★ What dataset did you use?

**A:** The CDC Behavioral Risk Factor Surveillance System (BRFSS) Diabetes Health Indicators Dataset from the UCI Machine Learning Repository (ID 891). It contains 253,680 respondents, 21 health indicator features (3 continuous: BMI, MentHlth, PhysHlth; 18 binary/ordinal), and a binary target `Diabetes_binary` (0 = No Diabetes, 1 = Pre-diabetes or Diabetes). The natural positive class prevalence is approximately 13.9%.

---

### Q4. ⚡ Why didn't you use real EHR data with naturally missing values?

**A:** Real EHR data with natural missingness patterns would be ideal but presents three barriers: (1) obtaining clinical EHR data requires IRB approval and data use agreements that are typically unavailable for student research projects; (2) BRFSS is a well-studied, freely available public health dataset that allows reproducible benchmarking; and (3) our framework is designed to handle ANY binary mask, so the methodology transfers to real EHR data — the current synthetic masking establishes algorithmic validity before clinical validation. This is a stated limitation of the work.

---

### Q5. How did you split the data?

**A:** Stratified 70/15/15 split using scikit-learn's `train_test_split` with `random_state=42` and `stratify=y`. "Stratified" means the ~13.9% positive class ratio is preserved in all three splits. This gives approximately 177,576 training, 38,052 validation, and 38,052 test patients. The validation set is used for threshold tuning and early stopping; the test set is used only for final evaluation.

---

### Q6. ⚡ How do you simulate missing data?

**A:** For each patient, we sample a uniform random fraction f ∈ [0.20, 0.60], then randomly select k = round(f × 21) features to observe (m_j = 1); the remaining features are masked (m_j = 0). This generates Missing Completely at Random (MCAR) masks. Each split (train, val, test) uses an independent random number generator stream to prevent correlation. On training, mask augmentation replicates each patient K=3 times with different mask patterns to increase robustness.

---

### Q7. ⚡ What is the difference between MCAR, MAR, and MNAR missingness, and which does your system handle?

**A:** MCAR (Missing Completely at Random): missingness is independent of all data — both observed and unobserved values. MAR (Missing at Random): missingness depends on observed values but not on the missing values themselves. MNAR (Missing Not at Random): missingness depends on the unobserved values — e.g., a doctor doesn't order a cholesterol test because the patient looks healthy (but actually has high cholesterol). My system generates MCAR masks. It can technically handle any binary mask pattern at inference time, but it was TRAINED on MCAR data, so its learned distributions may not generalize to MAR or MNAR settings without retraining.

---

### Q8. Why not just impute missing values with the mean?

**A:** Mean imputation replaces uncertainty with a single point estimate, discarding all information about what the missing value COULD be. This leads to two problems: (1) the model cannot distinguish between a patient who genuinely has the average BMI and one whose BMI was never measured, and (2) we lose the ability to quantify how much the prediction depends on the missing information. Our CVAE generates M=100 plausible completions, preserving the diversity of possible values and enabling uncertainty quantification through CS_t and PD_t.

---

### Q9. What is the clinical significance of your work?

**A:** The clinical significance is in decision support triage: when a patient presents with incomplete records, the system can (1) still produce a risk estimate using available information, (2) quantify how confident that estimate is, and (3) recommend the single most cost-effective test to order next. This could reduce unnecessary testing while maintaining diagnostic reliability. However, this is a RESEARCH contribution — it has not been clinically validated and should not be used for medical decisions without physician oversight.

---

### Q10. What is novel about your approach compared to standard active learning?

**A:** Standard active learning selects the most informative SAMPLES (patients) for labeling. Our approach selects the most informative FEATURES (clinical tests) for a specific patient. The acquisition decision integrates four factors — feature relevance, graph connectivity, expected DSS gain, and acquisition cost — rather than using a single information-theoretic criterion. Additionally, the DSS stopping criterion explicitly combines class-level and probability-level stability, providing a clinically interpretable measure of decision sufficiency.

---

## SECTION B: MODEL ARCHITECTURE AND TRAINING (Questions 11–25)

---

### Q11. ★★★ What is your primary classifier and why?

**A:** LightGBM (Light Gradient Boosting Machine) — a gradient-boosted decision tree framework. Three reasons: (1) native NaN handling, which is essential when features are missing — LightGBM can learn optimal split directions for NaN entries without requiring imputation; (2) `class_weight="balanced"` for handling the 86%/14% class imbalance naturally; (3) computational efficiency — LightGBM's histogram-based algorithms are faster than XGBoost, which matters during the CVAE-intensive acquisition loop.

---

### Q12. ★★★ How does LightGBM handle missing features?

**A:** The input is a 42-column matrix: [x_nan | m]. The first 21 columns contain the feature values where observed entries keep their actual values and unobserved entries are set to NaN (not zero). The last 21 columns are the binary mask indicators. LightGBM natively handles NaN by learning, during training, whether NaN values should go to the left or right child of each decision tree split. The mask columns provide an additional explicit signal so the model can directly split on "was this feature measured?"

---

### Q13. ⚡ Why encode missing values as NaN instead of 0?

**A:** Zero-imputation is ambiguous. For binary features like HighBP, a zero means "no high blood pressure" — but it could also mean "blood pressure was not measured." NaN encoding eliminates this ambiguity: NaN distinctly represents "not measured" while 0 distinctly represents "measured and the value is zero." The appended mask column provides a redundant but complementary signal for this distinction.

---

### Q14. What is the CVAE and what role does it play?

**A:** The Conditional Variational Autoencoder models the conditional distribution P(x_miss | x_obs). During training, it sees the full ground-truth features x and the mask m, and learns to reconstruct x from a compressed latent code z. At inference, we fix the observed features and sample latent codes from the prior N(0,I) to generate M=100 plausible completions of the missing features. These completions are used to measure prediction stability (CS_t, PD_t) — NOT for the final prediction itself.

---

### Q15. ★★★ Explain the CVAE loss function.

**A:** The CVAE optimizes the Evidence Lower Bound (ELBO), which has two terms:
1. **Reconstruction loss:** MSE between the decoder's output x̂ and the true x, computed over all features (both observed and missing). This encourages the decoder to produce accurate completions.
2. **KL divergence:** `KL(q(z|x,m) || N(0,I))`, which regularizes the encoder's posterior to stay close to the standard Gaussian prior. This ensures the latent space is well-structured for sampling at inference time.

The total loss is: `ELBO = MSE_recon + β × KL`, with β=1.0 in our system.

---

### Q16. What is the reparameterization trick?

**A:** During training, we need to backpropagate through the stochastic sampling z ~ q(z|x,m). Direct sampling breaks the gradient computation. The reparameterization trick reformulates sampling as: `z = μ + ε × exp(0.5 × log σ²)` where ε ~ N(0,I) is an external random variable. This makes z a deterministic, differentiable function of the encoder outputs (μ, log σ²) plus fixed noise ε, enabling gradient-based training.

---

### Q17. ⚡ Why does the CVAE encoder see the full ground-truth x during training but only x_obs at inference?

**A:** During training, the encoder needs to see the true x so that the reconstruction loss can push the decoder to generate accurate completions. The encoder's job is to compress the full information (x, m) into a meaningful latent representation z. At inference, we do NOT use the encoder at all — we sample z directly from the prior N(0,I) and run only the decoder conditioned on (x_obs, m). The KL regularization during training ensures that sampling from the prior produces latent codes in the same space the decoder was trained to reconstruct from.

---

### Q18. ⚡ What guarantees that the CVAE doesn't alter observed feature values?

**A:** The identity constraint in `CVAE.sample_plausible()` (line 122): `x_plausible = x_obs × m + x̂ × (1-m)`. This explicitly overwrites the decoder's output for observed positions with the actual observed values. The decoder can output anything for those positions, but the identity constraint forces the plausible profile to match x_obs exactly for all observed features and to use the decoder's output only for missing features.

---

### Q19. ★★★ What is the clinical information graph?

**A:** A graph with 21 nodes (one per clinical feature) and edges weighted by mutual information between feature pairs, computed from the training data. The graph serves two purposes: (1) feature relevance R_j = MI(X_j, Y) measures how informative each feature is for the diabetes diagnosis; (2) connectivity Conn_j = mean edge weight from feature j to currently observed features, which measures how related a candidate feature is to what we already know. Both are normalized to [0,1] and used in the acquisition score formula.

---

### Q20. ⚡ Does the clinical graph represent causal relationships?

**A:** No. The graph uses mutual information, which measures statistical association, not causation. High MI between HighBP and HighChol means knowing one reduces uncertainty about the other, but it does not establish that high blood pressure causes high cholesterol. Establishing causal relationships would require interventional data or formal causal inference methods (do-calculus, instrumental variables).

---

### Q21. How did you handle class imbalance?

**A:** Three strategies: (1) `class_weight="balanced"` in LightGBM, which automatically upweights minority class (diabetic) samples during training; (2) decision threshold tuning on the validation set to maximize balanced accuracy rather than raw accuracy; (3) reporting balanced accuracy, sensitivity, and specificity as primary metrics, not raw accuracy.

---

### Q22. What is threshold tuning and why is it needed?

**A:** LightGBM outputs a probability P(Y=1). To get a binary prediction, we need a threshold τ: predict "Diabetic" if P(Y=1) ≥ τ. The default τ=0.5 is suboptimal for imbalanced data — with 86% non-diabetic prevalence, a 0.5 threshold tends to classify most patients as non-diabetic. We sweep 91 thresholds from 0.05 to 0.95 on the validation set and select the one maximizing balanced accuracy. This typically selects a lower threshold (e.g., ~0.14 in the unweighted full-feature experiment), increasing sensitivity at the cost of some specificity.

---

### Q23. What is early stopping and why use it?

**A:** Early stopping halts LightGBM training when the validation logloss stops improving for 30 consecutive boosting rounds. Without it, the model would continue adding trees that overfit the training data, memorizing noise rather than learning generalizable patterns. The model checkpoint from the best validation round is retained.

---

### Q24. ⚡ You have XGBoost and TabResNet code in the repository. Why aren't they used?

**A:** The reference design specified an XGBoost + TabResNet ensemble with Bayesian fusion. During implementation, I found that LightGBM alone achieved comparable performance with three advantages: (1) native NaN support (XGBoost requires explicit imputation or DMatrix construction); (2) faster training and inference (critical when the acquisition loop calls the classifier hundreds of times per patient); (3) simpler pipeline with fewer failure modes. XGBoost and TabResNet code are retained in `models/` as reference implementations for benchmarking and potential future ensemble experiments.

---

### Q25. ⚡ Your TrainedSystem has properties .xgb and .tabresnet that return the LightGBM model. Isn't that misleading?

**A:** Yes, this is a backward-compatibility artifact. When the pipeline was refactored from three models to one, the property names were kept so that downstream code (e.g., acquisition.py) that referenced `system.xgb.predict_proba_y1()` would not break. The properties are documented as aliases in the TrainedSystem docstring. For clarity, new code should use `system.lightgbm` directly. I acknowledge this naming could be improved.

---

## SECTION C: DSS AND ADAPTIVE ACQUISITION (Questions 26–40)

---

### Q26. ★★★ What is the Decision Sufficiency Score?

**A:** DSS_t = α × CS_t + (1−α) × (1 − PD̃_t), where α=0.5. CS_t is the class stability — the fraction of M=100 plausible completions that predict the same class as the current prediction. PD̃_t is the normalized Jensen-Shannon divergence measuring how much the predicted probability varies across completions. DSS ranges from 0 (completely unstable) to 1 (perfectly stable). When DSS_t ≥ τ_DSS = 0.85, the system considers the current evidence sufficient and stops acquiring features.

---

### Q27. ★★★ Walk me through one iteration of the adaptive acquisition loop.

**A:** Starting with patient data (x_obs, m):
1. **Predict:** LightGBM produces p_t = 0.72, ŷ = 1 (Diabetic).
2. **Generate completions:** CVAE generates 100 plausible complete profiles, keeping observed values fixed.
3. **Classify completions:** LightGBM classifies each completion. Suppose 85 predict Diabetic, 15 predict No Diabetes.
4. **Compute stability:** CS_t = 85/100 = 0.85. PD̃_t ≈ 0.04. DSS_t = 0.5×0.85 + 0.5×0.96 = 0.905.
5. **Check stopping:** 0.905 ≥ 0.85? Yes → STOP. Final diagnosis: Diabetic.

If DSS_t were below 0.85:
6. **Compute DSG_j** for each missing feature using K=5 CVAE candidate values.
7. **Score:** S_j = 0.35×R_j + 0.20×Conn_j + 0.35×DSG_j − 0.10×C_j.
8. **Select:** j* = argmax S_j. Suppose j* = "HighChol."
9. **Acquire:** Reveal HighChol's true value from X_true. Update x_obs and m.
10. **Repeat** from step 1 with updated information.

---

### Q28. ⚡ What if DSS is high but the prediction is wrong?

**A:** DSS measures prediction STABILITY, not correctness. A biased model could consistently predict the wrong class with high confidence across all plausible completions, resulting in high CS_t and low PD_t — therefore high DSS. This is an inherent limitation: DSS tells us the model is confident, not that it's right. Clinical accuracy must be validated separately through evaluation metrics on held-out test data.

---

### Q29. Why use Jensen-Shannon divergence instead of simpler metrics?

**A:** JSD has four advantages over variance or absolute difference: (1) it is symmetric (D_JS(p||q) = D_JS(q||p)); (2) it is bounded in [0, log 2], allowing normalization to [0,1]; (3) it has information-theoretic grounding as the expected information gain from observing which of two distributions generated a sample; (4) it properly accounts for the non-linearity of decision boundaries — a probability shift from 0.50 to 0.55 (crosses decision boundary) is more impactful than a shift from 0.85 to 0.90.

---

### Q30. ★★★ How does the system decide which feature to acquire next?

**A:** Each candidate missing feature j receives a score S_j combining four factors: (1) **Relevance R_j** (35% weight) — mutual information with the diabetes label, measuring how diagnostic the feature is; (2) **Connectivity Conn_j** (20% weight) — average edge weight to already-observed features in the clinical graph; (3) **Expected DSS Gain DSG_j** (35% weight) — how much acquiring j is expected to improve decision sufficiency; (4) **Cost C_j** (10% penalty) — a cost term discouraging expensive tests. The feature with the highest S_j is selected.

---

### Q31. ⚡ Why is DSG_j the most computationally expensive component?

**A:** For each candidate feature j, DSG_j requires: (1) sampling K=5 candidate values from the CVAE; (2) for each candidate value, updating x_obs and m to hypothetically include feature j; (3) generating M=100 plausible completions of the REMAINING missing features; (4) running LightGBM on each completion; (5) computing CS, PD, and DSS for each candidate; (6) averaging across K candidates. This means K × M = 500 LightGBM evaluations per candidate feature. With ~12 missing features on average, that's ~6,000 classifier evaluations per patient per acquisition step.

---

### Q32. ⚡ How are candidate feature values sampled in Step 18?

**A:** The `sample_candidate_values` function draws K=5 values from the CVAE decoder's implicit distribution. It generates CVAE plausible completions and extracts the sampled values for feature j. Since the CVAE samples from the learned conditional distribution P(x_j | x_obs), the candidate values are approximately posterior-weighted. The code treats each candidate with equal weight (1/K) for simplicity, which is equivalent to Monte Carlo estimation under the posterior.

---

### Q33. What is the stopping condition?

**A:** Two stopping conditions (whichever comes first):
1. **DSS threshold:** DSS_t ≥ τ_DSS = 0.85 — decision sufficiency achieved.
2. **Budget limit:** max_acquisitions = 15 features acquired — safety cap preventing infinite loops.

From the demo cohort: 100% of patients achieved the DSS threshold before hitting the budget limit, with a mean of 0.17 features acquired per patient.

---

### Q34. ★★★ What do the ablation study results show?

**A:** The ablation study (metrics_report.json) systematically removes each λ component:
- **Full model** (0.35, 0.20, 0.35, 0.10): balanced accuracy = 0.8621, cost = 0.033.
- **Remove relevance** (λ₁=0): balanced accuracy drops to 0.8448. Feature relevance matters.
- **Remove DSG** (λ₃=0): balanced accuracy drops to 0.8448. DSG matters.
- **Remove connectivity** (λ₂=0): balanced accuracy stays 0.8621 but cost increases to 0.043. Connectivity helps efficiency.
- **Remove cost** (λ₄=0): balanced accuracy stays 0.8621 but no cost savings. Cost penalty matters for efficiency.
- **Random baseline** (all λ=0): balanced accuracy = 0.8621 but cost doubles to 0.076.

Key takeaway: the full scoring formula achieves the same accuracy as simpler strategies but at significantly lower cost.

---

### Q35. ⚡ The random baseline achieves the same balanced accuracy as the full model. Why?

**A:** On the 30-patient demo cohort, the adaptive acquisition loop runs very few iterations (mean 0.17 features acquired per patient) because most patients already have DSS ≥ 0.85 at initialization. With so few acquisitions occurring, the feature selection strategy has minimal opportunity to differentiate itself from random. The advantage of the full model manifests in cost — 0.033 vs. 0.076 — meaning the full model achieves the same result while acquiring cheaper tests. On patients requiring more acquisitions (e.g., at 80% missing rate), the differences would be more pronounced.

---

### Q36. How does the system perform at different missing rates?

**A:** From the robustness analysis (metrics_report.json):
- 20% missing: balanced accuracy = 0.81, mean DSS = 0.99, mean acquisitions = 0.17
- 40% missing: balanced accuracy = 0.79, mean DSS = 0.98, mean acquisitions = 0.17
- 60% missing: balanced accuracy = 0.81, mean DSS = 0.99, mean acquisitions = 0.10
- 80% missing: balanced accuracy = 0.91, mean DSS = 0.98, mean acquisitions = 0.63

The system maintains high DSS across all missing rates. At 80% missing, more acquisitions are needed (0.63 vs 0.10-0.17), but the balanced accuracy actually improves, likely because the acquired features are the most informative ones.

---

### Q37. ⚡ How does the tau_DSS threshold affect behavior?

**A:** Lower τ_DSS → fewer acquisitions (easier to satisfy), potentially less accurate.  
Higher τ_DSS → more acquisitions (harder to satisfy), potentially more accurate but higher cost.

From robustness analysis:
- τ=0.70: 0.07 acquisitions/patient, cost=0.017, balanced accuracy=0.8448
- τ=0.85: 0.23 acquisitions/patient, cost=0.033, balanced accuracy=0.8621
- τ=0.95: 0.37 acquisitions/patient, cost=0.033, balanced accuracy=0.8621

The balanced accuracy plateaus at τ=0.85 — raising τ beyond 0.85 increases acquisitions without improving accuracy. This validates the default τ=0.85 choice.

---

### Q38. What is the oracle simulation?

**A:** When the system "acquires" a feature, it reads the true value from the ground-truth matrix X_true: `X_obs[patient, j*] = X_true[patient, j*]`. In real deployment, this would be an actual medical test or EHR query. The oracle simulation verifies the algorithm works correctly; clinical validation requires real-world deployment.

---

### Q39. ⚡ Can DSG_j ever be negative, and what does that mean?

**A:** Yes. A negative DSG_j means that acquiring feature j is expected to DECREASE decision sufficiency — it might reveal information that conflicts with the current prediction, making the system less confident. For example, if the model currently predicts "Diabetic" with high confidence, but hypothetical HighChol=0 values (low cholesterol — unusual for diabetics) would weaken the prediction, DSG_HighChol could be negative. The system avoids such features by assigning them a low acquisition score.

---

### Q40. ⚡ Why not use information gain or expected model change instead of DSG?

**A:** Information gain and expected model change are valid alternatives used in classical active learning. Our DSG formulation has a specific advantage: it directly measures the expected improvement in DECISION STABILITY, not just information. A feature might be highly informative (high information gain) but not change the prediction — DSG correctly identifies this as low gain. Conversely, a feature might have modest information content but dramatically resolve a borderline case — DSG correctly identifies this as high gain. DSG is decision-aware, not just information-aware.

---

## SECTION D: EVALUATION AND RESULTS (Questions 41–50)

---

### Q41. ★★★ What are your key experimental results?

**A:** Two sets of results:

**Full dataset (~38k test patients, from `dss_diagnostic_results.csv`):**
- Full features + balanced weights: balanced accuracy = 0.7502, ROC-AUC = 0.8285
- Masked (K=5 augmentation) + balanced weights: balanced accuracy = 0.7004, ROC-AUC = 0.7678

**Demo cohort (30 test patients, from `metrics_report.json`):**
- Initial (partial observation): balanced accuracy = 0.8276, DSS = 0.9466
- Post-adaptive acquisition: balanced accuracy = 0.8621, DSS = 0.9800
- Mean acquisitions: 0.17 features/patient
- Early stop rate: 100%

---

### Q42. ⚡ Why are there two different sets of results?

**A:** `diagnose_dss.py` runs four LightGBM experiments on the full dataset (~38k test patients) with independent splits and masks. `metrics.py` runs the complete DSS/DSG adaptive acquisition pipeline on a smaller demo cohort (30 patients from the test split) for tractable computation. The two scripts use slightly different split seeds, so their test sets are not identical. Both are valid — the full-dataset results show classifier performance at scale, while the demo cohort results demonstrate the adaptive acquisition framework.

---

### Q43. ★★★ What is the performance drop from full features to masked input?

**A:** On the full dataset:
- Full features (balanced weights, val-tuned): balanced accuracy = 0.7502, ROC-AUC = 0.8285
- Masked K=1 (balanced weights, val-tuned): balanced accuracy = 0.6949, ROC-AUC = 0.7639
- Masked K=5 augmentation (balanced weights): balanced accuracy = 0.7004, ROC-AUC = 0.7678

The drop in balanced accuracy is about 5–6 percentage points (0.7502 → 0.6949), which is expected given that 40–80% of features are missing. Mask augmentation (K=5) recovers ~0.5 percentage points over K=1.

---

### Q44. Does the adaptive acquisition loop improve performance?

**A:** On the 30-patient demo cohort: balanced accuracy improves from 0.8276 to 0.8621 (+3.5 pp), specificity improves from 0.6552 to 0.7241, and mean DSS increases from 0.9466 to 0.9800, with a mean of only 0.17 additional features acquired per patient. The improvement is modest in absolute terms but achieved at very low cost.

---

### Q45. ⚡ Isn't 30 patients too small for reliable evaluation?

**A:** Yes — 30 patients is a demo cohort, not a full evaluation. With only 1 true positive in the test subset, metrics like precision (0.09–0.11) and F1 are unreliable due to small counts. The demo cohort demonstrates that the pipeline runs correctly and produces reasonable outputs, but the specific numbers should not be quoted as definitive performance benchmarks. The full-dataset results from `dss_diagnostic_results.csv` are more representative.

---

### Q46. How did mask augmentation help?

**A:** Mask augmentation (K=3 or K=5 copies of each training patient with different random masks) increases training set diversity. Instead of seeing each patient with only one random mask, LightGBM sees the same patient with 3–5 different observation patterns. This teaches the model to be robust to different missingness patterns. On the full dataset, K=5 augmentation improved balanced accuracy from 0.6949 (K=1) to 0.7004 (+0.5 pp) and ROC-AUC from 0.7639 to 0.7678.

---

### Q47. What evaluation metrics did you use and why?

**A:** Primary: balanced accuracy (handles class imbalance), ROC-AUC (threshold-independent ranking quality), sensitivity (catching diabetics), specificity (correctly dismissing non-diabetics). Secondary: Brier score (calibration), PR-AUC (precision-recall trade-off), macro-F1. For the adaptive loop: DSS, delta-DSS, mean acquisitions, mean cost, early stop rate. We do NOT use raw accuracy as a primary metric because the 86% non-diabetic majority makes it misleading.

---

### Q48. ⚡ Your precision is very low (0.25-0.27). Is that acceptable?

**A:** Given the 13.9% positive prevalence, low precision with a sensitive classifier is expected. Even with 73% sensitivity and 67% specificity on ~38,000 patients, precision = TP/(TP+FP) ≈ 3870/(3870+10808) ≈ 0.26. This is a known consequence of the prevalence-precision relationship. In clinical screening contexts, high sensitivity (catching all diabetics) is often prioritized over precision, because the cost of missing a diabetic patient typically exceeds the cost of a false alarm that triggers follow-up testing.

---

### Q49. ★★★ What does the Pareto analysis show?

**A:** The Pareto frontier (metrics_report.json) shows the trade-off between acquisition cost and diagnostic accuracy:
- Budget 0 (no acquisitions): balanced accuracy = 0.8276, DSS = 0.9466, cost = 0.
- Budget max=3: balanced accuracy = 0.8621, DSS = 0.9898, cost = 0.033.

This shows that a very small number of acquisitions (~0.23 per patient at max budget) yields measurable improvements in both accuracy and decision sufficiency. The AU-BAC (Area Under Budget-Accuracy Curve) = 0.70 summarizes this trade-off.

---

### Q50. ⚡ Your system shows 100% early stop rate. Is that suspicious?

**A:** Not necessarily, given the experimental setup. With masks exposing 20–60% of features (mean ~40% observed), many patients already have enough information for a stable prediction — the CVAE completions largely agree with each other because the observed features are sufficiently constraining. DSS ≥ 0.85 is achieved at initialization for most patients, so few acquisitions are needed. At 80% missing rate, the early stop rate is still 100% but mean acquisitions increase to 0.63, showing that more extreme missingness does require more effort.

---

## SECTION E: LIMITATIONS AND FUTURE WORK (Questions 51–60)

---

### Q51. ★★★ What are the main limitations of your project?

**A:** Five key limitations:
1. **Synthetic masking:** MCAR masks may not reflect real clinical MNAR missingness.
2. **Oracle simulation:** "Acquired" values come from ground truth, not actual tests.
3. **Small demo cohort:** The 30-patient adaptive acquisition evaluation is demonstrative, not statistically definitive.
4. **Single-model pipeline:** The reference ensemble (XGBoost + TabResNet) was replaced by single LightGBM.
5. **No clinical validation:** The system has not been tested in a hospital, lacks external validation, and has no regulatory approval.

---

### Q52. What would you do differently if you could start over?

**A:** (1) Use a larger evaluation cohort (hundreds or thousands of patients) for the adaptive acquisition benchmark. (2) Implement the full ensemble as specified in the reference design and compare it against the single-model pipeline. (3) Explore MNAR-aware mask generation based on clinician behavior models. (4) Add formal probability calibration (Platt scaling) to LightGBM outputs. (5) Train multiple CVAE checkpoints and ensemble their completions for more diverse plausible profiles.

---

### Q53. ⚡ Your system uses ordinal cost values (1-10) instead of real hospital costs. How does this affect results?

**A:** The ordinal scale preserves the relative ranking of feature acquisition costs — self-report questions remain cheap relative to blood draws. The absolute values affect the exact trade-off between cost and DSS gain, but the multi-objective framework remains valid because only the normalized cost C̃_j enters the acquisition score. Replacing ordinal values with real dollar amounts in `DEFAULT_RAW_COSTS` would change the specific feature ordering but not the algorithmic structure.

---

### Q54. ★★★ Is this system safe for clinical deployment?

**A:** No. This is a research prototype. It lacks: (1) external validation on independent populations; (2) prospective clinical trials; (3) FDA/CE regulatory approval; (4) integration with EHR systems; (5) physician-in-the-loop safeguards. The system should be used only for academic research. Any clinical decision MUST involve a licensed physician.

---

### Q55. What future work do you recommend?

**A:** Five directions:
1. **Real clinical data:** Validate on EHR data with natural missingness (MNAR).
2. **Ensemble activation:** Train XGBoost + TabResNet and evaluate Bayesian fusion against single LightGBM.
3. **Calibration:** Apply Platt scaling or isotonic regression to improve probability calibration.
4. **Larger evaluation:** Run adaptive acquisition on 1,000+ test patients for reliable performance estimates.
5. **Cost optimization:** Integrate real hospital cost data and optimize λ weights via Bayesian optimization.

---

### Q56. ⚡ How would you validate this system clinically?

**A:** A clinical validation study would require: (1) IRB approval and a clinical data use agreement; (2) retrospective validation on EHR data from multiple hospitals with naturally missing values; (3) prospective pilot study where the system's feature acquisition recommendations are compared against physician-selected tests; (4) subgroup analysis across age, sex, and ethnicity to assess fairness; (5) formal calibration and clinical utility analysis. This is a multi-year effort requiring clinical partnerships.

---

### Q57. ⚡ Could your system be unfair to certain demographic groups?

**A:** Potentially. The BRFSS dataset has demographic biases (overrepresentation of certain age/income groups, self-selection into phone surveys). If the mutual information graph and LightGBM learn different feature importance patterns for different demographic subgroups, the acquisition strategy could systematically recommend different tests for different populations. Fairness analysis across Age, Sex, Income, and Education subgroups is an important direction for future work.

---

### Q58. How does your approach compare to existing active feature acquisition methods?

**A:** Previous work on active feature acquisition typically uses: (1) Expected Prediction Change (EPC) — greedy information gain without cost awareness; (2) Reinforcement Learning — trains a policy network to select features (e.g., JAFA, ASAC); (3) Bayesian approaches — use posterior uncertainty for acquisition decisions. Our approach differs by combining four factors (relevance, connectivity, DSG, cost) into a single interpretable score, using the DSS stopping criterion instead of a fixed budget, and employing a CVAE for counterfactual reasoning rather than a simple imputation model.

---

### Q59. ⚡ If the CVAE is poorly trained, how does that affect DSS?

**A:** A poorly trained CVAE would generate unrealistic completions. If completions are too similar (collapsed posterior), CS_t and PD_t would artificially indicate high stability, leading to premature stopping with potentially wrong predictions. If completions are too diverse (noisy), CS_t would be artificially low, causing unnecessary acquisitions. CVAE quality is therefore critical. The ELBO loss during training should be monitored, and the plausible completions should be spot-checked for realism.

---

### Q60. ★★★ Summarize the contribution of your project in one paragraph.

**A:** This project implements and evaluates a Decision-Sufficiency-Guided framework for adaptive diabetes risk prediction under partial clinical observability. Given a patient with incomplete health indicators, the system uses a LightGBM classifier on masked inputs, a Conditional Variational Autoencoder for uncertainty quantification via plausible profile completions, and a Clinical Information Graph for feature-relationship modeling. The Decision Sufficiency Score combines class stability and predictive divergence to determine when evidence is sufficient, and a multi-objective acquisition score integrating feature relevance, graph connectivity, expected DSS gain, and cost guides the selection of the next most informative clinical test. Empirical evaluation on the CDC BRFSS dataset demonstrates that the framework achieves balanced accuracy improvements with fewer than one additional feature acquisition per patient on average, and ablation studies confirm that all four scoring components contribute to cost-effective acquisition.
