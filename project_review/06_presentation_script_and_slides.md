# 06 — Presentation Script and Slide Outline

---

## Part 1: The 30-Second Elevator Pitch

> "My project tackles diabetes risk prediction when patient information is incomplete. I built a system that uses a LightGBM classifier on partial observations, a Conditional VAE to model what missing features might look like, and a Decision Sufficiency Score to know when the evidence is enough. When it's not enough, the system recommends the most cost-effective clinical test to order next. On the CDC diabetes dataset with synthetic missingness, the system achieves balanced accuracy improvements with fewer than one additional feature per patient."

---

## Part 2: The 2-Minute Presentation Script

> "Good morning, my project is titled 'A Decision-Sufficiency-Guided Framework for Adaptive Diabetes Risk Prediction under Partial Clinical Observability.'
>
> **The problem:** In clinical practice, you rarely have all the information you need. When a patient arrives with incomplete health records, a clinician faces two questions: Is what I have enough to make a reliable assessment? If not, what should I test next?
>
> **Our approach:** We built a three-component framework using the CDC Behavioral Risk Factor Surveillance dataset — 253,000 patients, 21 health indicators, about 14% diabetic.
>
> Component one: a LightGBM gradient-boosted classifier that receives observed feature values alongside explicit missing-data indicators. It's trained on masked inputs where 40 to 80 percent of features are hidden, using mask augmentation and class-weight balancing.
>
> Component two: a Conditional Variational Autoencoder that generates 100 plausible completions of the missing features, preserving known values exactly. These completions let us measure prediction stability.
>
> Component three: the Decision Sufficiency Score, which combines class stability — do the completions agree on the predicted class? — with predictive divergence — are the probability values consistent? When DSS exceeds 0.85, we stop. When it doesn't, we compute the expected DSS gain from each missing feature, combine it with feature relevance, graph connectivity, and acquisition cost, and greedily select the best next test.
>
> **Results:** On the CDC dataset, the classifier achieves a balanced accuracy of 0.75 on the full dataset with balanced weights. With partial observations, the adaptive framework raises DSS from 0.95 to 0.98 with fewer than one additional feature per patient on average.
>
> **Conclusion:** This is a research prototype demonstrating that decision-sufficiency-guided acquisition can reduce unnecessary testing while maintaining diagnostic reliability. It requires clinical validation before deployment."

---

## Part 3: The 5-Minute Extended Presentation Script

### Opening (30 seconds)

> "Thank you for the opportunity to present my project: a Decision-Sufficiency-Guided Framework for Adaptive Diabetes Risk Prediction under Partial Clinical Observability.
>
> The core question: when you have incomplete patient information, how do you know when you have enough to make a reliable prediction, and if you don't, what should you collect next?"

### Background and Motivation (60 seconds)

> "Diabetes affects over 37 million Americans. Early risk identification through population-level screening can guide preventive interventions. However, screening questionnaires and medical tests are often incomplete — patients skip questions, lab results are pending, or budget constraints limit the tests that can be ordered.
>
> Existing classifiers assume complete feature vectors. When features are missing, common approaches either discard the patient, impute with means, or treat missing as zero — all of which lose information or introduce bias.
>
> Our framework takes a different approach: it reasons explicitly about what is known, what is unknown, and how the unknown affects the reliability of the prediction."

### System Architecture (90 seconds)

> "The framework has three main components.
>
> First, a LightGBM classifier trained on partially observed data. Each input is a 42-dimensional vector: 21 feature values — with NaN for missing entries, not zero — concatenated with 21 binary mask indicators. This lets the model learn different behavior for 'feature is zero' versus 'feature is unmeasured.' The model uses class-weight balancing for the imbalanced dataset and a validation-tuned decision threshold.
>
> Second, a Conditional Variational Autoencoder. The CVAE learns the conditional distribution of missing features given observed features. At inference, it generates 100 plausible complete patient profiles by sampling latent codes from the prior distribution and decoding them, while preserving all observed values exactly through an identity constraint. These completions are NOT used for the final prediction — they are used purely for uncertainty quantification.
>
> Third, the Decision Sufficiency Score. For each of the 100 plausible profiles, we run LightGBM and record the predicted class and probability. Class stability CS measures how often the completions agree with the current prediction. Predictive divergence PD measures how much the probabilities vary, using Jensen-Shannon divergence. DSS combines both: DSS = 0.5 times CS plus 0.5 times one minus PD. When DSS exceeds the threshold of 0.85, we stop — current evidence is sufficient."

### Adaptive Acquisition Loop (60 seconds)

> "When DSS is below the threshold, we need more information. But which feature should we acquire?
>
> For each missing feature j, we compute the Decision Stabilization Gain: how much would DSS improve if we observed feature j? We estimate this using 5 CVAE-sampled candidate values and computing future DSS for each.
>
> The acquisition score combines four factors: feature relevance — how informative is feature j for diabetes prediction? Graph connectivity — how related is j to what we already know? Expected DSS gain — how much will it help? And acquisition cost — how expensive or invasive is the test?
>
> We select the feature with the highest score, reveal its true value from the ground-truth matrix — this is an oracle simulation — update the patient record, and repeat.
>
> Our ablation study shows that all four scoring components contribute to cost-effective acquisition. The full scoring formula achieves the same accuracy as random feature selection but at less than half the cost."

### Results and Limitations (60 seconds)

> "On the full CDC dataset with approximately 38,000 test patients, the LightGBM classifier with balanced class weights achieves a balanced accuracy of 0.75 and ROC-AUC of 0.83 with all features available. Under partial observation with mask augmentation, balanced accuracy drops to 0.70 with ROC-AUC of 0.77 — a modest 5-point decrease despite having 40 to 80 percent of features missing.
>
> The adaptive acquisition pipeline, evaluated on a demonstration cohort, achieves a 100% early stop rate with fewer than one additional feature per patient, raising the DSS from 0.95 to 0.98.
>
> I want to be transparent about the limitations. First, our missing data is synthetic — random masks may not match real clinical missingness patterns. Second, 'acquired' features come from ground truth, not real tests. Third, the demo cohort is small. Fourth, the system has not been clinically validated. This is a methodological contribution, not a deployable medical device."

### Conclusion (30 seconds)

> "In summary, I have implemented a complete pipeline for decision-sufficiency-guided adaptive feature acquisition for diabetes risk prediction. The framework correctly integrates a masked-input classifier, a conditional generative model for uncertainty quantification, and a multi-objective scoring function for cost-aware feature selection. The code is modular, fully reproducible, and accompanied by comprehensive evaluation including ablation studies and robustness analyses.
>
> Thank you. I'm happy to take questions."

---

## Part 4: Ten-Slide Outline

### Slide 1: Title Slide
- Title: "A Decision-Sufficiency-Guided Framework for Adaptive Diabetes Risk Prediction under Partial Clinical Observability"
- Author name, affiliation, date
- Advisor/supervisor name

### Slide 2: Problem Statement
- "What if you don't have all the information?" — clinical scenario illustration
- Standard classifiers assume complete data
- Two key questions: Is current evidence sufficient? What to test next?
- CDC BRFSS dataset: 253,680 patients, 21 features, 13.9% diabetic

### Slide 3: System Architecture Overview
- Three-component diagram:
  1. LightGBM Classifier (42-input: [x_nan | m])
  2. CVAE Profile Generator (P(x_miss | x_obs))
  3. DSS + Adaptive Acquisition Loop
- Data flow: partial observation → prediction + uncertainty → stopping or acquisition

### Slide 4: Data Pipeline
- Stratified 70/15/15 split
- Preprocessing: z-score (3 continuous), integer codebook (18 categorical), train-only fitting
- Synthetic observation masks (20–60% observed)
- Mask augmentation (K=3 on training)

### Slide 5: LightGBM Classifier Design
- Input: [x_nan | m] = 42 features
- Missing entries → NaN (native handling)
- class_weight="balanced" for 86%/14% imbalance
- Validation-tuned decision threshold
- Early stopping on validation logloss

### Slide 6: CVAE and Plausible Completions
- Architecture: Encoder q(z|x,m) → latent z → Decoder p(x|z,x_obs,m)
- ELBO loss = MSE + β × KL divergence
- Identity constraint: x̃ = x_obs × m + x̂ × (1−m)
- Generates M=100 diverse plausible profiles
- Completions used for UNCERTAINTY QUANTIFICATION only, not for final prediction

### Slide 7: Decision Sufficiency Score (DSS)
- Class Stability: CS_t = fraction of completions agreeing with current prediction
- Predictive Divergence: PD_t via Jensen-Shannon divergence
- DSS_t = 0.5 × CS_t + 0.5 × (1 − PD_t)
- Stopping rule: DSS_t ≥ 0.85 → sufficient, STOP
- Diagram: DSS increasing with each acquisition step

### Slide 8: Adaptive Acquisition Scoring
- Acquisition score: S_j = 0.35 R_j + 0.20 Conn_j + 0.35 DSG_j − 0.10 C_j
- Feature relevance R_j (mutual information with label)
- Graph connectivity Conn_j (MI between features)
- Expected DSS gain DSG_j (counterfactual simulation)
- Cost penalty C_j (heuristic clinical cost model)
- Ablation results table showing each component's contribution

### Slide 9: Experimental Results
- Full-dataset table: full features vs. masked vs. augmented
- Demo cohort table: initial vs. post-acquisition (DSS, balanced accuracy, acquisitions)
- Robustness: performance across missing rates (20%–80%)
- Tau-DSS sensitivity analysis
- Key finding: balanced accuracy maintained with minimal acquisitions

### Slide 10: Limitations, Future Work, and Conclusion
- Limitations: synthetic masking, oracle simulation, small demo cohort, no clinical validation, single-model pipeline
- Future work: real EHR data, ensemble activation, probability calibration, larger evaluation, clinical pilot study
- Conclusion: complete, modular framework for decision-sufficiency-guided adaptive feature acquisition; methodological contribution requiring clinical validation

---

## Part 5: Key Phrases to Memorize for Viva

1. **Opening:** "My system predicts diabetes risk when patient information is incomplete, using a Decision Sufficiency Score to determine when current evidence is reliable enough."

2. **DSS formula:** "DSS equals alpha times class stability plus one-minus-alpha times one-minus normalized predictive divergence. Alpha is 0.5, giving equal weight to both stability measures."

3. **Acquisition score:** "S_j equals lambda-1 times relevance plus lambda-2 times connectivity plus lambda-3 times DSG minus lambda-4 times cost. It balances diagnostic value against clinical cost."

4. **CVAE role:** "The CVAE generates plausible completions of missing features for uncertainty quantification only — the final prediction is always made on actually observed features."

5. **LightGBM input:** "LightGBM receives 42 inputs: 21 feature values with NaN for missing entries, plus 21 binary mask indicators. This lets the model distinguish missing from measured-zero."

6. **Limitation acknowledgment:** "This is a research prototype evaluated with synthetic missingness and oracle feature acquisition. Clinical validation on real EHR data is required before any deployment."

7. **Ablation takeaway:** "The ablation study shows that all four scoring components contribute to cost-effective acquisition — the full model achieves equivalent accuracy to simpler strategies at half the cost."

8. **Class imbalance handling:** "We use balanced class weights in LightGBM, validation-tuned decision threshold, and balanced accuracy as the primary metric to handle the 86%-to-14% class imbalance."

9. **Identity constraint:** "Observed values are never changed. The CVAE completion formula x-tilde equals x-obs times m plus x-hat times one-minus-m ensures that known facts about the patient are preserved exactly."

10. **Why not imputation:** "We don't impute missing values for the final prediction. LightGBM handles NaN natively. CVAE completions are used only to measure how stable the prediction would be across different possible values of the missing features."

---

## Part 6: Anticipated Tough Questions and Prepared Responses

### "Your LightGBM input uses NaN for missing features AND appends the mask. Isn't the mask redundant?"

**Prepared response:** "Partially. LightGBM can learn to handle NaN values at each tree split, so in principle the mask is redundant. However, the explicit mask provides two benefits: (1) it allows the model to make global splits on feature availability — for example, 'if more than 10 features are missing, go left' — which NaN alone cannot express; (2) it creates explicit feature interaction terms between availability patterns and observed values. In practice, having both representations marginally improves performance by making the missingness pattern explicitly available as a first-class input signal."

### "How do you know your CVAE isn't just memorizing training patients?"

**Prepared response:** "Three safeguards against memorization: (1) the KL divergence term in the ELBO loss regularizes the latent space toward a standard Gaussian, preventing the encoder from simply memorizing patient-specific codes; (2) dropout (0.1) in both encoder and decoder provides stochastic regularization; (3) the latent dimension (16) is much smaller than the input dimension (63 for the encoder), forcing information compression. Additionally, the identity constraint at inference means the CVAE only needs to generate values for missing features — a much smaller subspace than the full patient record."

### "Your system acquires features from ground truth. How do you validate it would work with real test results?"

**Prepared response:** "The oracle simulation validates the algorithmic correctness of the acquisition strategy — the DSG scoring, DSS stopping, and feature selection logic all work correctly. What it does NOT validate is the practical integration: real test results might arrive with measurement noise, time delays, or reporting errors that the oracle does not simulate. A clinical pilot study would be needed to assess these factors. The framework itself is agnostic to how the acquired value arrives — the `acquire_actual_value` function simply sets `X_obs[patient, j] = value` and `m[patient, j] = 1` — so swapping the oracle with an EHR API requires changing only that function."

### "Why is your balanced accuracy only 0.70 for masked input when the full-feature model gets 0.75?"

**Prepared response:** "A 5-percentage-point drop in balanced accuracy when 40–80% of features are missing is actually a modest degradation. The model is predicting with an average of only 8–12 out of 21 features available. That it maintains 0.70 balanced accuracy and 0.77 ROC-AUC in this regime demonstrates that the masked-input representation ([x_nan | m]) effectively preserves diagnostic signal. The adaptive acquisition framework then recovers some of this gap by identifying the most informative features to collect."

### "Is your project actually novel, or is it just implementing known techniques?"

**Prepared response:** "The individual components — LightGBM, CVAE, mutual information graphs, greedy feature selection — are established techniques. The novelty lies in their specific integration: (1) the Decision Sufficiency Score combining class stability and JS-divergence-based predictive divergence as a stopping criterion; (2) the Decision Stabilization Gain as a feature acquisition criterion that is decision-aware, not just information-aware; (3) the four-factor acquisition score integrating relevance, connectivity, DSG, and cost in a single framework; (4) the application of this full pipeline to diabetes risk prediction under simulated partial clinical observability. The contribution is the framework design and its empirical validation, not the individual components."
