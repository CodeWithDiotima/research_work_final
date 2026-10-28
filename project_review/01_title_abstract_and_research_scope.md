# 01 — Title, Abstract, and Research Scope

---

## 1. Recommended Final Research Paper Title

**"A Decision-Sufficiency-Guided Framework for Adaptive Diabetes Risk Prediction under Partial Clinical Observability"**

---

## 2. Alternative Titles

1. "Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition for Diabetes Risk Prediction with Incomplete Health Indicators"
2. "Uncertainty-Aware Sequential Feature Acquisition for Diabetes Risk Prediction Using Conditional Variational Autoencoders and LightGBM"

---

## 3. Short Project Title

**"DSS-Guided Adaptive Diabetes Risk Prediction"**

---

## 4. Academic Subtitle

*A Simulation-Based Study Using the CDC Behavioral Risk Factor Surveillance System Dataset*

---

## 5. Why This Title Matches the Implementation

The recommended title is justified by the code audit as follows:

| Title Component | Justification from Code |
|-----------------|------------------------|
| "Decision-Sufficiency-Guided" | `stability.py`: DSS_t = alpha*CS_t + (1-alpha)*(1-PD_t) is the core stopping criterion. |
| "Adaptive" | `pipeline.py:155-339`: The iterative acquisition loop selects features dynamically based on DSG_j scores. |
| "Diabetes Risk Prediction" | `config.py:25`: TARGET_COLUMN = "Diabetes_binary"; classifier produces a risk probability P(Y=1). |
| "Partial Clinical Observability" | `data_processing.py:232-253`: Synthetic random masks enforce 20-60% feature availability. |
| "Framework" (not "System") | XGBoost and TabResNet are coded but not active; calling this a "framework" honestly acknowledges the architectural flexibility without overstating the single-model deployment. |

The working title is NOT significantly reduced because the complete pipeline IS implemented. However, "Adaptive Clinical Information Acquisition" overstates the claim slightly, since the "acquisition" is a simulation using ground-truth values from X_true (not a real clinical ordering system). "Adaptive Diabetes Risk Prediction" accurately describes what the system does.

---

## 6. Abstract

Accurate diabetes risk prediction in clinical settings is frequently complicated by incomplete patient data, where only a subset of health indicators is available at decision time. This paper presents a Decision-Sufficiency-Guided (DSS) framework that combines a LightGBM gradient-boosted classifier with a Conditional Variational Autoencoder (CVAE) and a Clinical Information Graph to enable adaptive, cost-aware feature acquisition under partial observability.

The system operates on the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset comprising 21 binary, ordinal, and continuous health indicators from over 250,000 respondents. A stratified 70/15/15 train/validation/test split preserves the natural class prevalence (~13.9% diabetic). Synthetic observation masks are applied to simulate partial information, exposing between 20% and 60% of features per patient.

The LightGBM classifier receives a 42-dimensional masked input (observed values encoded as NaN for missing entries, concatenated with binary mask indicators), enabling the model to distinguish true zero values from missing information. A CVAE, trained with Evidence Lower Bound (ELBO) optimization, generates M plausible completions of unobserved features to quantify decision stability. Class Stability (CS_t) and Jensen-Shannon Divergence-based Predictive Divergence (PD_t) are combined into the Decision Sufficiency Score (DSS_t = alpha*CS_t + (1-alpha)*(1-PD_t)) to determine when current evidence is sufficient for a stable prediction.

When DSS falls below a threshold tau_DSS, the framework selects the next most informative feature using a multi-objective acquisition score S_j that integrates feature relevance, graph connectivity, expected DSS gain (DSG_j), and normalized acquisition cost. Oracle simulation experiments demonstrate that the adaptive loop achieves a 100% early-stop rate with a mean of fewer than 1 additional feature acquired per patient on the demo cohort, accompanied by improvements in balanced accuracy and ROC-AUC over the initial partial-observation baseline.

This work provides an evidence-based framework for sequential, uncertainty-aware health indicator acquisition that respects diagnostic cost, though it is not a clinically validated medical diagnostic tool and requires prospective validation before deployment.

---

## 7. Keywords

1. Diabetes risk prediction
2. Partial observability
3. Adaptive feature acquisition
4. Decision Sufficiency Score
5. Conditional Variational Autoencoder
6. Clinical information graph
7. LightGBM
8. Class stability

---

## 8. Problem Statement

Standard machine learning classifiers for diabetes risk prediction assume that all patient features are available at prediction time. In practice, clinical data is often incomplete: some laboratory tests may be pending, some survey questions may be unanswered, and cost or privacy concerns may prevent collecting all 21 BRFSS health indicators for every patient.

The fundamental research question is: **Given an initially incomplete patient record, how should a system decide (a) whether current information is sufficient for a reliable prediction, and (b) if not, which additional information should be collected next?**

---

## 9. Research Objectives

1. **Design a partial-observability classifier** that correctly handles missing features by treating them as informatively absent (not as zero) using NaN encoding and explicit mask indicators.

2. **Implement a CVAE-based plausibility generator** that models the conditional distribution P(x_miss | x_obs) to produce diverse, realistic patient profiles for uncertainty quantification.

3. **Formulate and implement the Decision Sufficiency Score (DSS)** combining class stability and predictive distribution stability to measure whether the current observed evidence supports a reliable prediction.

4. **Design and implement a cost-aware adaptive acquisition loop** that greedily selects the next-best clinical feature based on expected DSS gain, feature relevance, graph connectivity, and acquisition cost.

5. **Evaluate the framework empirically** on the CDC BRFSS dataset, comparing prediction performance before and after adaptive acquisition and conducting ablation studies on acquisition scoring weights.

---

## 10. Scope and Limitations of the Title

### Within scope:
- The title claims a "framework," which is accurate: all major pipeline components are implemented and executable.
- "Adaptive" refers to the DSG-based dynamic selection loop, which is fully implemented in `acquisition.py` and `pipeline.py`.
- "Partial Clinical Observability" refers to synthetic masking, not naturally missing data from a clinical EHR system.

### Outside scope (must not claim):
- The system does NOT perform real-time clinical information acquisition; "acquired" features are pulled from the ground-truth X_true matrix (oracle simulation).
- The clinical graph captures statistical associations (mutual information), NOT causal relationships.
- The system is NOT clinically validated; it has not been tested in a hospital environment or approved by any regulatory body.
- The XGBoost/TabResNet ensemble fusion described in the reference paper is NOT the active classifier; LightGBM is.
- The 30-patient demo cohort results in metrics_report.json are NOT generalizable as full-dataset performance benchmarks.
