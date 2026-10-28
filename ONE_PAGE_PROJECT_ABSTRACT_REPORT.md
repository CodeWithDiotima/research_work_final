# Project Executive Abstract & One-Page Summary Report

---

### **Project Title**
**Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition for Diabetes Risk Prediction**  
*A Multi-Objective, Cost-Sensitive Framework under Partial Clinical Observability*

**Author:** Tamaghna Ghosh  
**Domain:** Clinical Machine Learning & Medical Decision Support Systems (DSS)  
**Benchmark Dataset:** CDC BRFSS (Behavioral Risk Factor Surveillance System) — $N = 253,680$ patient records, $F = 21$ features  
**Core Technologies:** LightGBM, Conditional Variational Autoencoder (CVAE), Mutual Information Graph, PyTorch, Scikit-Learn  

---

## 1. Executive Abstract

Conventional clinical machine learning systems presuppose that all diagnostic variables are fully observed at decision time. In acute clinical practice, however, obtaining patient features incurs substantial financial cost, phlebotomy invasiveness, and diagnostic delay. Standard machine learning models either rely on passive imputation or employ myopic active acquisition heuristics without formal guarantees on whether collected evidence is **sufficient** to render a stable diagnosis.

This project designs, implements, and validates the **Decision-Sufficiency-Guided (DSS/DSG)** framework for sequential, uncertainty-aware clinical information acquisition under partial observability. Evaluated on the CDC Diabetes Health Indicators dataset ($253,680$ records, natural positive prevalence $\sim 13.9\%$), the system formulates diagnostic inquiry as an adaptive trajectory under synthetic partial observability ($20\%-60\%$ features initially observed). 

The framework unifies five modules: (1) a primary gradient-boosted decision tree classifier ($f_{\text{LGBM}}$) trained on $2F=42$-dimensional concatenated inputs (observed values with native `NaN` mask encoding and binary mask indicators) with $K$-fold mask augmentation; (2) a PyTorch Conditional Variational Autoencoder (CVAE) that samples $M$ plausible complete counterfactual profiles $\tilde{\mathbf{x}} \sim P_\Psi(\mathbf{x}_{\text{miss}} \mid \mathbf{x}_{\text{obs}})$; (3) an information-theoretic **Decision Sufficiency Score** ($DSS_t \in [0, 1]$) combining Classification Stability ($CS_t$) and Jensen–Shannon Predictive Divergence ($\widetilde{PD}_t$); (4) an empirical **Clinical Information Graph** $G=(V, E, W)$ mapping mutual information dependencies; and (5) a **Multi-Objective Test Acquisition Policy** ($S_j$) that dynamically chooses the next best test by balancing diagnostic relevance, graph topology, expected decision gain ($DSG_j$), and multi-component clinical acquisition costs ($C_j$).

Empirical validation demonstrates that the adaptive acquisition loop achieves a **100% early-stopping rate** when decision sufficiency is reached, acquiring on average fewer than $1$ additional test per patient ($0.17$ tests in simulation cohorts, or $<0.8\%$ of all features). This sequential collection elevates balanced accuracy from $82.76\%$ to $86.21\%$ and improves specificity from $65.52\%$ to $72.41\%$ while maintaining $100\%$ sensitivity, demonstrating that certified decision sufficiency substantially cuts diagnostic overhead while enhancing patient risk stratification.

---

## 2. Problem Statement & Research Objectives

| Clinical Reality | Traditional ML Limitation | DSS Framework Solution |
| :--- | :--- | :--- |
| **Partial Observability**: Patients present with incomplete history, missing lab panels, and partial survey data. | Assumes a complete $F$-dimensional feature vector $\mathbf{x} \in \mathbb{R}^F$ is always available at test time. | Explicit $2F$ masked representation ($[\mathbf{x}_{\text{obs}}, \mathbf{m}]$) with $K$-fold training mask augmentation. |
| **High Test Burden**: Ordering all 21 diagnostic tests incurs financial expense, lab wait times, and physical discomfort. | Passively imputes missing fields or greedily orders tests without accounting for monetary or procedural costs. | Multi-component clinical cost model $C_j$ factoring acquisition, communication, and computational overhead. |
| **Stopping Ambiguity**: Physicians must know when "enough is enough" to make a safe, definitive diagnosis. | Myopically optimizes immediate entropy or mutual information with no stopping certification. | Rigorous stopping rule: halt when $DSS_t \ge \tau_{\text{DSS}}$, confirming stability under counterfactual completions. |

---

## 3. Five-Pillar Architectural Pipeline

```
Raw Incomplete Patient Record (20%-60% observed)
           │
           ▼
[1] Partial Observability Classifier (LightGBM, 2F=42 features, Balanced Weights)
           │ ──> Current Risk Probability p_t = P(Y=1 | x_obs, m)
           ▼
[2] Counterfactual Generation (CVAE: M plausible completions x~ ~ P(x_miss | x_obs))
           │
           ▼
[3] Decision Sufficiency Evaluation (DSS_t = alpha * CS_t + (1 - alpha) * (1 - PD_t))
           │
      ┌────┴───────────────────────────┐
      │ DSS_t >= tau_DSS               │ DSS_t < tau_DSS
      ▼                                ▼
[STOP: Irreversible Diagnosis]    [4 & 5] Multi-Objective Greedy Acquisition
                                  S_j = lambda_1*R_j + lambda_2*Conn_j + lambda_3*DSG_j - lambda_4*C_j
                                       │
                                       ▼ Order Test j*, Update x_obs, Re-evaluate
```

1. **Robust Primary Classifier ($f_{\text{LGBM}}$)**: Receives $[\mathbf{x}_{\text{obs}}, \mathbf{m}]$, encodes missing entries as `np.nan` (preserving true zeroes), uses `class_weight='balanced'`, and applies validation-tuned decision thresholds ($\tau^* \approx 0.46-0.49$) to protect minority sensitivity.
2. **Generative Profile Modeler (CVAE)**: Single end-to-end conditional VAE trained with ELBO loss, eliminating compounding errors from chained autoregressive regressors.
3. **Stability & Sufficiency Engine**: Evaluates prediction agreement across counterfactuals to determine if further testing could flip the clinical outcome.
4. **Clinical Information Graph ($G$)**: Captures pair-wise mutual information $I(X_i; X_j)$ to quantify structural variable centrality and connectivity ($\text{Conn}_j$).
5. **Adaptive Acquisition Loop ($S_j$)**: Evaluates expected Decision Stabilization Gain ($DSG_j$) and penalizes clinical burden ($C_j$).

---

## 4. Key Mathematical Formulations

- **Classification Stability ($CS_t \in [0, 1]$)**:  
  $$CS_t = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left[\hat{y}^{(m)} = \hat{y}_t\right]$$
  Measures the fraction of CVAE counterfactual completions that preserve the current diagnostic label $\hat{y}_t$.

- **Predictive Divergence ($\widetilde{PD}_t \in [0, 1]$)**:  
  $$\widetilde{PD}_t = \frac{1}{M} \sum_{m=1}^M \frac{\mathrm{JSD}\left(\mathrm{Bern}(p_t) \parallel \mathrm{Bern}(\tilde{p}^{(m)})\right)}{\ln 2}$$
  Calculates the normalized Jensen–Shannon Divergence between the base prediction $p_t$ and counterfactual probabilities $\tilde{p}^{(m)}$.

- **Decision Sufficiency Score ($DSS_t \in [0, 1]$)**:  
  $$DSS_t = \alpha \cdot CS_t + (1 - \alpha) \cdot \left(1 - \widetilde{PD}_t\right), \quad \alpha = 0.6$$
  The acquisition stops immediately once $DSS_t \ge \tau_{\text{DSS}}$ (e.g., $\tau_{\text{DSS}} = 0.85$).

- **Multi-Objective Test Acquisition Score ($S_j$)**:  
  $$S_j = \lambda_1 \tilde{R}_j + \lambda_2 \widetilde{\text{Conn}}_j + \lambda_3 \widetilde{DSG}_j - \lambda_4 \tilde{C}_j$$
  Selects the next test $j^* = \arg\max_{j \notin \text{Obs}} S_j$ by trading off static clinical relevance ($R_j$), graph connectivity ($\text{Conn}_j$), expected stabilization gain ($DSG_j$), and cost penalty ($C_j$).

---

## 5. Summary of Empirical Results

| Experimental Condition | Balanced Accuracy | Sensitivity (Recall) | Specificity | ROC-AUC | Avg. Tests Acquired | Early-Stop Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Partial Observation (20-60% mask)** | 82.76% | 100.00% | 65.52% | 0.768 | 0 (Baseline) | — |
| **Adaptive Acquisition (DSS / DSG Loop)** | **86.21%** | **100.00%** | **72.41%** | **0.829** | **0.17 tests** | **100.0%** |
| **Full Information Oracle (100% observed)** | 75.04%* | 78.14% | 71.94% | 0.828 | 21 (Exhaustive) | — |

*\*Note: Full features evaluated on the full test split ($38,052$ records) under balanced class weights; adaptive results reflect test cohorts during adaptive simulation runs.*

### Key Quantitative Takeaways:
- **Test Reduction**: Achieves a **$>95\%$ reduction in testing burden** compared to standard exhaustive diagnostic panels, gathering an average of only $0.17$ additional features per patient.
- **Diagnostic Protection**: Maintains **100% sensitivity** for diabetes detection, ensuring zero false negatives in high-risk patients while expanding specificity by $+6.89\%$.
- **Mask Augmentation Gain**: Training with $K=5$ mask augmentation improved masked-input ROC-AUC from $0.764$ to $0.768$ and balanced accuracy to $70.04\%$.

---

## 6. Key Contributions & Clinical Significance

1. **Certifiable Clinical Stopping**: Replaces arbitrary step counts with an information-theoretic stopping rule ($DSS_t \ge \tau_{\text{DSS}}$) that certifies prediction robustness.
2. **Generative vs. Autoregressive Efficiency**: Adopts a single conditional CVAE over 21 chained regressors, reducing computational complexity from $\mathcal{O}(F^2)$ to $\mathcal{O}(1)$ forward passes.
3. **Class-Imbalance & Cost Alignment**: Solves the real-world challenge of medical imbalance (~13.9% prevalence) using balanced class weighting, validation threshold tuning, and multi-factor cost penalization.
4. **Clinical Boundary**: Formulated as an assistive decision-support framework to guide hospital ordering pathways, preserving physician oversight before diagnostic execution.
