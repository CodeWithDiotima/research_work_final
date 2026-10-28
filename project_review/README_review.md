# Project Review — Master Index
## Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition System

> **Generated:** 2026-10-27 | **Audit scope:** Read-only static code audit of `dss_system_final/`

---

## Document Index

| # | File | Contents |
|---|------|----------|
| — | `README_review.md` *(this file)* | Master index + code audit summary |
| 1 | `01_title_abstract_and_research_scope.md` | Final title, abstract, keywords, objectives, scope |
| 2 | `02_implementation_audit.md` | Repository inventory, pipeline-compliance matrix, gaps, recommendations |
| 3 | `03_actual_pipeline_explained.md` | Stage-by-stage explanation with examples, Mermaid diagrams, 2-minute script |
| 4 | `04_core_concepts_and_mathematics.md` | 35-concept glossary with formulas, examples, viva questions |
| 5 | `05_viva_questionnaire_with_answers.md` | 60+ viva Q&A, high-priority marks, difficult follow-ups |
| 6 | `06_presentation_script_and_slides.md` | 30s / 2-min / 5-min scripts, 10-slide outline |

---

## Code Audit Summary (Read-Only, No Files Modified)

### Repository: `dss_system_final/`

**Dataset:** CDC Diabetes Health Indicators Dataset (UCI ML Repository ID 891)  
**Target:** `Diabetes_binary` (0 = No diabetes, 1 = Pre-diabetes/Diabetes)  
**Features:** 21 clinical indicators (3 continuous: BMI, MentHlth, PhysHlth; 18 categorical)  
**Natural class prevalence:** ~13.9% positive

---

### What Is Actually Implemented (Verified by Source Code)

| Component | File | Status |
|-----------|------|--------|
| Stratified 70/15/15 data split | `data_processing.py:146-173` | IMPLEMENTED |
| Leak-free preprocessing (fit on train only) | `data_processing.py:179-218` | IMPLEMENTED |
| Random observation masks (20-60% observed) | `data_processing.py:232-253` | IMPLEMENTED |
| Mask augmentation (K=3 per training patient) | `data_processing.py:336-349` | IMPLEMENTED |
| LightGBM classifier with 2F=42 NaN-masked inputs | `models/lightgbm_model.py:59-265` | IMPLEMENTED |
| Validation-tuned decision threshold | `models/lightgbm_model.py:160-191` | IMPLEMENTED |
| Conditional VAE (CVAE) generative model | `models/cvae.py:71-183` | IMPLEMENTED |
| CVAE ELBO loss (reconstruction + KL) | `models/cvae.py:94-103` | IMPLEMENTED |
| Clinical Information Graph (MI-weighted edges) | `graph.py:97-122` | IMPLEMENTED |
| Feature relevance R_j (mutual information) | `graph.py:112-120` | IMPLEMENTED |
| Graph connectivity Conn_j | `graph.py:82-94` | IMPLEMENTED |
| Class stability CS_t | `stability.py:33-43` | IMPLEMENTED |
| Jensen-Shannon divergence PD_t | `stability.py:46-62` | IMPLEMENTED |
| Decision Sufficiency Score DSS_t | `stability.py:65-68` | IMPLEMENTED |
| DSS stopping rule (DSS_t >= tau_DSS) | `stability.py:71-73` | IMPLEMENTED |
| Decision Stabilization Gain DSG_j | `acquisition.py:110-175` | IMPLEMENTED |
| Multi-factor acquisition score S_j | `acquisition.py:182-207` | IMPLEMENTED |
| Adaptive acquisition loop | `pipeline.py:155-339` | IMPLEMENTED |
| Ground-truth oracle value reveal | `acquisition.py:223-236` | IMPLEMENTED |
| Full evaluation suite (ROC, PR, CM, stability) | `evaluate.py`, `metrics.py` | IMPLEMENTED |
| Ablation study (lambda weights) | `metrics.py` | IMPLEMENTED |
| Experiment A vs B comparison | `evaluate.py:138-196` | IMPLEMENTED |

### Reference Models (Coded but Not in Active Pipeline)

| Component | File | Status |
|-----------|------|--------|
| XGBoost classifier | `models/xgboost_model.py` | Coded, not called by train_full_system() |
| TabResNet (Residual MLP) | `models/tabresnet.py` | Coded, not called by train_full_system() |
| CatBoost classifier | `models/catboost_model.py` | Coded, not called by train_full_system() |
| Two-model fusion | `fusion.py:31-44` | Coded, w_LGBM=1.0 in active pipeline |

### Key Architectural Difference from Reference Paper

The reference paper describes an XGBoost + TabResNet ensemble fused via Bayesian weighting.
The actual implemented primary classifier is **LightGBM** (gradient-boosted trees with native NaN support, critical for partial-observability).
The `TrainedSystem.xgb` and `TrainedSystem.tabresnet` properties are backward-compatibility aliases that both return the LightGBM model (pipeline.py:88-95).

### Verified Experimental Results

NOTE: `metrics_report.json` results were produced on a 30-patient demo cohort. `dss_diagnostic_results.csv` covers full dataset (~38,052 test patients).

**Demo cohort (30 patients), held-out test split:**

| Metric | Initial (Partial Obs.) | Post-Adaptive Acquisition |
|--------|------------------------|---------------------------|
| Balanced Accuracy | 0.8276 | 0.8621 |
| ROC-AUC | 0.7241 | 0.7586 |
| Recall/Sensitivity | 1.0000 | 1.0000 |
| Specificity | 0.6552 | 0.7241 |
| Mean DSS | 0.9466 | 0.9800 |
| Mean acquisitions | 0 | 0.17 features/patient |
| Early stop rate | 100% | 100% |

**Full dataset (~38,052 test patients), from dss_diagnostic_results.csv:**

| Experiment | Balanced Acc | ROC-AUC | Sensitivity | Specificity |
|------------|-------------|---------|-------------|-------------|
| Full features, unweighted (default threshold) | 0.5714 | 0.8298 | 0.1630 | 0.9798 |
| Full features, balanced (val-tuned) | 0.7502 | 0.8285 | 0.7891 | 0.7114 |
| Masked K=1, balanced (val-tuned) | 0.6949 | 0.7639 | 0.7556 | 0.6342 |
| Masked K=5 aug, balanced (default) | 0.7004 | 0.7678 | 0.7303 | 0.6706 |

### Five Most Important Audit Findings

1. Active pipeline is LightGBM-only; XGBoost and TabResNet exist in code but are not trained by the main pipeline.
2. CVAE is fully implemented with PyTorch ELBO loss and observed-value identity constraint.
3. The complete adaptive acquisition loop (DSG computation, scoring, greedy selection, oracle reveal, DSS stopping) is fully implemented in pipeline.py and acquisition.py.
4. The clinical graph uses statistical mutual information, not causal inference.
5. The benchmark metrics_report.json used a 30-patient demo cohort; the full-dataset results are in dss_diagnostic_results.csv.
