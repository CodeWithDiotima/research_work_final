# Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition System

Decision-Sufficiency-Guided (DSS/DSG) clinical information acquisition system using a single **CatBoost** diagnostic classifier, **Conditional Variational Autoencoder (CVAE)** profile generator, and Clinical Information Graph.

---

## Architecture Overview

1. **Stratified Preprocessing & Data Split**:
   - Performs a clean **70/15/15** stratified split on `Diabetes_binary` preserving natural class prevalence (~13.9% positive in the CDC BRFSS dataset).
   - Fits standardizers on the **training split only** and transforms validation/test splits to eliminate data leakage.
   - Generates independent, reproducible observation masks for each split using sequential sub-RNGs.

2. **Single CatBoost Diagnostic Classifier ($f_{CB}$)**:
   - Evaluates patient risk under partial observability: $f_{CB}(x_{obs}, m) \to p_t = P(Y=1 \mid x_{obs}, m)$.
   - Augments inputs by encoding unobserved entries ($m_j = 0$) as `np.nan` (derived strictly from observation masks, leaving true clinical zeros intact) concatenated with binary mask indicators $m$ ($2 \times F = 42$ features).
   - Utilizes `auto_class_weights="Balanced"` to naturally manage class imbalance without destructive majority undersampling.
   - Utilizes `nan_mode="Min"` and early stopping on validation logloss.

3. **CVAE Counterfactual Completions**:
   - Generates $M$ plausible complete patient profiles $x \sim P(x_{miss} \mid x_{obs})$ to evaluate decision stability.

4. **Clinical Information Graph & Adaptive Acquisition**:
   - Calculates Classification Stability ($CS_t$), Predictive Divergence ($PD_t$), and Decision Sufficiency Score ($DSS_t$).
   - Dynamically selects next-best clinical tests by maximizing acquisition score $S_j = \lambda_1 R_j + \lambda_2 \text{Conn}_j + \lambda_3 \text{DSG}_j - \lambda_4 C_j$.

---

## File Layout

| File | What it does |
|---|---|
| `dss_system/config.py` | Pipeline hyperparameters, `CatBoostConfig`, thresholds (`tau_DSS`, `alpha`, weights) |
| `dss_system/data_processing.py` | Stratified 70/15/15 split, leak-free preprocessing, observation masks |
| `dss_system/models/catboost_model.py` | Primary CatBoost diagnostic classifier with NaN masking ($2 \times F$) |
| `dss_system/models/cvae.py` | Conditional VAE generative model ($g_1 \dots g_F$) for plausible completions |
| `dss_system/fusion.py` | Single-classifier pass-through ($p_t = p_{\text{catboost}}$) & thresholding |
| `dss_system/stability.py` | $CS_t$, Jensen-Shannon divergence $PD_t$, $DSS_t$, and stopping rule |
| `dss_system/graph.py` | Clinical Information Graph $G=(V, E, W)$, relevance $R_j$, connectivity $\text{Conn}_j$ |
| `dss_system/acquisition.py` | Candidate evaluation, future DSS, $\text{DSG}_j$, cost model $C_j$, test selection |
| `dss_system/pipeline.py` | End-to-end training and per-patient adaptive acquisition loop |
| `dss_system/evaluate.py` | Masked accuracy, confusion matrix heatmap, ROC/PR curves, plausible stability |
| `dss_system/extra_plots.py` | Non-blocking plotting utilities (loss/accuracy curves, heatmaps) |
| `dss_system/metrics.py` | Comprehensive evaluation suite, benchmark reporting, checkpointing |
| `dss_system/train.py` | CLI entry point: dataset preparation and CatBoost system training |
| `dss_system/main.py` | Full end-to-end demo: training, 3 evaluation checkpoints, adaptive simulation |

---

## Quick Start (PowerShell / Windows)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Train System
```powershell
python -m dss_system.train
# Or pass a local CDC CSV file:
python -m dss_system.train --data path/to/cdc_diabetes.csv
```

### 3. Run Full Pipeline & Adaptive Acquisition Demo
```powershell
python -m dss_system.main
```

### 4. Run Comprehensive Metrics Benchmark
```powershell
python -m dss_system.metrics --quick --n-patients 20
```
