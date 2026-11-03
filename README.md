# Decision-Sufficiency-Guided Adaptive Clinical Information Acquisition System

Boilerplate implementation of the uploaded 29-step pipeline (CDC Diabetes
Health Indicators dataset, XGBoost + TabResNet fused classifier, CVAE
missing-value generator, DSS/DSG-driven adaptive feature acquisition).

## Layout

| File | Spec steps | What it does |
|---|---|---|
| `config.py` | — | All hyperparameters / thresholds (`tau_DSS`, `alpha`, `lambda_1..4`, etc.) |
| `data_processing.py` | 1-3 | Load, exact-balance undersample, 70/15/15 split, encode/normalize, observation masks |
| `models/xgboost_model.py` | 4 | `f_XGB(x_obs, m) -> p_XGB` |
| `models/tabresnet.py` | 5 | `f_TR(x_obs, m) -> p_TR` (residual MLP, PyTorch) |
| `models/cvae.py` | 7-8 | Conditional VAE realising `g_1..g_F`; samples `M` plausible complete profiles |
| `fusion.py` | 6 | `p_t = w_XGB*p_XGB + w_TR*p_TR`, weight fitting on validation set |
| `stability.py` | 12-15 | `CS_t`, JS-divergence `PD_t`, `DSS_t`, stopping rule |
| `graph.py` | 16-17, 25 | Clinical information graph `G=(V,E,W)`, feature relevance `R_j`, connectivity `Conn_j` |
| `acquisition.py` | 18-29 | Candidate-value evaluation, future DSS, `DSG_j`, cost `C_j`, score `S_j`, `argmax` selection |
| `pipeline.py` | all | Trains the full system and runs the per-patient adaptive-acquisition loop |
| `train.py` | — | CLI entry point: prepare data, train, report static fused test accuracy |
| `main.py` | — | CLI entry point: train, then run the adaptive loop on a demo batch of patients |

## Wiring in the real dataset

Everything currently runs against synthetic data shaped like the schema
in `config.py` (21 features + `ID` + `Diabetes_binary`). Once the real
CDC Diabetes Health Indicators file is available:

```python
python train.py --data /path/to/cdc_diabetes_health_indicators.csv
python main.py   # after editing main() to pass data_path=...
```

`data_processing.load_raw_dataframe` is the only function that needs a
real file path — nothing else needs to change.

## Notes / places you'll likely want to tune

- `CONTINUOUS_FEATURES` / `CATEGORICAL_FEATURES` in `config.py` — adjust
  once the real dataset's dtypes are confirmed (currently only `BMI`,
  `MentHlth`, `PhysHlth` are treated as continuous, matching the CDC
  schema; everything else is treated as categorical/ordinal).
- `acquisition.DEFAULT_RAW_COSTS` — currently every feature defaults to
  a flat acquisition/communication/computation cost of 1.0; populate
  with real clinical costs (invasiveness, lab turnaround, etc.) once
  available.
- `graph._mutual_info_association` / relevance use a simple binned
  mutual-information estimator — swap in a domain-specific association
  measure if one is preferred.
- `pipeline.run_adaptive_acquisition` loops over candidate features and
  Monte-Carlo profiles in plain Python for clarity; vectorize/batch on
  GPU before scaling to the full test set.
- `torch` is required for `tabresnet.py` and `cvae.py` (not needed for
  `xgboost_model.py`, `fusion.py`, `stability.py`, `graph.py`, or
  `data_processing.py`).
  Copyright (c) {YEAR} {OWNER}. All rights reserved.
Author: {TAMAGHNA GHOSH} (https://github.com/Tamaghna01)
 
1. OWNERSHIP
   The {PROJECT} project, including its source code, documentation, models,
   scripts, and original materials (the "Work"), is the exclusive intellectual
   property of {OWNER} (the "Copyright Holder").
 
2. NO PERMISSION GRANTED
   No permission is granted to any person to reproduce, redistribute, modify,
   adapt, create derivative works from, publish, sublicense, sell, or
   commercially exploit the Work, in whole or in part, in any form or by any
   means, without the prior written permission of the Copyright Holder.
 
3. NO TRANSFER OF RIGHTS
   Accessing, viewing, downloading, forking, or cloning this repository does
   not transfer any ownership rights, title, or license to the Work. All rights
   not expressly granted in writing are reserved by the Copyright Holder.
 
4. THIRD-PARTY MATERIALS
   Third-party libraries, datasets, and other externally licensed materials
   that may be referenced by or included with the Work are not covered by this
   license. They remain subject to their respective licenses and terms, and
   nothing in this document alters those terms.
 
5. ACADEMIC CITATION
   Referencing the Work in scholarly writing with proper attribution to the
   Copyright Holder is not restricted by this license, provided no substantial
   portion of the Work is reproduced without written permission.
 
6. PLATFORM TERMS
   Nothing in this license limits the rights that GitHub's Terms of Service
   grant to other users to view and fork public repositories on the GitHub
   platform. Those rights do not extend beyond the platform's functionality.
 
7. NO WARRANTY
   THE WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY
   ARISING FROM, OUT OF, OR IN CONNECTION WITH THE WORK OR ITS USE.
 
8. PERMISSION REQUESTS
   To request written permission, contact the Copyright Holder via
   {[PROFILE](https://github.com/Tamaghna01)}.
"""
