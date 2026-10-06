# Workspace Memory
This file is maintained automatically by Code Janitor so Claude, Codex, Bob, and any other AI agent can reuse repo context without rescanning everything from scratch.
Generated: 2026-10-05T15:38:14.083Z
Workspace: dss_system
Workspace root: c:\Users\Tamaghna Ghosh\Downloads\dss_system
Refresh reason: startup
Output path: graphify-out/WORKSPACE_MEMORY.md
Shared mirror: workspacememory.md
Structured manifest: workspace.json
## Handoff Guidance
- Read `graphify-out/GRAPH_REPORT.md` first when the request is about architecture, dependencies, file ownership, or codebase navigation.
- Use this memory file and the workspace-root `workspacememory.md` mirror for recent activity, hot files, Git-aware status, and GitHub-enriched project context.
- Use the workspace-root `workspace.json` file when an AI agent wants machine-readable repo metadata, file inventory, package details, and Git/Graphify summaries without rescanning the repository.
- Refresh this file with the `Code Janitor: Refresh Workspace Memory` command after significant edits or branch changes.
## Repository Blueprint
- Audience: any AI agent working in this repository can treat this file as the current handoff ledger.
- Graphify report: not available yet
- Graphify graph: not available yet
- Last activity: 2026-10-05T14:42:51.093Z
## Workspace Focus
- Active file in focus: main.py
- Hottest files right now: data_processing.py (10), main.py (10), .cursor/rules/main.py (7), evaluate.py (7)
- Suggested starting points: main.py, data_processing.py, .cursor/rules/main.py, evaluate.py, train.py, extra_plots.py
## Current Workspace
- Active file: main.py
- Tracked files in snapshot: 26208
- Top-level areas: .conda (26162), dss_system (17), [root] (16), evaluation_plots (8), models (4), .cursor (1)
- Primary file types: .py (10922), .h (9786), [no extension] (2198), .pyi (476), .pyd (298), .lib (174), .txt (150), .msg (147)
- Key files: .conda/.gitignore, .conda/Lib/site-packages/sklearn/externals/array_api_compat/README.md, .conda/Lib/site-packages/sklearn/externals/array_api_extra/README.md, README.md, dss_system/README.md
## Package Snapshot
- Package metadata unavailable: package.json was not found.
## Current Stack
- Logged change events: 40
- Change mix: save (39), create (1)
- Remembered file snapshots: 8
- Working tree summary: fatal: not a git repository (or any of the parent directories): .git
## Tracked Snapshots
- evaluate.py | 337 lines | 13130 chars | hash a06a655adafb
  Last snapshot: 2026-10-05T14:42:51.093Z
  Preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- extra_plots.py | 117 lines | 4332 chars | hash 394f7ef73b50
  Last snapshot: 2026-10-05T12:43:06.988Z
  Preview: """" / extra_plots.py / ============== / Drop-in plotting helpers for evaluate.py: / * plot_confusion_heatmap()  -> seaborn heatmap (counts + row-normalised %) / * plot_training_history()   -> loss-vs-epoch and train/v..."
- main.py | 64 lines | 2521 chars | hash 2320fc84bccf
  Last snapshot: 2026-10-02T12:38:37.055Z
  Preview: """" / main.py / ======= / Demonstrates the complete Decision-Sufficiency-Guided adaptive / acquisition loop end to end: / train the system (train.py) -> take a batch of test patients with / only a few features initial..."
- data_processing.py | 323 lines | 12256 chars | hash 8e5ce24e7a2e
  Last snapshot: 2026-10-02T12:32:33.486Z
  Preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- train.py | 78 lines | 2734 chars | hash ae7bfa887e8e
  Last snapshot: 2026-10-01T16:25:55.327Z
  Preview: """" / train.py / ======== / Entry point: prepare the dataset (Steps 1-3), train f_XGB, f_TR, and the / CVAE generator, fit fusion weights, build the clinical graph, and report / static test-set accuracy for the fused..."
- .cursor/rules/main.py | 66 lines | 2219 chars | hash 42cab3e74b03
  Last snapshot: 2026-10-01T16:24:03.160Z
  Preview: """" / main.py / ======= / Demonstrates the complete Decision-Sufficiency-Guided adaptive / acquisition loop end to end: / train the system (train.py) -> take a batch of test patients with / only a few features initial..."
- fusion.py | 58 lines | 1803 chars | hash 733f8871f389
  Last snapshot: 2026-10-01T16:18:16.247Z
  Preview: """" / fusion.py / ========= / Step 6 — Fusion of the two classifiers. / p_t = w_XGB * p_XGB,t + w_TR * p_TR,t     with  w_XGB + w_TR = 1 / y_hat_t = argmax_y p_t(y)                 ( -> 1 if p_t >= 0.5 else 0 ) / Also..."
- pipeline.py | 263 lines | 10161 chars | hash 49e73cfa3b56
  Last snapshot: 2026-10-01T16:18:06.245Z
  Preview: """" / pipeline.py / =========== / End-to-end orchestration of the Decision-Sufficiency-Guided Adaptive / Clinical Information Acquisition System. / D -> Z -> (x_obs, m) -> {f_XGB, f_TR} -> p_t -> y_hat_t / -> P_Psi(x_..."

## Recent Changes
### 2026-10-05T14:42:51.093Z | saved | evaluate.py
- Summary: Line 1: replaced 119 lines with 337 lines.
- Before: 119 lines | 2,958 chars | hash 1b360dfd10b4 | preview: "plot_metric_comparison(results) / # XGBoost does not use epochs in the same sense as a neural network. / # The epoch plots therefore use the TabResNet training history. / training_history = history if history is not N..."
- After: 337 lines | 13,130 chars | hash a06a655adafb | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- Previous fragment: "plot_metric_comparison(results) / # XGBoost does not use epochs in the same sense as a neural network. / # The epoch plots therefore use the TabResNet training history. / traini..."
- Current fragment: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed..."

### 2026-10-05T12:43:55.228Z | created | extra_plots.py
- Summary: Created file.
- After: 117 lines | 4,332 chars | hash 394f7ef73b50 | preview: """" / extra_plots.py / ============== / Drop-in plotting helpers for evaluate.py: / * plot_confusion_heatmap()  -> seaborn heatmap (counts + row-normalised %) / * plot_training_history()   -> loss-vs-epoch and train/v..."

### 2026-10-05T12:32:13.482Z | saved | evaluate.py
- Summary: Line 1: replaced 336 lines with 119 lines.
- Before: 336 lines | 12,927 chars | hash 0130b88b7ca9 | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- After: 119 lines | 2,958 chars | hash 1b360dfd10b4 | preview: "plot_metric_comparison(results) / # XGBoost does not use epochs in the same sense as a neural network. / # The epoch plots therefore use the TabResNet training history. / training_history = history if history is not N..."
- Previous fragment: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed..."
- Current fragment: "plot_metric_comparison(results) / # XGBoost does not use epochs in the same sense as a neural network. / # The epoch plots therefore use the TabResNet training history. / traini..."

### 2026-10-05T12:15:04.651Z | saved | evaluate.py
- Summary: Saved without a textual diff.
- Before: 336 lines | 12,927 chars | hash 0130b88b7ca9 | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- After: 336 lines | 12,927 chars | hash 0130b88b7ca9 | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."

### 2026-10-05T12:14:59.120Z | saved | evaluate.py
- Summary: Line 28: replaced 252 lines with 249 lines.
- Before: 339 lines | 13,017 chars | hash 507f3e9d4193 | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- After: 336 lines | 12,927 chars | hash 0130b88b7ca9 | preview: """" / evaluate.py / ============ / Diagnostic evaluation for f_XGB, f_TR and the fused classifier. / IMPORTANT: this project never evaluates the classifiers on fully- / observed data — f_XGB/f_TR are always called as..."
- Previous fragment: "matplotlib.use("Agg") / import matplotlib.pyplot as plt / from sklearn.metrics import ( / confusion_matrix, / classification_report, / roc_curve, / auc, / precision_recall_curve..."
- Current fragment: ".pyplot as plt / from sklearn.metrics import ( / confusion_matrix, / classification_report, / roc_curve, / auc, / precision_recall_curve, / average_precision_score, / accuracy_s..."

### 2026-10-02T12:38:37.055Z | saved | main.py
- Summary: Line 26: inserted 5 lines.
- Before: 60 lines | 2,338 chars | hash a12176b8740b | preview: """" / main.py / ======= / Demonstrates the complete Decision-Sufficiency-Guided adaptive / acquisition loop end to end: / train the system (train.py) -> take a batch of test patients with / only a few features initial..."
- After: 64 lines | 2,521 chars | hash 2320fc84bccf | preview: """" / main.py / ======= / Demonstrates the complete Decision-Sufficiency-Guided adaptive / acquisition loop end to end: / train the system (train.py) -> take a batch of test patients with / only a few features initial..."
- Current fragment: "print("xgb:", system.xgb) / print("tabresnet:",system.tabresnet) / print("cvae=",system.cvae) / print("fusion_weights=",system.fusion_weights) / print("graph=",system.graph)"

### 2026-10-02T12:32:33.486Z | saved | data_processing.py
- Summary: Line 317: inserted 1 line.
- Before: 323 lines | 12,254 chars | hash 9852aac48318 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,256 chars | hash 8e5ce24e7a2e | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Current fragment: "#"

### 2026-10-02T12:31:24.125Z | saved | data_processing.py
- Summary: Line 317: inserted 1 line.
- Before: 323 lines | 12,251 chars | hash 5422df322696 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,254 chars | hash 9852aac48318 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Current fragment: "[0]"

### 2026-10-02T12:30:03.769Z | saved | data_processing.py
- Summary: Line 317: replaced 1 line with 1 line.
- Before: 323 lines | 12,254 chars | hash 5deee389ab59 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,251 chars | hash 5422df322696 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Previous fragment: ".shape"
- Current fragment: "[0]"

### 2026-10-02T12:28:38.483Z | saved | data_processing.py
- Summary: Line 317: inserted 1 line.
- Before: 323 lines | 12,248 chars | hash 4cf8de3daf5e | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,254 chars | hash 5deee389ab59 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Current fragment: ".shape"

### 2026-10-02T12:27:23.720Z | saved | data_processing.py
- Summary: Line 315: replaced 9 lines with 9 lines.
- Before: 323 lines | 12,238 chars | hash 88aefb7597e0 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,248 chars | hash 4cf8de3daf5e | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Previous fragment: "print("X_train",final_dataset.X_train.shape) / print("y_train",final_dataset.y_train.shape) / print("m_train",final_dataset.m_train.shape) / print("X_val",final_dataset.X_val.sh..."
- Current fragment: "# print("X_train",final_dataset.X_train.shape) / # print("y_train",final_dataset.y_train.shape) / print("m_train",final_dataset.m_train) / # print("X_val",final_dataset.X_val.sh..."

### 2026-10-02T12:24:35.045Z | saved | data_processing.py
- Summary: Line 315: replaced 9 lines with 9 lines.
- Before: 323 lines | 12,256 chars | hash 87aaa631f61f | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,238 chars | hash 88aefb7597e0 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Previous fragment: "()) / print("y_train",final_dataset.y_train.shape()) / print("m_train",final_dataset.m_train.shape()) / print("X_val",final_dataset.X_val.shape()) / print("X_val",final_dataset...."
- Current fragment: ") / print("y_train",final_dataset.y_train.shape) / print("m_train",final_dataset.m_train.shape) / print("X_val",final_dataset.X_val.shape) / print("X_val",final_dataset.y_val.sh..."

### 2026-10-02T12:21:20.675Z | saved | data_processing.py
- Summary: Line 315: replaced 9 lines with 9 lines.
- Before: 323 lines | 12,184 chars | hash bac7ec1025e1 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,256 chars | hash 87aaa631f61f | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Previous fragment: ") / print("y_train",final_dataset.y_train) / print("m_train",final_dataset.m_train) / print("X_val",final_dataset.X_val) / print("X_val",final_dataset.y_val) / print("m_val",fin..."
- Current fragment: ".shape()) / print("y_train",final_dataset.y_train.shape()) / print("m_train",final_dataset.m_train.shape()) / print("X_val",final_dataset.X_val.shape()) / print("X_val",final_da..."

### 2026-10-02T12:19:20.958Z | saved | data_processing.py
- Summary: Line 315: replaced 9 lines with 9 lines.
- Before: 323 lines | 12,103 chars | hash 8aa37096d029 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,184 chars | hash bac7ec1025e1 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Previous fragment: "final_dataset.X_train) / print(final_dataset.y_train) / print(final_dataset.m_train) / print(final_dataset.X_val) / print(final_dataset.y_val) / print(final_dataset.m_val) / pri..."
- Current fragment: ""X_train",final_dataset.X_train) / print("y_train",final_dataset.y_train) / print("m_train",final_dataset.m_train) / print("X_val",final_dataset.X_val) / print("X_val",final_dat..."

### 2026-10-02T12:15:56.569Z | saved | data_processing.py
- Summary: Line 314: inserted 10 lines.
- Before: 314 lines | 11,779 chars | hash aa77c3f7664d | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- After: 323 lines | 12,103 chars | hash 8aa37096d029 | preview: """" / data_processing.py / =================== / Implements Steps 1-3 of the uploaded specification: / Step 1 — Dataset representation      D = {(x_i, y_i)}_{i=1}^N / Step 2 — Preprocessing               z_ij = (x_ij..."
- Current fragment: "final_dataset= prepare_dataset() / print(final_dataset.X_train) / print(final_dataset.y_train) / print(final_dataset.m_train) / print(final_dataset.X_val) / print(final_dataset...."


## Hot Files
- data_processing.py (10 tracked changes)
- main.py (10 tracked changes)
- .cursor/rules/main.py (7 tracked changes)
- evaluate.py (7 tracked changes)
- train.py (3 tracked changes)
- extra_plots.py (1 tracked changes)
- fusion.py (1 tracked changes)
- pipeline.py (1 tracked changes)

## Git Snapshot
- fatal: not a git repository (or any of the parent directories): .git

## GitHub Snapshot
GitHub context unavailable: Could not determine the GitHub repository from origin remote: fatal: not a git repository (or any of the parent directories): .git

## Graphify Snapshot
Graphify report not found. Generate Graphify output if you want architecture-aware memory excerpts here.

## Project Planner
- Project planner is not configured yet. Enable it in the chat panel to generate a time-based todo list and progress rescue briefs.

## Agent Notes
- If a future task asks what changed recently, start with `Recent Changes`, `Tracked Snapshots`, `Hot Files`, and `Git Snapshot`.
- If a future task asks how the project is organized, combine this file with `graphify-out/GRAPH_REPORT.md`.
- If a future task needs repository-level context, use `Package Snapshot`, the GitHub snapshot, and the Graphify snapshot before rescanning broad parts of the repo.
