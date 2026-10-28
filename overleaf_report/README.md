# Overleaf Upload & Compilation Guide

This folder contains the complete, self-contained LaTeX source for the comprehensive research report:
**"Decision-Sufficiency-Guided Adaptive Clinical Acquisition System for Diabetes Risk Prediction"**.

---

## 1. Quick Upload to Overleaf

1. Download or locate the pre-packaged ZIP archive:
   ```text
   overleaf_report.zip
   ```
2. Open your web browser and log in to [Overleaf](https://www.overleaf.com/).
3. Click the green **"New Project"** button in the top-left corner.
4. Select **"Upload Project"** from the drop-down menu.
5. Drag and drop `overleaf_report.zip` (or click "Select a .zip file" and choose it).
6. Overleaf will automatically unpack all files into your project workspace.

---

## 2. Recommended Compiler Settings

Ensure your Overleaf project settings match the following (available under the **Menu** icon in the top-left of the Overleaf editor):

* **Compiler**: `pdfLaTeX` (Default)
* **TeX Live Version**: `2023` or `2024`
* **Main Document**: `main.tex`

Click the green **"Recompile"** button. The document will compile cleanly into a publication-grade PDF.

---

## 3. Project Directory Contents

```text
overleaf_report/
├── main.tex           # Master LaTeX research document (all chapters, TikZ diagrams, tables)
├── references.bib     # Curated bibliography with genuine, verifiable citations & DOIs
├── AUDIT_NOTES.md     # Codebase verification audit matching code to report claims
├── README.md          # Overleaf compilation instructions
└── figures/           # Genuine empirical plot assets generated from evaluation runs:
    ├── confusion_matrix_lightgbm.png
    ├── roc_lightgbm.png
    ├── pr_lightgbm.png
    ├── model_comparison.png
    ├── experiment_a_vs_b_comparison.png
    └── training_history_lightgbm.png
```

---

## 4. Local Compilation (Optional)

If you have a local TeX distribution installed (TeX Live, MacTeX, or MiKTeX):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
