# Diabetes Readmission EDA Final Project

This repository contains the code and selected reproducibility materials for the course project:

**How Longitudinal History Features Improve 30-Day Readmission Prediction for Diabetes Patients**

Author: 花浩文, 2462404009  
Affiliation: School of Future Science and Engineering, Soochow University  
Course: Exploratory Data Analysis Final Project, June 2026

## Project Summary

The project studies 30-day hospital readmission prediction using the Diabetes 130-US Hospitals dataset. The analysis focuses on feature engineering, patient-level validation, class imbalance, and whether strictly defined longitudinal patient-history features improve risk ranking.

Main conclusions:

- The final cohort contains 99,340 encounters and 69,987 patients after excluding death/hospice discharge records and invalid gender records.
- All primary train/validation/test splits are mutually exclusive at the `patient_nbr` level.
- The strongest non-history ensemble reaches AP/PR-AUC 0.2423 on the independent test set.
- The final strict-history ensemble reaches AP/PR-AUC 0.2607, ROC-AUC 0.6895, Brier score 0.0925, and Top 5% precision 36.75%.
- Patient-level bootstrap, five-fold patient-level cross-validation, OOF stacking checks, calibration, decision curves, and encounter-id gap ablation are used to audit robustness.

## Repository Contents

```text
scripts/                         Full Python scripts used in the project
scripts/latex_report/            Figure-generation scripts for the LaTeX paper
paper/report.pdf                 Final submitted report
paper/main.tex                   LaTeX source of the final report
paper/core_code_appendix.pdf     Printable shortened code appendix
code/complete_code_with_notes.txt Full code listing with Chinese notes
results/key_tables/              Selected key result tables used in the report
requirements.txt                 Python package versions used in the experiment environment
```

## Data

The raw course data files are not included in this repository:

- `diabetic_data.csv`
- `IDS_mapping.csv`

In the original experiment environment, these files were stored under:

```text
/Users/huahaowen/Downloads/期末大作业/
```

The scripts use explicit paths for reproducibility in the submitted local environment. To run the project elsewhere, update `ROOT` and `DATA_DIR` near the top of `scripts/diabetes_readmission_project.py` and related scripts.

## Suggested Reproduction Order

```bash
cd scripts

# 1. Data cleaning, EDA, patient-level split, baseline models
python diabetes_readmission_project.py

# 2. Static feature engineering and non-history ensemble
python optimization_extra_search.py
python optimization_extra_focused.py

# 3. Research branches and strict longitudinal history model
python optimization_research_pass.py
python optimization_history_refinement.py

# 4. Robustness checks
python optimization_history_gap_ablation.py
python optimization_history_cv.py
python optimization_history_oof_stacking.py

# 5. Final report figures
cd latex_report
python make_history_figures.py
python make_final_history_diagnostics.py
```

## Notes on Evaluation

AP/PR-AUC is reported using `sklearn.metrics.average_precision_score` and is treated as the primary metric because the positive class is rare. Accuracy is not used as the main model-selection criterion. The final paper distinguishes between the deployable strict-history model and a target-history sensitivity upper bound.

