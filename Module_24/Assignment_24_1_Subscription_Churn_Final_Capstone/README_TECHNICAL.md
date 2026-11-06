# Predicting Subscription Renewal Risk and Customer Churn Using Machine Learning

**Technical Capstone Report | UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence — Capstone 24.1**  
**Author:** Trilok Reddy Arava

[Main project README](README.md) · [Nontechnical report](README_NON_TECHNICAL.md) · [Project notebooks](#outline-of-project)

## Table of contents

- [Executive summary](#executive-summary)
  - [Project overview and goals](#project-overview-and-goals)
  - [Findings](#findings)
  - [Results and conclusion](#results-and-conclusion)
  - [Future research and development](#future-research-and-development)
  - [Next steps and recommendations](#next-steps-and-recommendations)
- [Rationale](#rationale)
- [Research Question](#research-question)
- [Data Sources](#data-sources)
  - [Exploratory data analysis](#exploratory-data-analysis)
  - [Cleaning and preparation](#cleaning-and-preparation)
  - [Preprocessing](#preprocessing)
  - [Final Dataset](#final-dataset)
- [Methodology](#methodology)
  - [Naive Bayes Model](#naive-bayes-model)
  - [Logistic Regression Model](#logistic-regression-model)
  - [Decision Tree Model](#decision-tree-model)
  - [Support Vector Classifier Model](#support-vector-classifier-model)
  - [Random Forest Model](#random-forest-model)
  - [Model evaluation and results](#model-evaluation-and-results)
  - [Model interpretation and validation](#model-interpretation-and-validation)
- [Outline of project](#outline-of-project)
  - [Reproducibility and execution status](#reproducibility-and-execution-status)
- [Project files and output navigation](#project-files-and-output-navigation)
  - [Notebooks and documentation](#notebooks-and-documentation)
  - [Reports](#reports)
  - [Figures](#figures)
  - [Execution screenshots](#execution-screenshots)
- [Contact and Further Information](#contact-and-further-information)

## Executive summary

### Project overview and goals

This supervised binary-classification study uses the public IBM Telco Customer Churn dataset (7,043 rows, 21 original columns) to estimate churn risk. It extends the Module 20 Logistic Regression baseline with Naive Bayes, Decision Tree, linear Support Vector Classifier, and Random Forest. All comparisons preserve the original stratified 80/20 holdout partition (`random_state=42`).

### Findings

The observed churn prevalence is **26.54%**. Segment associations include month-to-month contracts (**42.71%** churn), electronic-check payment (**45.29%**), and fiber-optic internet (**41.89%**). These associations are not causal effects.

### Results and conclusion

Logistic Regression achieved the highest **training-only three-fold stratified cross-validation average precision (0.6602)** and was selected before test evaluation. On the 1,409-row holdout, it achieved **accuracy 0.8055**, **churn precision 0.6572**, **recall 0.5588**, **F1 0.6040**, **ROC-AUC 0.8419**, and **average precision 0.6334**. At threshold 0.50, its confusion matrix is TN=926, FP=109, FN=165, TP=209.

### Future research and development

Evaluate temporal and external validation, probability calibration, threshold optimization using separate validation data, subgroup stability, drift monitoring, alternative feature sets, and causal impact of retention interventions.

### Next steps and recommendations

Use the model only as a candidate risk-ranking tool for a consent- and privacy-reviewed pilot. Set outreach thresholds based on business costs and validation data, and evaluate retention uplift through randomized testing. Do not treat feature importance as causal evidence.

## Rationale

The target is customer churn (`ChurnFlag` = 1 for `Yes`, 0 for `No`). Because churn is the minority class, accuracy alone can mask missed churners. Average precision is the **primary model-selection criterion**; precision, recall, F1, ROC-AUC, and confusion matrices provide complementary diagnostics.

## Research Question

**Can machine learning identify customers at greater risk of churn and highlight patterns that could help prioritize retention outreach?**

## Data Sources

- [IBM Telco Customer Churn CSV](data/Telco-Customer-Churn.csv): 7,043 rows and 21 original columns.
- Target: `Churn` → binary `ChurnFlag`.
- Features include tenure, contract, internet service, payment method, monthly charges, total charges, and add-on services.
- This is a public historical telecom proxy; it does not contain actual SaaS renewals or confidential CRM data.

### Exploratory data analysis

The exploratory notebook examines class distribution, categorical churn rates, numeric distributions, correlations, and engineered customer segments. Supporting figures include [segment churn rates](outputs/figures/02_categorical_churn_rates.png), [numeric distributions](outputs/figures/03_numeric_distributions.png), and [correlations](outputs/figures/04_numeric_correlations.png). A [data dictionary](outputs/reports/data_dictionary.csv) and [data-quality audit](outputs/reports/data_quality_summary.csv) document the source fields.

### Cleaning and preparation

Convert `TotalCharges` to numeric with invalid/blank values coerced to missing (**11 blank entries**); check duplicate customer identifiers and exclude IDs from predictors. Feature engineering includes `AddOnServiceCount`; `TenureGroup` and `MonthlyChargeBand` support exploratory analysis but are excluded from the final predictor set. Missing-value handling occurs within fitted preprocessing pipelines to avoid test leakage.

### Preprocessing

`ColumnTransformer` applies median imputation and `StandardScaler` to numeric predictors, and most-frequent imputation plus `OneHotEncoder` to categorical predictors. Gaussian Naive Bayes receives a dense transformed matrix. The same training/test rows are used across all models; model fitting and hyperparameter searches use training data only. This is tabular data, so the TF-IDF text preprocessing in the exemplar is **not applicable**.

### Final Dataset

7,043 rows; **20 final predictor columns**, including engineered `AddOnServiceCount`. Stratified split: **5,634 training** and **1,409 test** observations, `test_size=0.20`, `random_state=42`.

## Methodology

Pipelines encapsulate preprocessing and estimation. **Stratified 3-fold cross-validation** with `GridSearchCV` tunes each classifier on the training partition. The primary selection metric is **average precision**; ROC-AUC, recall, F1, precision, and accuracy are reported. The held-out test partition is evaluated at the prespecified threshold of **0.50**. Selecting a different winner after inspecting holdout scores would invalidate the training-only selection procedure.

### Naive Bayes Model

`GaussianNB` provides a probabilistic benchmark with conditional-independence and Gaussian assumptions that may not hold for all encoded attributes. **CV AP 0.6134; CV ROC-AUC 0.8202.** Test: accuracy 0.6927, precision 0.4567, recall 0.8316, F1 0.5896, ROC-AUC 0.8070, AP 0.5846.

### Logistic Regression Model

The Module 20 Logistic Regression baseline is preserved. The expanded training notebook tunes regularization/class-weight settings using the same holdout split. **CV AP 0.6602; CV ROC-AUC 0.8458.** Test: accuracy 0.8055, precision 0.6572, recall 0.5588, F1 0.6040, ROC-AUC 0.8419, AP 0.6334. **Selected by training CV AP.**

### Decision Tree Model

A tuned `DecisionTreeClassifier` models nonlinear interactions and may overfit if unconstrained. **CV AP 0.5909; CV ROC-AUC 0.8013.** Test: accuracy 0.7793, precision 0.5782, recall 0.6230, F1 0.5997, ROC-AUC 0.8048, AP 0.6026.

### Support Vector Classifier Model

A linear-kernel `SVC` with probability estimates supports ROC and precision–recall analysis. **CV AP 0.6494; CV ROC-AUC 0.8342.** Test: accuracy 0.7949, precision 0.6421, recall 0.5134, F1 0.5706, ROC-AUC 0.8266, AP 0.6211.

### Random Forest Model

`RandomForestClassifier` is an additional ensemble benchmark. **CV AP 0.6531; CV ROC-AUC 0.8437.** Test: accuracy 0.7999, precision 0.6691, recall 0.4866, F1 0.5635, ROC-AUC 0.8394, AP 0.6502.

### Model evaluation and results

| Model | CV average precision | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression (selected)** | 0.6602 | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 | 0.6334 |
| Random Forest | 0.6531 | 0.7999 | 0.6691 | 0.4866 | 0.5635 | 0.8394 | 0.6502 |
| Support Vector Classifier | 0.6494 | 0.7949 | 0.6421 | 0.5134 | 0.5706 | 0.8266 | 0.6211 |
| Naive Bayes | 0.6134 | 0.6927 | 0.4567 | 0.8316 | 0.5896 | 0.8070 | 0.5846 |
| Decision Tree | 0.5909 | 0.7793 | 0.5782 | 0.6230 | 0.5997 | 0.8048 | 0.6026 |

Training CV AP for Logistic Regression narrowly exceeded Random Forest and SVC; this does not establish statistically significant superiority. Naive Bayes has highest test recall at threshold 0.50 but incurs more false positives. Random Forest has highest holdout AP, which is **not** the precommitted model-selection criterion. All test figures are from the original holdout, not an independent external validation cohort.

![Model comparison](outputs/figures/08_model_comparison.png)

![ROC and PR curves](outputs/figures/09_roc_pr_comparison.png)

![Confusion matrices](outputs/figures/10_confusion_matrices.png)

### Model interpretation and validation

[Permutation importance](outputs/figures/12_final_feature_importance.png) highlights tenure, internet service, monthly charges, contract type, and total charges as important predictors for the selected model. Permutation importance describes predictive reliance, not causation. [Illustrative customer predictions](outputs/reports/illustrative_customer_risk_examples.csv) and [top-decile risk report](outputs/reports/illustrative_top_decile_risk_report.csv) demonstrate possible prioritization, but do not validate outreach effectiveness.

## Outline of project

The final project contains **two notebooks**, executed in order:

1. **[Notebook 01 — Subscription Churn Prediction](notebooks/Subscription_Churn_Prediction.ipynb)** — research question, EDA, data cleaning, feature engineering, preprocessing, baseline Logistic Regression, five-model training, cross-validation, hyperparameter tuning, comparison, and model selection. Produces fitted model artifacts and reports.
2. **[Notebook 02 — Subscription Churn Evaluation](notebooks/Subscription_Churn_Evaluation.ipynb)** — loads saved model artifacts and reports; examines held-out performance, confusion matrix, threshold diagnostics, feature importance, coefficients, customer-level risk examples, business implications, and limitations.

Related files: [Dataset](data/Telco-Customer-Churn.csv) · [CV results](outputs/reports/cross_validation_results.csv) · [Holdout results](outputs/reports/final_model_comparison.csv) · [Dependencies](requirements.txt) · [Main README](README.md).

### Reproducibility and execution status

From the project root:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Run `notebooks/Subscription_Churn_Prediction.ipynb` first with **Restart Kernel and Run All Cells**. Then run `notebooks/Subscription_Churn_Evaluation.ipynb` the same way. Notebook 02 depends on reports and saved fitted models created by Notebook 01; it does not repeat hyperparameter tuning.

**Local verification:** The author reports restarting and executing both final notebooks without errors; key required paths passed a dependency check, and saved charts and reports were generated. This does **not** constitute an independent execution on a clean machine. Keep the held-out test set out of model selection and threshold tuning.

## Project files and output navigation

### Notebooks and documentation

| Resource | Link |
|---|---|
| Prediction, preprocessing, model training and selection | [Notebook 01 — Subscription Churn Prediction](notebooks/Subscription_Churn_Prediction.ipynb) |
| Held-out evaluation, interpretation and business findings | [Notebook 02 — Subscription Churn Evaluation](notebooks/Subscription_Churn_Evaluation.ipynb) |
| Main project guide | [README.md](README.md) |
| Technical methodology | [README_TECHNICAL.md](README_TECHNICAL.md) |
| Business-facing findings | [README_NON_TECHNICAL.md](README_NON_TECHNICAL.md) |
| Source dataset | [Telco Customer Churn CSV](data/Telco-Customer-Churn.csv) |
| Dependencies | [requirements.txt](requirements.txt) |
| Shared utilities | [capstone_utils.py](capstone_utils.py) |

### Reports

| Area | Files |
|---|---|
| Data audit | [Data dictionary](outputs/reports/data_dictionary.csv) · [Data quality summary](outputs/reports/data_quality_summary.csv) · [Outlier audit](outputs/reports/outlier_audit.csv) |
| Baseline | [Baseline metrics](outputs/reports/baseline_metrics.csv) · [Baseline coefficients](outputs/reports/baseline_coefficients.csv) |
| Model selection | [Cross-validation results](outputs/reports/cross_validation_results.csv) · [Final model comparison](outputs/reports/final_model_comparison.csv) · [Selected model](outputs/reports/selected_model.txt) |
| Test predictions | [Test predictions](outputs/reports/test_predictions.csv) |
| Interpretation | [Permutation importance](outputs/reports/final_permutation_importance.csv) · [Logistic Regression coefficients](outputs/reports/selected_logistic_regression_coefficients.csv) |
| Illustrative diagnostics | [Threshold trade-offs](outputs/reports/illustrative_threshold_tradeoffs.csv) · [Customer risk examples](outputs/reports/illustrative_customer_risk_examples.csv) · [Top-decile risk](outputs/reports/illustrative_top_decile_risk_report.csv) · [Local sensitivity](outputs/reports/illustrative_local_sensitivity.csv) |

### Figures

| Figure | Link |
|---|---|
| 01 · Churn distribution | [View figure](outputs/figures/01_churn_distribution.png) |
| 02 · Categorical churn rates | [View figure](outputs/figures/02_categorical_churn_rates.png) |
| 03 · Numeric distributions | [View figure](outputs/figures/03_numeric_distributions.png) |
| 04 · Numeric correlations | [View figure](outputs/figures/04_numeric_correlations.png) |
| 05 · Engineered features | [View figure](outputs/figures/05_engineered_features.png) |
| 06 · Baseline evaluation | [View figure](outputs/figures/06_baseline_evaluation.png) |
| 07 · Model coefficients | [View figure](outputs/figures/07_model_coefficients.png) |
| 08 · Model comparison | [View figure](outputs/figures/08_model_comparison.png) |
| 09 · ROC and precision–recall comparison | [View figure](outputs/figures/09_roc_pr_comparison.png) |
| 10 · Confusion matrices | [View figure](outputs/figures/10_confusion_matrices.png) |
| 11 · Threshold trade-off | [View figure](outputs/figures/11_threshold_tradeoff.png) |
| 12 · Final feature importance | [View figure](outputs/figures/12_final_feature_importance.png) |
| Selected model confusion matrix | [View figure](outputs/figures/final_selected_model_confusion_matrix.png) |
| Selected Logistic Regression coefficients | [View figure](outputs/figures/selected_logistic_regression_coefficients.png) |

### Execution screenshots

These screenshots document the author's successful local notebook execution; they are not a substitute for reproducing the environment.

| Notebook | Screenshots |
|---|---|
| Notebook 01 — Prediction | [Execution screenshot 1](Execution_Screenshots/jupyter_notebook1_execution_1.png) · [Execution screenshot 2](Execution_Screenshots/jupyter_notebook1_execution_2.png) |
| Notebook 02 — Evaluation | [Execution screenshot 1](Execution_Screenshots/jupyter_notebook2_execution_1.png) · [Execution screenshot 2](Execution_Screenshots/jupyter_notebook2_execution_2.png) |

> **Link verification:** Relative paths follow the documented repository layout. Some matching generated artifacts were available for local filename checks, but the full GitHub directory tree was not supplied. Verify all links on the live GitHub repository before submission; local upload suffixes such as `(3)` do not establish GitHub filenames.

## Contact and Further Information

**Trilok Reddy Arava**  
UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence
