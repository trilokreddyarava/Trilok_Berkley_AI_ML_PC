# Predicting Subscription Renewal Risk and Customer Churn Using Machine Learning

**Author:** Trilok Reddy Arava  
**Program:** UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence  
**Assignment:** Final Capstone 24.1 (extending Module 20.1)

## Project Overview

This capstone uses the public IBM Telco Customer Churn dataset (**7,043 customer records; 26.54% churn**) to predict which customers are at risk of leaving. It follows the two-notebook progression of the supplied Berkeley sample: **first prepare data, train and compare models; then evaluate and interpret the selected model.** This is an educational telecom dataset used as a proxy for subscription renewal risk, not actual Salesforce or SaaS renewal records.

## Start Here — Only Two Notebooks

1. **[Notebook 01 — Subscription Churn Prediction](notebooks/Subscription_Churn_Prediction.ipynb)**: Goal, Data Problem, Expected Results, Executive Summary, Business Rationale, Research Question, Data Overview, EDA, Data Setup, Preprocessing, original Module 20 Logistic Regression baseline, **Model Training, Hyperparameter Tuning, and Comparison** (retained original code), and Best Model Selection. Trains **Naive Bayes, Logistic Regression, Decision Tree, SVC, and Random Forest**.
2. **[Notebook 02 — Subscription Churn Evaluation](notebooks/Subscription_Churn_Evaluation.ipynb)**: Essential data setup, final selected Logistic Regression model statistics, confusion matrix, individual customer predictions, global feature importance, coefficients, local sensitivity explanations, business implications, conclusions, and future work.

**Run Notebook 01 first, then Notebook 02.** Notebook 01 writes fitted models and comparison files to `outputs/reports/`, which Notebook 02 reads. The original seven overlapping notebook versions are deliberately **not included** in this final submission folder.

## Reports for Two Audiences

- **[Technical README](README_TECHNICAL.md)** — methodology, preprocessing, model selection, validation, metrics, interpretation, and reproducibility.
- **[Nontechnical README](README_NON_TECHNICAL.md)** — business question, executive summary, findings, retention implications, limitations, and recommendations.

## Research Question

**Can machine learning identify subscription-based customers at greatest risk of churn and explain the factors associated with that risk, enabling businesses to prioritize retention efforts and improve renewal outcomes?**

## Key Findings

The original Logistic Regression baseline achieved **80.55% accuracy, 55.88% churn recall, and 0.8419 ROC-AUC** on the 1,409-customer held-out set. The five-model comparison uses **three-fold stratified cross-validation average precision on training data** for model selection. Logistic Regression was selected in the original project run. Observed patterns and coefficients are associations, **not causal evidence** that any specific intervention prevents churn.

## Data and Reproducibility

- Dataset: [`data/Telco-Customer-Churn.csv`](data/Telco-Customer-Churn.csv)
- Python dependencies: [`requirements.txt`](requirements.txt)
- Utility: [`capstone_utils.py`](capstone_utils.py)
- Outputs: `outputs/figures/` and `outputs/reports/`
- Data cleaning includes numeric conversion of `TotalCharges`, imputation inside model pipelines, and exclusion of customer IDs.
- Split: 80% training / 20% test, stratified by churn, `random_state=42`.

From the project root:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open and run the notebooks in the numbered order above. **Do not use the holdout test set to choose a model or tune a deployment threshold.**

## Limitations and Future Work

The source is historical telecom data, not real software renewals. Future work should test external and temporal generalization, calibration, fairness, operational thresholds, and the causal impact of outreach through controlled pilots.
