# Predicting Subscription Renewal Risk and Customer Churn Using Machine Learning

**UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence**  
**Required Capstone Assignment 16.1 — Finalizing Your Problem Statement**  
**Section D**

## Table of Contents

- [Research Question](#1-research-question)
- [Expected Data Sources and Structure](#2-expected-data-sources-and-structure)
- [Proposed Analysis and Techniques](#3-proposed-analysis-and-techniques)
- [Expected Results](#4-expected-results)
- [Why This Question Matters](#5-why-this-question-matters)
- [Planned Next Steps](#6-planned-next-steps)
- [Original Submission](#original-submission)
- [Continuation in Module 20](#continuation-in-module-20)

## 1. Research Question

Can machine learning identify subscription-based customers at greatest risk of churn and explain the factors associated with that risk, enabling businesses to prioritize retention efforts and improve renewal outcomes?

## 2. Expected Data Sources and Structure

I plan to use the publicly available **IBM Telco Customer Churn** dataset as a proxy for subscription-based customer relationships. The dataset contains customer-level records with demographics, tenure, contract terms, services, monthly and total charges, payment methods, and a binary churn outcome (Yes/No).

The target variable will be **Churn**. Predictors will include characteristics available before the churn outcome. Although the data concerns telecommunications customers, the methodology may inform future analyses in SaaS subscription and renewal management. It does not directly contain Salesforce renewal opportunities or verified SaaS renewal outcomes.

## 3. Proposed Analysis and Techniques

I will organize the project using the **CRISP-DM** methodology: business understanding, data understanding, preparation, modeling, evaluation, and recommendations.

### Data Preparation and Feature Engineering

- Inspect data quality, duplicates, missing values, inconsistent categories, and the balance of churn outcomes.
- Encode categorical variables, scale numerical features where appropriate, and apply transformations within training-only pipelines to avoid data leakage.
- Explore tenure groups, monthly-charge segments, service counts, contract commitment, and payment methods as candidate predictors; assess whether engineered features improve results.

### Exploratory Visualizations

- Bar charts of churn rates by contract type, payment method, and internet service.
- Histograms of tenure and monthly charges, and box plots comparing charges for churned and retained customers.
- Correlation heatmaps for numeric relationships, plus confusion matrices and ROC/precision–recall curves for model evaluation.

### Predictive Modeling and Evaluation

I will compare **Logistic Regression, Decision Tree, and Random Forest** classifiers using a stratified train/test split, cross-validation, and hyperparameter tuning. Performance will be assessed using recall, precision, F1-score, ROC-AUC, and precision–recall measures. Because missing at-risk customers may be costly, I will examine recall alongside precision and the practical workload created by false positives. Model coefficients or feature-importance analyses will be used to communicate associations in understandable terms, without claiming causation.

## 4. Expected Results

I expect to identify customer segments associated with elevated churn risk, compare the predictive performance of several models, and develop an interpretable customer-risk ranking. Possible associations include tenure, contract length, pricing, payment method, and subscribed services; these are hypotheses to test, not predetermined findings.

The final project will present clear charts, model comparisons, risk-segment summaries, and recommendations that nontechnical business stakeholders can evaluate.

## 5. Why This Question Matters

Subscription businesses depend on recurring customer relationships. When customers leave, organizations lose recurring revenue and may spend additional resources acquiring replacements. Without early warning signals, customer success and renewal teams may identify problems only after a customer has already canceled.

A practical churn-risk model could help teams focus outreach on customers who may need support rather than applying the same intervention to everyone. For example, if short-tenure customers show elevated risk, a business could evaluate stronger onboarding; if certain contract or pricing segments are associated with churn, it could investigate service value or renewal options. These would be proposals for testing, not proven effects of the model.

The intended business value is better prioritization of retention resources, clearer visibility into churn risk, and informed opportunities to protect recurring revenue. Any operational recommendations should be validated through later-period data or a controlled pilot, with attention to outreach costs, fairness, privacy, and customer consent.

## 6. Planned Next Steps

1. Confirm the exact dataset version and document its fields and limitations.
2. Perform exploratory analysis, preprocessing, model comparison, and interpretation.
3. Translate validated findings into a concise retention playbook for nontechnical decision-makers.
4. Recommend a prospective pilot and monitoring approach before any real-world deployment.

## Original Submission

The original Word document is retained alongside this README as the submitted problem statement. This README reformats its proposal content for easy viewing on GitHub.

## Continuation in Module 20

The initial exploratory analysis and Logistic Regression baseline are documented in [Module 20 — Capstone Initial Report and EDA](../../Module_20/Assignment_20_2_Capstone_Initial_Report_EDA/README.md).

*This document describes the **Module 16 proposal**, so the models and results above are planned rather than claims that all analyses were completed at this stage.*
