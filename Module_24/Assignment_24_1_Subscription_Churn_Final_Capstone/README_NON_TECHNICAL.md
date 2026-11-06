# Predicting Subscription Renewal Risk and Customer Churn Using Machine Learning

**Nontechnical Capstone Report | UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence — Capstone 24.1**  
**Author:** Trilok Reddy Arava

[Main project README](README.md) · [Technical report](README_TECHNICAL.md) · [Project notebooks](#outline-of-project)

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
- [Outline of project](#outline-of-project)
- [Project files and output navigation](#project-files-and-output-navigation)
  - [Notebooks and documentation](#notebooks-and-documentation)
  - [Reports](#reports)
  - [Figures](#figures)
  - [Execution screenshots](#execution-screenshots)
- [Contact and Further Information](#contact-and-further-information)

## Executive summary

### Project overview and goals

Subscription businesses need to know which customers may cancel so they can offer timely, relevant support. I analyzed **7,043 customers** from the public IBM Telco Customer Churn dataset and compared five machine-learning approaches to identify customers at higher risk of leaving. This is an educational study using telecom data, **not** proprietary SaaS or Salesforce renewal records.

### Findings

**26.54%** of customers in the dataset churned. Churn was more common among customers on **month-to-month contracts (42.71%)**, those using **electronic checks (45.29%)**, and those with **fiber-optic internet service (41.89%)**. These patterns show associations, not proof of why a customer canceled.

### Results and conclusion

The project compared **Naive Bayes, Logistic Regression, Decision Tree, Support Vector Classifier, and Random Forest**. **Logistic Regression** was selected because it achieved the best score on the model-selection measure applied during training. On a separate test group of **1,409 customers**, it achieved **80.55% accuracy** and identified **209 of the 374** customers who actually churned. It missed **165 churners** and incorrectly flagged **109** customers who stayed. This illustrates why prediction accuracy alone is not enough.

Naive Bayes identified more churners at the standard cutoff, but it also produced many more false alarms. Random Forest was competitive. There is no evidence from this dataset that contacting customers, offering discounts, or changing contract terms would itself prevent churn.

### Future research and development

Test the approach on more recent and appropriately authorized subscription-renewal data. Study whether predicted risk remains reliable across customer segments and over time, and whether probability estimates accurately reflect observed churn rates.

### Next steps and recommendations

1. **Pilot targeted outreach** with a limited group of high-risk customers, rather than contacting everyone.
2. **Investigate underlying customer needs**, including onboarding, service quality, and billing experience.
3. **Test what works** using a randomized comparison between outreach and business-as-usual groups.
4. **Measure business outcomes**, including actual retained customers, outreach cost, and customer experience—not model accuracy alone.
5. **Review privacy and fairness** before using any customer-level prediction operationally.

## Rationale

Keeping existing customers can support recurring revenue, but customer-success teams have limited time. A risk-ranking model could help prioritize investigation and support while avoiding unnecessary outreach.

## Research Question

**Can machine learning identify customers at greater risk of churn and highlight patterns that could help prioritize retention outreach?**

## Data Sources

The project uses the public **IBM Telco Customer Churn** dataset: 7,043 customer records with information about contract type, tenure, billing, internet service, and other subscriptions. The outcome is whether each customer churned. [Dataset](data/Telco-Customer-Churn.csv).

### Exploratory data analysis

The analysis compared churn patterns across customer segments and reviewed tenure, monthly charges, and other customer characteristics. These charts help identify groups worth further investigation:

![Customer churn by contract, payment method, and internet service](outputs/figures/02_categorical_churn_rates.png)

### Cleaning and preparation

Customer records were checked for quality issues. Blank total-charge values were handled, and customer identifiers were excluded from prediction.

### Preprocessing

Customer characteristics were prepared consistently for model comparison. Details are in the [technical report](README_TECHNICAL.md).

### Final Dataset

The same 7,043 customers were divided into training and testing groups. The separate test group contained **1,409 customers** and was held aside while the models were developed.

## Methodology

Five approaches were compared using the same underlying customer data. Models were selected using training-data validation, then evaluated on the separate test group.

### Naive Bayes Model

A simple probability-based model. It identified **83.16%** of actual churners in the test group, but its lower precision means more unnecessary outreach if used at the standard cutoff.

### Logistic Regression Model

The original Module 20 baseline and the **selected model**. Test accuracy was **80.55%**, with **55.88% churn recall**. It balances competitive prediction quality with relatively straightforward interpretation.

### Decision Tree Model

A rule-based model that can express customer patterns in understandable splits. It identified **62.30%** of churners in the test group but was less accurate overall than Logistic Regression.

### Support Vector Classifier Model

A model that separates customers into predicted risk groups using patterns in their characteristics. It achieved **79.49%** test accuracy and **51.34%** churn recall.

### Random Forest Model

A collection of decision trees used together. It achieved **79.99%** test accuracy and **48.66%** churn recall. Its overall ranking performance was competitive, but the training-only selection criterion favored Logistic Regression.

### Model evaluation and results

| Model | Accuracy | Churn recall | Meaning for outreach |
|---|---:|---:|---|
| Naive Bayes | 69.27% | **83.16%** | Finds more churners but raises more false alarms |
| **Logistic Regression (selected)** | **80.55%** | 55.88% | Strong overall balance and interpretable baseline |
| Decision Tree | 77.93% | 62.30% | Finds more churners than Logistic Regression, with tradeoffs |
| Support Vector Classifier | 79.49% | 51.34% | Competitive but misses more churners |
| Random Forest | 79.99% | 48.66% | Competitive ranking; lower recall at standard cutoff |

**What this means:** A model can prioritize whom to review, but it cannot establish that a retention action will succeed. The right outreach threshold depends on capacity, cost, and customer experience.

![Model comparison](outputs/figures/08_model_comparison.png)

## Outline of project

The final project is organized into **two notebooks**:

1. **[Notebook 01 — Subscription Churn Prediction](notebooks/Subscription_Churn_Prediction.ipynb)** — explains the business problem, explores customer patterns, prepares data, trains and compares five machine-learning models, and selects the final model using training-data validation.
2. **[Notebook 02 — Subscription Churn Evaluation](notebooks/Subscription_Churn_Evaluation.ipynb)** — evaluates the selected model on held-out customers, explains important predictors, illustrates customer risk prioritization, and discusses business recommendations and limitations.

**Execution order:** Run Notebook 01 first, then Notebook 02. Both were successfully executed locally after kernel restarts, with no errors reported. Independent reproduction on a different machine has not been verified.

See the [technical report](README_TECHNICAL.md) for implementation details or the [main README](README.md) for setup instructions.

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

**Limitations:** Historical telecom data may not generalize to SaaS customers. Results are predictive associations, not causal evidence of successful retention actions.
