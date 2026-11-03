# Module 20.1 — Comparing Aggregating Models for Regression

**UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence**  
**Required Try-It Activity 20.1 | VotingRegressor**

This project evaluates whether combining several regression models with scikit-learn's `VotingRegressor` improves predictions of hourly wages compared with individual models. It also examines which input features contribute most to the ensemble's predictions.

## Table of Contents

1. [Project Objective](#project-objective)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Model Performance](#model-performance)
5. [Key Findings](#key-findings)
6. [Feature Importance and Interpretability](#feature-importance-and-interpretability)
7. [Visualizations](#visualizations)
8. [Project Files and Navigation](#project-files-and-navigation)
9. [How to Run the Notebook](#how-to-run-the-notebook)
10. [Conclusion](#conclusion)

## Project Objective

Determine whether an equal-weight or hyperparameter-tuned `VotingRegressor` outperforms five standalone regression algorithms on a held-out test set, and assess how interpretable the ensemble's wage predictions are.

## Dataset

- **Source:** OpenML Census Wage dataset, ID **534**
- **Observations:** 534
- **Columns:** 11, comprising 10 input features and one wage target
- **Target:** Hourly wage (`WAGE`, normalized to `wage` during preparation)
- **Predictors:** Education, occupation, experience, sex, age, sector, race, marital status, southern region, and union membership

Dataset reference: [OpenML dataset 534](https://www.openml.org/d/534).

## Methodology

1. Load the wage dataset and normalize column names.
2. Prepare numeric and categorical inputs using preprocessing appropriate to the estimators.
3. Split observations into training and held-out test sets.
4. Train five individual regressors:
   - Linear Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Ridge Regression
   - Support Vector Regression (SVR)
5. Fit an **equal-weight VotingRegressor** combining estimators.
6. Tune a **VotingRegressor using `RandomizedSearchCV`** and cross-validation on the training set.
7. Compare all seven approaches using **RMSE, MAE, and R²** on the held-out test set.
8. Use **permutation importance** to examine the tuned ensemble's reliance on original input features.

The full implementation is available in the [executed Jupyter notebook](notebooks/VotingRegressor_Completed.ipynb).

## Model Performance

Held-out test results, sorted by increasing RMSE (lower is better):

| Rank | Model | RMSE ↓ | MAE ↓ | R² ↑ |
|---:|---|---:|---:|---:|
| **1** | **SVR** | **4.3604** | **2.9226** | **0.4036** |
| 2 | VotingRegressor (equal weights) | 4.3880 | 3.1019 | 0.3960 |
| 3 | Linear Regression | 4.4162 | 3.2132 | 0.3882 |
| 4 | VotingRegressor (tuned) | 4.4250 | 3.2059 | 0.3858 |
| 5 | Ridge Regression | 4.4361 | 3.2498 | 0.3827 |
| 6 | KNN | 4.4753 | 3.2129 | 0.3717 |
| 7 | Decision Tree | 4.9460 | 3.5370 | 0.2326 |

**Full results:** [model_comparison.csv](outputs/reports/model_comparison.csv)  
**Cross-validation tuning results:** [voting_tuning_cv_results.csv](outputs/reports/voting_tuning_cv_results.csv)

## Key Findings

- **SVR performed best** on the held-out test set (RMSE **4.3604**).
- The **equal-weight ensemble** was close but slightly worse (RMSE **4.3880**).
- The **tuned ensemble** scored **4.4250 RMSE**, so tuning did not improve holdout performance relative to equal weighting.
- Consequently, **the “wisdom of the crowd” did not outperform the strongest individual regressor in this experiment**.
- These results apply to this dataset and evaluation split; they do not establish that voting ensembles are generally inferior.

## Feature Importance and Interpretability

A VotingRegressor combines predictions from different model types and does not expose one universal coefficient for each original feature. **Permutation importance** estimates the change in prediction error when a feature is shuffled.

| Rank | Feature | Increase in RMSE when shuffled |
|---:|---|---:|
| 1 | Education | 1.0828 |
| 2 | Occupation | 0.4282 |
| 3 | Experience | 0.4006 |
| 4 | Sex | 0.2070 |
| 5 | Age | 0.1068 |

**Top predictors:** Education, occupation, and experience.

These are **predictive associations, not causal effects**. Correlated predictors can influence permutation importance rankings, and demographic predictors merit additional fairness review before any real-world use.

**Full results:** [permutation_feature_importance.csv](outputs/reports/permutation_feature_importance.csv)

## Visualizations

### Model Comparison

![Held-out RMSE comparison for individual regressors and voting ensembles](outputs/figures/model_comparison.png)

[Open model comparison chart](outputs/figures/model_comparison.png)

### Permutation Feature Importance

![Permutation feature importance for tuned VotingRegressor](outputs/figures/feature_importance.png)

[Open feature importance chart](outputs/figures/feature_importance.png)

## Project Files and Navigation

All links below are relative to **this assignment folder** and will work on GitHub when the files retain the names and locations shown.

| Folder / File | Description | Link |
|---|---|---|
| `notebooks/` | Complete, executed analysis notebook | [VotingRegressor_Completed.ipynb](notebooks/VotingRegressor_Completed.ipynb) |
| `Execution_Screenshots/` | Notebook execution evidence | [Screenshot 1](Execution_Screenshots/jupyter_notebook_execution_1.png) · [Screenshot 2](Execution_Screenshots/jupyter_notebook_execution_2.png) |
| `outputs/figures/` | Charts | [Model comparison](outputs/figures/model_comparison.png) · [Feature importance](outputs/figures/feature_importance.png) |
| `outputs/reports/` | Model metrics, tuning, feature importance, discussion text | [Model comparison CSV](outputs/reports/model_comparison.csv) · [Tuning CSV](outputs/reports/voting_tuning_cv_results.csv) · [Importance CSV](outputs/reports/permutation_feature_importance.csv) · [Discussion post](outputs/reports/discussion_post.txt) |
| `../README.md` | Module 20 overview | [Module 20 README](../README.md) |
| `../../README.md` | Repository home (if present) | [Repository README](../../README.md) |

### Directory Structure

```text
Module_20/
├── README.md
└── Assignment_20_1_Comparing Aggregating Models/
    ├── README.md
    ├── notebooks/
    │   └── VotingRegressor_Completed.ipynb
    ├── Execution_Screenshots/
    │   ├── jupyter_notebook_execution_1.png
    │   └── jupyter_notebook_execution_2.png
    └── outputs/
        ├── figures/
        │   ├── model_comparison.png
        │   └── feature_importance.png
        └── reports/
            ├── discussion_post.txt
            ├── model_comparison.csv
            ├── voting_tuning_cv_results.csv
            └── permutation_feature_importance.csv
```

## How to Run the Notebook

1. Clone or download the repository.
2. Install the required Python packages in your Jupyter environment:

   ```bash
   pip install numpy pandas matplotlib scikit-learn openml jupyter
   ```

3. Open [`notebooks/VotingRegressor_Completed.ipynb`](notebooks/VotingRegressor_Completed.ipynb).
4. In Jupyter, select **Kernel → Restart & Run All**.
5. Review the metrics, charts, and saved reports. Initial dataset retrieval may require internet access.

> Package requirements are indicative; the notebook itself is the source of truth for its imports.

## Conclusion

On the OpenML wage dataset, **SVR achieved the lowest test RMSE**, and neither the equal-weight nor tuned VotingRegressor improved upon it. Permutation importance identified **education, occupation, and experience** as the most influential predictors in the tuned ensemble. The exercise illustrates why ensembles should be evaluated against strong individual baselines and why prediction accuracy and interpretability should be considered together.
