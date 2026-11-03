# Required Discussion 16.1 — Comparing Classification Models

**UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence**  
**Module:** 16 · **Section:** D  
**Author:** Trilok Reddy Arava

## Overview

For this discussion, I compared **Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM)** for two classification tasks: predicting customer churn and recognizing handwritten digits. I considered class imbalance, training speed, interpretability, and the requirements of each task when selecting a suitable model.

## Datasets and Tasks

| Attribute | Task 1: Customer Churn | Task 2: Handwritten Digits |
|---|---|---|
| Dataset | Provided telecom customer churn dataset | scikit-learn handwritten digits dataset |
| Dataset size | 3,333 customers | 1,797 images |
| Features | 19 predictors | 64 pixel features |
| Target | Churn (`True` / `False`) | Digit (`0`–`9`) |
| Class distribution | Imbalanced; approximately 14.5% churn | Approximately balanced |
| Preferred model | Logistic Regression | SVM |
| Main consideration | Interpretability and identifying customers likely to churn | Classification accuracy |

## Comparison of Classification Models

| Model | Advantages | Limitations |
|---|---|---|
| **Logistic Regression** | Trains quickly, provides interpretable coefficients, and supports class weighting for imbalanced datasets | May struggle with complex nonlinear relationships |
| **Decision Trees** | Easy to visualize and interpret, handle nonlinear patterns, and support class weighting | Deep trees may overfit without tuning |
| **KNN** | Simple and effective for pattern recognition | Sensitive to feature scaling and computationally expensive during prediction |
| **SVM** | Effective for complex decision boundaries and supports class weighting | Can be slower to train and more difficult to interpret |

## Task 1: Customer Churn Prediction

I would choose **Logistic Regression** because interpretability is important for understanding the factors associated with customer churn. Since approximately **14.5%** of customers churned in this discussion dataset, I would consider **precision, recall, and F1-score** rather than relying on accuracy alone.

## Task 2: Handwritten Digit Recognition

I would prefer **SVM** because it handles complex classification boundaries effectively. The digit classes are relatively balanced, making **accuracy** a useful evaluation metric.

## Conclusion

The best classification model depends on **interpretability, training speed, class imbalance, predictive performance, and business requirements**. For customer churn, I prioritize an interpretable model that can help explain risk; for handwritten digits, I prioritize recognition accuracy.

These are **model-selection recommendations from the discussion**, not a claim that all four models were trained and benchmarked in the notebook.

## Data and Notebook Links

- [Open the data folder](data/)
- [Download the customer churn dataset](data/telecom_churn.csv)
- [Open the notebooks folder](notebooks/)
- [View the discussion notebook](notebooks/discussion_16.1.ipynb)

## Repository Files

```text
Assignment_16_1_Comparing_Models/
├── README.md
├── data/
│   └── telecom_churn.csv
└── notebooks/
    └── discussion_16.1.ipynb
```

The notebook contains the accompanying starter exploration and visualizations. The dataset used here is **separate from the IBM Telco Customer Churn dataset** used in the later capstone project.

## Related Capstone Work

My separate Module 16 capstone problem statement establishes the research question developed further in Module 20:

- [Module 16 — Capstone Problem Statement](../Assignment_16_2_Capstone_Finalizing_Problem_Statement/)
- [Module 20 — Initial Capstone Report and EDA](../../Module_20/Assignment_20_2_Capstone_Initial_Report_EDA/)
