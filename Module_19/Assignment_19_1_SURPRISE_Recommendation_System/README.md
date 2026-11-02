# Building a Recommendation System with SURPRISE — MovieLens 100K

**UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence — Module 19, Required Discussion 19.1**  
**Author:** Trilok Reddy Arava  
**Status:** Implementation prepared; **cross-validation results are pending local execution**.

> **Executive takeaway:** Compare five collaborative-filtering algorithms on MovieLens 100K using the same 5-fold cross-validation split, selecting the model with the lowest mean squared error (MSE). No winning algorithm or scores are claimed until the notebook runs.

## Contents
- [Business objective](#business-objective)
- [Dataset](#dataset)
- [Methods](#methods)
- [Project structure](#project-structure)
- [Run instructions](#run-instructions)
- [Results and deliverables](#results-and-deliverables)
- [Limitations](#limitations)
- [Discussion submission](#discussion-submission)

## Business objective
Predict users’ movie ratings to support personalization. Compare algorithms fairly and explain which best predicts held-out ratings; rating accuracy alone does not establish the quality of top-N recommendations.

## Dataset
**MovieLens 100K**, GroupLens: https://grouplens.org/datasets/movielens/ . This is the **original 100K dataset**, not MovieLens Latest Small. It contains **100,000 ratings**, **943 users**, **1,682 movies**, and a **1–5 rating scale**. The SURPRISE built-in `ml-100k` loader retrieves the official dataset on first run. The raw dataset is not redistributed in this repository; consult GroupLens licensing/usage terms.

Fields used: `user_id`, `movie_id`, `rating`; `timestamp` is present but not used for modeling.

## Methods
- **Algorithms:** KNNBasic, SVD, NMF, SlopeOne, CoClustering.
- **Validation:** seeded, shuffled 5-fold cross-validation (`random_state=42`), identical folds for each algorithm.
- **Primary selection metric:** mean fold MSE = average of each fold's squared RMSE. **Secondary:** RMSE, MAE, standard deviation of fold MSE, fit time.
- **Selection rule:** lowest mean cross-validation MSE; no hyperparameter tuning in this baseline comparison.

## Project structure
```text
Module_19/Assignment_19_1_SURPRISE_Recommendation_System/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── discussion19_1_completed.ipynb
├── outputs/
│   ├── figures/
│   │   └── model_comparison.png              # created on execution
│   └── reports/
│       ├── cross_validation_summary.csv      # created on execution
│       ├── cross_validation_folds.csv        # created on execution
│       └── discussion_post.txt               # created on execution
└── Execution_Screenshots/
    └── README.md
```

## Run instructions
1. Install Python **3.10 or 3.11** and create/activate a virtual environment (SURPRISE uses compiled extensions and may require build tools). Example:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   jupyter notebook
   ```
2. From the **project root**, open [`notebooks/discussion19_1_completed.ipynb`](notebooks/discussion19_1_completed.ipynb).
3. Run all cells in order. On first use, the SURPRISE loader downloads the original MovieLens 100K dataset; an internet connection is required for that download.
4. Check the cross-validation summary and best-model printout. The notebook creates the figure, CSV reports, and a **≤200-word discussion post** automatically.
5. Save the notebook with its executed outputs, then commit the generated files and optional screenshots to GitHub.

**Troubleshooting:** The training-time calculation uses `np.mean(scores['fit_time'])` because `fit_time` may be returned as a tuple. If SURPRISE fails to import, verify Python and NumPy compatibility and restart the kernel after installation.

## Results and deliverables
**Not yet executed in this package.** Once run, view:
- [Notebook](notebooks/discussion19_1_completed.ipynb)
- [Model comparison figure](outputs/figures/model_comparison.png) *(generated after execution)*
- [Cross-validation summary](outputs/reports/cross_validation_summary.csv) *(generated after execution)*
- [Per-fold metrics](outputs/reports/cross_validation_folds.csv) *(generated after execution)*
- [Final discussion post](outputs/reports/discussion_post.txt) *(generated after execution)*
- [Execution screenshots](Execution_Screenshots/) *(optional)*

The model with the lowest mean MSE will be reported as the best **among these five configurations**. Results must be copied from your executed notebook; none are fabricated here.

## Limitations
- A randomly shuffled rating split does not measure performance on future ratings or new users/items.
- Cold-start users and items are not specifically evaluated.
- MSE measures rating prediction accuracy, not ranking quality, diversity, fairness, or user satisfaction.
- No hyperparameter search is included; the comparison is a baseline, not a claim of global optimality.

## Discussion submission
Post the generated `outputs/reports/discussion_post.txt` after reviewing the real numbers. The course limits the **initial post to 200 words** and requires **two peer replies, each ≤150 words**. Include the GitHub project link if desired; the post is automatically word-count checked.
