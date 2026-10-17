# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Xəlilli, Yaqub         | yagub-k       | Data pipeline, cleaning, feature engineering, and Decision Tree implementation from scratch (data_prep.py, decision_tree.py). |
| Sadiqov, Ruslan        | ruslansadiqov | SVM model implementation from scratch via Pegasos, RFF integration, and hyperparameter tuning (svm.py, run_all.py). |
| Əsgərov, Elmir         | elmiresgerov  | Evaluation metrics, cross-validation, error analysis, and visualization scripts (evaluate.py, run_all.py). |

## Per-member summary
**Xəlilli, Yaqub —** Designed and implemented the complete data preprocessing and cleaning pipeline in data_prep.py, handling raw CSV parsing, text-to-number extraction, unit conversions, and strict train-only statistics to prevent data leakage. Implemented the Decision Tree classifier and regressor entirely from scratch using NumPy in decision_tree.py with an optimized prefix-sum split search. Also structured the documentation and learning guide.

**Sadiqov, Ruslan —** Implemented the soft-margin Support Vector Machine from scratch using the Pegasos stochastic sub-gradient descent algorithm in svm.py, including mini-batch updates and weight projection. Developed and integrated Random Fourier Features (RFF) for non-linear kernel approximation. Contributed significantly to the experiment pipeline orchestration in run_all.py for SVM training and parameter searching.

**Əsgərov, Elmir —** Implemented all performance evaluation metrics from scratch in evaluate.py (including RMSE, MAE, R^2, Precision, Recall, F1, and ROC-AUC). Designed and integrated the 3-fold cross-validation logic, error analysis across data slices, and comprehensive data visualization scripts for EDA, validation curves, and confusion matrices in run_all.py.

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Xəlilli, Yaqub         | 34%       |
| Sadiqov, Ruslan        | 33%       |
| Əsgərov, Elmir         | 33%       |

*Note: The effort was distributed evenly across core components: data ingestion & decision trees, SVM optimization, and evaluation & diagnostics.*