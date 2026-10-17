# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Xəlilli, Yaqub         | yagubbx       | Data pipeline, pipeline orchestration, and final report drafting (data_prep.py, run_all.py, report.tex/pdf). |
| Əsgərov, Elmir         | elmiresgerov  | SVM implementation, setup documentation, and presentation design (svm.py, START.md, presentation files). |
| Sadiqov, Ruslan        | ruslansadiqov | Decision tree model, evaluation metrics, and contribution report finalization (decision_tree.py, evaluate.py, contribution_report.md). |

## Per-member summary
**Xəlilli, Yaqub —** Designed and implemented the data preprocessing, cleaning, and train-only transformation pipeline in data_prep.py. Orchestrated the full reproducible experiment pipeline and baseline comparisons in run_all.py, and authored the final project report (report.tex/pdf) detailing methods, results, and limitations.

**Əsgərov, Elmir —** Implemented the soft-margin Support Vector Machine (Pegasos SVM) from scratch along with margin diagnostics in svm.py. Documented the system setup and reproducible run commands in START.md, and created the final presentation slides and speaker notes (presentation/slides.pdf and slides.pptx).

**Sadiqov, Ruslan —** Implemented the NumPy decision tree supporting both regression and classification tasks in decision_tree.py. Developed the custom regression and classification performance evaluation metrics in evaluate.py, and finalized the diagnostic figures alongside the team contribution record in contribution_report.md.

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Xəlilli, Yaqub         | 34%       |
| Əsgərov, Elmir         | 33%       |
| Sadiqov, Ruslan        | 33%       |

*Note: The effort was divided symmetrically across data engineering & reporting, core SVM modeling & documentation, and decision tree development & metrics evaluation.*