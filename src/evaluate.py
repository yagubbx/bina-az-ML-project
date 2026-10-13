"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    def _pair(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape:
            raise ValueError("Expected matching, nonempty 1D arrays.")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("Metric inputs must be finite.")
        return y, prediction
    y, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y_true, y_pred) -> float:
    def _pair(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape:
            raise ValueError("Expected matching, nonempty 1D arrays.")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("Metric inputs must be finite.")
        return y, prediction
    y, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def r2(y_true, y_pred) -> float:
    def _pair(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape:
            raise ValueError("Expected matching, nonempty 1D arrays.")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("Metric inputs must be finite.")
        return y, prediction
    y, p = _pair(y_true, y_pred)
    if len(y) < 2:
        return float('nan')
    residual, total = (np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2))
    return float(1 - residual / total) if total > 0 else float(residual == 0)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    def _pair(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape:
            raise ValueError("Expected matching, nonempty 1D arrays.")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("Metric inputs must be finite.")
        return y, prediction
    y, p = _pair(y_true, y_pred)
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('Confusion matrix expects binary 0/1 labels.')
    return np.bincount((2 * y + p).astype(int), minlength=4).reshape(2, 2)


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    def _pair(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape:
            raise ValueError("Expected matching, nonempty 1D arrays.")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("Metric inputs must be finite.")
        return y, prediction
    y, scores = _pair(y_true, scores)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('ROC AUC expects binary labels.')
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return float('nan')
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    import pandas as pd
    table = pd.DataFrame.from_dict(name_to_metrics, orient="index")
    table.index.name = "model"
    print(table.to_string(float_format=lambda value: f"{value:.4f}", na_rep="-"))
