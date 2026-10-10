"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        y = np.asarray(y)
        if not len(y):
            return 0.0
        if self.task == "regression":
            return float(np.var(y))
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1 - np.sum(p * p) if self.criterion == "gini" else -np.sum(p * np.log2(p)))

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        parent_impurity = self._impurity(y)
        def class_impurities(counts, sizes):
            p = counts / sizes[:, None]
            if self.criterion == "gini":
                return 1 - np.sum(p * p, axis=1)
            return -np.sum(p * np.log2(np.maximum(p, np.finfo(float).tiny)), axis=1)
        n = len(y)
        best = None
        best_gain = 0.0
        positions = np.arange(1, n)
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind='stable')
            x = X[order, feature]
            valid = (x[:-1] < x[1:]) & (positions >= self.min_samples_leaf) & (n - positions >= self.min_samples_leaf)
            if not valid.any():
                continue
            idx = np.flatnonzero(valid)
            left_n = positions[idx]
            right_n = n - left_n
            sorted_y = y[order]
            if self.task == 'regression':
                centered = sorted_y - sorted_y.mean()
                sums = np.cumsum(centered)
                squares = np.cumsum(centered * centered)
                left_imp = np.maximum(0, squares[idx] / left_n - (sums[idx] / left_n) ** 2)
                right_imp = np.maximum(0, (squares[-1] - squares[idx]) / right_n - ((sums[-1] - sums[idx]) / right_n) ** 2)
            else:
                counts = np.cumsum(np.eye(len(self.classes_))[sorted_y], axis=0)
                left_imp = class_impurities(counts[idx], left_n)
                right_imp = class_impurities(counts[-1] - counts[idx], right_n)
            gains = parent_impurity - (left_n * left_imp + right_n * right_imp) / n
            j = int(np.argmax(gains))
            if gains[j] > best_gain + 1e-14:
                k = idx[j]
                threshold = x[k] + (x[k + 1] - x[k]) / 2
                if threshold >= x[k + 1]:
                    threshold = x[k]
                best_gain = float(gains[j])
                best = (feature, threshold, best_gain)
        return best

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        if depth == 0:
            X, y = (np.asarray(X, dtype=float), np.asarray(y))
            if X.ndim != 2 or not X.shape[0] or (not X.shape[1]) or (y.shape != (len(X),)):
                raise ValueError('Expected a nonempty 2D X and a matching 1D y.')
            if not np.isfinite(X).all():
                raise ValueError('X must contain finite values.')
            if self.task not in {'regression', 'classification'}:
                raise ValueError('Unknown task.')
            allowed = {'mse'} if self.task == 'regression' else {'gini', 'entropy'}
            if self.criterion not in allowed:
                raise ValueError('Criterion is incompatible with task.')
            if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 0) or not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1 or (not isinstance(self.min_samples_split, int)) or (self.min_samples_split < 2) or (self.min_impurity_decrease < 0):
                raise ValueError('Invalid stopping parameters.')
            if self.task == 'classification':
                self.classes_, y = np.unique(y, return_inverse=True)
            else:
                y = y.astype(float)
                if not np.isfinite(y).all():
                    raise ValueError('y must contain finite values.')
            self.n_samples_, self.n_features_in_ = X.shape
            self.feature_importances_ = np.zeros(X.shape[1])
            self.depth_, self.n_leaves_ = (0, 0)
        if self.task == 'regression':
            node = Node()
            node.value = float(np.mean(y))
        else:
            probabilities = np.bincount(y, minlength=len(self.classes_)) / len(y)
            node = Node()
            node.value = self.classes_[np.argmax(probabilities)]
            node.probabilities = probabilities
        self.depth_ = max(self.depth_, depth)
        impurity = self._impurity(y)
        if self.max_depth is not None and depth >= self.max_depth or len(y) < max(self.min_samples_split, 2 * self.min_samples_leaf) or impurity <= 1e-14:
            self.n_leaves_ += 1
            return node
        split = self._best_split(X, y)
        if split is None or len(y) / self.n_samples_ * split[2] < self.min_impurity_decrease:
            self.n_leaves_ += 1
            return node
        feature, threshold, gain = split
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = (feature, float(threshold))
        self.feature_importances_[feature] += len(y) * gain
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        if depth == 0 and self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        if self.root is None:
            raise ValueError("Fit the tree before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid prediction features.")

        return np.asarray([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.root is None:
            raise ValueError("Fit the tree before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid prediction features.")

        if self.task != "classification":
            raise ValueError("Probabilities require a classification tree.")
        probabilities = []
        for x in X:
            node = self.root
            while not node.is_leaf():
                node = node.left if x[node.feature] <= node.threshold else node.right
            probabilities.append(node.probabilities)
        return np.asarray(probabilities).reshape(-1, len(self.classes_))
