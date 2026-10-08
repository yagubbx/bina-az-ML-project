"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias

    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        y = np.asarray(y)
        classes, encoded = np.unique(y, return_inverse=True)
        if y.ndim != 1 or len(classes) != 2:
            raise ValueError("Exactly two classes are required.")
        return 2 * encoded - 1

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or y.shape != (len(X),):
            raise ValueError("Expected a nonempty 2D X and matching 1D y.")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite.")
        if not np.isfinite(self.lambda_) or self.lambda_ <= 0 or not isinstance(self.n_iters, int) or self.n_iters < 1:
            raise ValueError("Invalid lambda or iteration count.")
        batch_size = getattr(self, "batch_size", 256)
        schedule = getattr(self, "learning_rate", "pegasos")
        eta0 = getattr(self, "eta0", .1)
        if not isinstance(batch_size, int) or batch_size < 1 or schedule not in {"pegasos", "inverse_sqrt"} or not np.isfinite(eta0) or eta0 <= 0:
            raise ValueError("Invalid optimizer settings.")
        self.classes_ = np.unique(y)
        signed = self._to_pm1(y)
        self.n_features_in_ = X.shape[1]
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        self.objective_history_ = [1.0]
        self.history_steps_ = [0]
        rng = np.random.default_rng(self.random_state)
        step = 0
        while step < self.n_iters:
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                ids = order[start:start + batch_size]
                xb, yb = X[ids], signed[ids]
                step += 1
                eta = 1 / (self.lambda_ * step) if schedule == "pegasos" else eta0 / np.sqrt(step)
                active = yb * (xb @ self.w + self.b) < 1
                self.w *= 1 - eta * self.lambda_
                if active.any():
                    self.w += eta * (yb[active] @ xb[active]) / len(ids)
                    self.b += eta * yb[active].sum() / len(ids)
                norm = np.linalg.norm(self.w)
                if norm > 1 / np.sqrt(self.lambda_):
                    self.w /= np.sqrt(self.lambda_) * norm
                if step == self.n_iters:
                    break
            objective = self.lambda_ / 2 * (self.w @ self.w) + np.maximum(0, 1 - signed * (X @ self.w + self.b)).mean()
            self.objective_history_.append(float(objective))
            self.history_steps_.append(step)
        self.weight_norm_ = float(np.linalg.norm(self.w))
        self.margin_width_ = 2 / self.weight_norm_ if self.weight_norm_ else float("inf")
        self.support_mask_ = signed * (X @ self.w + self.b) <= 1 + 1e-10
        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        if self.w is None:
            raise ValueError('Fit the SVM before prediction.')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or (not np.isfinite(X).all()):
            raise ValueError('Invalid prediction features.')
        return X @ self.w + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]
