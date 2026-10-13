"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator should:
  1. load + clean the data,
  2. build features, make the splits, standardize (for the SVM),
  3. train YOUR decision tree (Task A regression, Task B classification),
  4. train YOUR Pegasos SVM (Task B classification),
  5. train the scikit-learn baselines with matched settings,
  6. compute metrics and SAVE all figures to report/figures/,
  7. print a final comparison table.

Everything must be deterministic (fixed seeds). Keep print output tidy so a
grader can read the headline numbers straight from the console.
"""

from __future__ import annotations

from . import data_prep
# from .decision_tree import DecisionTree
# from .svm import PegasosSVM
# from . import evaluate


def main() -> None:
    # 1. data -------------------------------------------------------------
    # df = data_prep.load_raw()
    # df = data_prep.clean(df)
    # X, y, feat_names = data_prep.make_features(df)
    # splits = data_prep.train_val_test_split(X, y)
    # ...

    # 2. Task A: regression (price) --------------------------------------
    #    - your DecisionTree(task="regression", criterion="mse")
    #    - sklearn DecisionTreeRegressor baseline

    # 3. Task B: classification (price tier) -----------------------------
    #    - make_tier_label using TRAIN median
    #    - your DecisionTree(task="classification")
    #    - your PegasosSVM (standardized features)
    #    - sklearn baselines (DecisionTreeClassifier, SVC/LinearSVC)

    # 4. metrics + figures + comparison ----------------------------------
    #    evaluate.compare({...}); save plots to report/figures/

    import argparse
    import hashlib
    import json
    from importlib.metadata import version
    from pathlib import Path
    import platform
    from time import perf_counter
    from itertools import product
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from threadpoolctl import threadpool_limits
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM
    from . import evaluate
    from .evaluate import confusion_matrix
    ROOT = Path(__file__).resolve().parents[1]
    SEED = data_prep.SEED
    DATA_PATH = ROOT / data_prep.DATA_PATH
    DATA_URL = "https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project"
    make_tier_label = data_prep.make_tier_label
    standardize = data_prep.standardize
    load_raw = data_prep.load_raw

    class Preprocessor:
        def fit(self, frame):
            training = frame.copy()
            training.attrs.pop("preprocessing", None)
            _, _, self.feature_names_ = data_prep.make_features(training)
            self.state_ = training.attrs["preprocessing"]
            return self

        def transform(self, frame):
            heldout = frame.copy()
            heldout.attrs["preprocessing"] = self.state_
            return data_prep.make_features(heldout)[0]

    def train_svm(X, y, params):
        params = params.copy()
        epochs = params.pop("epochs")
        batch_size = params.pop("batch_size", 256)
        schedule = params.pop("learning_rate", "pegasos")
        eta0 = params.pop("eta0", .1)
        steps_per_epoch = (len(X) + batch_size - 1) // batch_size
        model = PegasosSVM(n_iters=epochs * steps_per_epoch, **params)
        model.batch_size, model.learning_rate, model.eta0 = batch_size, schedule, eta0
        return model.fit(X, y)

    class RandomFourierFeatures:
        """Approximate exp(-gamma * ||x-z||^2) using random cosine features.

        Reference: Rahimi and Recht (2007), Random Features for Large-Scale
        Kernel Machines, https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        """

        def __init__(self, gamma=0.01, n_components=128, random_state=42):
            self.gamma = gamma
            self.n_components = n_components
            self.random_state = random_state

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            if (X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all()
                    or not np.isfinite(self.gamma) or self.gamma <= 0
                    or not isinstance(self.n_components, int) or self.n_components < 1):
                raise ValueError("Invalid random-feature input or parameters.")
            rng = np.random.default_rng(self.random_state)
            self.weights_ = rng.normal(0, np.sqrt(2 * self.gamma), (X.shape[1], self.n_components))
            self.offset_ = rng.uniform(0, 2 * np.pi, self.n_components)
            return self

        def transform(self, X):
            if not hasattr(self, "weights_"):
                raise ValueError("Fit the feature map before transforming.")
            X = np.asarray(X, dtype=float)
            if X.ndim != 2 or X.shape[1] != self.weights_.shape[0] or not np.isfinite(X).all():
                raise ValueError("Invalid random-feature input.")
            return np.sqrt(2 / self.n_components) * np.cos(X @ self.weights_ + self.offset_)

        def fit_transform(self, X, y=None):
            return self.fit(X).transform(X)

    def stratified_folds(labels, n_splits=3, seed=SEED):
        labels = np.asarray(labels)
        rng = np.random.default_rng(seed)
        folds = [[] for _ in range(n_splits)]
        for label in np.unique(labels):
            ids = rng.permutation(np.flatnonzero(labels == label))
            if len(ids) < n_splits:
                raise ValueError("Each class must have at least n_splits rows.")
            for fold, part in zip(folds, np.array_split(ids, n_splits)):
                fold.extend(part.tolist())
        all_ids = np.arange(len(labels))
        for fold in folds:
            held = np.asarray(sorted(fold), dtype=int)
            yield np.setdiff1d(all_ids, held), held

    def regression_metrics(y, p):
        return {"rmse": evaluate.rmse(y, p), "mae": evaluate.mae(y, p), "r2": evaluate.r2(y, p),
                "log_rmse": evaluate.rmse(np.log(y), np.log(np.maximum(p, 1e-12)))}

    def classification_metrics(y, p, scores):
        precision, recall, f1 = evaluate.precision_recall_f1(y, p)
        tn, fp, fn, tp = evaluate.confusion_matrix(y, p).ravel()
        return {"precision": precision, "recall": recall, "f1": f1,
                "roc_auc": evaluate.roc_auc(y, scores), "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)}

    class SVMRun:
        """Keep the training scaler and optional feature map with the NumPy model."""

        def __init__(self, gamma=None, n_components=128, **params):
            self.gamma = gamma
            self.n_components = n_components
            self.params = params

        def fit(self, X, y):
            self.mean_ = X.mean(axis=0)
            scale = X.std(axis=0)
            self.scale_ = np.where(scale > 1e-12, scale, 1.0)
            scaled = (X - self.mean_) / self.scale_
            self.mapping = None
            if self.gamma is not None:
                self.mapping = RandomFourierFeatures(gamma=self.gamma, n_components=self.n_components, random_state=SEED)
                scaled = self.mapping.fit_transform(scaled)
            self.estimator = train_svm(scaled, y, self.params)
            return self

        def transform(self, X):
            scaled = (X - self.mean_) / self.scale_
            return self.mapping.transform(scaled) if self.mapping else scaled

        def predict(self, X):
            return self.estimator.predict(self.transform(X))

        def decision_function(self, X):
            return self.estimator.decision_function(self.transform(X))

    def prepare(train, validation, threshold):
        prep = Preprocessor().fit(train)
        X, Xv = prep.transform(train), prep.transform(validation)
        y, yv = train.price.to_numpy(), validation.price.to_numpy()
        labels, _ = make_tier_label(y, threshold)
        labels_v, _ = make_tier_label(yv, threshold)
        return prep, X, Xv, y, yv, labels, labels_v

    def tune(train, validation, threshold, quick=False):
        prep, X, Xv, y, yv, labels, labels_v = prepare(train, validation, threshold)
        rows, selected = [], {}
        depths = [3, 6] if quick else [3, 6, 10]
        leaves = [5, 20]
        for task, criteria in [("regression", ["mse"]), ("classification", ["gini", "entropy"])]:
            candidates = []
            task_depths = depths + ([14, 18] if task == "regression" and not quick else [])
            for criterion, depth, leaf in product(criteria, task_depths, leaves):
                params = dict(criterion=criterion, max_depth=depth, min_samples_leaf=leaf)
                tree = DecisionTree(task=task, **params)
                target = np.log(y) if task == "regression" else labels
                started = perf_counter()
                tree.fit(X, target)
                if task == "regression":
                    train_score = regression_metrics(y, np.exp(tree.predict(X)))["log_rmse"]
                    val_score = regression_metrics(yv, np.exp(tree.predict(Xv)))["log_rmse"]
                else:
                    train_score = classification_metrics(labels, tree.predict(X), tree.predict_proba(X)[:, 1])["f1"]
                    val_score = classification_metrics(labels_v, tree.predict(Xv), tree.predict_proba(Xv)[:, 1])["f1"]
                rows.append(dict(model=f"tree_{task}", **params, train_score=train_score,
                                 validation_score=val_score, fit_seconds=perf_counter() - started))
                candidates.append((val_score if task == "regression" else -val_score, params))
                print(f"  Tree {task}: depth={depth}, leaf={leaf}, {criterion}: {val_score:.4f}", flush=True)
            selected[f"tree_{task}"] = min(candidates, key=lambda pair: pair[0])[1]

        Xs, Xvs = standardize(X, Xv)
        lambdas = [.001, .01] if quick else [.0001, .001, .01]
        epochs_list = [5, 10] if quick else [20, 40]
        for kind in ["linear", "rff"]:
            candidates = []
            feature_settings = [(None, 128)] if kind == "linear" else [(.01, 128), (.1, 128)]
            if kind == "rff" and not quick:
                feature_settings += [(gamma, 512) for gamma in [.001, .003, .01]]
            for gamma, n_components in feature_settings:
                if gamma is None:
                    features, features_v = Xs, Xvs
                else:
                    mapping = RandomFourierFeatures(gamma=gamma, n_components=n_components, random_state=SEED)
                    features, features_v = mapping.fit_transform(Xs), mapping.transform(Xvs)
                candidate_lambdas = [.0001, .001] if n_components == 512 else lambdas
                candidate_epochs = [80] if n_components == 512 else epochs_list
                for lambda_, epochs in product(candidate_lambdas, candidate_epochs):
                    params = dict(lambda_=lambda_, epochs=epochs, learning_rate="pegasos")
                    model = train_svm(features, labels, params)
                    train_f1 = classification_metrics(labels, model.predict(features), model.decision_function(features))["f1"]
                    val_f1 = classification_metrics(labels_v, model.predict(features_v), model.decision_function(features_v))["f1"]
                    rows.append(dict(model=f"svm_{kind}", **params, gamma=gamma, n_components=n_components,
                                     train_score=train_f1, validation_score=val_f1,
                                     weight_norm=model.weight_norm_, margin_width=model.margin_width_,
                                     objective=model.objective_history_[-1]))
                    candidates.append((-val_f1, {**params, "gamma": gamma, "n_components": n_components}))
                    print(f"  SVM {kind}: lambda={lambda_}, epochs={epochs}, gamma={gamma}, features={features.shape[1]}: {val_f1:.4f}", flush=True)
            selected[f"svm_{kind}"] = min(candidates, key=lambda pair: pair[0])[1]
        return selected, pd.DataFrame(rows)

    def fit_models(train, threshold, selected):
        """Fit every model on the same training rows; do not inspect test data."""
        prep = Preprocessor().fit(train)
        X = prep.transform(train)
        y = train.price.to_numpy()
        labels, _ = make_tier_label(y, threshold)
        runs = {}
        for task in ["regression", "classification"]:
            params = selected[f"tree_{task}"]
            own = DecisionTree(task=task, **params)
            sklearn_params = {**params, "random_state": SEED}
            if task == "regression":
                sklearn_params["criterion"] = "squared_error"
                baseline = DecisionTreeRegressor(**sklearn_params)
            else:
                baseline = DecisionTreeClassifier(**sklearn_params)
            for prefix, model in [("scratch", own), ("sklearn", baseline)]:
                start = perf_counter()
                model.fit(X, np.log(y) if task == "regression" else labels)
                runs[f"{prefix}_tree_{task}"] = dict(model=model, task=task, fit_seconds=perf_counter() - start)

        for kind in ["linear", "rff"]:
            params = selected[f"svm_{kind}"].copy()
            gamma = params.pop("gamma")
            n_components = params.pop("n_components", 128)
            # Same features and population variance; independent optimizer updates.
            for prefix in ["scratch", "sklearn"]:
                if prefix == "scratch":
                    model = SVMRun(gamma=gamma, n_components=n_components, **params)
                else:
                    steps = [("scale", StandardScaler())]
                    if gamma is not None:
                        steps.append(("rff", RandomFourierFeatures(gamma=gamma, n_components=n_components, random_state=SEED)))
                    estimator = SGDClassifier(loss="hinge", penalty="l2", alpha=params["lambda_"],
                                              max_iter=params["epochs"], tol=None, learning_rate="optimal",
                                              random_state=SEED, shuffle=True)
                    steps.append(("svm", estimator))
                    model = Pipeline(steps)
                start = perf_counter()
                model.fit(X, labels)
                runs[f"{prefix}_svm_{kind}"] = dict(model=model, task="classification", fit_seconds=perf_counter() - start)
        return prep, runs

    def evaluate_models(prep, runs, heldout, threshold):
        X = prep.transform(heldout)
        y = heldout.price.to_numpy()
        labels, _ = make_tier_label(y, threshold)
        rows, predictions = [], {}
        for name, run in runs.items():
            model = run["model"]
            start = perf_counter()
            if run["task"] == "regression":
                pred = np.exp(model.predict(X))
                scores = None
            else:
                pred = model.predict(X)
                scores = (model.decision_function(X) if "svm" in name else model.predict_proba(X)[:, 1])
            elapsed = perf_counter() - start
            metrics = (regression_metrics(y, pred) if run["task"] == "regression"
                       else classification_metrics(labels, pred, scores))
            rows.append(dict(model=name, task=run["task"], fit_seconds=run["fit_seconds"],
                             predict_seconds=elapsed, **metrics))
            predictions[name] = pred
        return pd.DataFrame(rows), predictions

    def cross_validate(train, selected, n_splits=3):
        """Fold results describe fixed selected configurations, not nested-CV selection."""
        labels, _ = make_tier_label(train.price)
        rows = []
        for fold, (train_ids, held_ids) in enumerate(stratified_folds(labels, n_splits), 1):
            fold_train, fold_val = train.iloc[train_ids], train.iloc[held_ids]
            _, threshold = make_tier_label(fold_train.price)
            prep, runs = fit_models(fold_train, threshold, selected)
            metrics, _ = evaluate_models(prep, runs, fold_val, threshold)
            metrics["fold"] = fold
            metrics["tier_threshold_azn"] = threshold
            rows.append(metrics)
            print(f"  Cross-validation fold {fold}/{n_splits} complete", flush=True)
        return pd.concat(rows, ignore_index=True)

    def error_analysis(train, test, predictions, threshold):
        """Aggregate errors only; do not export addresses, names or listing records."""
        labels, _ = make_tier_label(test.price, threshold)
        luxury_threshold = float(train.price.quantile(.9))
        slices = {"all": np.ones(len(test), dtype=bool),
                  "price_above_train_p90": test.price.to_numpy() > luxury_threshold,
                  "price_at_or_below_train_p90": test.price.to_numpy() <= luxury_threshold}
        train_counts = train.location.value_counts()
        slices["location_train_count_below_100"] = test.location.map(train_counts).fillna(0).to_numpy() < 100
        for column in ["city", "category", "location"]:
            for value in sorted(test[column].unique()):
                mask = (test[column] == value).to_numpy()
                if mask.sum() >= 20:
                    slices[f"{column}={value}"] = mask
        rows = []
        for name, pred in predictions.items():
            for group, mask in slices.items():
                if not mask.any():
                    continue
                row = dict(model=name, group=group, n=int(mask.sum()))
                if "regression" in name:
                    row.update(regression_metrics(test.price.to_numpy()[mask], pred[mask]))
                else:
                    precision_recall_f1 = evaluate.precision_recall_f1
                    precision, recall, f1 = precision_recall_f1(labels[mask], pred[mask])
                    row.update(precision=precision, recall=recall, f1=f1,
                               error_rate=float(np.mean(labels[mask] != pred[mask])))
                rows.append(row)
        return pd.DataFrame(rows)

    def _save(fig, folder, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def eda(train, threshold, folder):
        labels, _ = make_tier_label(train.price, threshold)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes[0, 0].hist(train.price, bins=60, color="#287c8e")
        axes[0, 0].set(title="Training sale prices", xlabel="Price (AZN)", ylabel="Listings", yscale="log")
        axes[0, 1].hist(np.log(train.price), bins=50, color="#287c8e")
        axes[0, 1].set(title="Log-price distribution", xlabel="Natural log of price", ylabel="Listings")
        valid = train.area.notna() & (train.area > 0)
        axes[0, 2].hexbin(np.log(train.loc[valid, "area"]), np.log(train.loc[valid, "price"]), gridsize=35, mincnt=1)
        axes[0, 2].set(title="Area and price", xlabel="Log area (m²)", ylabel="Log price (AZN)")
        rooms = train.loc[train.rooms.between(1, 10)].groupby("rooms").price.median()
        axes[1, 0].bar(rooms.index, rooms.values, color="#287c8e")
        axes[1, 0].set(title="Median price by room count", xlabel="Rooms", ylabel="Price (AZN)")
        valid = train.lat.between(38, 42) & train.lng.between(44, 51)
        if valid.any():
            geo = axes[1, 1].hexbin(train.loc[valid, "lng"], train.loc[valid, "lat"],
                                  C=np.log(train.loc[valid, "price"]), reduce_C_function=np.median, gridsize=35)
            fig.colorbar(geo, ax=axes[1, 1], label="Median log price")
        axes[1, 1].set(title="Location and prices (regional view)", xlabel="Longitude", ylabel="Latitude")
        axes[1, 2].bar(["Standard", "Premium"], np.bincount(labels, minlength=2), color=["#287c8e", "#e89946"])
        axes[1, 2].set(title=f"Training tiers: threshold {threshold:,.0f} AZN", ylabel="Listings")
        _save(fig, folder, "eda.png")

    def tuning_curves(tuning, folder):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, task in zip(axes, ["regression", "classification"]):
            part = tuning[tuning.model == f"tree_{task}"]
            for (criterion, leaf), group in part.groupby(["criterion", "min_samples_leaf"]):
                group = group.sort_values("max_depth")
                line, = ax.plot(group.max_depth, group.validation_score, "o-", label=f"{criterion}, leaf={int(leaf)} val")
                ax.plot(group.max_depth, group.train_score, "--", color=line.get_color(), alpha=.6)
            ax.set(title=f"Tree {task}: solid validation, dashed train", xlabel="Maximum depth",
                   ylabel="Log RMSE" if task == "regression" else "F1")
            ax.legend(fontsize=7)
        _save(fig, folder, "tree_validation.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        part = tuning[tuning.model == "svm_linear"]
        for epochs, group in part.groupby("epochs"):
            group = group.sort_values("lambda_")
            axes[0].semilogx(group.lambda_, group.validation_score, "o-", label=f"{int(epochs)} epochs")
            axes[1].semilogx(group.lambda_, group.margin_width, "o-", label=f"{int(epochs)} epochs")
        axes[0].set(title="Linear SVM validation", xlabel="Lambda", ylabel="F1")
        axes[1].set(title="Linear SVM margin", xlabel="Lambda", ylabel="2 / ||w||")
        for ax in axes:
            ax.legend()
        _save(fig, folder, "svm_validation.png")
        fig, ax = plt.subplots(figsize=(10, 5))
        part = tuning[tuning.model == "svm_rff"]
        for (gamma, epochs, dimensions), group in part.groupby(["gamma", "epochs", "n_components"]):
            group = group.sort_values("lambda_")
            ax.semilogx(group.lambda_, group.validation_score, "o-", label=f"gamma={gamma}, epochs={int(epochs)}, D={int(dimensions)}")
        ax.set(title="RFF SVM validation", xlabel="Lambda", ylabel="F1")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
        _save(fig, folder, "rff_validation.png")

    def model_diagnostics(prep, runs, test, predictions, threshold, folder):
        labels, _ = make_tier_label(test.price, threshold)
        class_names = [name for name in runs if runs[name]["task"] == "classification"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, name in zip(axes.flat, class_names):
            matrix = confusion_matrix(labels, predictions[name])
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                color = "white" if value > matrix.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color)
            ax.set(title=name.replace("_", " "), xlabel="Predicted", ylabel="Actual",
                   xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Premium"], yticklabels=["Standard", "Premium"])
        _save(fig, folder, "confusion_matrices.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for kind in ["linear", "rff"]:
            model = runs[f"scratch_svm_{kind}"]["model"].estimator
            axes[0].plot(model.objective_history_, label=kind)
        axes[0].set(title="Full training objective by epoch", xlabel="Epoch", ylabel="Regularized hinge objective", yscale="log")
        axes[0].legend()
        pred = predictions["scratch_tree_regression"]
        axes[1].scatter(np.log(test.price), np.log(pred), s=4, alpha=.15)
        limits = [min(np.log(test.price).min(), np.log(pred).min()), max(np.log(test.price).max(), np.log(pred).max())]
        axes[1].plot(limits, limits, "k--")
        axes[1].set(title="Tree predictions on held-out listings", xlabel="Actual log price", ylabel="Predicted log price")
        _save(fig, folder, "training_and_predictions.png")
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        names = np.asarray(prep.feature_names_)
        for ax, task in zip(axes[:2], ["regression", "classification"]):
            importance = runs[f"scratch_tree_{task}"]["model"].feature_importances_
            ids = np.argsort(importance)[-12:]
            ax.barh(names[ids], importance[ids], color="#287c8e")
            ax.set(title=f"Tree {task}", xlabel="Impurity-decrease importance")
        weights = runs["scratch_svm_linear"]["model"].estimator.w
        ids = np.argsort(np.abs(weights))[-12:]
        axes[2].barh(names[ids], weights[ids], color=np.where(weights[ids] >= 0, "#287c8e", "#e89946"))
        axes[2].set(title="Linear SVM standardized weights", xlabel="Weight (positive: premium)")
        _save(fig, folder, "feature_effects.png")

    def boundary_sensitivity(train, threshold, folder):
        """Two-feature training diagnostic, separate from the final all-feature models."""
        sample = train.loc[train.area.notna() & train.rooms.notna()].sample(
            frac=1, random_state=SEED).head(800)
        if len(sample) < 10:
            return pd.DataFrame()
        X = np.column_stack([np.log(sample.area), sample.rooms])
        X, = standardize(X)
        y, _ = make_tier_label(sample.price, threshold)
        if len(np.unique(y)) < 2:
            return pd.DataFrame()
        xlim = np.quantile(X[:, 0], [.01, .99]) + [-.5, .5]
        ylim = np.quantile(X[:, 1], [.01, .99]) + [-.5, .5]
        xx, yy = np.meshgrid(np.linspace(*xlim, 120), np.linspace(*ylim, 120))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        configs = [(0.001, 10, "pegasos"), (0.001, 80, "pegasos"),
                   (0.1, 80, "pegasos"), (0.001, 80, "inverse_sqrt")]
        rows = []
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, (lambda_, epochs, schedule) in zip(axes.flat, configs):
            model = train_svm(X, y, dict(lambda_=lambda_, epochs=epochs, learning_rate=schedule, batch_size=32, random_state=SEED))
            scores = model.decision_function(grid).reshape(xx.shape)
            ax.contourf(xx, yy, scores >= 0, levels=[-.5, .5, 1.5], colors=["#e6eff4", "#fff0da"])
            levels = [level for level in [-1, 0, 1] if scores.min() < level < scores.max()]
            if levels:
                ax.contour(xx, yy, scores, levels=levels, colors="grey", linewidths=.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=9, alpha=.35)
            support = X[model.support_mask_]
            ax.scatter(support[:, 0], support[:, 1], facecolors="none", edgecolors="grey", s=16, linewidths=.3)
            ax.set(xlim=xlim, ylim=ylim, xlabel="Standardized log area", ylabel="Standardized rooms",
                   title=f"lambda={lambda_}, epochs={epochs}, {schedule}\nmargin={model.margin_width_:.3f}")
            rows.append(dict(lambda_=lambda_, epochs=epochs, learning_rate=schedule,
                             weight_norm=model.weight_norm_, margin_width=model.margin_width_,
                             support_count=int(model.support_mask_.sum()), objective=model.objective_history_[-1]))
        fig.suptitle("Two-feature training illustration; not the final model", fontsize=12)
        _save(fig, folder, "svm_boundary_sensitivity.png")
        return pd.DataFrame(rows)

    def save_json(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def run(data_path=DATA_PATH, output=None, quick=False):
        output = Path(output) if output else ROOT / "report" / "figures" / ("quick" if quick else "results")
        output.mkdir(parents=True, exist_ok=True)
        figures = output if quick else output.parent
        figures.mkdir(exist_ok=True)
        started = perf_counter()
        print("[1/7] Loading and cleaning the dataset", flush=True)
        frame = data_prep.clean(load_raw(data_path))
        audit = frame.attrs["cleaning_audit"]
        if quick and len(frame) > 1200:
            frame = frame.sample(1200, random_state=SEED).reset_index(drop=True)
        train, _, validation, _, test, _ = data_prep.train_val_test_split(frame, frame.price.to_numpy())
        _, threshold = make_tier_label(train.price)
        print(f"  Train: {len(train)}, validation: {len(validation)}, test: {len(test)}", flush=True)
        print(f"  Training price-tier threshold: {threshold:,.0f} AZN", flush=True)
        save_json(output / "cleaning.json", audit)
        print("[2/7] Training-only EDA and validation search", flush=True)
        eda(train, threshold, figures)
        selected, tuning = tune(train, validation, threshold, quick)
        tuning.to_csv(output / "tuning.csv", index=False)
        save_json(output / "selected_config.json", selected)
        tuning_curves(tuning, figures)
        print("[3/7] Three-fold cross-validation on the training partition", flush=True)
        cv = cross_validate(train, selected)
        cv.to_csv(output / "cv_folds.csv", index=False)
        metric_cols = ["rmse", "mae", "r2", "log_rmse", "precision", "recall", "f1", "roc_auc"]
        summary = cv.groupby("model")[metric_cols].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(output / "cv_summary.csv")
        print("[4/7] Fitting fixed configurations and evaluating the held-out test", flush=True)
        prep, runs = fit_models(train, threshold, selected)
        metrics, predictions = evaluate_models(prep, runs, test, threshold)
        metrics.to_csv(output / "metrics.csv", index=False)
        # Fit constant baselines on training prices only, then compare on both splits.
        baseline_rows = []
        constants = {"training_mean": float(train.price.mean()),
                     "training_median": float(train.price.median())}
        for split_name, partition in [("validation", validation), ("test", test)]:
            target = partition.price.to_numpy(dtype=float)
            for name, value in constants.items():
                baseline_rows.append(dict(model=name, split=split_name,
                                          prediction_azn=value,
                                          **regression_metrics(target, np.full(len(target), value))))
        baselines = pd.DataFrame(baseline_rows)
        baselines.to_csv(output / "regression_baselines.csv", index=False)
        print("  Constant regression baselines (fitted on training prices):", flush=True)
        print(baselines.round(4).to_string(index=False), flush=True)
        print("[5/7] Error analysis and model diagnostics", flush=True)
        error_analysis(train, test, predictions, threshold).to_csv(output / "error_analysis.csv", index=False)
        model_diagnostics(prep, runs, test, predictions, threshold, figures)
        boundary_sensitivity(train, threshold, figures).to_csv(output / "boundary_sensitivity.csv", index=False)
        effects = pd.DataFrame({"feature": prep.feature_names_,
                               "tree_regression_importance": runs["scratch_tree_regression"]["model"].feature_importances_,
                               "tree_classification_importance": runs["scratch_tree_classification"]["model"].feature_importances_,
                               "linear_svm_weight": runs["scratch_svm_linear"]["model"].estimator.w})
        effects.to_csv(output / "feature_effects.csv", index=False)
        diagnostics, objectives = {}, []
        for kind in ["linear", "rff"]:
            model = runs[f"scratch_svm_{kind}"]["model"].estimator
            diagnostics[kind] = dict(weight_norm=model.weight_norm_, margin_width=model.margin_width_,
                                     support_count=int(model.support_mask_.sum()),
                                     support_fraction=float(model.support_mask_.mean()),
                                     initial_objective=model.objective_history_[0],
                                     final_objective=model.objective_history_[-1])
            for epoch, value in enumerate(model.objective_history_):
                objectives.append(dict(model=kind, epoch=epoch, objective=value))
        pd.DataFrame(objectives).to_csv(output / "objectives.csv", index=False)
        # Positions refer to the seeded training partition, not personal listing IDs.
        pd.DataFrame({"training_position": np.arange(len(train)),
                      "linear_support": runs["scratch_svm_linear"]["model"].estimator.support_mask_,
                      "rff_support": runs["scratch_svm_rff"]["model"].estimator.support_mask_}).to_csv(
                          output / "support_vectors.csv", index=False)
        print("[6/7] Recording reproducibility information", flush=True)
        with Path(data_path).open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        metadata = dict(seed=SEED, quick=quick, dataset_url=DATA_URL, dataset_sha256=digest,
                        clean_rows_before_sampling=audit["clean_rows"], rows_used=len(frame),
                        split_sizes=dict(train=len(train), validation=len(validation), test=len(test)),
                        tier_threshold_azn=threshold, feature_count=len(prep.feature_names_),
                        python=platform.python_version(), platform=platform.platform(),
                        versions={name: version(name) for name in ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "threadpoolctl"]},
                        elapsed_seconds=perf_counter() - started, svm=diagnostics,
                        protocol=[
                            "Fixed 70% training partition; remaining rows split by training-derived tier into validation/test.",
                            "Train/validation/test share no listing URLs after cleaning.",
                            "Train-only 1%/99% predictor clipping, imputation and categorical vocabulary.",
                            "Unknown categorical values map to all-zero indicators.",
                            "Regression models learn natural log price; predictions are exponentiated without bias correction.",
                            "Regression selection minimizes validation log RMSE; classification selection maximizes validation F1.",
                            "Revision expands regression depth and RFF bandwidth/dimension/epoch candidates using validation only.",
                            "The historical test results were already observed before this revision; this is not a fresh blind test.",
                            "Final models retain the original training partition; validation is not added to training.",
                            "CV uses only training rows, refits preprocessing and tier thresholds within each fold.",
                            "CV mean/std measures fixed-configuration variation, not a test confidence interval or nested CV.",
                            "SGDClassifier uses the same lambda (alpha), epochs, scaling and RFF map; optimizer and projection differ.",
                            "Tree differences may reflect float32 sklearn thresholds and split tie-breaking.",
                            "Fit/predict timing excludes shared data preparation; SVM timing includes its scaling and mapping.",
                            "RFF margin is measured in the approximate feature space; support mask uses signed margin <= 1.",
                            "Extreme valid prices remain; inspect price-tail and sparse-location errors.",
                            "Listing deduplication cannot guarantee distinct properties advertised under different URLs.",
                            "One BLAS thread is used for repeatable timing; hardware still affects elapsed time.",
                        ])
        save_json(output / "run_metadata.json", metadata)
        print("[7/7] Complete", flush=True)
        evaluate.compare(metrics.set_index("model").to_dict(orient="index"))
        print(f"Results: {output.resolve()}", flush=True)
        return metrics

    def check():
        from sklearn import metrics
        # Unambiguous splits check the implementation independently of this dataset.
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        for criterion in ["gini", "entropy"]:
            tree = DecisionTree(criterion=criterion).fit(X, y)
            reference = DecisionTreeClassifier(criterion=criterion).fit(X, y)
            assert tree.root.threshold == reference.tree_.threshold[0] == 3.5
            np.testing.assert_array_equal(tree.predict(X), reference.predict(X))
            np.testing.assert_allclose(tree.predict_proba(X), reference.predict_proba(X))
        for name, value in [("max_depth", 0), ("min_samples_split", 20), ("min_samples_leaf", 5), ("min_impurity_decrease", 1.0)]:
            assert DecisionTree(**{name: value}).fit(X, y).root.is_leaf()
        target = y * 7 + 2.
        tree = DecisionTree(task="regression", criterion="mse").fit(X, target)
        reference = DecisionTreeRegressor().fit(X, target)
        np.testing.assert_allclose(tree.predict(X), reference.predict(X))
        tree.fit(X, np.ones(8))
        assert tree.root.is_leaf()
        np.testing.assert_allclose(tree.feature_importances_, 0)
        np.testing.assert_array_equal(DecisionTree().fit([[0], [1]], ["a", "b"]).predict([[0], [1]]), ["a", "b"])
        svm = PegasosSVM(lambda_=1, n_iters=1).fit([[-1.], [1.]], [0, 1])
        np.testing.assert_allclose(svm.w, [1.])
        assert svm.b == 0 and svm.objective_history_[-1] == .5
        rng = np.random.default_rng(3)
        features = np.r_[rng.normal(-2, .2, (60, 2)), rng.normal(2, .2, (60, 2))]
        labels = np.array(["low"] * 60 + ["high"] * 60)
        a = PegasosSVM(lambda_=.01, n_iters=400).fit(features, labels)
        b = PegasosSVM(lambda_=.01, n_iters=400).fit(features, labels)
        assert np.mean(a.predict(features) == labels) > .98
        assert a.objective_history_[-1] < 1
        np.testing.assert_array_equal(a.w, b.w)
        signed = PegasosSVM._to_pm1(labels)
        np.testing.assert_array_equal(a.support_mask_, signed * a.decision_function(features) <= 1 + 1e-10)
        scores = np.array([.2, .2, .8, .6, .8, .1])
        truth = np.array([0, 1, 1, 0, 1, 0])
        assert np.isclose(evaluate.roc_auc(truth, scores), metrics.roc_auc_score(truth, scores))
        np.testing.assert_array_equal(evaluate.confusion_matrix(truth, scores > .5), metrics.confusion_matrix(truth, scores > .5))
        assert evaluate.precision_recall_f1([0, 1], [0, 0]) == (0., 0., 0.)
        assert np.isclose(evaluate.rmse([1, 4, 8], [2, 3, 6]), np.sqrt(metrics.mean_squared_error([1, 4, 8], [2, 3, 6])))
        assert evaluate.r2([2, 2], [2, 2]) == 1
        assert evaluate.roc_auc([0, 1], [.5, .5]) == .5
        raw = pd.DataFrame({"price": np.arange(100) * 1000 + 10000, "Sahə": np.arange(100) + 40, "currency_x": ["AZN"] * 100, "location": ["central"] * 100})
        frame = data_prep.clean(raw)
        tr, _, va, _, te, _ = data_prep.train_val_test_split(frame, frame.price.to_numpy())
        assert not set(tr.index) & set(va.index) and not set(tr.index) & set(te.index)
        prep = Preprocessor().fit(tr)
        va["area"] = 1e12
        va["location"] = "unseen"
        assert prep.transform(va)[:, 0].max() == prep.state_["upper"][0]
        assert "unseen" not in prep.state_["categories"]["location"]
        scaled, held = standardize(np.array([[1., 1.], [3., 1.]]), np.array([[5., 1.]]))
        np.testing.assert_allclose(held, [[3, 0]])
        from sklearn.metrics.pairwise import rbf_kernel
        features = rng.normal(size=(10, 3))
        mapping = RandomFourierFeatures(gamma=.3, n_components=12000)
        transformed = mapping.fit_transform(features)
        np.testing.assert_allclose(transformed @ transformed.T, rbf_kernel(features, gamma=.3), atol=.04)
        print("All built-in model, metric, split, preprocessing and RFF checks passed.")

    parser = argparse.ArgumentParser(description="Reproduce the real-estate experiments.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--quick", action="store_true", help="Small smoke run, not final results.")
    parser.add_argument("--check", action="store_true", help="Run correctness checks without the dataset.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.check:
            check()
        else:
            run(args.data, quick=args.quick)


if __name__ == "__main__":
    main()
