"""Build ensembles from the existing classic-model configurations."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.classic.train import build_model, build_scaler

METHODS = ("average", "voting_hard", "voting_soft", "stacking_linear", "stacking_ridge")


class OOFStackingClassifier(ClassifierMixin, BaseEstimator):
    """Fit a regression meta-model on held-out probabilities of binary classifiers.

    The supplied estimators must include their trainable preprocessing. Each OOF
    row is predicted without fitting either the classifier or scaler on that row.
    After meta-model fitting, base estimators are refitted on all supplied data.
    """

    def __init__(self, estimators, final_estimator, cv=5, random_state=42, n_jobs=1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Generate inner OOF features, fit the meta-model and refit base models."""
        X, y = np.asarray(X), np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Stacking requires binary labels 0 and 1.")
        if self.cv < 2 or counts.min() < self.cv:
            raise ValueError("Each class must contain at least cv rows for inner stacking CV.")

        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        splits = list(splitter.split(X, y))
        self.oof_fold_ = np.empty(len(y), dtype=int)
        for fold, (_, val_idx) in enumerate(splits, start=1):
            self.oof_fold_[val_idx] = fold

        self.oof_predictions_ = np.column_stack(
            [
                cross_val_predict(
                    estimator, X, y, cv=splits, method="predict_proba", n_jobs=self.n_jobs
                )[:, 1]
                for _, estimator in self.estimators
            ]
        )
        self.final_estimator_ = clone(self.final_estimator).fit(self.oof_predictions_, y)
        self.estimators_ = [clone(estimator).fit(X, y) for _, estimator in self.estimators]
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        """Return raw regression scores, preserving their ordering for ROC AUC."""
        check_is_fitted(self, "final_estimator_")
        meta_features = np.column_stack(
            [estimator.predict_proba(X)[:, 1] for estimator in self.estimators_]
        )
        return self.final_estimator_.predict(meta_features)

    def predict_proba(self, X):
        """Clip regression scores to [0, 1]; these are not calibrated probabilities."""
        positive = np.clip(self.decision_function(X), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        """Convert regression scores to binary predictions at threshold 0.5."""
        return (self.decision_function(X) >= 0.5).astype(int)


def build_estimators(base_configs: list[dict]) -> list[tuple[str, Pipeline]]:
    """Wrap each base model and its scaler in a fold-local sklearn pipeline."""
    return [
        (
            f"base_{index}",
            Pipeline([("scaler", build_scaler(config)), ("model", build_model(config))]),
        )
        for index, config in enumerate(base_configs)
    ]


def build_ensemble(method: str, base_configs: list[dict], config: dict):
    """Create an unfitted ensemble; all learning happens inside fit()."""
    estimators = build_estimators(base_configs)
    if method in ("average", "voting_hard", "voting_soft"):
        return VotingClassifier(
            estimators=estimators,
            voting="hard" if method == "voting_hard" else "soft",
            weights=None if method == "average" else config["voting"].get("weights"),
            n_jobs=1,
        )
    if method in ("stacking_linear", "stacking_ridge"):
        params = config["stacking"].get(method.removeprefix("stacking_"), {})
        final_estimator = (
            LinearRegression(**params) if method == "stacking_linear" else Ridge(**params)
        )
        return OOFStackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=config["stacking"]["n_splits"],
            random_state=config["experiment"]["seed"],
        )
    raise ValueError(f"Unknown ensemble method: {method}")


def prediction_scores(model, X):
    """Return continuous scores for AUC, or weighted vote shares for hard voting."""
    if isinstance(model, OOFStackingClassifier):
        return model.decision_function(X)
    if isinstance(model, VotingClassifier) and model.voting == "hard":
        return np.average(model.transform(X), axis=1, weights=model.weights)
    return model.predict_proba(X)[:, 1]
