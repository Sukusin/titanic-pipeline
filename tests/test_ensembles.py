"""Check OOF isolation, voting rules and saved ensemble inference."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression

from src.ensembles.model import METHODS, OOFStackingClassifier, build_ensemble, prediction_scores
from src.ensembles.submission_predictions import create_submission
from src.ensembles.train import run_cv, validate_config


class MemorizingClassifier(ClassifierMixin, BaseEstimator):
    """Expose leakage by predicting labels only for rows seen during fitting."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.seen_ = dict(zip(np.asarray(X)[:, 0], y, strict=True))
        return self

    def predict_proba(self, X):
        positive = np.array([self.seen_.get(row[0], 0.25) for row in np.asarray(X)])
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EnsembleTests(unittest.TestCase):
    """Exercise small deterministic ensembles without expensive boosting runs."""

    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=42
        )
        self.base_configs = [
            {
                "experiment": {"name": "logreg"},
                "preprocessing": {"scaler": "standard"},
                "model": {"name": "logistic_regression", "params": {"max_iter": 200}},
            },
            {
                "experiment": {"name": "tree"},
                "preprocessing": {"scaler": "standard"},
                "model": {
                    "name": "decision_tree_classifier",
                    "params": {"max_depth": 3, "random_state": 42},
                },
            },
        ]
        self.config = {
            "experiment": {"seed": 42},
            "validation": {"n_splits": 3, "shuffle": True, "random_state": 42},
            "metrics": {"primary": "f1", "log": ["accuracy", "f1", "roc_auc"]},
            "methods": list(METHODS),
            "voting": {"weights": [1, 2]},
            "stacking": {"n_splits": 3, "linear": {}, "ridge": {"alpha": 1.0}},
        }

    def test_meta_training_uses_only_unseen_rows(self):
        X = np.arange(60).reshape(-1, 1)
        y = np.arange(60) % 2
        model = OOFStackingClassifier(
            [("memory", MemorizingClassifier())], LinearRegression(), cv=3
        ).fit(X, y)
        np.testing.assert_allclose(model.oof_predictions_, 0.25)
        np.testing.assert_array_equal(model.estimators_[0].predict_proba(X)[:, 1], y)
        self.assertEqual(set(model.oof_fold_), {1, 2, 3})

    def test_averaging_and_weighted_voting(self):
        for method in ("average", "voting_hard", "voting_soft"):
            model = build_ensemble(method, self.base_configs, self.config).fit(self.X, self.y)
            if method == "voting_hard":
                votes = model.transform(self.X)
                expected_scores = np.average(votes, axis=1, weights=[1, 2])
                expected_labels = (expected_scores > 0.5).astype(int)
            else:
                probabilities = np.column_stack(
                    [estimator.predict_proba(self.X)[:, 1] for estimator in model.estimators_]
                )
                weights = None if method == "average" else [1, 2]
                expected_scores = np.average(probabilities, axis=1, weights=weights)
                expected_labels = (expected_scores > 0.5).astype(int)
            np.testing.assert_allclose(prediction_scores(model, self.X), expected_scores)
            np.testing.assert_array_equal(model.predict(self.X), expected_labels)

    def test_nested_cv_separates_meta_training_from_outer_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics = run_cv(self.X, self.y, self.base_configs, self.config, run_dir)
            self.assertEqual(metrics.height, (len(METHODS) + 2) * 3)
            outer = pl.read_csv(run_dir / "oof_predictions.csv")
            for method in METHODS:
                rows = outer.filter(pl.col("model_name") == method)
                self.assertEqual(sorted(rows.get_column("row_id").to_list()), list(range(60)))
            for method in ("stacking_linear", "stacking_ridge"):
                for fold in range(1, 4):
                    inner = pl.read_csv(run_dir / f"{method}_fold_{fold}_inner_oof.csv")
                    outer_ids = set(
                        outer.filter(
                            (pl.col("model_name") == method) & (pl.col("fold") == fold)
                        ).get_column("row_id")
                    )
                    inner_ids = set(inner.get_column("row_id"))
                    self.assertFalse(inner_ids.intersection(outer_ids))
                    self.assertEqual(inner_ids | outer_ids, set(range(60)))
                    self.assertEqual(inner.height, len(inner_ids))

    def test_stacking_serialization_and_submission_schema(self):
        for method in ("stacking_linear", "stacking_ridge"):
            with tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory)
                test_path = artifact / "test.parquet"
                features = [f"x_{index}" for index in range(4)]
                pl.DataFrame(self.X[:7], schema=features, orient="row").with_columns(
                    pl.Series("PassengerId", np.arange(900, 907))
                ).write_parquet(test_path)
                model = build_ensemble(method, self.base_configs, self.config).fit(self.X, self.y)
                joblib.dump(model, artifact / "model.joblib")
                metadata = {"features": features, "test_path": str(test_path)}
                (artifact / "metadata.json").write_text(json.dumps(metadata))
                output_path = artifact / "submission.csv"
                create_submission(artifact, output_path)
                submission = pl.read_csv(output_path)
                self.assertEqual(submission.columns, ["PassengerId", "Survived"])
                self.assertEqual(
                    submission.get_column("PassengerId").to_list(), list(range(900, 907))
                )
                np.testing.assert_array_equal(
                    submission.get_column("Survived"), model.predict(self.X[:7])
                )
                with self.assertRaisesRegex(ValueError, "missing columns"):
                    pl.DataFrame({"PassengerId": [900]}).write_parquet(test_path)
                    create_submission(artifact, output_path)

    def test_invalid_weights_and_insufficient_inner_folds(self):
        self.config["voting"]["weights"] = [1]
        with self.assertRaisesRegex(ValueError, "Voting weights"):
            validate_config(self.config, self.base_configs, self.y)
        model = OOFStackingClassifier(
            [("memory", MemorizingClassifier())], LinearRegression(), cv=5
        )
        with self.assertRaisesRegex(ValueError, "at least cv rows"):
            model.fit(np.arange(6).reshape(-1, 1), [0, 0, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
