"""Train-fold preprocessing for mixed numerical and categorical DNN inputs."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class EmbeddingPreprocessor(TransformerMixin, BaseEstimator):
    """Scale numerical columns and map categories to IDs, reserving 0 for unknowns."""

    def __init__(self, scaler, feature_names: list[str], categorical_features: list[str]):
        """Store the numeric scaler and categorical column layout."""
        self.scaler = scaler
        self.feature_names = feature_names
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        """Fit category dictionaries and numerical statistics on supplied training rows."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Input columns must match feature_names.")
        if not self.categorical_features:
            raise ValueError("An embedding model needs at least one categorical feature.")
        if len(set(self.categorical_features)) != len(self.categorical_features):
            raise ValueError("Categorical features must be unique.")
        missing = set(self.categorical_features).difference(self.feature_names)
        if missing:
            raise ValueError(f"Unknown categorical features: {sorted(missing)}")

        self.n_features_in_ = X.shape[1]
        self.categorical_indices_ = [
            self.feature_names.index(name) for name in self.categorical_features
        ]
        self.numerical_indices_ = [
            index for index in range(X.shape[1]) if index not in self.categorical_indices_
        ]
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        ).fit(X[:, self.categorical_indices_])
        self.scaler_ = clone(self.scaler)
        if self.numerical_indices_:
            self.scaler_.fit(X[:, self.numerical_indices_])
        self.model_input_params_ = {
            "categorical_indices": self.categorical_indices_,
            "numerical_indices": self.numerical_indices_,
            "categorical_cardinalities": [len(values) + 1 for values in self.encoder_.categories_],
        }
        return self

    def transform(self, X):
        """Preserve column order; emit scaled numbers and nonnegative integral category IDs."""
        check_is_fitted(self, "encoder_")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Input shape differs from the fitted embedding preprocessor.")
        transformed = np.empty(X.shape, dtype=np.float32)
        # OrdinalEncoder uses -1 for unseen values; shifting reserves embedding ID 0 for them.
        transformed[:, self.categorical_indices_] = (
            self.encoder_.transform(X[:, self.categorical_indices_]) + 1
        )
        if self.numerical_indices_:
            transformed[:, self.numerical_indices_] = self.scaler_.transform(
                X[:, self.numerical_indices_]
            )
        return transformed


def model_input_params(preprocessor) -> dict:
    """Return data-derived embedding dimensions, or no extra parameters for a plain MLP."""
    if isinstance(preprocessor, Pipeline):
        preprocessor = preprocessor.named_steps["model_preprocessor"]
    if isinstance(preprocessor, EmbeddingPreprocessor):
        check_is_fitted(preprocessor, "encoder_")
        return preprocessor.model_input_params_
    return {}
