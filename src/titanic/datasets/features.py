"""Fold-local Titanic feature engineering for classic, neural, and ensemble models."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RAW_FEATURES = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
ORIGINAL_FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare", "EmbarkedCode",
    "FamilySize", "IsSingleCode", "SexCode", "TitleCode",
]
BINNED_FEATURES = [
    "Pclass", "SexCode", "AgeBinCode", "FareBinCode", "EmbarkedCode",
    "FamilySize", "IsSingleCode", "TitleCode",
]


class TitanicFeatureTransformer(TransformerMixin, BaseEstimator):
    """Learn imputation, title grouping, quantiles, and category codes on fit rows only."""

    def __init__(self, dataset_type: str = "original"):
        """Select the original or binned output feature layout."""
        self.dataset_type = dataset_type

    def _frame(self, X) -> pd.DataFrame:
        """Validate raw feature shape and restore named columns."""
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != len(RAW_FEATURES):
            raise ValueError(f"Expected a 2D array with {len(RAW_FEATURES)} Titanic features.")
        return pd.DataFrame(values, columns=RAW_FEATURES)

    @staticmethod
    def _titles(frame: pd.DataFrame) -> pd.Series:
        """Extract honorific titles from passenger names."""
        return frame["Name"].astype("string").str.extract(r",\s*([^\.]+)\.", expand=False)

    def fit(self, X, y=None):
        """Learn imputation, bins, and category codes from training rows."""
        if self.dataset_type not in ("original", "binned"):
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        frame = self._frame(X)
        self.feature_names_out_ = (
            ORIGINAL_FEATURES if self.dataset_type == "original" else BINNED_FEATURES
        )

        self.age_median_ = float(pd.to_numeric(frame["Age"], errors="coerce").median())
        self.fare_median_ = float(pd.to_numeric(frame["Fare"], errors="coerce").median())
        embarked = frame["Embarked"].dropna().mode()
        self.embarked_mode_ = str(embarked.iloc[0]) if not embarked.empty else "Unknown"
        titles = self._titles(frame)
        # Rare titles share one code, so validation-only titles cannot create new categories.
        self.common_titles_ = set(titles.value_counts()[lambda counts: counts >= 10].index)
        if self.dataset_type == "binned":
            age = pd.to_numeric(frame["Age"], errors="coerce").fillna(self.age_median_)
            fare = pd.to_numeric(frame["Fare"], errors="coerce").fillna(self.fare_median_)
            self.age_breaks_ = np.unique(np.quantile(age, [0.25, 0.5, 0.75]))
            self.fare_breaks_ = np.unique(np.quantile(fare, [0.25, 0.5, 0.75]))
        prepared = self._prepare(frame)
        category_columns = ["Sex", "Embarked", "Title", "IsSingle"]
        if self.dataset_type == "binned":
            category_columns.extend(["AgeBin", "FareBin"])
        self.categories_ = {
            column: {value: index for index, value in enumerate(sorted(prepared[column].unique()))}
            for column in category_columns
        }
        return self

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputations and derive family and title features."""
        result = frame.copy()
        result["Age"] = pd.to_numeric(result["Age"], errors="coerce").fillna(self.age_median_)
        result["Fare"] = pd.to_numeric(result["Fare"], errors="coerce").fillna(self.fare_median_)
        result["Embarked"] = result["Embarked"].fillna(self.embarked_mode_).astype(str)
        result["Sex"] = result["Sex"].fillna("Unknown").astype(str)
        titles = self._titles(result)
        result["Title"] = titles.where(titles.isin(self.common_titles_), "Misc")
        result["Title"] = result["Title"].fillna("Misc").astype(str)
        result["FamilySize"] = result["SibSp"].astype(float) + result["Parch"].astype(float) + 1
        result["IsSingle"] = (result["FamilySize"] == 1).astype(str)
        if self.dataset_type == "binned":
            result["AgeBin"] = np.searchsorted(
                self.age_breaks_, result["Age"], side="right"
            ).astype(str)
            result["FareBin"] = np.searchsorted(
                self.fare_breaks_, result["Fare"], side="right"
            ).astype(str)
        return result

    def transform(self, X):
        """Return numeric features in the learned column order."""
        check_is_fitted(self, "feature_names_out_")
        frame = self._frame(X)
        prepared = self._prepare(frame)
        category_columns = ["Sex", "Embarked", "Title", "IsSingle"]
        if self.dataset_type == "binned":
            category_columns.extend(["AgeBin", "FareBin"])
        for column in category_columns:
            mapping = self.categories_[column]
            # A category absent from the training fold gets a stable fallback code.
            prepared[f"{column}Code"] = prepared[column].map(mapping).fillna(-1).astype(float)
        return prepared[self.feature_names_out_].to_numpy(dtype=float)
