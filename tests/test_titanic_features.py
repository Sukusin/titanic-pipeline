"""Check that learned Titanic feature statistics stay inside each training fold."""

import unittest

import numpy as np
import polars as pl

from src.datasets.features import RAW_FEATURES, TitanicFeatureTransformer


class TitanicFeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = pl.DataFrame({
            "Pclass": [1, 2, 3, 1],
            "Name": ["A, Mr. One", "B, Mrs. Two", "C, Mr. Three", "D, Miss. Four"],
            "Sex": ["male", "female", "male", "female"],
            "Age": [20.0, 30.0, None, 40.0],
            "SibSp": [0, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Fare": [10.0, 20.0, None, 40.0],
            "Embarked": ["S", "C", None, "S"],
        }).select(RAW_FEATURES)

    def test_validation_rows_do_not_change_fitted_statistics(self):
        validation = self.train.head(1).with_columns(
            pl.lit(1000.0).alias("Age"),
            pl.lit(9000.0).alias("Fare"),
            pl.lit("unseen").alias("Sex"),
        )
        transformer = TitanicFeatureTransformer("original").fit(self.train.to_numpy())
        self.assertEqual(transformer.age_median_, 30.0)
        self.assertEqual(transformer.fare_median_, 20.0)
        self.assertEqual(transformer.embarked_mode_, "S")
        features = transformer.transform(validation.to_numpy())
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(features[0, transformer.feature_names_out_.index("SexCode")], -1)
        self.assertTrue(np.isfinite(transformer.transform(self.train.to_numpy())).all())

    def test_binned_variant_uses_train_quantiles_and_output_schema(self):
        transformer = TitanicFeatureTransformer("binned").fit(self.train.to_numpy())
        transformed = transformer.transform(self.train.to_numpy())
        self.assertEqual(transformed.shape, (4, 8))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertIn("AgeBinCode", transformer.feature_names_out_)
        self.assertNotIn("Age", transformer.feature_names_out_)
        self.assertLess(transformer.age_breaks_.max(), 1000)

    def test_rejects_wrong_input_width(self):
        with self.assertRaisesRegex(ValueError, "8 Titanic features"):
            TitanicFeatureTransformer().fit(np.ones((4, 7)))


if __name__ == "__main__":
    unittest.main()
