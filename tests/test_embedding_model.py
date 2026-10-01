"""Validate category isolation, embedding gradients and DNN artifact compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler

from src.nn import engine
from src.nn.data_setup import build_kfold_dataloader
from src.nn.embedding_model import EmbeddingModel
from src.nn.preprocessing import EmbeddingPreprocessor, model_input_params
from src.nn.submission_predictions import (
    build_model_from_config,
    load_artifacts,
    make_predictions,
)


class EmbeddingTests(unittest.TestCase):
    """Use small mixed datasets to verify training and inference contracts."""

    def setUp(self):
        torch.manual_seed(42)
        self.X = np.array([[10, 1.0], [20, 3.0], [10, 5.0], [20, 7.0]])
        self.preprocessor = EmbeddingPreprocessor(
            StandardScaler(), ["Category", "Value"], ["Category"]
        ).fit(self.X)
        self.config = {
            "model": {
                "name": "embedding_model",
                "params": {"hidden_features": [8], "embedding_dims": 3},
            }
        }

    def test_only_numbers_are_scaled_and_unknown_categories_use_zero(self):
        transformed = self.preprocessor.transform(self.X)
        np.testing.assert_array_equal(transformed[:, 0], [1, 2, 1, 2])
        np.testing.assert_allclose(transformed[:, 1].mean(), 0, atol=1e-7)
        unseen = self.preprocessor.transform([[999, 9.0]])
        self.assertEqual(unseen[0, 0], 0)
        np.testing.assert_allclose(self.preprocessor.scaler_.mean_, [4.0])
        self.assertEqual(model_input_params(self.preprocessor)["categorical_cardinalities"], [3])

    def test_embeddings_receive_gradients_and_unknown_vector_stays_zero(self):
        model = build_model_from_config(2, self.config, model_input_params(self.preprocessor))
        X = torch.tensor(self.preprocessor.transform(self.X))
        y = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        before = model.embeddings[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = torch.nn.BCEWithLogitsLoss()(model(X), y)
        loss.backward()
        self.assertGreater(model.embeddings[0].weight.grad[1:].abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before[1:], model.embeddings[0].weight[1:]))
        torch.testing.assert_close(model.embeddings[0].weight[0], torch.zeros(3))
        self.assertEqual(model(X).shape, (4, 1))

    def test_each_fold_fits_its_own_vocabulary_and_numeric_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pl.DataFrame(
                {
                    "Category": np.arange(1000, 1012),
                    "Value": np.arange(12, dtype=float),
                    "Survived": np.arange(12) % 2,
                }
            ).write_parquet(path)
            folds, count = build_kfold_dataloader(
                path,
                "Survived",
                "standard",
                4,
                n_splits=3,
                shuffle=False,
                categorical_features=["Category"],
            )
            self.assertEqual(count, 2)
            for train_loader, val_loader, preprocessor in folds:
                train_X = train_loader.dataset.tensors[0]
                val_X = val_loader.dataset.tensors[0]
                self.assertTrue(torch.all(train_X[:, 0] > 0))
                self.assertTrue(torch.all(val_X[:, 0] == 0))
                self.assertEqual(len(preprocessor.encoder_.categories_[0]), 8)
                np.testing.assert_allclose(train_X[:, 1].mean(), 0, atol=1e-7)
                model = build_model_from_config(2, self.config, model_input_params(preprocessor))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                loss, accuracy = engine.train_step(
                    model,
                    train_loader,
                    torch.nn.BCEWithLogitsLoss(),
                    optimizer,
                    torch.device("cpu"),
                )
                self.assertTrue(np.isfinite(loss))
                self.assertTrue(0 <= accuracy <= 1)
                self.assertEqual(model(val_X).shape, (4, 1))

    def test_saved_preprocessor_and_model_reproduce_predictions(self):
        model = build_model_from_config(2, self.config, model_input_params(self.preprocessor))
        X = self.preprocessor.transform([[10, 2], [999, 8]])
        expected = make_predictions(X, model, model.state_dict(), torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            metadata = {"model_input_params": model_input_params(self.preprocessor)}
            (artifact / "metadata.json").write_text(json.dumps(metadata))
            joblib.dump(self.preprocessor, artifact / "scaler.joblib")
            torch.save(model.state_dict(), artifact / "model.pt")
            loaded_metadata, state, preprocessor = load_artifacts(artifact, torch.device("cpu"))
            loaded_model = build_model_from_config(
                2, self.config, loaded_metadata["model_input_params"]
            )
            actual_X = preprocessor.transform([[10, 2], [999, 8]])
            np.testing.assert_allclose(actual_X, X)
            actual = make_predictions(actual_X, loaded_model, state, torch.device("cpu"))
            self.assertEqual(expected, actual)

    def test_plain_mlp_and_all_categorical_input_remain_supported(self):
        config = {
            "model": {"name": "model_one", "params": {"hidden_features": 4, "out_features": 1}}
        }
        model = build_model_from_config(2, config)
        self.assertEqual(model(torch.zeros(3, 2)).shape, (3, 1))
        self.assertEqual(model_input_params(StandardScaler()), {})
        preprocessor = EmbeddingPreprocessor(StandardScaler(), ["Category"], ["Category"]).fit(
            self.X[:, :1]
        )
        embedded = EmbeddingModel(1, **model_input_params(preprocessor), hidden_features=[4])
        output = embedded(torch.tensor(preprocessor.transform([[10], [999]])))
        self.assertEqual(output.shape, (2, 1))

    def test_invalid_categorical_schema_is_rejected(self):
        for categories in ([], ["Missing"], ["Category", "Category"]):
            with self.assertRaises(ValueError):
                EmbeddingPreprocessor(StandardScaler(), ["Category", "Value"], categories).fit(
                    self.X
                )
        with self.assertRaisesRegex(ValueError, "Embedding dimensions"):
            EmbeddingModel(2, [0], [1], [3], [8], embedding_dims=[2, 2])


if __name__ == "__main__":
    unittest.main()
