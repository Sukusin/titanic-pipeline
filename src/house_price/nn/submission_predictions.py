"""Generate a House Prices submission from a saved neural network."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch

from src.house_price.nn.model import MODEL_MAP


def main() -> None:
    """Restore the best saved network and write nonnegative price predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--output-path", type=Path,
                        default=Path("submissions/house_price/deepnn.csv"))
    args = parser.parse_args()
    artifact_path = args.artifact_path
    if artifact_path is None:
        leaderboard = pl.read_csv("logs/house_price/deepnn/leaderboard.csv")
        artifact_path = Path(leaderboard.sort("mean_rmsle")[0, "artifact_dir"])

    with (artifact_path / "metadata.json").open() as file:
        metadata = json.load(file)
    test = pd.read_csv(args.test_path or metadata["test_path"],
                       keep_default_na=False, na_values=[""])
    preprocessor = joblib.load(artifact_path / "preprocessor.joblib")
    features = preprocessor.transform(test[metadata["features"]])
    model_class = MODEL_MAP[metadata["model_name"]]
    model = model_class(in_features=metadata["in_features"], **metadata["model_params"])
    model.load_state_dict(torch.load(artifact_path / "model.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    with torch.inference_mode():
        predicted_log = model(torch.tensor(features, dtype=torch.float32)).numpy().ravel()
    predicted = np.maximum(np.expm1(predicted_log), 0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({metadata["id"]: test[metadata["id"]],
                  metadata["target"]: predicted}).to_csv(args.output_path, index=False)
    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()
