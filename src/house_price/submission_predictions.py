"""Generate a House Prices submission from a trained regressor."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--output-path", type=Path,
                        default=Path("submissions/house_price/classic.csv"))
    args = parser.parse_args()
    artifact_path = args.artifact_path
    if artifact_path is None:
        leaderboard = pl.read_csv("logs/house_price/classic/leaderboard.csv")
        artifact_path = Path(leaderboard.sort("mean_rmsle")[0, "artifact_dir"])

    with (artifact_path / "metadata.json").open() as file:
        metadata = json.load(file)
    test = pd.read_csv(args.test_path or metadata["test_path"],
                       keep_default_na=False, na_values=[""])
    model = joblib.load(artifact_path / "model.joblib")
    predicted = np.maximum(model.predict(test[metadata["features"]]), 0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({metadata["id"]: test[metadata["id"]],
                  metadata["target"]: predicted}).to_csv(args.output_path, index=False)
    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()
