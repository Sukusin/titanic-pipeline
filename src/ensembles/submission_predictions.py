"""Generate a submission with a saved ensemble and its matching test dataset."""

import argparse
import json
from pathlib import Path

import joblib
import polars as pl


def create_submission(artifact_path: Path, output_path: Path, test_path: Path | None = None):
    """Load a complete ensemble pipeline and preserve PassengerId order."""
    with open(artifact_path / "metadata.json") as file:
        metadata = json.load(file)
    test_path = test_path or Path(metadata["test_path"])
    test = pl.read_parquet(test_path)
    missing = set(metadata["features"] + ["PassengerId"]).difference(test.columns)
    if missing:
        raise ValueError(f"Test data {test_path} is missing columns: {sorted(missing)}")
    model = joblib.load(artifact_path / "model.joblib")
    predicted = model.predict(test.select(metadata["features"]).to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "PassengerId": test.get_column("PassengerId"),
            "Survived": predicted,
        }
    ).write_csv(output_path)


def main() -> None:
    """Select the best ensemble by the configured CV metric unless an artifact is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument(
        "--output-path", type=Path, default=Path("submissions/submission_ensemble.csv")
    )
    args = parser.parse_args()
    artifact_path = args.artifact_path
    if artifact_path is None:
        leaderboard = pl.read_csv("logs/ensembles/leaderboard.csv").filter(
            pl.col("model_type") == "ensemble"
        )
        primary = leaderboard.get_column("primary_metric")[0]
        artifact_path = Path(
            leaderboard.sort(f"mean_{primary}", descending=True).get_column("artifact_dir")[0]
        )
    create_submission(artifact_path, args.output_path, args.test_path)
    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()
