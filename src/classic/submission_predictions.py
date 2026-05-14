import argparse
import json
from pathlib import Path

import joblib
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-path",
        required=False,
        type=Path,
        help="Path to artifact"
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Path to test/submission csv data you want to predict"   
    )

    parser.add_argument(
        "--output-path",
        required=False,
        default=Path("submissions/submission_classic.csv"),
        help="Path where to store submission predictions",
    )
    return parser.parse_args()

def get_artifact_path(args: argparse.Namespace) -> str | Path:
    if args.artifact_path is not None:
        return args.artifact_path
    else:
        leaderboard = pl.read_csv("logs/classic/leaderboard.csv")
        primary_metric = leaderboard.select("primary_metric")[0].item()
        artifact_path = (
            leaderboard
            .sort(by=f"mean_{primary_metric}", descending=True)[0]
            .get_column("artifact_dir")
            .item())
        return artifact_path

def load_artifacts(
        artifact_path: str | Path,
        ):
    metadata_path = f"{artifact_path}/metadata.json"
    with open(file=metadata_path) as f:
        metadata = json.load(f)
    model = joblib.load(f"{artifact_path}/model.joblib")
    scaler = joblib.load(f"{artifact_path}/scaler.joblib")
    return metadata, model, scaler

def make_predictions(X, model):
    y_pred = model.predict(X)
    return y_pred

def save_submission(passenger_id, y_pred, output_path) -> None:
    data = {
        "PassengerId": passenger_id,
        "Survived": y_pred,
    }
    submission_dataset = pl.DataFrame(data)

    submission_dataset.write_csv(output_path)

def main() -> None:
    args = parse_args()
    artifact_path = get_artifact_path(args=args)
    metadata, model, scaler = load_artifacts(artifact_path=artifact_path)
    features = metadata["features"]
    test_data = pl.read_parquet(args.test_path)
    passenger_id = test_data.get_column("PassengerId")

    X = test_data.select(features)

    X_scaled = scaler.transform(X)
    y_pred = make_predictions(X=X_scaled, model=model)
    save_submission(passenger_id=passenger_id, y_pred=y_pred, output_path=args.output_path)

if __name__ == "__main__":
    main()
