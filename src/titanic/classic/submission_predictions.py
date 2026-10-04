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
        help="Optional override; defaults to the selected model's test dataset",
    )

    parser.add_argument(
        "--output-path",
        required=False,
        default=Path("submissions/titanic/classic.csv"),
        help="Path where to store submission predictions",
    )
    return parser.parse_args()

def get_artifact_path(args: argparse.Namespace) -> str | Path:
    if args.artifact_path is not None:
        return args.artifact_path
    else:
        leaderboard = pl.read_csv("logs/titanic/classic/leaderboard.csv")
        if "preprocessing_version" in leaderboard.columns:
            current = leaderboard.filter(pl.col("preprocessing_version") == 2)
            if not current.is_empty():
                leaderboard = current
        artifact_path = (
            leaderboard
            .sort(by="mean_f1", descending=True)[0]
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
    feature_transformer = (
        joblib.load(Path(artifact_path) / "features.joblib")
        if "features_path" in metadata else None
    )
    return metadata, model, scaler, feature_transformer

def make_predictions(X, model):
    y_pred = model.predict(X)
    return y_pred

def save_submission(passenger_id, y_pred, output_path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "PassengerId": passenger_id,
        "Survived": y_pred,
    }
    submission_dataset = pl.DataFrame(data)

    submission_dataset.write_csv(output_path)

def main() -> None:
    args = parse_args()
    artifact_path = get_artifact_path(args=args)
    metadata, model, scaler, feature_transformer = load_artifacts(artifact_path=artifact_path)
    features = metadata["features"]
    test_path = args.test_path or metadata.get("test_path")
    if test_path is None:
        test_path = f"data/titanic/processed/{metadata['dataset_type']}/test_dataset.parquet"
    test_data = pl.read_parquet(test_path)
    passenger_id = test_data.get_column("PassengerId")

    X = test_data.select(features)
    X_features = (
        feature_transformer.transform(X.to_numpy()) if feature_transformer is not None else X
    )
    X_scaled = scaler.transform(X_features)
    y_pred = make_predictions(X=X_scaled, model=model)
    save_submission(passenger_id=passenger_id, y_pred=y_pred, output_path=args.output_path)

if __name__ == "__main__":
    main()
