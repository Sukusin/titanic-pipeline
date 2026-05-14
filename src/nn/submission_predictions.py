import argparse
import json
from pathlib import Path

import joblib
import polars as pl
import torch
import yaml

import src.nn.model as models

MODEL_MAP = {
    "custom_model": models.CustomModel,
    "model_one": models.ModelOne,
    "model_two": models.ModelTwo,
    "model_batch_norm": models.ModelBatchNorm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-path",
        required=False,
        type=Path,
        help="Path to artifact directory with model.pt, scaler.joblib, metadata.json",
    )

    parser.add_argument(
        "--config-path",
        required=False,
        type=Path,
        help="Path to config.yaml. Required if artifact-path is passed manually.",
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Path to test data you want to predict",
    )

    parser.add_argument(
        "--output-path",
        required=False,
        type=Path,
        default=Path("submissions/submission_deepnn.csv"),
        help="Path where to store submission predictions",
    )

    return parser.parse_args()


def get_artifact_config_path(args: argparse.Namespace) -> tuple:
    if args.artifact_path and args.config_path:
        return args.artifact_path, args.config_path
    else:
        leaderboard = pl.read_csv("logs/deepnn/leaderboard.csv")

        primary_metric = leaderboard.select("primary_metric")[0].item()

        best_row = (
            leaderboard
            .sort(by=f"mean_{primary_metric}", descending=True)[0]
            .row(0, named=True))

        artifact_path = Path(best_row["artifact_dir"])
        config = Path(best_row["run_dir"])/"config.yaml"

        return artifact_path, config


def load_artifacts(
        artifact_path: str | Path,
        device: torch.device,
        ):
    artifact_path = Path(artifact_path)

    metadata_path = artifact_path/"metadata.json"
    model_path = artifact_path/"model.pt"
    scaler_path = artifact_path/"scaler.joblib"

    with open(file=metadata_path) as f:
        metadata = json.load(f)

    model_state_dict = torch.load(model_path, map_location=device)
    scaler = joblib.load(scaler_path)
    return metadata, model_state_dict, scaler


def build_model_from_config(in_features: int, config: dict) ->torch.nn.Module:
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]

    model_class = MODEL_MAP[model_name]

    model = model_class(in_features=in_features, **model_params)

    return model


def make_predictions(X,
                     model: torch.nn.Module,
                     model_state_dict: dict,
                     device: torch.device,
                     ) -> list:
    model.load_state_dict(state_dict=model_state_dict)
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.inference_mode():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        return preds.cpu().numpy().reshape(-1).tolist()


def save_submission(passenger_id, y_pred, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_dataset = pl.DataFrame(
        {
            "PassengerId": passenger_id,
            "Survived": y_pred,
        }
    )

    submission_dataset.write_csv(output_path)


def main() -> None:
    args = parse_args()
    artifact_path, config = get_artifact_config_path(args=args)
    with open(config) as f:
        config = yaml.safe_load(f)

    device = config["experiment"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    
    metadata, model_state_dict, scaler = load_artifacts(
        artifact_path=artifact_path,
        device=device,
        )
    
    features = metadata["features"]
    test_data = pl.read_parquet(args.test_path)
    passenger_id = test_data.get_column("PassengerId")

    X = test_data.select(features).to_numpy()
    in_features = X.shape[1]

    X_scaled = scaler.transform(X)

    model = build_model_from_config(in_features=in_features, config=config)

    y_pred = make_predictions(
        X=X_scaled,
        model=model,
        model_state_dict=model_state_dict,
        device=device)
    
    save_submission(
        passenger_id=passenger_id,
        y_pred=y_pred,
        output_path=args.output_path)

if __name__ == "__main__":
    main()
