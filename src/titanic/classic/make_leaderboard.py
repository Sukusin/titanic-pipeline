import json
from pathlib import Path

import polars as pl

LOGS_DIR = Path("logs/titanic/classic")
OUTPUT_PATH = Path("logs/titanic/classic/leaderboard.csv")
LEADERBOARD_COLUMNS = [
    "experiment_name",
    "model_name",
    "dataset_type",
    "preprocessing",
    "preprocessing_version",
    "primary_metric",
    "mean_accuracy",
    "std_accuracy",
    "mean_precision",
    "std_precision",
    "mean_recall",
    "std_recall",
    "mean_f1",
    "std_f1",
    "mean_roc_auc",
    "std_roc_auc",
    "config_path",
    "run_dir",
    "artifact_dir",
]


def main() -> None:
    rows = []

    for summary_path in LOGS_DIR.glob("*/summary.json"):
        with open(summary_path) as f:
            row = json.load(f)

        if row.get("preprocessing_version") != 2:
            continue

        artifact_dir = Path(row["artifact_dir"])
        metadata_path = artifact_dir / "metadata.json"
        if not (artifact_dir / "model.joblib").is_file() or not metadata_path.is_file():
            continue
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
        if not Path(metadata["test_path"]).is_file():
            continue

        row["run_dir"] = str(summary_path.parent)
        row["primary_metric"] = "f1"
        row = {
            column: row.get(column)
            for column in LEADERBOARD_COLUMNS
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No classic summaries with fold-local preprocessing found.")

    leaderboard = (
        pl.DataFrame(rows)
        .sort("mean_f1", descending=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.write_csv(OUTPUT_PATH)

    print(leaderboard.head(5))
    print(f"Saved leaderboard to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
