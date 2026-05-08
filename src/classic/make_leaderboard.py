import json
from pathlib import Path

import polars as pl

LOGS_DIR = Path("logs/classic")
OUTPUT_PATH = Path("logs/classic/leaderboard.csv")
LEADERBOARD_COLUMNS = [
    "experiment_name",
    "model_name",
    "dataset_type",
    "preprocessing",
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

        row["run_dir"] = str(summary_path.parent)
        row = {
            column: row.get(column)
            for column in LEADERBOARD_COLUMNS
        }
        rows.append(row)

    if not rows:
        print("No experiment summaries found.")
        return

    leaderboard = (
        pl.DataFrame(rows)
        .sort("mean_accuracy", descending=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.write_csv(OUTPUT_PATH)

    print(leaderboard.head(5))
    print(f"Saved leaderboard to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
