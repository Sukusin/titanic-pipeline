"""Prepare data, train classic models, rank them by F1, and create a submission."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.datasets.process_dataset import main as prepare_data

DEFAULT_MODELS = [
    "knn", "logreg_l1", "logreg_l2", "logreg_elasticnet",
    "decision_tree_classifier", "random_forest_classifier",
    "catboost_classifier", "xgb_classifier", "lgbm_classifier",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset-types", nargs="+", choices=["original", "binned"],
                        default=["original", "binned"])
    parser.add_argument("--output-path", type=Path,
                        default=Path("submissions/submission_classic.csv"))
    args = parser.parse_args()

    prepare_data()
    for dataset_type in args.dataset_types:
        for model in args.models:
            config = Path("configs/classic_config") / f"{model}.yaml"
            if not config.is_file():
                raise FileNotFoundError(f"Classic model config does not exist: {config}")
            subprocess.run(
                [sys.executable, "-m", "src.classic.train", "--config", str(config),
                 "--dataset-type", dataset_type],
                check=True,
            )
    subprocess.run([sys.executable, "-m", "src.classic.make_leaderboard"], check=True)
    subprocess.run(
        [sys.executable, "-m", "src.classic.submission_predictions",
         "--output-path", str(args.output_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
