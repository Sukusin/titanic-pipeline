"""Prepare data, train models, and create a Kaggle submission."""

import argparse
import subprocess
import sys
from pathlib import Path

TITANIC_MODELS = [
    "knn", "logreg_l1", "logreg_l2", "logreg_elasticnet",
    "decision_tree_classifier", "random_forest_classifier",
    "catboost_classifier", "xgb_classifier", "lgbm_classifier",
]
HOUSE_PRICE_MODELS = ["linear_regression", "ridge", "random_forest", "catboost", "xgboost"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", choices=["titanic", "house_price"], default="titanic")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dataset-types", nargs="+", choices=["original", "binned"],
                        default=["original", "binned"])
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    if args.competition == "house_price":
        from src.house_price.datasets.process_dataset import main as prepare_data

        prepare_data()
        for model in args.models or HOUSE_PRICE_MODELS:
            subprocess.run(
                [sys.executable, "-m", "src.house_price.train", "--model", model],
                check=True,
            )
        module = "src.house_price"
        output_path = args.output_path or Path("submissions/house_price/classic.csv")
    else:
        from src.titanic.datasets.process_dataset import main as prepare_data

        prepare_data()
        for dataset_type in args.dataset_types:
            for model in args.models or TITANIC_MODELS:
                config = Path("configs/titanic/classic") / f"{model}.yaml"
                if not config.is_file():
                    raise FileNotFoundError(f"Classic model config does not exist: {config}")
                subprocess.run(
                    [sys.executable, "-m", "src.titanic.classic.train", "--config", str(config),
                     "--dataset-type", dataset_type],
                    check=True,
                )
        module = "src.titanic.classic"
        output_path = args.output_path or Path("submissions/titanic/classic.csv")
    subprocess.run([sys.executable, "-m", f"{module}.make_leaderboard"], check=True)
    subprocess.run(
        [sys.executable, "-m", f"{module}.submission_predictions",
         "--output-path", str(output_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
