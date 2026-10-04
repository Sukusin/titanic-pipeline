"""Prepare raw Titanic columns; learned transformations run inside model CV folds."""

from pathlib import Path

import polars as pl

from src.titanic.datasets.download_data import download_main_dataset
from src.titanic.datasets.features import RAW_FEATURES

RAW_DIR = Path("data/titanic/raw")
OUTPUT_DIR = Path("data/titanic/processed/raw")


def main() -> None:
    """Save raw Titanic features while leaving learned transforms for CV folds."""
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        download_main_dataset(handle="titanic", output_dir=RAW_DIR)

    train = pl.read_csv(train_path).select([*RAW_FEATURES, "Survived"])
    test = pl.read_csv(test_path).select(["PassengerId", *RAW_FEATURES])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.write_parquet(OUTPUT_DIR / "train_dataset.parquet")
    test.write_parquet(OUTPUT_DIR / "test_dataset.parquet")
    train.write_csv(OUTPUT_DIR / "train_dataset.csv")
    test.write_csv(OUTPUT_DIR / "test_dataset.csv")
    print(f"Saved raw train and test data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
