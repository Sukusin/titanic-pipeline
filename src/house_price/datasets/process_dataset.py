"""Prepare the Kaggle House Prices CSV files for model training."""

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/house_price/raw")
OUTPUT_DIR = Path("data/house_price/processed/raw")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        frame = pl.read_csv(RAW_DIR / f"{name}.csv", null_values="NA", infer_schema_length=None)
        frame.write_csv(OUTPUT_DIR / f"{name}_dataset.csv")
    print(f"Saved House Prices data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
