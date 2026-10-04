"""Rank House Prices models by cross-validated RMSLE."""

import json
from pathlib import Path

import polars as pl

LOGS_DIR = Path("logs/house_price/classic")
OUTPUT_PATH = LOGS_DIR / "leaderboard.csv"


def main() -> None:
    rows = []
    for path in LOGS_DIR.glob("*/summary.json"):
        with path.open() as file:
            rows.append(json.load(file))
    leaderboard = pl.DataFrame(rows).sort("mean_rmsle")
    leaderboard.write_csv(OUTPUT_PATH)
    print(leaderboard.select("model_name", "mean_rmsle", "mean_rmse", "mean_mae"))


if __name__ == "__main__":
    main()
