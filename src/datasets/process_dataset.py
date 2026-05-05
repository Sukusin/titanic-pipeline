from pathlib import Path

import polars as pl

from src.datasets.download_data import download_main_dataset


RAW_DIR = "data/raw"
PROCESSED_ORIGINAL_DIR = Path("data/processed/original")
PROCESSED_BINNED_DIR = Path("data/processed/binned")

TRAIN_PATH = f"{RAW_DIR}/train.csv"
TEST_PATH = f"{RAW_DIR}/test.csv"

DROP_COLUMNS = ["PassengerId", "Ticket", "Cabin"]

TARGET = "Survived"

FEATURES_ORIGINAL = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "EmbarkedCode",
    "FamilySize",
    "IsSingleCode",
    "SexCode",
    "TitleCode",
]

FEATURES_BINNED = [
    "Pclass",
    "SexCode",
    "AgeBinCode",
    "FareBinCode",
    "EmbarkedCode",
    "FamilySize",
    "IsSingleCode",
    "TitleCode",
]

COLUMNS_TO_CODE = [
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "FareBin",
    "IsSingle",
]


def value_count(df: pl.DataFrame, value: str) -> pl.DataFrame:
    return (
        df
        .group_by(value)
        .len()
        .sort("len", descending=True)
    )


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
        ((pl.col("SibSp") + pl.col("Parch") + 1) == 1).alias("IsSingle"),
        pl.col("Name").str.extract(r",\s*([^\.]+)\.", 1).alias("Title"),
    ])


def filter_title(
    df: pl.DataFrame,
    title_threshold: int = 10,
) -> pl.DataFrame:
    title_count = value_count(data_train, "Title")

    common_titles = (
        title_count
        .filter(pl.col("len") >= title_threshold)
        .get_column("Title")
        .to_list()
    )

    return (
        df
        .with_columns(
            pl.when(pl.col("Title").is_in(common_titles))
            .then(pl.col("Title"))
            .otherwise(pl.lit("Misc"))
            .alias("Title")
        )
        .drop("Name")
    )


def get_quntile_breaks(
    df: pl.DataFrame,
    column: str,
    quantiles: tuple = (0.25, 0.5, 0.75),
) -> list[float]:
    breaks = []

    for q in quantiles:
        value = df.select(
            pl.col(column).quantile(q)
        ).item()

        breaks.append(float(value))

    return breaks


def cut_into_quantiles(
    df: pl.DataFrame,
    column: str,
    breaks: list,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column).cut(breaks).alias(f"{column}Bin")
    )


def add_code_columns_from_train(
    df: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    categories = (
        data_train
        .select(pl.col(column).unique().sort())
        .get_column(column)
        .to_list()
    )

    mapping = {
        category: idx
        for idx, category in enumerate(categories)
    }

    return df.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .replace(mapping)
        .cast(pl.Int64)
        .alias(f"{column}Code")
    )


def fill_missing_values(
    df: pl.DataFrame,
    age_median: float,
    fare_median: float,
    embarked_mode: str,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("Age").fill_null(age_median),
        pl.col("Fare").fill_null(fare_median),
        pl.col("Embarked").fill_null(embarked_mode),
    )


def save_dataset(
    df: pl.DataFrame,
    output_dir: Path,
    filename: str,
) -> None:
    df.write_parquet(output_dir / f"{filename}.parquet")
    df.write_csv(output_dir / f"{filename}.csv")


download_main_dataset(
    handle="titanic",
    output_dir="data/raw",
)

print("=="*20)
print("Start preprocessing!\n")

data_train = pl.read_csv(TRAIN_PATH)
data_test = pl.read_csv(TEST_PATH)

test_passenger_id = data_test.get_column("PassengerId")

data_train = data_train.drop(DROP_COLUMNS)
data_test = data_test.drop(DROP_COLUMNS)

all_datasets = [data_train, data_test]

for dataset in all_datasets:
    missing = dataset.select([
        pl.col(col).is_null().sum().alias(col)
        for col in dataset.columns
    ])


age_median = data_train.select(pl.col("Age").median()).item()
fare_median = data_train.select(pl.col("Fare").median()).item()

embarked_mode = (
    data_train
    .filter(pl.col("Embarked").is_not_null())
    .group_by("Embarked")
    .len()
    .sort("len", descending=True)
    .get_column("Embarked")[0]
)


all_datasets = []

for dataset in [data_train, data_test]:
    dataset = fill_missing_values(
        df=dataset,
        age_median=age_median,
        fare_median=fare_median,
        embarked_mode=embarked_mode,
    )

    all_datasets.append(dataset)

    missing = dataset.select(
        pl.col(col).null_count()
        for col in dataset.columns
    )


data_train, data_test = all_datasets


data_train = add_features(data_train)
data_test = add_features(data_test)

data_train = filter_title(data_train)
data_test = filter_title(data_test)


age_breaks = get_quntile_breaks(
    df=data_train,
    column="Age",
)

fare_breaks = get_quntile_breaks(
    df=data_train,
    column="Fare",
)


data_train = cut_into_quantiles(
    df=data_train,
    column="Age",
    breaks=age_breaks,
)

data_train = cut_into_quantiles(
    df=data_train,
    column="Fare",
    breaks=fare_breaks,
)

data_test = cut_into_quantiles(
    df=data_test,
    column="Age",
    breaks=age_breaks,
)

data_test = cut_into_quantiles(
    df=data_test,
    column="Fare",
    breaks=fare_breaks,
)


for column in COLUMNS_TO_CODE:
    data_train = add_code_columns_from_train(
        df=data_train,
        column=column,
    )

    data_test = add_code_columns_from_train(
        df=data_test,
        column=column,
    )


X_train = data_train.select(FEATURES_ORIGINAL)
y_train = data_train.select(TARGET)
X_test = data_test.select(FEATURES_ORIGINAL)


original_train_dataset = data_train.select(FEATURES_ORIGINAL + [TARGET])
binned_train_dataset = data_train.select(FEATURES_BINNED + [TARGET])

original_test_dataset = data_test.select([test_passenger_id] + FEATURES_ORIGINAL)
binned_test_dataset = data_test.select([test_passenger_id] + FEATURES_BINNED)


PROCESSED_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_BINNED_DIR.mkdir(parents=True, exist_ok=True)


save_dataset(
    df=original_train_dataset,
    output_dir=PROCESSED_ORIGINAL_DIR,
    filename="train_dataset",
)

save_dataset(
    df=binned_train_dataset,
    output_dir=PROCESSED_BINNED_DIR,
    filename="train_dataset",
)

save_dataset(
    df=original_test_dataset,
    output_dir=PROCESSED_ORIGINAL_DIR,
    filename="test_dataset",
)

save_dataset(
    df=binned_test_dataset,
    output_dir=PROCESSED_BINNED_DIR,
    filename="test_dataset",
)

print("Prepocessing done!")
print("=="*20)