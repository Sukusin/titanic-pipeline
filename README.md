# Titanic Kaggle

Проект для соревнования [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
В репозитории есть полный пайплайн: загрузка данных Kaggle, препроцессинг, обучение классических ML-моделей и нейросетей, сбор leaderboard по локальной кросс-валидации и генерация `submission.csv`.

## Что внутри

- `src/datasets/` - загрузка и подготовка датасетов.
- `src/classic/` - обучение классических моделей: Logistic Regression, KNN, Decision Tree, Random Forest, CatBoost, XGBoost, LightGBM.
- `src/nn/` - обучение PyTorch-моделей.
- `configs/classic_config/` - YAML-конфиги классических моделей.
- `configs/deepnn_config/` - YAML-конфиги нейросетей.
- `data/raw/` - исходные файлы Kaggle.
- `data/processed/original/` - обработанные признаки без биннинга `Age` и `Fare`.
- `data/processed/binned/` - обработанные признаки с бинами для `Age` и `Fare`.
- `logs/` - метрики экспериментов и локальные leaderboard-файлы.
- `models/` - сохраненные модели, scaler и metadata.
- `submissions/` - CSV-файлы для отправки на Kaggle.
- `notebooks/` - EDA и проверка результатов.

## Требования

- Python `>=3.12`
- `uv`
- Kaggle credentials для загрузки данных через `kagglehub`

Зависимости описаны в `pyproject.toml` и фиксируются через `uv.lock`.

## Установка

```bash
uv sync
```

Если используешь GPU, PyTorch берется из индекса `pytorch-cu128`, который уже прописан в `pyproject.toml`.

## Подготовка данных

```bash
make process-data
```

Команда скачивает данные соревнования в `data/raw/`, строит признаки и сохраняет два варианта датасета:

- `data/processed/original/train_dataset.parquet`
- `data/processed/original/test_dataset.parquet`
- `data/processed/binned/train_dataset.parquet`
- `data/processed/binned/test_dataset.parquet`

Основные признаки:

- `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
- закодированные `Sex`, `Embarked`, `Title`
- `FamilySize`, `IsSingle`
- для `binned`-датасета дополнительно используются бины `Age` и `Fare`

## Обучение классических моделей

Одна модель:

```bash
make train-classic CLASSIC_MODEL=logreg_l1 DATASET_TYPE=original
```

Все классические модели на обоих вариантах датасета:

```bash
make train-classic-all
```

После обучения артефакты сохраняются в:

- `logs/classic/<run_name>/summary.json`
- `logs/classic/<run_name>/fold_metrics.csv`
- `models/classic/<run_name>/model.joblib`
- `models/classic/<run_name>/scaler.joblib`
- `models/classic/<run_name>/metadata.json`

Собрать локальный leaderboard:

```bash
make leaderboard-classic
```

Файл будет сохранен в `logs/classic/leaderboard.csv`.

Сгенерировать submission по лучшей классической модели:

```bash
make submission-predictions-classic
```

По умолчанию результат сохраняется в `submissions/submission_classic.csv`.

Полный classic-пайплайн:

```bash
make classic-pipeline
```

## Обучение нейросети

```bash
make train-deepnn DEEPNN_MODEL=custom_model DATASET_TYPE=binned
```

Доступные конфиги лежат в `configs/deepnn_config/`.

После обучения артефакты сохраняются в:

- `logs/deepnn/<run_name>/summary.json`
- `logs/deepnn/<run_name>/fold_metrics.csv`
- `logs/deepnn/<run_name>/final_history.csv`
- `models/deepnn/<run_name>/model.pt`
- `models/deepnn/<run_name>/scaler.joblib`
- `models/deepnn/<run_name>/metadata.json`

Собрать leaderboard для нейросетей:

```bash
make leaderboard-deepnn
```

Сгенерировать submission по лучшей нейросети:

```bash
make submission-predictions-deepnn
```

По умолчанию результат сохраняется в `submissions/submission_deepnn.csv`.

## Конфиги

Эксперименты настраиваются через YAML:

```yaml
experiment:
  name: "logreg_l1"
  seed: 42

validation:
  type: "stratified_kfold"
  n_splits: 5

metrics:
  primary: "f1"
  log:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "roc_auc"
```

Для классических моделей конфиг задает:

- путь к данным;
- scaler;
- модель и параметры;
- метрики;
- параметры Stratified K-Fold.

Для нейросетей дополнительно задаются:

- архитектура;
- loss;
- optimizer;
- batch size;
- количество эпох;
- устройство `cpu`, `cuda` или `auto`.

## Качество кода

Линтинг:

```bash
make lint
```

Проверка типов:

```bash
make type-check
```

## Docker

Сборка образа:

```bash
make docker-build
```

## Быстрый старт

```bash
uv sync
make process-data
make train-classic-all
make leaderboard-classic
make submission-predictions-classic
```

После этого файл для Kaggle будет лежать в `submissions/submission_classic.csv`.
