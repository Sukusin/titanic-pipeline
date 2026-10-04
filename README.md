# Titanic и House Prices

Два воспроизводимых ML-пайплайна для Kaggle: [Titanic](https://www.kaggle.com/competitions/titanic) — классификация выживаемости пассажиров, [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — прогноз стоимости домов.

Каждый пайплайн подготавливает данные, оценивает модели на кросс-валидации, обучает итоговую модель на всём train и создаёт CSV для Kaggle. Для обоих соревнований доступны классические модели и PyTorch-нейросети; для Titanic — также Voting и OOF Stacking. Код и артефакты разделены по соревнованиям, общие утилиты переиспользуются.

## Основные результаты

Ниже — **локальная 5-fold CV**, среднее ± стандартное отклонение по фолдам, `seed=42`. Titanic использует Stratified K-fold, House Prices — K-fold. Предобработка обучается внутри train-части каждого фолда. Это не оценки Kaggle leaderboard; результаты выбора модели по CV требуют независимой проверки.

### Titanic

Классические модели и базовая MLP повторно проверены на текущем коде 5 октября 2026 года. В таблице используется вариант признаков `original`.

| Модель | F1 ↑ | Accuracy ↑ |
| --- | ---: | ---: |
| XGBoost | **0.7815 ± 0.0245** | 0.8429 |
| LightGBM | 0.7742 ± 0.0390 | 0.8395 |
| CatBoost | 0.7686 ± 0.0226 | 0.8350 |
| MLP (`model_one`) | 0.7664 ± 0.0218 | 0.8260 |
| Random Forest | 0.7636 ± 0.0289 | 0.8305 |
| Logistic Regression L2 | 0.7384 ± 0.0283 | 0.8070 |

XGBoost показал лучшую среднюю F1; MLP улучшает линейный baseline. В отдельном [сравнении ансамблей](reports/titanic/ensemble_results.md) Soft Voting получил F1 0.7681: заметного прироста относительно CatBoost не обнаружено. Полное сравнение классических моделей на `original` и `binned` — в [отчёте Titanic](reports/titanic/baseline_results.md).

### House Prices

| Модель | RMSLE ↓ |
| --- | ---: |
| CatBoost | **0.1255 ± 0.0185** |
| XGBoost | 0.1293 ± 0.0202 |
| Random Forest | 0.1424 ± 0.0213 |
| Ridge | 0.1468 ± 0.0442 |
| Linear Regression | 0.1512 ± 0.0470 |
| MLP (`model_one`) | 0.1702 ± 0.0301 |

CatBoost — лучший из проверенных вариантов House Prices. Простая MLP, перенесённая из Titanic, уступает всем классическим моделям в этом запуске. Модели обучаются на `log1p(SalePrice)`; предсказания возвращаются в исходный масштаб. Подробности: [классические модели](reports/house_price/classic_results.md), [нейросеть](reports/house_price/deepnn_results.md).

## Установка и данные

Нужны **Python 3.12+**, **uv** и **Make**. Выполняйте команды из корня репозитория:

```bash
uv sync --locked
```

Зависимости зафиксированы в `uv.lock`. Нейросети работают на CPU; при `device: auto` используется CUDA, если она доступна. Для Linux и Windows PyTorch устанавливается из настроенного в проекте индекса CUDA 12.8.

Скачайте `train.csv` и `test.csv` со страниц соревнований и разместите их так:

```text
data/
├── titanic/raw/{train.csv,test.csv}
└── house_price/raw/{train.csv,test.csv}
```

House Prices требует эти файлы локально. Titanic при отсутствии CSV пытается скачать данные через `kagglehub`; для этого нужен настроенный доступ к соревнованию Kaggle. `sample_submission.csv` для запуска не требуется.

## Запуск

### Классические модели

Общая команда выполняет подготовку, CV, финальное обучение, сбор leaderboard и создание submission:

```bash
make pipeline COMPETITION=titanic
make pipeline COMPETITION=house_price
```

Titanic сравнивает девять конфигураций: KNN, три варианта Logistic Regression, Decision Tree, Random Forest, CatBoost, XGBoost и LightGBM — на признаках `original` и `binned`. House Prices сравнивает пять моделей из таблицы выше.

Для короткого первого запуска:

```bash
make pipeline COMPETITION=titanic CLASSIC_PIPELINE_MODELS=logreg_l2 CLASSIC_PIPELINE_DATASETS=original
make pipeline COMPETITION=house_price HOUSE_PRICE_PIPELINE_MODELS=ridge
```

Без Make можно запустить тот же пайплайн через Python:

```bash
uv run python -m src.main --competition titanic --models logreg_l2 --dataset-types original
uv run python -m src.main --competition house_price --models ridge
```

Результат: `submissions/<competition>/classic.csv`. Submission выбирает **лучший сохранённый запуск** из активного leaderboard своего семейства, включая предыдущие эксперименты. Для другого пути используйте `--output-path` в `src.main`.

### Нейросети

```bash
make deepnn-pipeline COMPETITION=titanic DEEPNN_MODEL=model_one DATASET_TYPE=original
make deepnn-pipeline COMPETITION=house_price DEEPNN_MODEL=model_one
```

Результат: `submissions/<competition>/deepnn.csv`. House Prices использует общую с Titanic двухслойную MLP, one-hot кодирование категорий и MSE на логарифме цены. Для Titanic доступны также `model_two`, `custom_model` и модель с embeddings:

```bash
make deepnn-pipeline COMPETITION=titanic DEEPNN_MODEL=embedding_model DATASET_TYPE=original
```

### Ансамбли и подбор параметров Titanic

```bash
make process-data COMPETITION=titanic
make train-ensembles DATASET_TYPE=original
make tune-classic CLASSIC_MODEL=logreg_l2_optuna DATASET_TYPE=original
```

Ансамбли сравнивают усреднение, hard/soft Voting и Stacking с Linear Regression или Ridge; submission сохраняется в `submissions/titanic/ensemble.csv`. Stacking строит OOF-признаки во внутренней CV каждого внешнего фолда. Optuna подбирает параметры на тех же фолдах, на которых затем выводятся метрики: такая оценка может быть оптимистичной и не включена в таблицы выше.

Полный список команд: `make help`. Отдельные classic-команды `train-classic`, `tune-classic` и `leaderboard-classic` относятся к Titanic; общий `pipeline` и команды DeepNN поддерживают оба соревнования.

## Конфиги и артефакты

| Назначение | Titanic | House Prices |
| --- | --- | --- |
| Классические модели | `configs/titanic/classic/*.yaml` | `configs/house_price/classic.yaml` |
| Нейросети | `configs/titanic/deepnn/*.yaml` | `configs/house_price/deepnn/model_one.yaml` |
| Ансамбли | `configs/titanic/ensembles/ensembles.yaml` | — |

В YAML задаются параметры моделей, CV и seed; для нейросетей — также устройство, optimizer, scheduler, epochs и batch size.

```text
src/{titanic,house_price}/       подготовка, обучение и предсказание
src/utils/                     общие метрики, конфиги, сохранение и компоненты MLP
data/<competition>/            raw/ и processed/
logs/<competition>/<family>/   summary.json, метрики фолдов, конфиг, leaderboard.csv
models/<competition>/<family>/ модель, предобработка и metadata
submissions/<competition>/     CSV для Kaggle
notebooks/<competition>/       EDA и анализ результатов
reports/<competition>/         отчёты экспериментов
```

`<family>` — `classic`, `deepnn` или `ensembles` (Titanic). CSV содержит `PassengerId,Survived` для Titanic и `Id,SalePrice` для House Prices. Логи и модели исключены из Git и создаются при запуске. Старые артефакты Titanic в `archive/` относятся к прежнему расположению модулей; для предсказаний их нужно переобучить.

## Docker

Положите исходные CSV в `data/<competition>/raw/` перед сборкой: образ включает их, а подготовленные данные создаёт при запуске.

```bash
make docker-build
docker run --rm -v "$PWD/submissions:/app/submissions" titanic-kaggle
docker run --rm -v "$PWD/submissions:/app/submissions" titanic-kaggle python -m src.main --competition house_price
docker run --rm -v "$PWD/submissions:/app/submissions" titanic-kaggle make deepnn-pipeline COMPETITION=house_price
```

По умолчанию контейнер запускает короткий Titanic-пайплайн с Logistic Regression L2 на `original`. CSV сохраняются в подключённый каталог `submissions/`. Чтобы сохранить метрики и модели, дополнительно подключите `logs/` к `/app/logs` и `models/` к `/app/models`. Нейросети могут работать на CPU; для CUDA нужен запуск с `--gpus all` и настроенная поддержка GPU в Docker.

## Проверки

```bash
uv run ruff check .
make type-check
```

Дополнительные материалы: [EDA Titanic](notebooks/titanic/eda.ipynb), [EDA House Prices](notebooks/house_price/eda.ipynb).
