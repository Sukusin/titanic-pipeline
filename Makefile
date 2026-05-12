PYTHON := uv run python

CLASSIC_CONFIG_DIR := configs/classic_config
DEEPNN_CONFIG_DIR := configs/deepnn_config

MODEL ?= logreg_l1
DATASET_TYPE ?= original

CLASSIC_CONFIG_PATH := $(CONFIG_DIR)/$(MODEL).yaml
DEEPNN_CONFIG_PATH := $(DEEPNN_CONFIG_DIR)/$(MODEL).yaml

TEST_PATH ?= data/processed/original/test_dataset.parquet

lint:
	uv run ruff check --fix

type-check:
	uv run mypy .

docker-build:
	docker build -it titanic-kaggle .

process-data:
	$(PYTHON) -m src.datasets.process_dataset

train-classic:
	$(PYTHON) -m src.classic.train --config $(CLASSIC_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

train-classic-all:
	make train-classic MODEL=knn DATASET_TYPE=binned
	make train-classic MODEL=knn DATASET_TYPE=original
	make train-classic MODEL=logreg_l1 DATASET_TYPE=binned
	make train-classic MODEL=logreg_l1 DATASET_TYPE=original
	make train-classic MODEL=logreg_l2 DATASET_TYPE=binned
	make train-classic MODEL=logreg_l2 DATASET_TYPE=original
	make train-classic MODEL=logreg_elasticnet DATASET_TYPE=binned
	make train-classic MODEL=logreg_elasticnet DATASET_TYPE=original
	make train-classic MODEL=decision_tree_classifier DATASET_TYPE=original
	make train-classic MODEL=decision_tree_classifier DATASET_TYPE=binned
	make train-classic MODEL=random_forest_classifier DATASET_TYPE=original
	make train-classic MODEL=random_forest_classifier DATASET_TYPE=binned
	make train-classic MODEL=catboost_classifier DATASET_TYPE=original
	make train-classic MODEL=catboost_classifier DATASET_TYPE=binned
	make train-classic MODEL=xgb_classifier DATASET_TYPE=original
	make train-classic MODEL=xgb_classifier DATASET_TYPE=binned
	make train-classic MODEL=lgbm_classifier DATASET_TYPE=original
	make train-classic MODEL=lgbm_classifier DATASET_TYPE=binned

leaderboard-classic:
	$(PYTHON) -m src.classic.make_leaderboard

submission-predictions-classic:
	$(PYTHON) -m src.classic.submission_predictions --test-path $(TEST_PATH)

classic-pipeline:
	make train-classic-all
	make leaderboard-classic
	make submission-predictions-classic

train-deepnn:
		$(PYTHON) -m src.nn.train --config $(DEEPNN_CONFIG_PATH) --dataset-type $(DATASET_TYPE)
