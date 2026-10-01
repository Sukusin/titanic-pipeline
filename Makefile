PYTHON := uv run python

CLASSIC_CONFIG_DIR := configs/classic_config
DEEPNN_CONFIG_DIR := configs/deepnn_config

CLASSIC_MODELS := \
	knn logreg_l1 logreg_l2 logreg_elasticnet \
	decision_tree_classifier random_forest_classifier \
	catboost_classifier xgb_classifier lgbm_classifier
DATASET_TYPES := original binned

CLASSIC_MODEL ?= logreg_l1
DEEPNN_MODEL ?= model_one
DATASET_TYPE ?= original

CLASSIC_CONFIG_PATH := $(CLASSIC_CONFIG_DIR)/$(CLASSIC_MODEL).yaml
DEEPNN_CONFIG_PATH := $(DEEPNN_CONFIG_DIR)/$(DEEPNN_MODEL).yaml
ENSEMBLE_CONFIG_PATH ?= configs/ensemble_config/ensembles.yaml

TEST_PATH ?= data/processed/original/test_dataset.parquet


.DEFAULT_GOAL := help
.PHONY: help lint type-check docker-build process-data \
	train-classic train-classic-all leaderboard-classic \
	submission-predictions-classic classic-pipeline \
	train-deepnn leaderboard-deepnn submission-predictions-deepnn \
	train-ensembles submission-predictions-ensembles test-ensembles

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-34s %s\n", $$1, $$2}'

lint: ## Run Ruff and apply safe fixes
	uv run ruff check --fix

type-check: ## Run static type checking
	uv run mypy .

docker-build: ## Build the Docker image
	docker build -t titanic-kaggle .

process-data: ## Download and preprocess the Titanic data
	$(PYTHON) -m src.datasets.process_dataset

train-classic: ## Train one classic model; set CLASSIC_MODEL and DATASET_TYPE
	$(PYTHON) -m src.classic.train --config $(CLASSIC_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

train-classic-all: ## Train every classic model on every dataset variant
	@for dataset in $(DATASET_TYPES); do \
		for model in $(CLASSIC_MODELS); do \
			$(MAKE) --no-print-directory train-classic CLASSIC_MODEL=$$model DATASET_TYPE=$$dataset || exit $$?; \
		done; \
	done

leaderboard-classic: ## Build the classic-model leaderboard
	$(PYTHON) -m src.classic.make_leaderboard

submission-predictions-classic: ## Create a classic-model Kaggle submission
	$(PYTHON) -m src.classic.submission_predictions --test-path $(TEST_PATH)

classic-pipeline: ## Train classic models, build leaderboard, create submission
	$(MAKE) --no-print-directory train-classic-all
	$(MAKE) --no-print-directory leaderboard-classic
	$(MAKE) --no-print-directory submission-predictions-classic

train-deepnn: ## Train one neural network; set DEEPNN_MODEL and DATASET_TYPE
	$(PYTHON) -m src.nn.train --config $(DEEPNN_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

leaderboard-deepnn: ## Build the neural-network leaderboard
	$(PYTHON) -m src.nn.make_leaderboard

submission-predictions-deepnn: leaderboard-deepnn ## Create a neural-network Kaggle submission
	$(PYTHON) -m src.nn.submission_predictions

train-ensembles: ## Compare averaging, voting and OOF stacking; save models and submission
	$(PYTHON) -m src.ensembles.train --config $(ENSEMBLE_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

submission-predictions-ensembles: ## Create a submission from the best saved ensemble
	$(PYTHON) -m src.ensembles.submission_predictions

test-ensembles: ## Check ensemble predictions and OOF isolation
	$(PYTHON) -m unittest discover -s tests -p 'test_ensembles.py'
