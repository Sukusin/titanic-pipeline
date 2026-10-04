PYTHON := uv run python

CLASSIC_CONFIG_DIR := configs/titanic/classic
DEEPNN_CONFIG_DIR = configs/$(COMPETITION)/deepnn

CLASSIC_MODELS := \
	knn logreg_l1 logreg_l2 logreg_elasticnet \
	decision_tree_classifier random_forest_classifier \
	catboost_classifier xgb_classifier lgbm_classifier
DATASET_TYPES := original binned

CLASSIC_MODEL ?= logreg_l1
DEEPNN_MODEL ?= model_one
DATASET_TYPE ?= original

CLASSIC_CONFIG_PATH := $(CLASSIC_CONFIG_DIR)/$(CLASSIC_MODEL).yaml
DEEPNN_CONFIG_PATH = $(DEEPNN_CONFIG_DIR)/$(DEEPNN_MODEL).yaml
ENSEMBLE_CONFIG_PATH ?= configs/titanic/ensembles/ensembles.yaml

CLASSIC_PIPELINE_MODELS ?= $(CLASSIC_MODELS)
CLASSIC_PIPELINE_DATASETS ?= $(DATASET_TYPES)
HOUSE_PRICE_PIPELINE_MODELS ?= linear_regression ridge random_forest catboost xgboost
TEST_PATH ?=
COMPETITION ?= titanic


.DEFAULT_GOAL := help
.PHONY: help lint type-check docker-build process-data pipeline \
	train-classic tune-classic train-classic-all leaderboard-classic \
	submission-predictions-classic classic-pipeline \
	train-deepnn leaderboard-deepnn submission-predictions-deepnn deepnn-pipeline \
	train-ensembles submission-predictions-ensembles

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-34s %s\n", $$1, $$2}'

lint: ## Run Ruff and apply safe fixes
	uv run ruff check --fix

type-check: ## Run static type checking
	uv run mypy .

docker-build: ## Build the Docker image
	docker build -t titanic-kaggle .

process-data: ## Prepare data; set COMPETITION=titanic or house_price
	$(PYTHON) -m src.$(COMPETITION).datasets.process_dataset

pipeline: ## Prepare, train, rank, and predict; set COMPETITION
	$(PYTHON) -m src.main --competition $(COMPETITION) $(if $(filter titanic,$(COMPETITION)),--models $(CLASSIC_PIPELINE_MODELS) --dataset-types $(CLASSIC_PIPELINE_DATASETS),--models $(HOUSE_PRICE_PIPELINE_MODELS))

train-classic: ## Train one classic model; set CLASSIC_MODEL and DATASET_TYPE
	$(PYTHON) -m src.titanic.classic.train --config $(CLASSIC_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

tune-classic: ## Tune and train one classic model; set CLASSIC_MODEL and DATASET_TYPE
	$(PYTHON) -m src.titanic.classic.train --config $(CLASSIC_CONFIG_PATH) --dataset-type $(DATASET_TYPE) --tune

train-classic-all: ## Train every classic model on every dataset variant
	@for dataset in $(DATASET_TYPES); do \
		for model in $(CLASSIC_MODELS); do \
			$(MAKE) --no-print-directory train-classic CLASSIC_MODEL=$$model DATASET_TYPE=$$dataset || exit $$?; \
		done; \
	done

leaderboard-classic: ## Build the classic-model leaderboard
	$(PYTHON) -m src.titanic.classic.make_leaderboard

submission-predictions-classic: ## Create a classic-model Kaggle submission
	$(PYTHON) -m src.titanic.classic.submission_predictions $(if $(TEST_PATH),--test-path $(TEST_PATH),)

classic-pipeline: pipeline ## Alias for the main pipeline

train-deepnn: ## Train one neural network; set COMPETITION and DEEPNN_MODEL
	$(PYTHON) -m src.$(COMPETITION).nn.train --config $(DEEPNN_CONFIG_PATH) $(if $(filter titanic,$(COMPETITION)),--dataset-type $(DATASET_TYPE),)

leaderboard-deepnn: ## Build the selected competition's neural-network leaderboard
	$(PYTHON) -m src.$(COMPETITION).nn.make_leaderboard

submission-predictions-deepnn: leaderboard-deepnn ## Create a neural-network Kaggle submission
	$(PYTHON) -m src.$(COMPETITION).nn.submission_predictions

deepnn-pipeline: process-data ## Prepare data, train a DNN, rank and create a submission
	$(MAKE) --no-print-directory train-deepnn COMPETITION=$(COMPETITION) DEEPNN_MODEL=$(DEEPNN_MODEL) DATASET_TYPE=$(DATASET_TYPE)
	$(MAKE) --no-print-directory submission-predictions-deepnn COMPETITION=$(COMPETITION)

train-ensembles: ## Compare averaging, voting and OOF stacking; save models and submission
	$(PYTHON) -m src.titanic.ensembles.train --config $(ENSEMBLE_CONFIG_PATH) --dataset-type $(DATASET_TYPE)

submission-predictions-ensembles: ## Create a submission from the best saved ensemble
	$(PYTHON) -m src.titanic.ensembles.submission_predictions
