PYTHON := uv run python

CONFIG_DIR := configs/classic_config

MODEL ?= logreg_l1
DATASET_TYPE ?= original

CONFIG_PATH := $(CONFIG_DIR)/$(MODEL).yaml

lint:
	uv run ruff check --fix

process-data:
	$(PYTHON) -m src.datasets.process_dataset

train-classic:
	$(PYTHON) -m src.classic.train --config $(CONFIG_PATH) --dataset-type $(DATASET_TYPE)

type-check:
	uv run mypy .