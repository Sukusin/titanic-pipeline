lint:
	uv run ruff check --fix

process-data:
	python -m src.datasets.process_dataset
	
# type-check:
# 	uv run mypy deep-nn/