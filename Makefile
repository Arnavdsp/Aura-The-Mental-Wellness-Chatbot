.PHONY: help install dev serve test lint format typecheck evaluate dataset train docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the runtime (echo engine, no GPU needed)
	pip install -e .

dev:  ## Install runtime + dev tooling
	pip install -e ".[dev]"

serve:  ## Run the app with hot reload at http://localhost:8000
	AURA_ENGINE=$${AURA_ENGINE:-auto} python -m aura.cli serve --reload

test:  ## Run the test suite
	pytest

lint:  ## Check style
	ruff check src tests

format:  ## Apply formatting fixes
	ruff check --fix src tests

typecheck:  ## Run mypy
	mypy

evaluate:  ## Score coaching behaviour on the current engine
	python -m aura.cli evaluate --engine $${AURA_ENGINE:-echo}

dataset:  ## Build the preference dataset and write JSONL to artifacts/data
	python -m aura.cli dataset --out artifacts/data --show 1

train:  ## Fine-tune with DPO (needs a GPU and the train extra)
	python -m aura.cli train --strategy dpo --max-steps 200

docker:  ## Build the container image
	docker build -t aura:latest .

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
