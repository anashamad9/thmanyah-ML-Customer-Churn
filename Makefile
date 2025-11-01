.PHONY: help setup lint format test train retrain api monitor

PYTHONPATH := src
export PYTHONPATH

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  setup    Install dependencies using uv (system mode)"
	@echo "  lint     Run ruff lint checks"
	@echo "  format   Format code with black"
	@echo "  test     Run unit tests with pytest"
	@echo "  train    Train the churn model"
	@echo "  retrain  Retrain the model using scripts/retrain.py"
	@echo "  api      Launch the FastAPI prediction service"
	@echo "  monitor  Execute data and performance drift checks"

setup:
	uv pip install --system ".[dev]"

lint:
	$(PYTHON) -m ruff check src scripts tests api

format:
	$(PYTHON) -m black src scripts tests api

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) scripts/train.py --config configs/training.yaml

retrain:
	$(PYTHON) scripts/retrain.py --config configs/training.yaml

api:
	$(PYTHON) -m uvicorn api.main:app --reload

monitor:
	$(PYTHON) scripts/monitor.py --config configs/monitoring.yaml
