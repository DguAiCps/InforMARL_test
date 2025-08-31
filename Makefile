# InforMARL Bottleneck Environment Makefile

.PHONY: install train demo test quicktest lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install package and dependencies"
	@echo "  train      - Run training (default: 100 episodes, 4 agents)"
	@echo "  demo       - Run animation demo (default: 4 agents)"
	@echo "  test       - Run quick movement test (default: 2 agents)"
	@echo "  quicktest  - Alias for test"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black and isort"
	@echo "  clean      - Clean build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Training and evaluation
train:
	python -m src.informarl_bneck.cli.train

train-long:
	python -m src.informarl_bneck.cli.train 200 6

demo:
	python -m src.informarl_bneck.cli.demo

demo-big:
	python -m src.informarl_bneck.cli.demo 6

test:
	python -m src.informarl_bneck.cli.quicktest

quicktest: test

# Code quality
lint:
	flake8 src/
	black --check src/
	isort --check-only src/

format:
	black src/
	isort src/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete