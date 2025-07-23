# ToyRL Makefile

.PHONY: help docs docs-clean docs-serve install install-dev test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,docs]"

docs:  ## Build documentation
	export PATH=$$PATH:/home/ubuntu/.local/bin && sphinx-build -M html docs docs/_build

docs-clean:  ## Clean documentation build
	rm -rf docs/_build docs/api

docs-serve:  ## Serve documentation locally
	export PATH=$$PATH:/home/ubuntu/.local/bin && sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000

test:  ## Run tests
	pytest tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/
	isort src/ tests/