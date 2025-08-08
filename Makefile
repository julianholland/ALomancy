# Makefile for alomancy package development and testing

.PHONY: help install install-dev test test-unit test-integration test-slow test-all coverage clean lint format type-check docs build upload

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install the package"
	@echo "  install-dev    - Install package with development dependencies"
	@echo "  test           - Run unit tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-slow      - Run slow tests"
	@echo "  test-all       - Run all tests"
	@echo "  coverage       - Run tests with coverage reporting"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  type-check     - Run type checking with mypy"
	@echo "  docs           - Build documentation"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build package"
	@echo "  upload         - Upload to PyPI (test)"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Test targets
test: test-unit

test-unit:
	python -m pytest -m "unit or not (integration or slow or requires_external)" tests/

test-integration:
	python -m pytest -m integration tests/

test-slow:
	python -m pytest -m slow tests/

test-all:
	python -m pytest tests/

# Coverage
coverage:
	python -m pytest --cov=alomancy --cov-report=term-missing --cov-report=html tests/

coverage-xml:
	python -m pytest --cov=alomancy --cov-report=xml tests/

# Code quality
lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests

format:
	isort src tests
	black src tests

type-check:
	mypy src/alomancy

# Documentation
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

# Development convenience targets
dev-setup: install-dev
	pre-commit install

test-quick:
	python -m pytest -x --lf tests/

test-parallel:
	python -m pytest -n auto tests/

test-verbose:
	python -m pytest -v tests/

# CI targets
ci-test:
	python -m pytest --cov=alomancy --cov-report=xml --cov-fail-under=80 tests/

ci-lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests
	mypy src/alomancy

# Environment and dependency management
requirements:
	pip-compile requirements.in

requirements-dev:
	pip-compile requirements-dev.in

update-deps:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in
