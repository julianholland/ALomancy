# Contributing

We welcome contributions to ALomnacy! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/julianholland/ALomancy.git
cd ALomancy
pip install -e ".[dev,docs]"
pre-commit install
```

## Running Tests

```bash
pytest
pytest --cov=alomancy
```

## Documentation

Build documentation locally:

```bash
cd docs
make html
```

## Code Style

We use `ruff` for formatting and linting:

```bash
ruff check .
ruff format .
```
