# Contributing

We welcome contributions to ALomancy! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/julianholland/ALomancy.git
cd ALomancy
uv sync --extra docs        # alomancy (editable) + dev tools + docs, from uv.lock
uv run pre-commit install
```

Dependencies are locked in `uv.lock`. After changing `pyproject.toml`, run
`uv lock` and commit the updated lockfile (CI installs with `uv sync --locked`).
Build the package with `uv build` (sdist and wheel in `dist/`).

## Running Tests

```bash
uv run pytest
uv run pytest --cov=alomancy
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
