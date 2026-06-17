# Alomancy Testing Guide

This document describes the testing framework and how to run, understand, and add tests to alomancy.

## Overview

The alomancy package uses **pytest** as the primary testing framework. The test suite is organized by module and covers:

- **Unit tests**: Individual functions and classes (marked `@pytest.mark.unit`)
- **Integration tests**: Component interactions (marked `@pytest.mark.integration`)
- **External tests**: Require external software like MACE or Quantum Espresso (marked `@pytest.mark.requires_external`)

**Current test results**: 155 passed, 4 skipped (requires_external), 0 failed

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── core_tests/
│   ├── test_base_active_learning.py
│   └── test_standard_active_learning.py
├── database_tests/
│   └── test_global_database.py
├── initialize_tests/
│   └── test_initialization.py
├── mlip_train_tests/
│   └── test_mace_training.py
├── struc_gen_tests/
│   └── test_structure_generation.py
├── high_acc_tests/
│   └── test_quantum_espresso.py
└── utils_tests/
    └── test_utilities.py
```

## Running Tests

### Quick Start (Recommended for Development)

```bash
# Fast — skip coverage overhead
pytest -m unit --no-cov

# Single file
pytest tests/database_tests/test_global_database.py --no-cov

# With verbose output
pytest -m unit --no-cov -v
```

### Full Test Suite

```bash
# All tests (includes integration and skipped tests)
pytest --cov=alomancy

# Skip slow tests
pytest -m "not slow" --no-cov

# Skip external dependency tests
pytest -m "not requires_external" --no-cov
```

### Test Categories

```bash
# Unit tests only (fast, no external deps)
pytest -m unit --no-cov

# Integration tests
pytest -m integration --no-cov

# Slow tests (typically skipped in development)
pytest -m slow --no-cov

# External dependency tests (skipped unless deps available)
pytest -m requires_external --no-cov

# All except slow and external
pytest -m "not slow and not requires_external" --no-cov
```

### Debugging

```bash
# Stop on first failure
pytest -x

# Run only tests that failed last time
pytest --lf

# Verbose output with local variables
pytest -v -l

# Drop into debugger on failure
pytest --pdb
```

## Test Markers

Tests are marked with `@pytest.mark.` to control execution:

| Marker | When Used |
|--------|-----------|
| `@pytest.mark.unit` | Fast tests, no external dependencies |
| `@pytest.mark.integration` | Tests component interactions |
| `@pytest.mark.slow` | Long-running operations (skipped in normal development) |
| `@pytest.mark.requires_external` | Require MACE, QE, or other external software |

## Fixtures (from conftest.py)

### Atom Creation

```python
make_atoms(symbols, config_type=None, ref_energy=None, ref_forces=None)
```
Factory function (not a fixture) to create test ASE Atoms objects.

### Pre-built Atoms

- `h_atom`: H isolated atom with REF_energy and REF_forces
- `o_atom`: O isolated atom  
- `h2_dimer`: H2 molecule with init_dimer config_type and REF_forces
- `h2o_mol`: H2O structure with al_loop_0 config_type
- `sample_atoms`: Alias for h2o_mol
- `sample_atoms_list`: List of 5 H2O structures

### Files & Config

- `sample_xyz_file(tmp_path)`: Writes h_atom and h2o_mol to extxyz, returns Path
- `minimal_jobs_dict`: Minimal valid jobs_dict for workflow testing
- `mock_job_dict`: Alias for minimal_jobs_dict
- `temp_dir`: Alias for tmp_path (pytest built-in)

### Database Testing

GlobalDatabase tests use real sage_lib via:
```python
db = GlobalDatabase(str(tmp_path / "db"))
```

### Mocking External Calls

For `compute_initialization_needs()` tests that call the database, use `MagicMock`:
```python
mock_db = MagicMock()
mock_db.get_structure_counts.return_value = {...}
```

## Key Testing Patterns

### Pattern 1: Testing Real Code (No Mock-Testing)

Tests exercise actual code paths. Do NOT mock a function just to test it returns what you told the mock to return — test real behavior instead.

```python
@pytest.mark.unit
def test_make_atoms_creates_atoms():
    """Test that make_atoms creates ASE Atoms correctly."""
    atoms = make_atoms("H2O", config_type="test", ref_energy=5.0)
    assert atoms.get_chemical_formula() == "H2O"
    assert atoms.info["config_type"] == "test"
    assert atoms.info["REF_energy"] == 5.0
```

### Pattern 2: GlobalDatabase with Real sage_lib

Database tests use real sage_lib Partition objects:

```python
@pytest.mark.unit
def test_global_database_add_structures(tmp_path, sample_atoms_list):
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(sample_atoms_list)
    # Test real behavior of database
```

### Pattern 3: Controlled Mock Returns for External Interfaces

Mock external dependencies at the boundary only:

```python
@pytest.mark.unit
def test_compute_initialization_needs():
    """Mock only the database, test real initialization logic."""
    mock_db = MagicMock()
    mock_db.get_structure_counts.return_value = {
        ("init_dimer", "H2"): 2,
        ("init_trimer", "H2O"): 1,
    }
    # Test real initialization logic against controlled counts
```

### Pattern 4: Logging Tests

**Important**: pytest's caplog does NOT intercept alomancy logs (propagate=False).

To test log output, attach a handler directly:

```python
@pytest.mark.unit
def test_logging_output():
    import logging
    handler = logging.StreamHandler()
    logger = logging.getLogger("alomancy")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Now test code that logs
```

### Pattern 5: MACE and wfl Handling

MACE is patched at the top of `test_standard_active_learning.py`:

```python
sys.modules.setdefault('mace', MagicMock())
sys.modules.setdefault('mace.calculators', MagicMock())
```

**Never** import wfl in tests — it is not installed. If you need to mock wfl behavior, patch it via sys.modules.

## Testing Coverage

- **Minimum**: 80% overall (enforced by `--cov-fail-under=80` in pyproject.toml)
- **Target**: Coverage grows as features are added

Run coverage reports:

```bash
# Terminal report
pytest --cov=alomancy --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=alomancy --cov-report=html
open htmlcov/index.html
```

## Adding New Tests

### Step 1: Choose Location and Marker

- Place test in appropriate subdirectory (e.g., `database_tests/` for database code)
- Mark with `@pytest.mark.unit` (fast) or `@pytest.mark.integration` (component interactions)

### Step 2: Use Existing Fixtures

```python
@pytest.mark.unit
def test_my_function(sample_atoms, sample_atoms_list, tmp_path):
    """Test my_function with real data."""
    result = my_function(sample_atoms, sample_atoms_list)
    assert result is not None
```

### Step 3: Mock External Dependencies Only

```python
@pytest.mark.unit
def test_external_interface(monkeypatch):
    """Test function that calls external code."""
    # Mock at the boundary
    mock_external = MagicMock(return_value="expected")
    monkeypatch.setattr("alomancy.module.external_call", mock_external)
    
    result = my_function()
    assert result == "expected"
    mock_external.assert_called_once()
```

### Step 4: Include Edge Cases

```python
@pytest.mark.unit
def test_my_function_empty_list():
    """Test that function handles empty input."""
    result = my_function([])
    assert result == []

@pytest.mark.unit
def test_my_function_raises_on_invalid():
    """Test error handling."""
    with pytest.raises(ValueError, match="Invalid"):
        my_function(None)
```

### Checklist for New Tests

- [ ] Placed in correct subdirectory
- [ ] Marked with `@pytest.mark.unit` or `@pytest.mark.integration`
- [ ] Uses existing fixtures where possible
- [ ] Mocks only external boundaries (not internal functions)
- [ ] Tests happy path, edge cases, and errors
- [ ] Test name clearly describes what is tested
- [ ] Passes locally with `pytest -m unit --no-cov`

## Best Practices

1. **Test names are documentation**: `test_global_database_adds_and_retrieves_structures` describes the behavior better than `test_db_add`
2. **One concept per test**: Each test should verify one behavior
3. **Use descriptive assertions**: `assert result == expected` is clear
4. **Keep tests fast**: Unit tests should complete in milliseconds
5. **Make tests deterministic**: Use fixed seeds (alomancy uses `seed=803` by default)
6. **Test real code paths**: Avoid mock-testing (mocking just to assert the mock was called)
7. **Shared setup goes in fixtures**: Use conftest.py for common test data

## Linting and Type Checking

This project uses **ruff** for both linting and formatting (not black/isort/flake8).

```bash
# Check code style
ruff check src/ tests/

# Auto-fix style issues
ruff format src/ tests/

# Type checking (requires mypy)
mypy src/alomancy
```

## Pre-Commit Hooks

Pre-commit hooks run `ruff` (with `--fix`) and unit tests on every commit. Install with:

```bash
pre-commit install
```

## Continuous Integration

- Pre-commit hooks run `ruff` check and unit tests on each commit
- Full test suite runs on push to ensure no regressions
- Coverage must stay above 80%

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [ASE testing guide](https://wiki.fysik.dtu.dk/ase/development/testing.html)
