# Alomancy Testing Guide

This document describes the testing framework and strategies for the alomancy package.

## Overview

The alomancy package uses pytest as the primary testing framework with a comprehensive test suite covering:

- **Unit tests**: Individual functions and classes
- **Integration tests**: Component interactions
- **Mock tests**: External dependencies (MACE, Quantum Espresso)
- **Performance tests**: Slow/compute-intensive operations

## Test Structure

```
tests/
├── conftest.py                     # Global fixtures and configuration
├── core_tests/                     # Core active learning workflow tests
│   ├── test_base_active_learning.py
│   └── test_standard_active_learning.py
├── mlip_train_tests/              # MLIP training and evaluation tests
│   └── test_mace_training.py
├── struc_gen_tests/               # Structure generation tests
│   └── test_structure_generation.py
├── high_acc_tests/                # High-accuracy evaluation tests
│   └── test_quantum_espresso.py
└── utils_tests/                   # Utility function tests
    └── test_utilities.py
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods
- Use mocks for external dependencies
- Fast execution (< 1 second per test)
- High coverage of edge cases

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Use real data flow but mocked external calls
- Medium execution time (1-10 seconds per test)
- Test realistic workflows

### Slow Tests (`@pytest.mark.slow`)
- Performance and stress tests
- Large data processing
- Long-running operations
- Typically skipped in CI

### External Tests (`@pytest.mark.requires_external`)
- Require external software (MACE, QE)
- Real hardware requirements (GPU)
- Only run when dependencies are available

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Exclude slow tests
pytest -m "not requires_external" # Exclude external dependency tests

# Run specific test files
pytest tests/core_tests/test_base_active_learning.py

# Run with coverage
pytest --cov=alomancy --cov-report=html

# Parallel execution
pytest -n auto
```

### Using the Test Runner

```bash
# Convenient test runner script
python run_tests.py --type unit
python run_tests.py --type integration --coverage
python run_tests.py --type all --parallel
```

### Using Makefile

```bash
make test              # Unit tests
make test-integration  # Integration tests
make test-all         # All tests
make coverage         # Tests with coverage
```

## Test Configuration

### pytest.ini / pyproject.toml
- Test discovery patterns
- Coverage settings
- Marker definitions
- Warning filters

### conftest.py
- Global fixtures
- Test environment setup
- Mock configuration
- Shared test utilities

## Fixtures

### Core Fixtures
- `sample_atoms`: Basic ASE Atoms objects
- `sample_training_data`: Training datasets
- `temp_dir`: Temporary directories
- `mock_job_dict`: Job configuration mocks

### External Mocks
- `mock_mace_calculator`: MACE calculator mock
- `mock_espresso_calculator`: QE calculator mock
- `mock_remote_info`: Remote execution mock

### Environment Control
- `skip_if_no_external`: Skip if external deps missing
- `skip_if_no_gpu`: Skip if GPU not available
- `skip_if_no_mace`: Skip if MACE not installed

## Mocking Strategy

### External Dependencies
```python
@patch('mace.calculators.MACECalculator')
def test_with_mace_mock(mock_mace):
    # Test logic without requiring MACE installation
```

### File Operations
```python
@patch('ase.io.write')
@patch('pathlib.Path.mkdir')
def test_file_operations(mock_mkdir, mock_write):
    # Test without actual file I/O
```

### Remote Execution
```python
@patch('alomancy.utils.remote_job_executor.RemoteJobExecutor')
def test_remote_jobs(mock_executor):
    # Test without HPC access
```

## Testing Guidelines

### Test Naming
- `test_<functionality>`: Basic functionality
- `test_<functionality>_edge_case`: Edge cases
- `test_<functionality>_error`: Error conditions
- `test_<functionality>_integration`: Integration scenarios

### Test Structure
```python
def test_functionality():
    # Arrange
    setup_data()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

### Assertions
- Use descriptive assertions
- Test both positive and negative cases
- Verify types and shapes for arrays
- Check error messages for exceptions

### Mock Guidelines
- Mock at the boundary (external interfaces)
- Keep mocks simple and focused
- Verify mock calls when behavior matters
- Use fixtures for complex mock setup

## Coverage Requirements

- **Minimum**: 80% overall coverage
- **Target**: 90% for core modules
- **Exclusions**: Version files, CLI entry points
- **Focus**: Logic branches over line coverage

## Continuous Integration

### GitHub Actions Workflow
- Runs on Python 3.9, 3.10, 3.11
- Tests on Ubuntu and macOS
- Includes linting and type checking
- Uploads coverage to Codecov

### Quality Checks
- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking
- bandit security scanning

## Performance Testing

### Benchmarks
- Time critical functions
- Memory usage monitoring
- Scalability testing
- Regression detection

### Example
```python
@pytest.mark.slow
def test_large_system_performance():
    # Create large system
    atoms_list = create_large_dataset(10000)
    
    # Time the operation
    start_time = time.time()
    result = process_large_dataset(atoms_list)
    elapsed = time.time() - start_time
    
    # Assert performance requirements
    assert elapsed < 60.0  # Should complete in under 1 minute
```

## Debugging Tests

### Running Failed Tests
```bash
# Run only failed tests from last run
pytest --lf

# Stop on first failure
pytest -x

# Increase verbosity
pytest -v

# Show local variables in tracebacks
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Common Issues

1. **Import Errors**: Check PYTHONPATH and package installation
2. **Missing Dependencies**: Install test dependencies with `pip install -e ".[dev]"`
3. **Fixture Issues**: Check fixture scope and dependencies
4. **Mock Problems**: Verify patch targets and mock setup

## Adding New Tests

### Checklist
- [ ] Choose appropriate test category (unit/integration/slow)
- [ ] Add relevant markers
- [ ] Use existing fixtures when possible
- [ ] Mock external dependencies
- [ ] Include edge cases and error conditions
- [ ] Update documentation if needed

### Example New Test
```python
import pytest
from unittest.mock import patch
from alomancy.new_module import new_function

@pytest.mark.unit
def test_new_function_basic(sample_atoms):
    """Test basic functionality of new_function."""
    result = new_function(sample_atoms)
    assert result is not None

@pytest.mark.unit
def test_new_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError, match="Invalid input"):
        new_function(None)

@pytest.mark.integration
@patch('external.dependency')
def test_new_function_integration(mock_ext, sample_training_data):
    """Test integration with other components."""
    mock_ext.return_value = "success"
    result = new_function(sample_training_data)
    assert result == "expected_output"
    mock_ext.assert_called_once()
```

## Best Practices

1. **Test First**: Write tests before or alongside implementation
2. **Single Responsibility**: One concept per test
3. **Clear Names**: Test names should describe what is being tested
4. **Fast Feedback**: Keep unit tests fast
5. **Reliable**: Tests should be deterministic
6. **Maintainable**: Keep tests simple and readable
7. **Comprehensive**: Cover happy path, edge cases, and errors

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [ASE testing guide](https://wiki.fysik.dtu.dk/ase/development/testing.html)
- [Python testing best practices](https://realpython.com/python-testing/)
