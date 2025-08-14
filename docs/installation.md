# Installation

## From PyPI (Recommended)

```bash
pip install alomancy
```

## From Source

```bash
git clone https://github.com/julianholland/ALomancy.git
cd ALomancy
pip install -e ".[dev]"
```

## Dependencies

- Python 3.9+
- [ASE](https://wiki.fysik.dtu.dk/ase/) - Atomic Simulation Environment
- [Expyre](https://github.com/libAtoms/ExPyRe) - Remote job execution
- [MACE](https://github.com/ACEsuit/mace) - MLIP training

## Development Installation

For contributors and developers:

```bash
# Clone and install with development dependencies
git clone https://github.com/julianholland/ALomancy.git
cd ALomancy
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```
