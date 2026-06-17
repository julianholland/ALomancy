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
- [ASE](https://wiki.fysik.dtu.dk/ase/) — Atomic Simulation Environment
- [expyre-wfl](https://github.com/libAtoms/ExPyRe) — Remote HPC job execution (pip package: expyre-wfl)
- [MACE](https://github.com/ACEsuit/mace) — Machine Learning Accelerated Computational Engine (pip package: mace-torch)
- [sage-lib](https://github.com/sage-lib/sage-lib) — Hybrid HDF5+SQLite structure database (GlobalDatabase backend)
- [mp-api](https://github.com/materialsproject/api) — Materials Project API for fetching reference structures
- numpy, pandas, polars, scipy, matplotlib, seaborn, tqdm, pyyaml

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
