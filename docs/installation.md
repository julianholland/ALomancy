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

## HPC Setup

After installing, run the interactive wizard to register your HPC system(s):

```bash
alomancy add-hpc
```

The wizard configures:
- `~/.expyre/config.json` — ExPyRe scheduler entry (Slurm, partitions, scratch directory)
- `~/.alomancy/hpc_config.yaml` — ALomancy HPC profile (venv, DFT paths, node info)

Run it once per HPC system (or profile). See [Examples](examples.md) for a full walkthrough.

### Clearing local ExPyRe state

```bash
alomancy nuke
```

Deletes all local ExPyRe job state (job cache, unsynced stage directories) under
`~/.expyre`, leaving `config.json` untouched. Prompts for confirmation before deleting.
Useful when local ExPyRe state gets out of sync with the remote HPC system and jobs
won't resubmit cleanly.

## Materials Project API

Ensure you have a `MP_API_KEY` accessible in the local environment. You can generate your API key on their [website](https://next-gen.materialsproject.org/api)

then you can ensure it is always accessible by adding the following line to your `.bashrc`

```bash
export MP_API_KEY <your key here>
```
or if you use fish, in your `~/.config/fish/config.fish`

```fish
set -gx MP_API_KEY <your key here>
```

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
