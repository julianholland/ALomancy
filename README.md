<div align="center">
  <h1><code>ALomancy</code>🔮</h1>
  <p><i>An active learning package using wfl for MACE</i></p>
</div>

***
## Key Features 🎰

- Modular AL routines
- Easy means to switch between multiple HPCs during the workflow
- Consistant evaluation


## Package Layout

```
ALomancy/
├── pyproject.toml                 # Modern Python packaging
├── README.md                      # Package documentation
├── LICENSE                        # License file
├── .gitignore                     # Git ignore file
├── alomancy/                      # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── version.py                # Version information
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── defaults.py           # Default configurations
│   │   └── systems.py            # HPC system configurations
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── active_learning.py    # Main AL workflow (from al_wfl.py)
│   │   └── remote_info.py        # Remote execution utilities
│   ├── md/                       # Molecular dynamics
│   │   ├── __init__.py
│   │   ├── workflows.py          # MD workflows (from md_wfl.py)
│   │   └── structure_selection.py # Structure selection logic
│   ├── dft/                      # DFT calculations
│   │   ├── __init__.py
│   │   └── workflows.py          # DFT workflows (from dft_wfl.py)
│   ├── mace/                     # MACE-specific functionality
│   │   ├── __init__.py
│   │   └── training.py           # MACE training workflows
│   ├── analysis/                 # Analysis and plotting
│   │   ├── __init__.py
│   │   ├── plotting.py           # Plotting utilities
│   │   └── metrics.py            # Analysis metrics
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── io.py                 # File I/O utilities
│   │   └── structures.py         # Structure manipulation
│   └── cli/                      # Command-line interface
│       ├── __init__.py
│       └── main.py               # CLI entry points
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_core/
│   ├── test_md/
│   ├── test_dft/
│   └── test_analysis/
├── docs/                         # Documentation
│   ├── conf.py
│   ├── index.rst
│   └── tutorials/
├── examples/                     # Example scripts
│   ├── basic_workflow.py
│   └── advanced_workflow.py
└── configs/                      # Configuration files
    ├── systems/
    │   ├── raven.json
    │   ├── nersc.json
    │   └── fhi-raccoon.json
    └── job_defaults.json
```
