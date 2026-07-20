"""Sphinx configuration for ALomancy documentation."""

# import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "ALomancy"
copyright = "2025, Julian Holland"
author = "Julian Holland"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",  # For markdown support
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Autosummary settings
autosummary_generate = True

# Source files
source_suffix = {
    ".rst": None,
    ".md": None,
}

# Master document
master_doc = "index"

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

_logo_path = Path(__file__).parent / "_static" / "alomancy_logo.png"
if _logo_path.exists():
    html_logo = "_static/alomancy_logo.png"

html_theme_options = {
    "logo_only": False,
    "display_version": True,
}

html_context = {
    "display_github": True,
    "github_user": "julianholland",
    "github_repo": "ALomancy",
    "github_version": "master",
    "conf_py_path": "/docs/",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}
