"""FHI-aims backend (registry ``dft_evaluator`` / ``aims``).

FHI-aims is an all-electron code with numeric atom-centred basis sets:
no pseudopotentials. Basis and integration grids come from the species
defaults (``species_defaults/defaults_2020/<setting>/NN_El_default``),
which ASE copies into ``control.in`` on the compute node. The HPC
profile's ``aims_species_dir`` names the defaults root and
``aims_kwargs.species_defaults`` the setting (default ``intermediate``).

Its energies are on another absolute scale than any pseudopotential code
(the database check before initialisation raises when the database holds
labels from another code; see core/active_learning_workflow.py).

Boundary conditions: FHI-aims runs fully periodic or fully non-periodic.
Mixed pbc is padded to fully periodic along the open axes
(utils/dft_utils.pad_non_periodic_axes); ``IsolatedAtom`` always runs
without a cell (the exact free-atom energy, and no k-points). Periodic
structures get a k-grid per structure from ``k_spacing``, clusters none.

Forces (and, periodic only, the analytical stress) are always requested
in ``control.in``: ASE adds them only for the property asked for, and
the workers ask for the energy first, so otherwise each structure would
run its SCF twice.

Launched like QE/VASP, through srun (_build_srun_command). FHI-aims is
MPI-only and needs a large stack: the HPC profile's ``pre_cmds`` should
run ``ulimit -s unlimited`` (validate_kwargs warns otherwise).

Runs remotely (module-level functions only): changes need
``alomancy upgrade-hpc``.
"""

import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.aims import Aims, AimsProfile

from alomancy.high_accuracy_evaluation.level_of_theory import (
    LevelOfTheory,
    functional_rung,
    is_hartree_fock,
)
from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    kpoint_grid,
    pad_non_periodic_axes,
)

logger = logging.getLogger(__name__)

# Registry entry points (see registry.py).
MAX_OUTPUT_DIR_LEN: int | None = None
SUPPORTS_PERIODIC = True
SUPPORTS_NON_PERIODIC = True

# k-point spacing (1/Angstrom, as generate_kpts takes it), shared with
# QE/VASP; aims_kwargs.k_spacing overrides it.
KSPACING = 0.15
# Vacuum added on each open axis of a mixed-pbc structure (Angstrom).
_VACUUM = 10.0

# The species defaults settings FHI-aims ships (folder names).
SPECIES_SETTINGS = ("light", "intermediate", "tight", "really_tight")

_WHERE = "high_accuracy_evaluation.aims_kwargs"

# Any other key is a control.in keyword passed through to ASE.
AIMS_KWARGS_DEFAULTS: dict[str, Any] = {
    # ALomancy's own keys (not written to control.in as such).
    "species_defaults": "intermediate",
    "k_spacing": KSPACING,
    # control.in keywords.
    "xc": "pbe",
    "relativistic": "atomic_zora scalar",
    "spin": "none",
    "charge": 0,
    "occupation_type": "gaussian 0.01",
    "sc_accuracy_rho": 1e-6,
    "sc_accuracy_eev": 1e-3,
    "sc_accuracy_etot": 1e-6,
    "sc_iter_limit": 500,
}

# Keys set per structure from k_spacing / the HPC profile: a fixed value
# can't be right for every cell (NPT frames, amorphous cells) and must not
# be written for clusters.
_REJECTED_KEYS = {
    "k_grid": (
        "set per structure from k_spacing (1/Angstrom) or k_grid_density "
        "(points per 1/Angstrom)"
    ),
    "kpts": (
        "set per structure from k_spacing (1/Angstrom) or k_grid_density "
        "(points per 1/Angstrom)"
    ),
    "species_dir": (
        "comes from the HPC profile's aims_species_dir and aims_kwargs.species_defaults"
    ),
    "plus_u": (
        "ASE's FHI-aims writer can't write it (it reads parameters.plus_u as "
        "an attribute of a dict); for DFT+U, write species files with the "
        "plus_u lines into your own species_defaults setting folder"
    ),
}


def resolve_aims_kwargs(aims_kwargs: dict | None) -> dict:
    """The defaults with the user's aims_kwargs merged over them. A user
    ``smearing`` (ASE's spelling) replaces the default occupation_type: ASE
    refuses both."""
    user = dict(aims_kwargs or {})
    merged = {**AIMS_KWARGS_DEFAULTS, **user}
    if "smearing" in user and "occupation_type" not in user:
        merged.pop("occupation_type")
    return merged


def resolve_effective_kwargs(aims_kwargs: dict) -> dict:
    """For the pre-run summary: the defaults merged in. The per-structure
    keys (k_grid, compute_forces, compute_analytical_stress, charge) are
    added by get_aims_parameters."""
    return resolve_aims_kwargs(aims_kwargs)


def effective_k_spacing(aims_kwargs: dict | None) -> float:
    """The k-point spacing (1/Angstrom, 2 pi included): from
    aims_kwargs.k_grid_density (points per 1/Angstrom, FHI-aims' own
    keyword) as 1 / density when set, else k_spacing (default KSPACING).
    ALomancy builds the grid itself, so a slab's out-of-plane axis gets
    one k-point."""
    kwargs = dict(aims_kwargs or {})
    if kwargs.get("k_grid_density") is not None:
        return 1.0 / float(kwargs["k_grid_density"])
    return float(kwargs.get("k_spacing", KSPACING))


def get_aims_parameters(
    aims_kwargs: dict | None, atoms: Atoms, k_grid: list[int] | None = None
) -> dict:
    """The ASE Aims parameters for one structure (after prepare_structure):
    control.in keywords, forces always, and for a periodic cell the k-grid
    and analytical stress (unless the user turned it off). *k_grid*: the
    grid computed on the structure's pbc before padding (prepare_structure);
    None computes it from *atoms* as they are."""
    params = resolve_aims_kwargs(aims_kwargs)
    params.pop("species_defaults")
    params.pop("k_spacing")
    params.pop("k_grid_density", None)
    params["charge"] = atoms.info.get("charge", params["charge"])
    params["compute_forces"] = True
    if atoms.pbc.all():
        params["k_grid"] = list(
            k_grid
            if k_grid is not None
            else kpoint_grid(atoms, effective_k_spacing(aims_kwargs))
        )
        params.setdefault("compute_analytical_stress", True)
    else:
        params.pop("compute_analytical_stress", None)
    return params


def prepare_structure(atoms: Atoms, k_spacing: float = KSPACING) -> list[int]:
    """In place, before the calculation (the stored result is what was
    computed): an IsolatedAtom loses its cell's periodicity (the exact free
    atom, no k-points); mixed pbc is padded to fully periodic along the
    open axes (FHI-aims has no slab or wire mode). Returns the k-grid,
    computed on the pbc before padding: a slab gets n x m x 1."""
    if atoms.info.get("config_type") == "IsolatedAtom":
        atoms.pbc = False
    k_grid = kpoint_grid(atoms, k_spacing)
    if atoms.pbc.any() and not atoms.pbc.all():
        pad_non_periodic_axes(atoms, vacuum=_VACUUM)
    return k_grid


def species_directory(hpc: dict, species_defaults: str) -> str:
    return str(Path(hpc["aims_species_dir"], species_defaults))


def create_aims_calc_object(
    atoms: Atoms, high_accuracy_eval_job_dict: dict, out_dir: str
) -> Aims:
    """An ASE Aims calculator for *atoms* (the stock interface; launched
    through srun like QE/VASP)."""
    hpc = high_accuracy_eval_job_dict["hpc"]
    for key in ("aims_path", "aims_species_dir"):
        if not hpc.get(key):
            raise ValueError(
                f"The HPC profile {hpc.get('hpc_name')!r} has no {key}, "
                "needed for FHI-aims."
            )
    aims_kwargs = high_accuracy_eval_job_dict.get("aims_kwargs") or {}
    k_grid = prepare_structure(atoms, effective_k_spacing(aims_kwargs))
    params = get_aims_parameters(aims_kwargs, atoms, k_grid)
    setting = resolve_aims_kwargs(aims_kwargs)["species_defaults"]
    return Aims(
        # OMP_NUM_THREADS=1 for FHI-aims only (MPI-only code; srun passes
        # the environment on), not in the profile's pre_cmds, which every
        # job on the HPC runs.
        profile=AimsProfile(
            command="env OMP_NUM_THREADS=1 "
            + _build_srun_command(hpc["node_info"], hpc["aims_path"])
        ),
        directory=out_dir,
        species_dir=species_directory(hpc, setting),
        **params,
    )


def run_sp_aims(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_sp(
        input_structure, out_dir, high_accuracy_eval_job_dict, create_aims_calc_object
    )


def run_go_aims(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_go(
        input_structure,
        out_dir,
        high_accuracy_eval_job_dict,
        create_aims_calc_object,
        opt_prefix="aims_opt",
    )


def max_num_of_atoms(config: dict) -> int | None:  # noqa: ARG001
    """Registry entry point: no size cap for FHI-aims."""
    return None


def _xc_name(xc: object) -> str:
    """The functional's name: the first token of aims' xc line
    (``hse06 0.11`` -> ``hse06``)."""
    return str(xc).split()[0] if str(xc).split() else ""


# FHI-aims' relativistic keyword -> the LevelOfTheory relativity vocabulary
# (scalar_ZORA | atomic_ZORA | none | ...). Both ZORA flavours are scalar:
# the "scalar" token that follows them in control.in adds nothing.
_RELATIVITY = {
    "atomic_zora": "atomic_ZORA",
    "zora": "scalar_ZORA",
    "none": "none",
}


def _relativity(value: object) -> str | None:
    tokens = str(value).lower().split()
    if not tokens:
        return None
    return _RELATIVITY.get(tokens[0], tokens[0])


def _dispersion(kwargs: dict) -> str:
    if kwargs.get("many_body_dispersion_nl") is not None:
        return "MBD-NL"
    if kwargs.get("many_body_dispersion") is not None:
        return "MBD"
    if kwargs.get("vdw_correction_hirshfeld"):
        return "TS"
    return "none"


def _smearing(occupation_type: object) -> dict | None:
    tokens = str(occupation_type).split() if occupation_type else []
    if len(tokens) < 2:
        return None
    try:
        width = float(tokens[1])
    except ValueError:
        return None
    return {"scheme": tokens[0], "width_eV": width}


def level_of_theory(config: dict, hpc: dict | None = None) -> LevelOfTheory:
    """Registry entry point (high_accuracy_evaluation/level_of_theory.py),
    from the effective aims_kwargs. The basis is the species defaults
    folder (e.g. defaults_2020/intermediate, so 2010 and 2020 species
    differ) plus the tier when one is set."""
    kwargs = resolve_aims_kwargs(config.get("aims_kwargs"))
    xc = _xc_name(kwargs.get("xc"))
    method, rung = functional_rung(xc)
    root = Path(str((hpc or {}).get("aims_species_dir") or "")).name
    basis = "/".join(p for p in (root, str(kwargs["species_defaults"])) if p)
    if kwargs.get("tier") is not None:
        basis += f" tier {kwargs['tier']}"
    spin = str(kwargs.get("spin", "none")).lower()
    smearing = _smearing(kwargs.get("occupation_type"))
    if smearing is None and isinstance(kwargs.get("smearing"), list | tuple):
        scheme, width = kwargs["smearing"][:2]
        smearing = {"scheme": str(scheme), "width_eV": float(width)}
    rho = kwargs.get("sc_accuracy_rho")
    solvent = str(kwargs.get("solvent", "")).lower()
    return LevelOfTheory(
        code="aims",
        method_class=("mean_field_hf" if is_hartree_fock(xc) else "dft")
        if xc
        else None,
        method=method,
        rung=rung,
        dispersion=_dispersion(kwargs),
        hubbard_u={},
        basis_type="numeric_atom_centred",
        basis=basis,
        core_treatment="all_electron",
        pseudopotentials=None,
        relativity=_relativity(kwargs.get("relativistic")),
        spin_treatment={"none": "restricted", "collinear": "collinear"}.get(spin, spin),
        solvation="MPE" if solvent.startswith("mpe") else "none",
        k_spacing=effective_k_spacing(config.get("aims_kwargs")),
        smearing=smearing,
        scf_threshold=None if rho is None else float(rho),
    )


def validate_kwargs(config: dict, hpc: dict | None) -> None:
    """Construction-time check: aims_kwargs readable, the species setting
    and k_spacing valid, no key that can't be honoured, the HPC profile
    complete. Warns on settings likely to fail on the node."""
    aims_kwargs = config.get("aims_kwargs") or {}
    if not isinstance(aims_kwargs, dict):
        raise ValueError(f"{_WHERE} must be a mapping, got {aims_kwargs!r}.")
    rejected = sorted(set(aims_kwargs) & set(_REJECTED_KEYS))
    if rejected:
        raise ValueError(
            f"{_WHERE}: "
            + "; ".join(
                f"{key} can't be set ({_REJECTED_KEYS[key]})" for key in rejected
            )
            + "."
        )
    kwargs = resolve_aims_kwargs(aims_kwargs)
    setting = kwargs["species_defaults"]
    if setting not in SPECIES_SETTINGS:
        raise ValueError(
            f"{_WHERE}.species_defaults must be one of {list(SPECIES_SETTINGS)}, "
            f"got {setting!r}."
        )
    if "k_spacing" in aims_kwargs and "k_grid_density" in aims_kwargs:
        raise ValueError(
            f"{_WHERE}: set k_spacing (1/Angstrom) or k_grid_density (points per "
            "1/Angstrom), not both."
        )
    for key in ("k_spacing", "k_grid_density"):
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(
                f"{_WHERE}.{key} must be a positive number, got {value!r}."
            )
    tier = kwargs.get("tier")
    if tier is not None and (isinstance(tier, bool) or not isinstance(tier, int)):
        raise ValueError(
            f"{_WHERE}.tier must be an integer (basis tier for every species) "
            f"or null, got {tier!r}."
        )
    charge = kwargs["charge"]
    if isinstance(charge, bool) or not isinstance(charge, int | float):
        raise ValueError(f"{_WHERE}.charge must be a number, got {charge!r}.")
    if kwargs.get("compute_forces") is False:
        raise ValueError(
            f"{_WHERE}.compute_forces can't be false: forces are training labels."
        )
    xc = _xc_name(kwargs.get("xc"))
    if not xc:
        raise ValueError(
            f"{_WHERE}.xc must name a functional, got {kwargs.get('xc')!r}."
        )
    if xc.lower() == "hse06" and "hse_unit" not in kwargs:
        raise ValueError(
            f"{_WHERE}: xc hse06 needs hse_unit (e.g. hse_unit: bohr-1 with "
            "xc: hse06 0.11); FHI-aims stops without it."
        )
    _, rung = functional_rung(xc)
    if rung == "meta-GGA" and kwargs.get("compute_analytical_stress") is not False:
        logger.warning(
            "%s.xc %s is a meta-GGA: check your FHI-aims version computes "
            "its analytical stress, or periodic runs stop; set "
            "compute_analytical_stress: false to train without stress.",
            _WHERE,
            xc,
            extra={"event": "aims_meta_gga_stress", "data": {"xc": xc}},
        )
    if rung is None and not is_hartree_fock(xc):
        logger.debug("Functional %r not in the rung table.", xc)

    if not isinstance(hpc, dict):
        return
    missing = [k for k in ("aims_path", "aims_species_dir") if not hpc.get(k)]
    if missing:
        raise ValueError(
            f"The HPC profile {hpc.get('hpc_name')!r} used by "
            f"high_accuracy_evaluation has no {' or '.join(missing)}: add "
            "aims_path (the aims executable) and aims_species_dir (the "
            "species_defaults/defaults_2020 folder on the HPC) to it in "
            "~/.alomancy/hpc_config.yaml, or run alomancy add-hpc."
        )
    pre_cmds = hpc.get("pre_cmds") or []
    if not any("ulimit -s" in str(cmd) for cmd in pre_cmds):
        logger.warning(
            "The HPC profile %r has no 'ulimit -s unlimited' in its pre_cmds: "
            "FHI-aims needs a large stack and may crash with a segmentation "
            "fault on larger structures.",
            hpc.get("hpc_name"),
            extra={"event": "aims_stack_limit", "data": {"hpc": hpc.get("hpc_name")}},
        )
    node_info = hpc.get("node_info") or {}
    if node_info.get("threads_per_rank", 1) != 1:
        logger.warning(
            "The HPC profile %r sets node_info.threads_per_rank %s; FHI-aims "
            "parallelises with MPI ranks only, so the extra threads stay idle.",
            hpc.get("hpc_name"),
            node_info.get("threads_per_rank"),
        )


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import dft_section

    return dft_section(stats, **kwargs)
