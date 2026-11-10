import logging
import os
from functools import partial
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp

from alomancy.high_accuracy_evaluation.level_of_theory import (
    LevelOfTheory,
    functional_rung,
)
from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    kpoint_grid,
    pad_non_periodic_axes,
)

logger = logging.getLogger(__name__)

# ASE's Vasp calculator rejects any pbc other than [True, True, True] — VASP
# itself has no non-periodic mode. Molecules generated for initialization
# (isolated atoms, dimers, trimers) are built non-periodic, so they need
# padding before they reach VASP or their periodic images will interact.
_MIN_VACUUM_PER_SIDE = 10.0  # Angstrom

# The longest remote job output directory VASP can work in (registry entry
# point max_output_dir_len, read by submitters.check_remote_path_length).
# VASP keeps file paths in a ~256-character Fortran buffer: a run directory
# of 255 characters left no room for the filename, so VASP tried to open the
# directory itself ("forrtl: severe (30): open failure ... Is a directory").
# 240 leaves room for "/vasprun.xml" and similar.
MAX_OUTPUT_DIR_LEN: int | None = 240

# Registry entry points: VASP is periodic only, but takes non-periodic
# structures by padding them with vacuum (_ensure_fully_periodic).
SUPPORTS_PERIODIC = True
SUPPORTS_NON_PERIODIC = True

# k-point spacing (1/Angstrom, as generate_kpts takes it).
KSPACING = 0.15


def _ensure_fully_periodic(atoms: Atoms, vacuum: float = _MIN_VACUUM_PER_SIDE) -> None:
    # Pads only the non-periodic axes: centring along all three used to
    # resize a slab's periodic in-plane cell too.
    pad_non_periodic_axes(atoms, vacuum=vacuum)


def create_vasp_command(para_info_dict: dict, vasp_path: str) -> str:
    return _build_srun_command(para_info_dict, vasp_path)


def get_vasp_input_kwargs(vasp_input_kwargs: dict) -> dict:
    """Default INCAR parameters merged with user overrides — analogous to get_qe_input_data."""
    defaults: dict = {
        "pp": "PBE",
        "encut": 500,
        "ediff": 1e-6,
        "nsw": 0,
        "ibrion": -1,
        "isif": 2,
        "sigma": 0.1,
        "ismear": 1,
        "prec": "Accurate",
        "lwave": False,
        "lcharg": False,
    }
    defaults.update(vasp_input_kwargs)
    return defaults


def resolve_effective_kwargs(vasp_kwargs: dict) -> dict:
    """The dft_evaluator registry's uniform defaults-resolution entry point
    (see registry.py) -- used only by the skeleton's pre-run config summary
    (active_learning_workflow.py's display_workflow_summary) to show
    the fully-resolved effective vasp_kwargs, not just what the user wrote.
    Thin alias for get_vasp_input_kwargs (above, unchanged) so this display
    can never drift out of sync with the real merge.
    """
    return get_vasp_input_kwargs(vasp_kwargs)


# vasp_kwargs keys about k-points: ALomancy builds the grid per structure.
_KPOINT_KEYS = ("kspacing", "kgamma")
# An explicit grid fits one cell only: raise (validate_kwargs).
_EXPLICIT_GRID_KEYS = ("kpts", "kpoints")


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip(".").lower() in ("true", "t", "1", "yes")
    return bool(value)


def _vasp_kpoint_settings(vasp_kwargs: dict) -> tuple[float, bool]:
    """(k-point spacing, Gamma-centred) from vasp_kwargs: KSPACING and
    KGAMMA as VASP spells them, else KSPACING (module default) and the
    ASE default (Monkhorst-Pack)."""
    lower = {str(k).lower(): v for k, v in vasp_kwargs.items()}
    spacing = float(lower["kspacing"]) if "kspacing" in lower else KSPACING
    gamma = (
        _truthy(lower["kgamma"])
        if "kgamma" in lower
        else bool(lower.get("gamma", False))
    )
    return spacing, gamma


def validate_kwargs(config: dict, hpc: dict | None) -> None:  # noqa: ARG001
    """Construction-time check: no explicit k-point grid (it fits one cell
    only; set kspacing instead), and kspacing positive when set."""
    vasp_kwargs = config.get("vasp_kwargs") or {}
    if not isinstance(vasp_kwargs, dict):
        raise ValueError(
            "high_accuracy_evaluation.vasp_kwargs must be a mapping, got "
            f"{vasp_kwargs!r}."
        )
    explicit = sorted(k for k in vasp_kwargs if str(k).lower() in _EXPLICIT_GRID_KEYS)
    if explicit:
        raise ValueError(
            f"high_accuracy_evaluation.vasp_kwargs: {', '.join(explicit)} can't "
            "be set: ALomancy builds each structure's k-point grid from a "
            "spacing (cells differ from structure to structure, and "
            "non-periodic axes get one k-point). Set kspacing (1/Angstrom, as "
            "VASP's KSPACING) instead, and kgamma: true for Gamma-centred grids."
        )
    spacing, _ = _vasp_kpoint_settings(vasp_kwargs)
    if not spacing > 0:
        raise ValueError(
            "high_accuracy_evaluation.vasp_kwargs.kspacing must be positive, "
            f"got {spacing!r}."
        )


def create_vasp_calc_object(
    atoms: Atoms,
    high_accuracy_eval_job_dict: dict,
    out_dir: str,
    is_relaxation: bool = False,
) -> Vasp:
    """Create an ASE Vasp calculator — analogous to create_qe_calc_object.
    The k-grid comes from the structure's own pbc, before padding (a slab
    gets n x m x 1, a padded molecule Gamma only), at vasp_kwargs.kspacing
    when set (removed from the INCAR: ALomancy writes KPOINTS, which VASP
    reads instead), else KSPACING; kgamma: true makes it Gamma-centred."""
    user_kwargs = dict(high_accuracy_eval_job_dict.get("vasp_input_kwargs") or {})
    k_spacing, gamma = _vasp_kpoint_settings(user_kwargs)
    kpt_arr = kpoint_grid(atoms, k_spacing)
    _ensure_fully_periodic(atoms)
    hpc = high_accuracy_eval_job_dict["hpc"]
    if hpc.get("pp_path"):
        # ASE's Vasp calculator reads POTCAR locations from this env var, not
        # from a constructor kwarg — must be set before Vasp(...) is built.
        os.environ["VASP_PP_PATH"] = hpc["pp_path"]
    vasp_kwargs = get_vasp_input_kwargs(
        {k: v for k, v in user_kwargs.items() if k.lower() not in _KPOINT_KEYS}
    )
    if is_relaxation:
        vasp_kwargs.update({"nsw": 0, "ibrion": -1})

    # NCORE ~ sqrt(ranks_per_node) is a sensible default for band parallelism
    ncore = max(1, int(hpc["node_info"]["ranks_per_node"] ** 0.5))
    vasp_kwargs.setdefault("ncore", ncore)

    return Vasp(
        command=create_vasp_command(hpc["node_info"], hpc["vasp_path"]),
        directory=out_dir,
        kpts=list(kpt_arr),
        gamma=gamma,
        setups=hpc.get("pseudo_dict", {}),
        **vasp_kwargs,
    )


def run_sp_vasp(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_sp(
        input_structure, out_dir, high_accuracy_eval_job_dict, create_vasp_calc_object
    )


def run_go_vasp(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    create_vasp_calc_go = partial(create_vasp_calc_object, is_relaxation=True)
    return _run_go(
        input_structure,
        out_dir,
        high_accuracy_eval_job_dict,
        create_vasp_calc_go,
        opt_prefix="vasp_opt",
    )


def max_num_of_atoms(config: dict) -> int | None:  # noqa: ARG001
    """Registry entry point: VASP has no size cap."""
    return None


# INCAR GGA tags -> functional.
_VASP_GGA = {
    "PE": "PBE",
    "PS": "PBEsol",
    "RP": "RPBE",
    "RE": "revPBE",
    "91": "PW91",
    "AM": "AM05",
    "CA": "LDA",
    "PZ": "LDA",
    "BO": "optB88",
    "MK": "optB86b",
    "OR": "optPBE",
    "ML": "vdW-DF2",
}
# IVDW -> dispersion.
_VASP_IVDW = {
    0: "none",
    1: "D2",
    10: "D2",
    11: "D3(0)",
    12: "D3(BJ)",
    13: "D4",
    2: "TS",
    20: "TS",
    21: "TS (iterative Hirshfeld)",
    202: "MBD",
    263: "MBD (FI)",
    4: "dDsC",
}
_VASP_ISMEAR = {
    -5: "tetrahedron (Bloechl)",
    -4: "tetrahedron",
    -1: "fermi-dirac",
    0: "gaussian",
}


def _vasp_method(kwargs: dict) -> str | None:
    if kwargs.get("xc"):
        return str(kwargs["xc"])
    if kwargs.get("lhfcalc"):
        screen = kwargs.get("hfscreen")
        if screen is not None and abs(float(screen) - 0.2) < 1e-6:
            return "HSE06"
        if screen is not None and abs(float(screen) - 0.3) < 1e-6:
            return "HSE03"
        if not screen and float(kwargs.get("aexx", 0.25)) == 0.25:
            return "PBE0"
        return None  # a custom hybrid: unspecified
    if kwargs.get("metagga"):
        return str(kwargs["metagga"])
    if kwargs.get("gga"):
        return _VASP_GGA.get(str(kwargs["gga"]).upper(), str(kwargs["gga"]))
    # Neither set: the functional of the POTCARs.
    return {"PBE": "PBE", "LDA": "LDA", "PW91": "PW91"}.get(
        str(kwargs.get("pp", "")).upper()
    )


def level_of_theory(config: dict, hpc: dict | None = None) -> LevelOfTheory:
    """Registry entry point (high_accuracy_evaluation/level_of_theory.py),
    from the effective INCAR settings."""
    kwargs = {
        k.lower(): v
        for k, v in get_vasp_input_kwargs(config.get("vasp_kwargs") or {}).items()
        if k.lower() not in _KPOINT_KEYS
    }
    method, rung = functional_rung(_vasp_method(kwargs))
    ivdw = int(kwargs.get("ivdw", 0))
    dispersion = _VASP_IVDW.get(ivdw, f"IVDW={ivdw}")
    if kwargs.get("luse_vdw"):
        dispersion = "nonlocal (vdW-DF)"
    hubbard = (
        {k: kwargs[k] for k in ("ldautype", "ldaul", "ldauu", "ldauj") if k in kwargs}
        if kwargs.get("ldau")
        else {}
    )
    if kwargs.get("lnoncollinear") or kwargs.get("lsorbit"):
        spin = "noncollinear"
    else:
        spin = "collinear" if int(kwargs.get("ispin", 1)) == 2 else "restricted"
    ismear = int(kwargs.get("ismear", 1))
    pseudopotentials = {"set": kwargs.get("pp")}
    pseudopotentials.update((hpc or {}).get("pseudo_dict") or {})
    numerical = {
        k: kwargs[k] for k in ("prec", "lreal", "lasph", "addgrid") if k in kwargs
    }
    return LevelOfTheory(
        code="vasp",
        method_class="dft" if method else None,
        method=method,
        rung=rung,
        dispersion=dispersion,
        hubbard_u=hubbard,
        basis_type="plane_wave",
        cutoff_eV=None if kwargs.get("encut") is None else float(kwargs["encut"]),
        core_treatment="paw",
        pseudopotentials=pseudopotentials,
        relativity="spin_orbit" if kwargs.get("lsorbit") else "in_pseudopotential",
        spin_treatment=spin,
        solvation="VASPsol" if kwargs.get("lsol") else "none",
        k_spacing=_vasp_kpoint_settings(config.get("vasp_kwargs") or {})[0],
        smearing={
            "scheme": _VASP_ISMEAR.get(ismear, f"methfessel-paxton {ismear}"),
            "width_eV": float(kwargs.get("sigma", 0.2)),
        },
        scf_threshold=None if kwargs.get("ediff") is None else float(kwargs["ediff"]),
        numerical_settings=numerical,
    )


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import dft_section

    return dft_section(stats, **kwargs)
