import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from alomancy.high_accuracy_evaluation.level_of_theory import (
    RYDBERG_EV,
    LevelOfTheory,
    functional_rung,
    is_hartree_fock,
)
from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    kpoint_grid,
)

logger = logging.getLogger(__name__)

# The longest remote job output directory QE can work in (registry entry
# point max_output_dir_len); None: no limit is checked. Only VASP has the
# short Fortran path buffer (run_vasp.MAX_OUTPUT_DIR_LEN).
MAX_OUTPUT_DIR_LEN: int | None = None

# Registry entry points: QE handles periodic and non-periodic cells.
SUPPORTS_PERIODIC = True
SUPPORTS_NON_PERIODIC = True

# k-point spacing (1/Angstrom, as generate_kpts takes it).
KSPACING = 0.15


def find_optimal_npool(
    ranks_per_system: int,
    total_kpoints: int,
    ranks_per_node: int | None = None,
    min_ranks_per_pool: int = 4,
) -> int:
    candidates = []
    for npool in range(1, min(ranks_per_system, total_kpoints) + 1):
        if ranks_per_system % npool != 0:
            continue

        ranks_per_pool = ranks_per_system // npool
        if ranks_per_pool < min_ranks_per_pool:
            continue

        score = 0

        if total_kpoints % npool == 0:
            score += 3

        if ranks_per_node is not None:
            pools_per_node = ranks_per_node / ranks_per_pool
            if pools_per_node.is_integer():
                score += 2

        score -= abs(ranks_per_pool - 8) / 8

        candidates.append((score, npool))

    if not candidates:
        return 1

    candidates.sort(reverse=True)
    return candidates[0][1]


def create_espresso_profile(
    para_info_dict: dict,
    npool: int,
    pwx_path: str,
    pp_path: str,
    ndiag: int | None = None,
    ntg: int | None = None,
) -> EspressoProfile:
    flags = [f"-nk {npool}"]

    if ndiag is not None and ndiag > 1:
        flags.append(f"-nd {ndiag}")

    if ntg is not None and ntg > 1:
        flags.append(f"-nt {ntg}")

    flag_str = " ".join(flags)

    command = _build_srun_command(para_info_dict, f"{pwx_path} {flag_str}")

    return EspressoProfile(
        command=command,
        pseudo_dir=pp_path,
    )


def get_qe_input_data(calculation_type: str, qe_input_kwargs: dict) -> dict:
    """Default pw.x namelists with user overrides merged in per namelist.

    Each namelist (``control``, ``system``, ...) is merged key-by-key, so
    overriding one key (e.g. ``control.tstress``) keeps every other default
    in that namelist. A shallow merge used to replace the whole namelist,
    silently dropping ``control.tprnfor``/``calculation`` so QE never
    printed forces and every job failed with "forces not present".
    Non-dict values (e.g. a top-level scalar) replace the default outright.
    """
    input_data: dict = {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "prefix": "qe",
            "nstep": 999,
            "tstress": True,
            "tprnfor": True,
            "disk_io": "none",
            "etot_conv_thr": 1.0e-5,
            "forc_conv_thr": 1.0e-5,
        },
        "system": {
            "ibrav": 0,
            "tot_charge": 0.0,
            "ecutwfc": 40.0,
            "ecutrho": 600,
            "occupations": "smearing",
            "degauss": 0.01,
            "smearing": "cold",
            "input_dft": "pbe",
            "nspin": 1,
        },
        "electrons": {
            "electron_maxstep": 999,
            "scf_must_converge": True,
            "conv_thr": 1.0e-12,
            "mixing_mode": "local-TF",
            "mixing_beta": 0.25,
            "startingwfc": "random",
            "diagonalization": "david",
        },
        "ions": {"ion_dynamics": "bfgs", "upscale": 1e8, "bfgs_ndim": 6},
        "cell": {"press_conv_thr": 0.1, "cell_dofree": "all"},
    }
    for section, overrides in qe_input_kwargs.items():
        if isinstance(overrides, dict) and isinstance(input_data.get(section), dict):
            input_data[section] = {**input_data[section], **overrides}
        else:
            input_data[section] = overrides
    return input_data


def resolve_effective_kwargs(qe_kwargs: dict) -> dict:
    """The dft_evaluator registry's uniform defaults-resolution entry point
    (see registry.py) -- used only by the skeleton's pre-run config summary
    (active_learning_workflow.py's display_workflow_summary) to show
    the fully-resolved effective qe_kwargs, not just what the user wrote.
    get_qe_input_data (above) already merges its own defaults with
    whatever's passed to it, per namelist -- calling it directly here,
    rather than re-deriving the same defaults separately, means this
    display can never drift out of sync with the real merge. "scf" is passed as calculation_type since it doesn't affect
    the input_data defaults shown (control.calculation itself does vary by
    scf/relax, but that distinction isn't relevant to a settings summary).
    """
    k_spacing, namelists = split_k_spacing(qe_kwargs)
    return {**get_qe_input_data("scf", namelists), "k_spacing": k_spacing}


def split_k_spacing(qe_kwargs: dict | None) -> tuple[float, dict]:
    """(k-point spacing, the namelists): qe_kwargs.k_spacing (1/Angstrom, 2
    pi included, as VASP's KSPACING) overrides KSPACING and is not a pw.x
    namelist, so it is taken out before the namelists are merged."""
    namelists = dict(qe_kwargs or {})
    k_spacing = namelists.pop("k_spacing", KSPACING)
    if isinstance(k_spacing, bool) or not isinstance(k_spacing, int | float):
        raise ValueError(
            "high_accuracy_evaluation.qe_kwargs.k_spacing must be a positive "
            f"number (1/Angstrom), got {k_spacing!r}."
        )
    if k_spacing <= 0:
        raise ValueError(
            "high_accuracy_evaluation.qe_kwargs.k_spacing must be positive, got "
            f"{k_spacing!r}."
        )
    return float(k_spacing), namelists


def validate_kwargs(config: dict, hpc: dict | None) -> None:  # noqa: ARG001
    """Construction-time check: qe_kwargs.k_spacing valid when set."""
    split_k_spacing(config.get("qe_kwargs"))


def create_qe_calc_object(
    atoms: Atoms, high_accuracy_eval_job_dict: dict, out_dir: str
) -> Espresso:
    """The k-grid comes from the structure's own pbc at qe_kwargs.k_spacing
    (default KSPACING): non-periodic axes get one k-point, so an isolated
    atom or molecule runs at Gamma only."""
    k_spacing, namelists = split_k_spacing(
        high_accuracy_eval_job_dict.get("qe_input_kwargs")
    )
    kpt_arr = kpoint_grid(atoms, k_spacing)
    npool = find_optimal_npool(
        total_kpoints=int(np.prod(kpt_arr)),
        ranks_per_system=high_accuracy_eval_job_dict["hpc"]["node_info"][
            "ranks_per_system"
        ],
        min_ranks_per_pool=8,
    )
    return Espresso(
        profile=create_espresso_profile(
            para_info_dict=high_accuracy_eval_job_dict["hpc"]["node_info"],
            npool=npool,
            pwx_path=high_accuracy_eval_job_dict["hpc"]["pwx_path"],
            pp_path=high_accuracy_eval_job_dict["hpc"]["pp_path"],
        ),
        input_data=get_qe_input_data("scf", namelists),
        kpts=list(kpt_arr),
        pseudopotentials=high_accuracy_eval_job_dict["hpc"]["pseudo_dict"],
        directory=out_dir,
    )


def run_sp_qe(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_sp(
        input_structure, out_dir, high_accuracy_eval_job_dict, create_qe_calc_object
    )


def run_go_qe(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_go(
        input_structure,
        out_dir,
        high_accuracy_eval_job_dict,
        create_qe_calc_object,
        opt_prefix="qe_opt",
    )


def max_num_of_atoms(config: dict) -> int | None:  # noqa: ARG001
    """Registry entry point: QE has no size cap."""
    return None


# system.vdw_corr spellings -> dispersion (dftd3_version picks the D3 damping).
_QE_VDW_CORR = {
    "grimme-d2": "D2",
    "dft-d": "D2",
    "d2": "D2",
    "ts": "TS",
    "ts-vdw": "TS",
    "tkatchenko-scheffler": "TS",
    "mbd": "MBD",
    "mbd_vdw": "MBD",
    "many-body dispersion": "MBD",
    "xdm": "XDM",
}
_QE_D3_VERSION = {2: "D2", 3: "D3(0)", 4: "D3(BJ)", 5: "D3M(0)", 6: "D3M(BJ)"}


def _qe_core_treatment(pseudo_dict: dict) -> str | None:
    """From the pseudopotential file names, when they say: SSSP/pslibrary
    names carry paw / rrkjus / uspp / oncv / nc."""
    kinds = set()
    for name in pseudo_dict.values():
        lower = str(name).lower()
        if "paw" in lower:
            kinds.add("paw")
        elif any(tag in lower for tag in ("rrkjus", "uspp", "gbrv", ".us.", "_us")):
            kinds.add("ultrasoft")
        elif any(tag in lower for tag in ("oncv", "sg15", "pseudodojo", ".nc.", "_nc")):
            kinds.add("norm_conserving")
        else:
            return None
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


def level_of_theory(config: dict, hpc: dict | None = None) -> LevelOfTheory:
    """Registry entry point (high_accuracy_evaluation/level_of_theory.py),
    from the effective pw.x namelists. Without system.input_dft QE takes
    the functional from the pseudopotentials: method left unspecified."""
    k_spacing, namelists = split_k_spacing(config.get("qe_kwargs"))
    input_data = get_qe_input_data("scf", namelists)
    system = {k.lower(): v for k, v in input_data.get("system", {}).items()}
    electrons = input_data.get("electrons", {})
    input_dft = system.get("input_dft")
    method, rung = functional_rung(input_dft)
    vdw_corr = str(system.get("vdw_corr", "none")).lower()
    if vdw_corr in ("grimme-d3", "dft-d3", "d3"):
        dispersion = _QE_D3_VERSION.get(int(system.get("dftd3_version", 3)), "D3")
    elif vdw_corr == "none":
        dispersion = "none"
        if input_dft and "vdw" in str(input_dft).lower():
            dispersion = f"nonlocal ({input_dft})"
    else:
        dispersion = _QE_VDW_CORR.get(vdw_corr, vdw_corr)
    hubbard = {
        k: v for k, v in system.items() if k.startswith(("hubbard_", "lda_plus_u"))
    }
    pseudo_dict = dict((hpc or {}).get("pseudo_dict") or {})
    if system.get("noncolin"):
        spin = "noncollinear"
    else:
        spin = "collinear" if int(system.get("nspin", 1)) == 2 else "restricted"
    smearing = None
    if system.get("occupations") == "smearing":
        smearing = {
            "scheme": system.get("smearing", "gaussian"),
            "width_eV": float(system.get("degauss", 0.0)) * RYDBERG_EV,
        }
    ecutwfc = system.get("ecutwfc")
    ecutrho = system.get("ecutrho")
    return LevelOfTheory(
        code="qe",
        method_class=(
            None
            if input_dft is None
            else "mean_field_hf"
            if is_hartree_fock(input_dft)
            else "dft"
        ),
        method=method,
        rung=rung,
        dispersion=dispersion,
        hubbard_u=hubbard,
        basis_type="plane_wave",
        cutoff_eV=None if ecutwfc is None else float(ecutwfc) * RYDBERG_EV,
        density_cutoff_eV=(None if ecutrho is None else float(ecutrho) * RYDBERG_EV),
        core_treatment=_qe_core_treatment(pseudo_dict) if pseudo_dict else None,
        pseudopotentials=pseudo_dict or None,
        relativity="spin_orbit" if system.get("lspinorb") else "in_pseudopotential",
        spin_treatment=spin,
        solvation="none",
        k_spacing=k_spacing,
        smearing=smearing,
        scf_threshold=(
            None
            if electrons.get("conv_thr") is None
            else float(electrons["conv_thr"]) * RYDBERG_EV
        ),
    )


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import dft_section

    return dft_section(stats, **kwargs)
