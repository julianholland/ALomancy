import logging

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    generate_kpts,
)

logger = logging.getLogger(__name__)


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
    return {
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
        **qe_input_kwargs,
    }


def resolve_effective_kwargs(qe_kwargs: dict) -> dict:
    """The dft_evaluator registry's uniform defaults-resolution entry point
    (see registry.py) -- used only by the skeleton's pre-run config summary
    (committee_uncertainty_workflow.py's display_workflow_summary) to show
    the fully-resolved effective qe_kwargs, not just what the user wrote.
    get_qe_input_data (above, unchanged) already merges its own defaults
    with whatever's passed to it, including its section-shallow merge
    behavior (overriding e.g. "system" replaces that whole sub-dict rather
    than merging individual keys within it) -- calling it directly here,
    rather than re-deriving the same defaults separately, means this
    display can never drift out of sync with the real merge, quirks
    included. "scf" is passed as calculation_type since it doesn't affect
    the input_data defaults shown (control.calculation itself does vary by
    scf/relax, but that distinction isn't relevant to a settings summary).
    """
    return get_qe_input_data("scf", qe_kwargs)


def create_qe_calc_object(
    atoms: Atoms, high_accuracy_eval_job_dict: dict, out_dir: str
) -> Espresso:
    kpt_arr = generate_kpts(cell=atoms.cell, periodic_3d=True, kspacing=0.15)
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
        input_data=get_qe_input_data(
            "scf", high_accuracy_eval_job_dict.get("qe_input_kwargs", {})
        ),
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
