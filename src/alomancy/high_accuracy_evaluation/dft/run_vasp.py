import logging
import os
from functools import partial

from ase import Atoms
from ase.calculators.vasp import Vasp

from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    generate_kpts,
)

logger = logging.getLogger(__name__)


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


def create_vasp_calc_object(
    atoms: Atoms,
    high_accuracy_eval_job_dict: dict,
    out_dir: str,
    is_relaxation: bool = False,
) -> Vasp:
    """Create an ASE Vasp calculator — analogous to create_qe_calc_object."""
    kpt_arr = generate_kpts(cell=atoms.cell, periodic_3d=True, kspacing=0.15)
    hpc = high_accuracy_eval_job_dict["hpc"]
    if hpc.get("pp_path"):
        # ASE's Vasp calculator reads POTCAR locations from this env var, not
        # from a constructor kwarg — must be set before Vasp(...) is built.
        os.environ["VASP_PP_PATH"] = hpc["pp_path"]
    vasp_kwargs = get_vasp_input_kwargs(
        high_accuracy_eval_job_dict.get("vasp_input_kwargs", {})
    )
    if is_relaxation:
        vasp_kwargs.update({"nsw": 200, "ibrion": 2})

    # NCORE ~ sqrt(ranks_per_node) is a sensible default for band parallelism
    ncore = max(1, int(hpc["node_info"]["ranks_per_node"] ** 0.5))
    vasp_kwargs.setdefault("ncore", ncore)

    return Vasp(
        command=create_vasp_command(hpc["node_info"], hpc["vasp_path"]),
        directory=out_dir,
        kpts=list(kpt_arr),
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
