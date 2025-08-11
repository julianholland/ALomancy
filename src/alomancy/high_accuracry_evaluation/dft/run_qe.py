from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import write


def get_qe_input_data(calculation_type: str) -> dict:
    return {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "prefix": "ac",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "disk_io": "none",
            "outdir": "./ac_data/",
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


def create_espresso_profile(
    para_info_dict: dict, npool: int, pwx_path: str, pp_path: str
) -> EspressoProfile:
    command = f"srun --ntasks={para_info_dict['ranks_per_system']} --tasks-per-node={para_info_dict['ranks_per_node']} --cpus-per-task={para_info_dict['threads_per_rank']} --distribution=block:block --hint=nomultithread --mem={para_info_dict['max_mem_per_node']} {pwx_path} -nk {npool}"

    print(command)
    return EspressoProfile(
        command=command,
        pseudo_dir=pp_path,
    )


def generate_kpts(
    cell: np.ndarray, periodic_3d: bool = True, kspacing: float = 0.1
) -> np.ndarray:
    cell_lengths = cell.diagonal()
    kpts = np.ceil(2 * np.pi / (cell_lengths * kspacing)).astype(int)
    return kpts if periodic_3d else np.array([kpts[0], kpts[1], 1])


def find_optimal_npool(
    ranks_per_system: int, total_kpoints: int, min_ranks_per_pool: int = 8
) -> int:
    # Get all possible values that divide total_cores evenly
    possible_npools = [
        i
        for i in range(1, ranks_per_system + 1)
        if ranks_per_system % i == 0
        and ranks_per_system / i >= min_ranks_per_pool
        and i <= total_kpoints
    ]
    target = ranks_per_system**0.5
    npool = min(possible_npools, key=lambda x: abs(x - target))

    return int(npool)


def create_qe_calc_object(atoms, high_accuracy_eval_hpc_job_dict, out_dir):
    kpt_arr = generate_kpts(cell=atoms.cell, periodic_3d=True, kspacing=0.15)
    npool = find_optimal_npool(
        total_kpoints=int(np.prod(kpt_arr)),
        ranks_per_system=high_accuracy_eval_hpc_job_dict["node_info"][
            "ranks_per_system"
        ],
        min_ranks_per_pool=8,
    )

    return Espresso(
        profile=create_espresso_profile(
            para_info_dict=high_accuracy_eval_hpc_job_dict["node_info"],
            npool=npool,
            pwx_path=high_accuracy_eval_hpc_job_dict["pwx_path"],
            pp_path=high_accuracy_eval_hpc_job_dict["pp_path"],
        ),
        input_data=get_qe_input_data("scf"),
        kpts=list(kpt_arr),
        pseudopotentials=high_accuracy_eval_hpc_job_dict["pseudo_dict"],
        directory=out_dir,
    )


def run_qe(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
):
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    calc = create_qe_calc_object(
        input_structure, high_accuracy_eval_job_dict["hpc"], out_dir
    )

    input_structure.calc = calc
    input_structure.get_potential_energy()

    write(
        Path(out_dir, f"{high_accuracy_eval_job_dict['name']}.xyz"),
        input_structure,
        format="extxyz",
    )

    return input_structure
