from ase.calculators.espresso import Espresso, EspressoProfile
from wfl.calculators import generic
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec
import numpy as np
from ase.io import read
# from al_wfl import get_remote_info


def get_qe_input_data(calculation_type: str) -> dict:
    return {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "prefix": "ac",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "disk_io": "low",
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


def create_espresso_profile(para_info_dict: dict, npool: int, pwx_path: str, pp_path: str):
    command = f"srun --ntasks={para_info_dict['ranks_per_system']} --tasks-per-node={para_info_dict['ranks_per_node']} --cpus-per-task={para_info_dict['threads_per_rank']} --distribution=block:block --hint=nomultithread --mem={para_info_dict['max_mem_per_node']} {pwx_path} -nk {npool}"

    print(command)
    return EspressoProfile(
        command=command,
        pseudo_dir=pp_path,
    )


def generate_kpts(
    cell: np.ndarray, periodic_3d: bool = False, kspacing: float = 0.1
) -> list:
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

    return npool

def create_calc_object(atoms, pseudo_dict, hpc_info_dict):
    kpt_arr = generate_kpts(cell=atoms.cell, periodic_3d=True, kspacing=0.15)
    npool = find_optimal_npool(
            total_kpoints=np.prod(kpt_arr),
            ranks_per_system=hpc_info_dict['node_info']['ranks_per_system'],
            min_ranks_per_pool=8,
        )    
    
    return Espresso(profile=create_espresso_profile(para_info_dict=hpc_info_dict['node_info'], npool=npool, pwx_path=hpc_info_dict['pwx_path'], pp_path=hpc_info_dict['pp_path']), 
                    input_data=get_qe_input_data("scf"), 
                    kpts=list(kpt_arr),
                    pseudopotentials=pseudo_dict)

def perform_dft_calculations(name, atoms_list, remote_info, hpc):
    pseudo_dict = {"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
                   "Na": "na_pbe_v1.5.uspp.F.UPF"}

    hpc_info_dict = {'nersc': {'node_info': {'ranks_per_system': 128,
                               'ranks_per_node':128,
                               'threads_per_rank':1,
                               'max_mem_per_node': '500GB'
                               },
                               'pwx_path' : '/global/common/software/nersc9/espresso/7.3.1-libxc-6.2.2-cpu/bin/pw.x',
                               'pp_path' : "pscratch/sd/j/joh1e19/pps/SSSP_1.3.0_PBE_efficiency"
                               },
                    'raven': {'node_info': {'ranks_per_system': 72,
                               'ranks_per_node':72,
                               'threads_per_rank':1,
                               'max_mem_per_node': '60GB'
                               },
                               'pwx_path' : '/raven/u/system/soft/SLE_15/packages/skylake/qe/gcc_13-13.1.0-openmpi_5.0-5.0.7/7_3_1/bin/pw.x',
                               'pp_path': '/u/jholl/pps/SSSP_1.3.0_PBE_efficiency'
                               },
                    }
    
    outputs = OutputSpec(f'{name}_out_structures.xyz')

    calc=create_calc_object(atoms_list[0], pseudo_dict, hpc_info_dict=hpc_info_dict[hpc]) # currently applies same calc to all, may need different k-grid for different systems

    generic.calculate(inputs=ConfigSet(atoms_list),
                      outputs=outputs,
                      calculator=calc,
                      properties = ["energy", "forces"],
                      output_prefix="Espresso_",
                      autopara_info=AutoparaInfo(
                         remote_info=remote_info,
                         )
                      )
    
if __name__ == '__main__':
    name='qe_test'
    hpc='raven'
    atoms_list=read('high_sd_structures.xyz', ':')
    # remote_info=get_remote_info(name, hpc)
    remote_info=None
    perform_dft_calculations('qe_test', atoms_list, remote_info, hpc)