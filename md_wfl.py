from genericpath import exists
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.units import fs
import numpy as np
import pandas as pd
from pathlib import Path

# calc=MACECalculator(model_paths=[f'MACE_model/fit_{i}/test_stagetwo_compiled.model' for i in range(2)],device='cpu')

def primary_run(out_dir, model_dir, base_mace=0, device='cuda', steps=100, temperature=300, number_of_md_runs=1, initial_structures: list=[]):
    calc=MACECalculator(model_paths=[str(Path(model_dir, f'/fit_{base_mace}/test_stagetwo.model'))], device=device)

    for i in range(number_of_md_runs):
        initial_structure = initial_structures[i].copy()
        initial_structure.calc=calc

        dyn= Langevin(atoms=initial_structure, timestep= 0.5*fs, temperature_K=temperature, friction=0.002, logfile=str(Path(out_dir, f'md_{temperature}.log')))
        
        traj= Trajectory(str(Path(out_dir, f'md_{i}_{temperature}.traj').mkdir()), 'w', initial_structure)
        dyn.attach(traj.write, interval=50)
        dyn.run(steps=steps)

def flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0]*3))

def std_deviation_of_forces(structure_forces_dict: dict[dict[dict[np.ndarray[float], float]]], verbose: int = 0) -> list[float]:
    """
    Calculate the standard deviation of forces for each structure in the dictionary.
    
    Parameters
    ----------
    structure_force_dict : dict
        A dictionary where keys are fit names and values are dictionaries with structure names as keys and forces as values.

        e.g.:
        {
            'base_mace': {
                'structure_0': {'forces': np.ndarray, 'energy': float},
                'structure_1': {'forces': np.ndarray, 'energy': float},
                ...
            },
            'fit_1': {
                ...
            },
        }
    
    Returns
    -------
    list
        A list of standard deviations of forces for each structure.
    """
    number_of_structures = len(structure_forces_dict['base_mace'])
    std_dev_array= np.zeros((number_of_structures, 3))
    for structure in range(number_of_structures):
        forces_array=np.concatenate([structure_forces_dict[fit][f'structure_{structure}']['forces'] for fit in structure_forces_dict], axis=0)
        std_dev_per_force_fragment = np.std(forces_array, axis=0)
        energy_array=np.array([structure_forces_dict[fit][f'structure_{structure}']['energy'] for fit in structure_forces_dict])
        std_dev_per_energy = np.std(energy_array)
        
        if verbose > 0:
            print(f"Structure {structure}, max std dev: {np.max(std_dev_per_force_fragment)}, mean std dev: {np.mean(std_dev_per_force_fragment)}, std dev of energy: {std_dev_per_energy}, energies: {energy_array}")
        
        std_dev_array[structure, :]= np.array([np.max(std_dev_per_force_fragment), np.mean(std_dev_per_force_fragment), std_dev_per_energy])
    
    return pd.DataFrame(std_dev_array, columns=['max_std_dev', 'mean_std_dev', 'std_dev_energy']).sort_values(by='max_std_dev', ascending=False)


def get_forces_for_all_maces(structure_list, base_mace=0, fits_to_use=[0]) -> np.ndarray:
    calc=MACECalculator(model_paths=f'MACE_model/fit_{base_mace}/test_stagetwo.model',device='cpu')

    for atoms in structure_list:
        atoms.calc=calc
    structure_forces_dict= {'base_mace':{f'structure_{i}' : {'forces': flatten_array_of_forces(structure_list[i].get_forces()), 'energy': structure_list[i].get_potential_energy()} for i in range(len(structure_list))}}

    for i in fits_to_use:
        calc=MACECalculator(model_paths=f'MACE_model/fit_{i}/test_stagetwo.model',device='cpu', default_dtype='float64')
        for atoms in structure_list:
            atoms.calc=calc            
        structure_forces_dict[f'fit_{i}']={f'structure_{i}' : {'forces': flatten_array_of_forces(structure_list[i].get_forces()), 'energy': structure_list[i].get_potential_energy()} for i in range(len(structure_list))}
    
    return structure_forces_dict

def get_structures_for_dft(md_name, initial_structures, read_md=False, number_of_structures=50, verbose=0, save_xyz=True) -> list:
    """
    Select structures for DFT calculations based on the standard deviation of forces.
    
    Parameters
    ----------
    md_name : str
        Name of the MD run.
    read_md : bool, optional
        Whether to read the MD file directly, by default False. If False, it will run
    number_of_structures : int, optional
        Number of structures to select, by default 50.
    
    Returns
    -------
    list
        List of structures selected for DFT calculations.
    """

    traj_files= sorted(Path('.').glob(f'{md_name}*.traj'))
    if not (read_md and len(traj_files)>0):
        primary_run(steps=1000, temperature=1200, number_of_md_runs=len(initial_structures), initial_structures=initial_structures)    
    
    traj_list=[Trajectory(traj_file, 'r') for traj_file in traj_files]
    structure_list=[a for a in traj_list for a in a]

    if verbose>0:    
        print(len(structure_list), "trajectories found in", traj_files)

    structure_forces_dict=get_forces_for_all_maces(structure_list, base_mace=0, fits_to_use=[1, 2, 3, 4])
    std_dev_df=std_deviation_of_forces(structure_forces_dict)
    index_list=list(std_dev_df[:number_of_structures].index)

    if verbose > 0:
        print(std_dev_df[:number_of_structures])
        print(f"total mean: {std_dev_df['mean_std_dev'].mean()}")
    
    if save_xyz:
        write('high_sd_structures.xyz', [structure_list[i] for i in index_list], format='extxyz')
    
    return [structure_list[i] for i in index_list]

if __name__ == "__main__":
    c10Na1_structures=[s for s in read('mace_general/ac_all_33_2025_07_11_ftrim_100_test_set.xyz', ':') if s.get_chemical_formula() == 'C10Na']
    print(c10Na1_structures)
    initial_structures=[c10Na1_structures[x] for x in np.random.choice(np.array(range(len(c10Na1_structures))), 5, replace=False)]
    print(initial_structures)
    # dft_structure_list=get_structures_for_dft('md', read_md=True, number_of_structures=10, verbose=1)
    