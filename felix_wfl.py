import os
from wrapper.remote_ba_nsga import run_ba_nsga
from wrapper.mace_fits import mace_multihead_fit
from wrapper.utils import ga_split, ga_read
from wrapper.vasp_dft import auto_DFT




start = 6
stop = 7

if not os.path.exists('MACE'):
    os.mkdir('MACE')

for i in range(start, stop):
    print(f'Starting Generation {i}')
    ga_run_folder = f'ga_dir/generation_{i}'
    if not os.path.exists(ga_run_folder):
        os.mkdir(ga_run_folder)
    mace_model = f'MACE/mace_{i-1}_stagetwo.model'
    DFT_in = f'ga_dir/generation_{i}/DFT_{i}_in.xyz'
    DFT_out = f'ga_dir/generation_{i}/DFT_{i}_out.xyz'
    

    if start == 1:
        mace_model = 'MACE/REMD_O2_bulk_stagetwo.model'


    ############### ba_nsga ##################
    print('################ GA running #################')
    run_ba_nsga(run_folder = ga_run_folder, mace_file = mace_model)


    ############### Random Sample  ##################
    ga_read(i, DFT_in) 
    

    ############### DFT #############
    print('############## DFT running #################')
    auto_DFT(DFT_in, DFT_out)
    ga_split(DFT_out,i)


    ############### MACE #############
    print('############## MACE running ################')

    mace_multihead_fit(i,f'ga_dir/generation_{i}/training_{i}.xyz',f'ga_dir/generation_{i}/valid_{i}.xyz',f'ga_dir/generation_{i}/test_{i}.xyz')