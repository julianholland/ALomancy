# TODO

### Plotting
- [ ] x axis on combined timings should only contain numbers (reduced if too many ticks)
- [ ] remove old timings plot(s) only the combined timings should be used
- [ ] only plot the best model of each loop for the totatl mlip mae plot
- [ ] add gold star watermark to top left of the subplot that has the lowest energy and lowest forces of the parity plots 

### Functionality 
- [ ] Optionally optimize a fraction of structures produced in the structure generation, until we are st
- [ ] check we can initialize froma a sage lib db. i.e. if I wanted to pickup from a former run it should be sufficient to use the global database folder alone
- [ ] save the md trajectories somewhere
- [ ] don't recalculae the descriptors every loop/restart they should be sotored in the db alongside the structure

### Refactor 

The general idea for the code so that it alligns more with the mission statement is to modularize the type of active learning we do 

Currently we have one type (commitee based) which is outlined in base_active_learning and flavoured with function calls from standard active learning. I want to move towards true modules that are correctly loaded and placed inside a chose active learning skeleton. 

To achieve this we will need to
- keep the core architecture of base_active_learning (renamed committee uncertainty active learning) However stanadard active learning should be broken up into its functions e.g. initialiser_interface.py and high_accuracy_calc_interface.py. The core purpose of them is to run a module the skeleton calls useing the relevant settings from the yaml file. the mlip train file will need extra work as it has two issues: it is hard coded for committee evaluation, it is hard coded fro use with MACE. The committee hard coding should get moved to the base active learning function and the mace hard coding to the respective mace handler. similarly the structure generatoin could be more elegantly handled. perhaps a dictionary of call methods which then load in the correct settings perhaps that is all the method needs to be. It may be better to have per module (i.e. per structure generator) interfacers responisble for loading in the correspondingly correct settings. regardless of how it is achieved they should be isolated modules that can be called in any order into the chose active learing workflow
- Room for new skeletons such as a single committee using furthest point sampling that only trains one potential not a committee and doesn't need reevaluation
