# Tutorials

After following the instructions in the [installation guide](installation.md) I reccomend following at least one of the following tutorials to understand how to best use the `alomancy` package

## 1. Creating a Standard AL workflow for water

For this tutorial we will
- create an initial starting data set for water
- set up input yaml files for an `alomancy` Standard AL Workflow object call
- run a lightweight local version of the code to perform the training

### Creating a Dataset
It is important that our initial data set contains
- All types of atoms that we wish to train upon (H and O)
- A diverse collection of potential stoichiometries, atom numbers, and cell densities

We will start with constructing atoms objects for the indiviual atoms

```python
from ase import atoms
from ase.calculators.emt import EMT

all_atoms_objects_list=[]

# construct single atom Atoms objects
h_atom = Atoms('H', position=[0,0,0], cell=[10,10,10], pbc=True)
o_atom = Atoms('O', position=[0,0,0], cell=[10,10,10], pbc=True)

# Evaluate the objects energetically, in reality you should use a much higher accuracy method than EMT such as DFT
for atom in [h_atom, o_atom]:
    atom.calc=EMT()
    atom.get_potential_energy()
    atom.info['config_type']='IsolatedAtom'
    all_atoms_objects_list.append(atom)

```
We assign config types to assist with the MACE training later on.

Next, we will add molecules to our data set. We will alsoe add deformed molecules to increase the amount of data in our training set.

```python
from ase.build import molecule

# define the paramater
max_contraction = 0.3
max_expansion = 0.5
deformation_samples = 10

molecule_list=['H2', 'O2']


h2=molecule('H2')
o2=molecule('O2')
