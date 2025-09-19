from mace.calculators import MACECalculator
from quansino.mc.fbmc import ForceBias, AdaptiveForceBias
from ase import Atoms
from pathlib import Path
from ase.calculators.emt import EMT
from ase.constraints import FixCom
from quansino.constraints import FixRot
from sys import stdout


calc = MACECalculator(model_paths=[f"test{i}_stagetwo.model" for i in range(3)], device="cpu", default_dtype="float64")


atoms=Atoms(
        "C3",
        positions=[[0, 0, 0],[1, 0.4, 0.5], [3, 0, 1]],
        cell=[5, 4, 4],
        pbc=True,
        calculator=calc,
    )
atoms.set_constraint([FixCom(), FixRot()])
fbmc_object = AdaptiveForceBias(
    atoms=atoms,
    temperature=300,
    min_delta=0.1,
    max_delta=0.5,
    trajectory=Path("demo_fbmc.xyz"),
    logfile=stdout,
    logging_interval=1,
    logging_mode="w",
)

fbmc_object.run(steps=100)
