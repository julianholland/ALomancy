from ase.build import bulk
from ase.calculators.emt import EMT

from wfl.calculators import generic
from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.autoparallelize.remoteinfo import RemoteInfo
from wfl.configset import ConfigSet, OutputSpec
from expyre.resources import Resources


atoms = []
for idx in range(5000):
    at = bulk("Cu", "fcc", a=3.6, cubic=True)
    at *= (2, 2, 2)
    at.rattle(stdev=0.01, seed=159+idx)
    atoms.append(at)

configset = ConfigSet(atoms)
outputspec = OutputSpec("configs.emt.xyz")
calculator = (EMT, [], {})  # (calculator_constructor_function, arguments, keyword_arguments)

remote_info = RemoteInfo(
        sys_name="raven",
        job_name="emt_calculations",
        num_inputs_per_queued_job=5000,
	pre_cmds=[
            "source /u/jholl/venvs/wfl/bin/activate",
            # "conda activate wfl",
        ],
        timeout=28000,
        check_interval=60,
        resources = Resources(
            max_time = "1h",
            num_nodes = 1,
            max_mem_tot = "10GB",
            partitions = ["general"],
            ),
        )

generic.calculate(
    inputs = configset,
    outputs = outputspec,
    calculator = calculator,
    output_prefix = "emt_",
    autopara_info = AutoparaInfo(
        remote_info = remote_info
    ))