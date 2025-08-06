def load_dictionaries():
    """
    Set the information for the HPCs and jobs used in the active learning workflow.

    three jobs are required
    mace_committee: advised to use a GPU based HPC
    structure_generation: advised to use a GPU based HPC
    high_accuracy_evaluation: advised to use a CPU based HPC

    for each job, the following information is required:
    - name: the name of the job, used to identify the job in the workflow
    - max_time: the maximum time allowed for the job to run
    - hpc: a dictionary containing information about the HPCs available for the job
        - pre_cmds: commands to run before starting the job, e.g. activating a virtual environment
        - partitions: the partition used for the job on the HPC

    Returns
    -------
    dict
        A dictionary containing the HPC and job information.
    """

    JOB_DICT = {
        "mlip_committee": {
            "name": "mlip_committee",
            "max_time": "5H",
            "hpc": {
                # "raven_gpu": {
                #     "pre_cmds": ["source /u/jholl/venvs/wfl/bin/activate"],
                #     "partitions": ["gpu"],
                "hpc_name": "fhi-raccoon",
                "pre_cmds": ["source /home/jholl/venvs/alomancy/bin/activate"],
                "partitions": ["gpusmall"],
            },
        },
        "structure_generation": {
            "name": "structure_generation",
            "max_time": "10H",
            "hpc": {
                # "raven_gpu": {
                #     "pre_cmds": ["source /u/jholl/venvs/wfl/bin/activate"],
                #     "partitions": ["gpu"],
                # },
                "hpc_name": "fhi-raccoon",
                "pre_cmds": ["source /home/jholl/venvs/alomancy/bin/activate"],
                "partitions": ["gpusmall"],
            },
        },
        "high_accuracy_evaluation": {
            "name": "high_accuracy_evaluation",
            "max_time": "30m",
            "hpc": {
                "hpc_name": "raven",
                "pre_cmds": ["source /u/jholl/venvs/wfl/bin/activate"],
                "partitions": ["gpu"],
            },
        },
    }

    return JOB_DICT
