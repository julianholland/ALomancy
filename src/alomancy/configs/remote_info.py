from wfl.autoparallelize.remoteinfo import RemoteInfo
from expyre.resources import Resources


def get_remote_info(job_dict, input_files: list[str] = []):
    """
    Returns a RemoteInfo object for running MACE fits on a GPU cluster.
    """

    print(f"HPC: {job_dict['hpc']['hpc_name']}, Job: {job_dict['name']}")
    return RemoteInfo(
        sys_name=job_dict["hpc"]["hpc_name"],
        job_name=job_dict["name"],
        num_inputs_per_queued_job=1,
        timeout=36000 * 3,
        input_files=input_files,
        pre_cmds=job_dict["hpc"]["pre_cmds"],
        resources=Resources(
            max_time=job_dict["max_time"],
            num_nodes=1,
            partitions=job_dict["hpc"]["partitions"],
        ),
    )
