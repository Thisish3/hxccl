import os


def fill_env_from_openmpi():
    """mpirun sets OMPI_* variables; torch.distributed reads RANK/WORLD_SIZE/MASTER_*."""
    m = {"RANK": "OMPI_COMM_WORLD_RANK", "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE",
         "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK"}
    for k, v in m.items():
        if k not in os.environ and v in os.environ:
            os.environ[k] = os.environ[v]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
