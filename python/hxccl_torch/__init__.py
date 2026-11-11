"""torch.distributed backend "hxccl". Run under mpirun: `mpirun -np 2 python train.py`."""
import atexit

import torch  # noqa: F401  loads libtorch before the extension
import torch.distributed as dist

from .launch import fill_env_from_openmpi
from . import _C  # initializes MPI on import

fill_env_from_openmpi()
dist.Backend.register_backend("hxccl", _C.create_backend, devices=["cuda"])
# mpirun treats a process that initialized MPI and exits without MPI_Finalize as failed
atexit.register(_C.finalize)
