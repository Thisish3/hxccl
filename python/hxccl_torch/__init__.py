"""torch.distributed backend "hxccl".

Launch with torchrun (`torchrun --nproc-per-node 2 train.py`) or with mpirun
(`mpirun -np 2 python train.py`); under mpirun, RANK/WORLD_SIZE come from OpenMPI.
"""
import atexit

import torch  # noqa: F401  loads libtorch before the extension
import torch.distributed as dist

from .launch import fill_env_from_openmpi
from . import _C

fill_env_from_openmpi()
dist.Backend.register_backend("hxccl", _C.create_backend, devices=["cuda"])
# finalizes MPI only if the program initialized it; otherwise a no-op
atexit.register(_C.finalize)
