# torchrun --nproc-per-node 2 experimental/torch_nccl_same_gpu.py
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
torch.cuda.set_device(0)
t = torch.ones(4, device="cuda") * (dist.get_rank() + 1)
try:
    dist.all_reduce(t)
    print(f"rank {dist.get_rank()}: OK {t.tolist()}")
except Exception as e:
    print(f"rank {dist.get_rank()}: ERROR {type(e).__name__}: {e}")
dist.destroy_process_group()
