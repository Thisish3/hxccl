# mpirun -np N python python/bench_allreduce.py --backend hxccl|gloo [--out file.csv]
# fp32 all_reduce (sum), 4 KiB .. 256 MiB in steps of 4x, warm-up 5, median of 20.
# hxccl uses CUDA tensors, gloo uses CPU tensors.
import argparse
import csv
import statistics
import time

import torch
import torch.distributed as dist

import hxccl_torch  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--backend", default="hxccl")
p.add_argument("--out")
a = p.parse_args()
dist.init_process_group(a.backend)
dev = "cuda" if a.backend != "gloo" else "cpu"
rows = []
for size in [1 << k for k in range(12, 29, 2)]:  # 4 KiB .. 256 MiB
    t = torch.ones(size // 4, device=dev)
    for _ in range(5):
        dist.all_reduce(t)
    times = []
    for _ in range(20):
        dist.barrier()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    rows.append((a.backend, dist.get_world_size(), size, med * 1e6, size / med / 1e9))
if dist.get_rank() == 0:
    for r in rows:
        print("%s np=%d bytes=%d median_us=%.1f GBps=%.2f" % r)
    if a.out:
        with open(a.out, "a", newline="") as f:
            csv.writer(f).writerows(rows)
dist.destroy_process_group()
