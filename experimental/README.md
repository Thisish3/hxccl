# Experiments

## X: device-driven ring AllReduce on one GPU (`device_allreduce.cu`)

One kernel per rank (1 block × 256 threads) runs ring reduce-scatter + all-gather. It writes into the right neighbor's inbox over CUDA IPC, fences with `__threadfence_system()`, raises the neighbor's flag, and spins until its own flag rises. Each step has its own inbox slot. The kernel records `%globaltimer`, a single nanosecond clock for the GPU, at start and at every send and every "seen". Setup: RTX 3060 Ti, WSL2, CUDA 12.2, driver 581.80, SM clock locked at 1500 MHz. MPS is not available in WSL (`nvidia-cuda-mps-control` is missing), so every run is without MPS. Raw output is in `docs/results/2026-11-11_x_device_allreduce.txt`.

### What was measured (2026-11-11)

| ranks | count (fp32) | result | kernel time |
|---|---|---|---|
| 1 | 1<<20 | PASS ×3 | 0.64–0.80 ms |
| 2 | 1<<20 | PASS ×6 (two batches) | 38–43 ms |
| 2 | 4096 | PASS ×3 (first batch) | 33–39 ms |
| 4 | 1<<20 | PASS ×4, then **hang** (see below) | 98–197 ms |
| 4 | 4096 | PASS ×3 (first batch) | 173–197 ms |

At np 2 and 4 the time hardly depends on the size (4096 vs 1<<20 elements). The device-driven version is about 10× slower than the IPC-copy path at np2 (3.4 ms at 4 MiB) and 10–20× slower at np4 (9.8 ms).

The step timeline from the second batch, np2, 1<<20, run 1 (ms, relative to the earliest kernel start):

```
rank 0: start +0.000  | s0 sent +0.729  seen +35.394 | s1 sent +36.882 seen +36.883
rank 1: start +19.928 | s0 sent +20.658 seen +20.658 | s1 sent +22.154 seen +37.796
```

- **The second process's kernel started about 20 ms after the first one,** although both launched right after an `MPI_Barrier`. The three np2 runs give 19.9, 21.7 and 21.2 ms.
- **A flag raised by one rank became visible to the other 14.7–15.6 ms later** (rank 1's s0 at +20.66 was seen by rank 0 at +35.39). Inside one kernel, the next step follows in 0.5–1.5 ms.
- **At np4 the kernel starts were staggered about 16 ms apart** (+0, +16.9, +31.7, +48.5 ms). Each wait lasted about 60 ms, and every rank finished at +192–194 ms.

**Reading:** the stamps fit kernels from different processes taking turns on the GPU (time slicing) instead of running at the same time. A spinning kernel then makes no progress until the GPU switches to the process that holds its data. This was **not confirmed with a GPU trace**: Nsight Systems 2023.2 in WSL records no kernel data (LEDGER R21). The roughly 16–20 ms period is read off the stamps; the scheduler's real time slice was not measured.

### The hang

The second np4 run with 1<<20 elements did not finish within the 60 s `timeout`. After `timeout` killed `mpirun`, its four processes stayed in uninterruptible sleep (`D`). Every later run hung too, including np1. `nvidia-smi` stopped answering in WSL and on Windows.

- **What is known:** the first hang came at np4 after four successful np4 runs, and the GPU did not recover when the processes were killed.
- **What is not known:** **cause not confirmed.** Nothing was recorded inside the hung run. One open question is whether the time-sliced spinning kernels can wait on each other forever, for example if a flag write is not visible in the slice in which the waiting kernel runs.
- **Recovery:** this needs `wsl --shutdown`, or a reboot if that does not free the GPU. Do not run this experiment at np ≥ 4 on a GPU that something else needs.

### Conclusion for hxccl

On one GPU without MPS, cross-process spinning kernels are slow (10–20× the IPC-copy path) and can hang the GPU. So hxccl keeps the host-synchronized IPC path (spec 3.4), and the device-driven kernel stays out of the library, as planned (spec 7).
