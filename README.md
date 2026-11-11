# hxccl

A small NCCL-style collective communication library, built step by step to understand how NCCL/RCCL work and to compare against them.

> **Status: A (core) and A2 (full CPU-side NCCL/RCCL API) done.** CPU buffers over OpenMPI. GPU direct transport (B), PyTorch `ProcessGroup` backend (C) and the comparison with RCCL/NCCL on real multi-GPU hardware (D) are next. Not affiliated with NVIDIA NCCL, AMD RCCL or Huawei HCCL.

## API

Same names, numbering and argument order as NCCL v2.32.3 (`nccl.h`), with an `hxccl` prefix. Result codes, datatypes and ops have NCCL's numeric values.

```c
hxcclComm_t comm;
hxcclCommInitMPI(&comm, MPI_COMM_WORLD);          // or hxcclGetUniqueId + hxcclCommInitRank[Config]
hxcclAllReduce(send, recv, count, hxcclFloat32, hxcclSum, comm, /*stream*/ NULL);
hxcclGroupStart();                                 // Send/Recv pairs that must progress together
hxcclSend(out, n, hxcclFloat32, right, comm, NULL);
hxcclRecv(in, n, hxcclFloat32, left, comm, NULL);
hxcclGroupEnd();
hxcclCommDestroy(comm);
```

| Area | Implemented |
|---|---|
| Collectives | AllReduce, AllGather, ReduceScatter, Broadcast, Bcast, Reduce, Gather, Scatter, AlltoAll; RCCL's AllToAll and AllToAllv |
| Point-to-point | Send, Recv, GroupStart, GroupEnd (nested groups run at the outermost End) |
| Datatypes | all 12: int8/uint8/int32/uint32/int64/uint64, fp16, bf16, fp8 e4m3/e5m2 (software conversion), fp32, fp64 |
| Ops | sum, prod, max, min, avg, RedOpCreatePreMulSum / RedOpDestroy |
| Communicator | InitRank, InitRankConfig, Split, Shrink, Abort, Finalize, Destroy, GetAsyncError, Count, UserRank, GetVersion |

Behaviour worth knowing:

- **Send/Recv outside a group block.** Two ranks that both Send then Recv a large message wait for each other forever, as in NCCL. Put them in a group.
- **Avg follows NCCL** (`src/enqueue/enqueue.cc`): integers are summed, then divided by N (truncating toward zero). Floats are first multiplied by 1/N (rounded to the datatype), then summed. For example, two fp16 values of 60000 average to 60000, not to inf.
- **fp8 overflow is undefined here.** e4m3 values beyond 448 become NaN and IEEE-style types become inf. NCCL's mode was not checked; B will compare the two on a GPU.
- **AllToAllv does one extra `MPI_Allreduce`** before sending. Block sizes differ per rank, so ranks first agree on whether any argument is invalid, and then all of them reject together instead of leaving the others waiting. The agreement runs at call time, also inside a group.
- **Known problems** are listed in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).
- **Abort is local.** It frees hxccl's state without talking to peers and leaks the MPI communicator on purpose.

Moved to later stages, because they need GPU memory, streams, or a socket bootstrap:

| API | Stage |
|---|---|
| MemAlloc/Free, CommRegister, Window*, CommCuDevice, MemStats, InitAll, PutSignal/Signal/WaitSignal, non-blocking communicators, Suspend/Resume | B (GPU) |
| AllReduceWithBias (RCCL) | D (needs RCCL's actual output to pin the meaning of `acc`) |
| CommGrow, InitRankScalable, CommGetUniqueId, Revoke, GroupSimulateEnd, Param*, per-collective `*Config`, SetEncryption, CommDump | E (bootstrap, tuner) |

## Design

```
api*.cc       public functions: check arguments now, run via dispatch (now, or at GroupEnd)
checks.cc     shared argument checks; resolves Avg / PreMulSum into an OpPlan
group.cc      group queue; Send/Recv are posted with MPI_Isend/Irecv, then waited together
reduce.cc     elementwise reduction for 12 dtypes; fp.h has the fp16/bf16/fp8 conversions
select.cc     allreduce algorithm by size: recursive doubling < 64 KB, ring above (HXCCL_ALGO=ring|rd forces one)
algo_ring.cc  ring allreduce (reduce-scatter + all-gather), allgather, reducescatter
algo_rd.cc    recursive doubling; non-power-of-two N folds the extra ranks into partners first
algo_bcast.cc binomial tree < 512 KB, 512 KB-segment chain above
algo_rooted.cc reduce (mirror of bcast), gather, scatter
algo_alltoall.cc pairwise exchange: step s sends to r+s, receives from r−s
transport_*   point-to-point layer; algorithms only see Transport (MPI_Sendrecv, Isend/Irecv)
```

Algorithms never call MPI directly, so the GPU IPC transport in B can be dropped in without touching them.

## Tests

`ctest` runs every test under `mpirun -np N` for N ∈ {1, 2, 3, 4, 5, 8} (78 test runs):

- **Correctness against MPI:** every collective is compared bit-for-bit with `MPI_Allreduce` / `MPI_Allgather` / `MPI_Reduce_scatter_block` / `MPI_Bcast`, for counts {0, 1, 3, 7, 1000, 4099}. Inputs are exactly representable, so the order of reduction does not matter. The tests cover all ops and dtypes, the ring / rd / auto algorithms, in-place calls, and broadcast from root 0 and root N−1.
- **Traffic:** the bytes and messages each rank sends must equal the schedule exactly. For ring this is the sum of the chunk sizes it forwards (2(N−1) messages). For rd it is log₂N messages × S when N is a power of two.
- **Errors:** invalid arguments return `hxcclInvalidArgument` and send nothing. An MPI failure returns `hxcclSystemError` and the process stays alive, because the communicator uses `MPI_ERRORS_RETURN`.

## Build

```bash
conda create -n hxccl -c conda-forge openmpi=5.0.11 cmake ninja
export PATH=/opt/anaconda3/envs/hxccl/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++   # Release by default
cmake --build build && ctest --test-dir build
mpirun --oversubscribe -np 4 build/hxccl_perf -b 256 -e 64M -f 4 -n 20
```

## Measurements (A: CPU, shared memory)

Setup:

- Machine: Apple M2 (4 performance + 4 efficiency cores).
- OpenMPI 5.0.11 (conda-forge), Apple clang 14, Release build.
- Date: 2026-11-06.
- `hxccl_perf` times each iteration as the slowest rank. The table shows the median of 20 iterations.
- busbw uses the nccl-tests definition (allreduce: algbw × 2(N−1)/N).

N = 4, allreduce float32:

| size | hxccl µs | MPI µs | hxccl busbw GB/s | MPI busbw GB/s | hxccl algo |
|---:|---:|---:|---:|---:|:---:|
| 256 B | 1 | 1 | 0.38 | 0.38 | rd |
| 16 KB | 11 | 10 | 2.23 | 2.46 | rd |
| 64 KB | 29 | 26 | 3.39 | 3.78 | ring |
| 1 MB | 321 | 273 | 4.90 | 5.76 | ring |
| 4 MB | 1250 | 1484 | 5.03 | 4.24 | ring |
| 16 MB | 7355 | 7874 | 3.42 | 3.20 | ring |
| 64 MB | 72210 | 69894 | 1.39 | 1.44 | ring |

What was measured:

- At N = 4, hxccl's time is 0.84× to 1.25× OpenMPI's `MPI_Allreduce` (hxccl ÷ MPI).
  - From 4 KB to 1 MB, OpenMPI is faster (1.09× to 1.25×).
  - At 4 MB and 16 MB, hxccl is faster (0.84× and 0.93×).
  - At 256 B, 1 KB and 64 MB, the two are about equal.
- At N = 8, up to 4 MB, the ratio is 0.86× to 1.1×, and neither library is consistently ahead (raw output in `docs/bench/`).
- At N = 8, from 16 MB up, both libraries vary a lot from run to run. Four runs at 16 MB gave:
  - hxccl: 17.6, 17.7, 18.2 and 127 ms
  - MPI: 61, 67, 85 and 98 ms
- Times come out in whole microseconds, which suggests a 1 µs timer resolution. The 256 B to 4 KB rows therefore carry about ±1 µs.
- A Debug build (no `-O`) made hxccl 15 to 20× slower. That is why the build now defaults to Release.

Not verified:

- **Why N = 8 is unstable.** Eight ranks on 4 performance + 4 efficiency cores is one candidate, but it was not tested.
- **Why bandwidth drops at 64 MB.** Cache and memory effects are possible, but they were not measured.
- **Which algorithm OpenMPI picks at each size.** Its `coll/tuned` decision was not inspected.

## Measurements (A2: other collectives, CPU, shared memory)

Same machine, build and method as above. Date: 2026-11-08. N = 4. Raw output is in `docs/bench/{reduce,gather,scatter,alltoall,sendrecv}_np4.txt`.

busbw factors come from the `*GetBw` functions in nccl-tests `src/*.cu`: Reduce and SendRecv use 1. Gather, Scatter and AlltoAll use (N−1)/N, with S = count per rank × N.

hxccl time ÷ OpenMPI time:

| size | Reduce | Gather | Scatter | AlltoAll | SendRecv |
|---:|---:|---:|---:|---:|---:|
| 64 KB | 1.59 | 0.78 | 0.45 | 1.27 | 1.10 |
| 1 MB | 1.09 | 0.94 | 1.48 | 1.05 | 1.09 |
| 4 MB | 0.78 | 0.93 | 0.93 | 0.95 | 1.01 |
| 16 MB | 0.82 | 1.09 | 1.03 | 0.88 | 1.01 |
| 64 MB | 0.83 | 1.02 | 0.88 | 0.96 | 1.08 |

- Each cell comes from one run of 20 iterations. Run-to-run spread was not measured for these collectives.
- The large deviations at 64 KB and 1 MB (Reduce 1.59, Scatter 0.45 and 1.48) were not investigated. OpenMPI's algorithm choice at those sizes was not inspected either.

## Measurements (B: GPU buffers over CUDA IPC, one GPU)

Date: 2026-11-11. RTX 3060 Ti 8 GB (sm_86), Windows driver 581.80, WSL2 Ubuntu 22.04, CUDA 12.2, OpenMPI 4.1.2, build `-DHXCCL_CUDA=ON` Release. SM clock locked at 1500 MHz with `nvidia-smi -lgc 1500,1500`; all 96 one-second samples in `docs/results/2026-11-11_b_clocks.csv` read 1500 MHz. The memory clock was not locked: 66 samples, taken during the host-buffer runs, read 810 MHz, and the rest read 6800–7000 MHz. Method: `scripts/measure_b.sh`, which runs `hxccl_perf -o allreduce` with fp32 sum, 5 warm-up calls, then the median of 20 rank-max wall times. Every call returns when the work is done. Raw tables are in `docs/results/2026-11-11_b_allreduce_np{2,4}_{host,device}.txt`.

**Limit: every rank shares one GPU.** The "transfer" is a copy inside one GPU's memory, so these numbers do not show GPU-to-GPU (NVLink/PCIe) bandwidth. They show where the IPC design spends time.

AllReduce median time (µs):

| size | np2 host | np2 device | np4 host | np4 device |
|---:|---:|---:|---:|---:|
| 4 KiB | 3.9 | 3293 | 7.8 | 6654 |
| 64 KiB | 18.7 | 3351 | 32.5 | 9286 |
| 1 MiB | 423 | 3476 | 970 | 9557 |
| 4 MiB | 2984 | 3581 | 5884 | 9818 |
| 16 MiB | 12219 | 4863 | 32330 | 11262 |
| 64 MiB | 53247 | 10970 | 128069 | 30065 |
| 256 MiB | 216332 | 43028 | 516918 | 110449 |

- **Up to 4 MiB the host path is faster.** The device path has a floor of about 3.1–3.6 ms at np2. At np4 the floor is 6.7 ms with recursive doubling (below 64 KiB) and 9.3–9.8 ms with ring.
- **From 8 MiB up the device path is faster.** The full table shows the crossover between 4 and 8 MiB at both np2 and np4. Speedup at np2: 1.8× at 8 MiB, 2.5× at 16 MiB, 4.9–5.0× at 64–256 MiB. At np4: 1.3× at 8 MiB, 2.9× at 16 MiB, 4.3–4.7× at 64–256 MiB. At 64 MiB and above, busbw is 6.1–6.2 GB/s at np2 and 3.3–3.6 GB/s at np4.

Time breakdown from Nsight Systems (`scripts/nsys_b.sh`, np2, 25 calls including warm-up, rank 0 / rank 1, ms per call = total ÷ 25). The NVTX ranges are CPU-side time. `ipc:copy` is `cudaMemcpyAsync` + `cudaStreamSynchronize` of one chunk. `reduce` is only the kernel launch; the kernel runs on the stream and its time lands in the next synchronize. Nsight Systems 2023.2 in WSL recorded no GPU kernel or memcpy trace ("does not contain CUDA kernel data"), so GPU-side time is not split out.

| size | `hxccl:allreduce` | `ipc:copy` | `reduce` (launch) | `cudaMallocAsync` | rest (MPI READY/DONE wait and other) |
|---:|---:|---:|---:|---:|---:|
| 64 KiB | 4.09 / 4.06 | 1.74 / 1.15 | 0.05 / 0.05 | 0.80 / 0.77 | 1.50 / 2.09 |
| 4 MiB | 4.09 / 4.11 | 1.00 / 2.00 | 0.04 / 0.03 | 0.75 / 0.75 | 2.31 / 1.33 |
| 64 MiB | 12.24 / 12.23 | 5.77 / 5.24 | 0.05 / 0.04 | 0.81 / 0.81 | 5.60 / 6.13 |

- **`cudaMallocAsync` costs a lot on every call.** It is called once per call (25 times in 25 calls), with a median of 0.31–0.37 ms. Why it does not reuse pool memory between calls is not confirmed: the pool's release threshold was not inspected.
- **Each chunk copy is slow.** The median `ipc:copy` is 0.14–0.15 ms, even for a 32 KiB chunk. Median `cudaStreamSynchronize` is 0.13–0.15 ms. Why one synchronize takes that long on WSL2 is not confirmed.
- **The "rest" column is not split further.** The MPI READY/DONE waits were not traced (`-t mpi` recorded only `MPI_Init`/`MPI_Finalize`), so how the remainder divides between waiting and other work is not confirmed.

Reduction kernel (Nsight Compute 2026.3.1, `--clock-control none`, `k_reduce<float>`, 1<<20 elements, fp32 sum). Results are in `docs/results/2026-11-11_b_ncu_reduce.txt`.

| collection | time | DRAM read + write | DRAM clock (measured) | throughput |
|---|---:|---:|---:|---:|
| metrics only (1 pass) | 31.5 µs | 8.39 + 3.76 MB | 6.79 GHz | 386 GB/s (86% of the 448 GB/s spec) |
| SpeedOfLight section (several passes) | 299.1 µs | 8.39 + 3.78 MB | 0.81 GHz | 78.7% of peak at that clock |

During the multi-pass SpeedOfLight replay the DRAM clock read 0.81 GHz (`dram__cycles_elapsed.avg.per_second`), so that row reflects a lowered memory clock. Why the memory clock dropped during the replay is not confirmed. The prod, max and min rows (28.7–29.2 µs, 386–392 GB/s) are in the same file.

Usage limits (B):

- Build with `-DHXCCL_CUDA=ON`. A call returns only after its device work finishes; it does not return early the way NCCL does.
- All ranks must pass the same kind of buffer (host or device) to a collective. The first device call starts the device path with a collective step.
- A group cannot mix host and device Send/Recv; GroupEnd returns `hxcclInvalidUsage`.

## PyTorch backend (C)

```bash
cmake -S . -B build-cuda -DHXCCL_CUDA=ON && cmake --build build-cuda -j
cd python && CUDA_HOME=/usr/local/cuda pip install -e . --no-build-isolation && cd ..
mpirun -np 2 python train.py
```
```python
import hxccl_torch                      # registers "hxccl"; fills RANK/WORLD_SIZE from OpenMPI
import torch.distributed as dist
dist.init_process_group("hxccl")
dist.all_reduce(t)                      # t: a contiguous CUDA tensor
```

- **Supported calls:** `all_reduce` (SUM, PRODUCT, MIN, MAX, AVG), `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast`, `send`, `recv` and `barrier`. Each call returns an already completed `Work`.
- **Launch with `mpirun`, not `torchrun`.** Communicators are built from `MPI_COMM_WORLD`, so `torchrun` is not supported. Socket bootstrap is left for stage E.
- **Tests:** `mpirun -np {2,4} python -m pytest -q python/tests` compares the results against the `gloo` backend: 37 passed, 3 skipped (gloo has no integer AVG).

### Measurements (C)

Date: 2026-11-11, same machine and SM clock lock as B (all 42 samples at 1500 MHz, `docs/results/2026-11-11_c_clocks.csv`). Method: `python/bench_allreduce.py`, fp32 `all_reduce` sum, 5 warm-up calls, then the median of 20, timed with `torch.cuda.synchronize()` around each call. hxccl uses CUDA tensors; gloo uses CPU tensors. Raw data: `docs/results/2026-11-11_c_torch_allreduce.csv`.

| size | np2 hxccl (µs) | np2 gloo (µs) | np4 hxccl (µs) | np4 gloo (µs) |
|---:|---:|---:|---:|---:|
| 4 KiB | 3182 | 447 | 6324 | 1296 |
| 64 KiB | 3370 | 431 | 9339 | 1307 |
| 1 MiB | 3415 | 705 | 9372 | 1842 |
| 4 MiB | 3443 | 2310 | 9456 | 6250 |
| 16 MiB | 4542 | 9411 | 10637 | 29633 |
| 64 MiB | 10377 | 34753 | 28847 | 121033 |
| 256 MiB | 41455 | 136131 | 106185 | 487899 |

- **Up to 4 MiB gloo is faster;** hxccl carries the B device-path floor of about 3.2 ms at np2 and 6.3–9.5 ms at np4. **From 16 MiB up hxccl is faster:** 2.1–3.3× at np2 and 2.8–4.6× at np4.
- **These are not the same kind of transfer.** gloo moves CPU memory over TCP, and hxccl copies inside one GPU, so this table compares the two backends as they run here, not as transports.
- **PyTorch NCCL is not in the table.** With 2 ranks on one GPU it fails at the first collective: `ncclInvalidUsage ... Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000` (NCCL 2.27.5, spec "환경 기록").

## License

Apache License 2.0. See [LICENSE](LICENSE).
