# hxccl

A small NCCL-style collective communication library, built step by step to understand how NCCL/RCCL work and to compare against them.

> **Status: A (core) done.** CPU buffers over OpenMPI. GPU direct transport (B), PyTorch `ProcessGroup` backend (C) and the comparison with RCCL/NCCL on real multi-GPU hardware (D) are next. Not affiliated with NVIDIA NCCL, AMD RCCL or Huawei HCCL.

## API

Same shape and argument order as NCCL (`nccl.h`), with an `hxccl` prefix:

```c
hxcclComm_t comm;
hxcclCommInitMPI(&comm, MPI_COMM_WORLD);          // or hxcclGetUniqueId + hxcclCommInitRank
hxcclAllReduce(send, recv, count, hxcclFloat32, hxcclSum, comm, /*stream*/ NULL);
hxcclAllGather / hxcclReduceScatter / hxcclBroadcast
hxcclCommGetStats(comm, &bytes_sent, &msgs_sent);  // used by the traffic tests
hxcclCommDestroy(comm);
```

Datatypes: int32, float32, float64. Ops: sum, prod, max, min. In-place calls follow NCCL's definitions.

## Design

```
api.cc        argument checks, error strings (hxcclGetLastError)
select.cc     allreduce algorithm by size: recursive doubling < 64 KB, ring above (HXCCL_ALGO=ring|rd forces one)
algo_ring.cc  ring allreduce (reduce-scatter + all-gather), allgather, reducescatter
algo_rd.cc    recursive doubling; non-power-of-two N folds the extra ranks into partners first
algo_bcast.cc binomial tree < 512 KB, 512 KB-segment chain above
transport_*   point-to-point layer; algorithms only see Transport (A: MPI_Sendrecv)
```

Algorithms never call MPI directly, so the GPU IPC transport in B can be dropped in without touching them.

## Tests

`ctest` runs every test under `mpirun -np N` for N ∈ {1, 2, 3, 4, 5, 8} (30 test runs):

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

- At N = 4, hxccl is within about 0.85× to 1.2× of OpenMPI's `MPI_Allreduce` at every size.
- Below 1 MB, OpenMPI is a little faster. At 4 to 16 MB, hxccl is a little faster.
- At N = 8, sizes up to 4 MB behave the same way (see the raw output in `docs/bench/`).
- At N = 8, from 16 MB up, both libraries vary a lot from run to run. Four runs at 16 MB gave:
  - hxccl: 17.6, 17.7, 18.2 and 127 ms
  - MPI: 61, 67, 85 and 98 ms
- A Debug build (no `-O`) made hxccl 15 to 20× slower. That is why the build now defaults to Release.

Not verified:

- **Why N = 8 is unstable.** Eight ranks on 4 performance + 4 efficiency cores is one candidate, but it was not tested.
- **Why bandwidth drops at 64 MB.** Cache and memory effects are possible, but they were not measured.
- **Which algorithm OpenMPI picks at each size.** Its `coll/tuned` decision was not inspected.
