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
- **AllToAllv does one extra `MPI_Allreduce`** before sending. Block sizes differ per rank, so ranks first agree on whether any argument is invalid, and then all of them reject together instead of leaving the others waiting.
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
