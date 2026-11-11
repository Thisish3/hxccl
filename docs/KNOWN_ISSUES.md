# Known issues

Open problems, deferred review findings and unexplained observations. Update this file whenever something is deferred or resolved; move resolved items to the bottom with the commit that fixed them.

## Open

### Behavior

| # | Issue | Where | Why deferred | Revisit |
|---|---|---|---|---|
| 1 | **AllToAllv's argument check runs at call time, even inside a group.** Ranks first agree on argument validity with one `MPI_Allreduce`. If ranks put AllToAllv calls on two communicators into a group in different orders, hxccl deadlocks where NCCL would not. | `src/api_alltoall.cc` | Moving the agreement to GroupEnd changes the group design | When groups get async execution (moved on from B, see #16) |
| 2 | **Argument errors that only some ranks see leave the others waiting.** This covers a bad color or config passed to `CommSplit` or `InitRankConfig` on some ranks only, and a bad `root` on some ranks only. | `src/comm.cc`, `src/api*.cc` | User error, and NCCL behaves the same. An agreement round on every call would cost one extra message. | If a real user hits it |
| 3 | **`async_error` stays set after a user-caused failure**, such as a Recv buffer smaller than the message (`MPI_ERR_TRUNCATE`), even though the communicator still works. | `src/comm.cc` (`comm_fail`) | Matches the spec text ("last failure") | C: decide what PyTorch's watchdog should treat as fatal |
| 4 | **fp8 overflow differs from CUDA's default conversion.** hxccl turns e4m3 overflow into NaN (0x7f/0xff) and e5m2 overflow into inf (0x7c/0xfc). CUDA 12.2's `__nv_fp8_e4m3(float)` / `__nv_fp8_e5m2(float)` call `__nv_cvt_float_to_fp8(f, __NV_SATFINITE, ...)` and saturate to the largest finite value (e4m3 0x7e/0xfe = ±448, e5m2 0x7b/0xfb = ±57344), also for ±inf. NaN: e4m3 is the same (0x7f); for e5m2 hxccl gives 0x7e and CUDA 0x7f. Values in range match. Table from `experimental/fp8_overflow.cu` (2026-11-11, desktop): 448/449/464 same for both types; 480, 1000 and 57344 differ for e4m3 only; 1e30, ±inf and 61440/65536 differ for both. Whether NCCL's reduction kernels use this saturating conversion was not checked. The code is unchanged; the GPU path matches the CPU path, not CUDA. | `src/fp.h` | Spec 3.3: record, do not fix | When an fp8 user needs NCCL-equal overflow |
| 5 | **RCCL `AllReduceWithBias` is not implemented.** The kernel stores `op(result, acc)` when it writes the output, but how often `acc` is added along the ring could not be pinned down from the source alone. | — | Needs RCCL's actual output | D (MI300X) |
| 6 | **The PreMulSum handle tag wraps after 32766 communicators in one process**, after which a handle from another communicator could be accepted again. | `src/comm.cc` (`make_comm`) | Needs that many communicators | — |
| 7 | **Out-of-range enum values are cast in tests** (for example `static_cast<hxcclDataType_t>(42)`), which UBSan reports as undefined behavior. | `tests/test_errors.cc`, `src/reduce.cc` | The library rejects them correctly, and only the cast itself is UB | When adding a UBSan CI build |
| 16 | **Device calls return only when the work is done.** NCCL queues the work on the stream and returns; hxccl waits for the stream (`sync_ops`). | `src/run.h`, `src/memspace_cuda.cc` | Spec 3.5: simple and certain first | When a stream-ordered design replaces the MPI READY/DONE handshake |
| 17 | **A group cannot mix host and device Send/Recv** and returns `hxcclInvalidUsage`. The host transport blocks in `MPI_Waitall` while the device transport polls, so mixing them could stall a peer. | `src/group.cc` | Avoids a possible deadlock | When both transports progress from one loop |
| 18 | **All ranks must pass the same kind of buffer (host or device) to a collective.** The first device call on a communicator runs a collective setup step (`MPI_Allgather` of IPC handles). There is no check, so a rank that passes host buffers while the others pass device buffers hangs the call. Count-0 calls with NULL buffers go down the host path. | `src/memspace_cuda.cc` | Checking would cost an agreement round on every call | If a real user hits it |
| 19 | **The PyTorch backend needs `mpirun`.** Communicators come from `MPI_COMM_WORLD`, so `torchrun` processes cannot find each other. | `python/hxccl_torch` | Needs a socket bootstrap | E |
| 20 | **Every PyTorch `Work` is complete when returned**, `async_op=True` included. Failures are raised as exceptions from the call itself, not through `Work.wait()`. | `python/hxccl_torch/csrc/backend.cpp` | Follows from #16 | With #16 |

### Performance and implementation

| # | Issue | Where |
|---|---|---|
| 8 | `ring_reducescatter` copies the whole input even when the call is not in place | `src/algo_ring.cc` |
| 9 | Ring sends zero-byte messages when count < N | `src/algo_ring.cc` |
| 10 | All collectives share tag 0 and rely on MPI's per-(source, tag, comm) ordering. Send/Recv use tag 1. | `src/transport.h` |
| 11 | `hxccl_perf` hard-codes the 64 KB rd/ring boundary in its "algo" column and ignores return codes | `bench/hxccl_perf.cc` |
| 21 | The device path has a floor of about 3.1–3.6 ms per AllReduce at np2 (README, B). Measured parts: `cudaMallocAsync` on every call (median 0.31–0.37 ms), and one `cudaStreamSynchronize` per chunk copy (median 0.13–0.15 ms). The rest is mostly MPI READY/DONE waits, which were not traced. Why the pool does not reuse memory and why one synchronize costs that much on WSL2 is not confirmed. | `src/memops_device.cc`, `src/transport_ipc.cc` |
| 22 | `IpcTransport` moves one chunk per round trip, with one staging buffer per peer pair. Chunks are not pipelined, and no CUDA IPC events are used. | `src/transport_ipc.cc` |
| 23 | `hxccl_perf -d` leaves the MPI column empty. Whether apt's OpenMPI 4.1.2 is CUDA-aware was not checked. | `bench/hxccl_perf.cc` |
| 27 | `IpcTransport::waitall` polls `MPI_Test` in a loop without sleeping, so a waiting rank keeps a CPU core busy. CPU use was not measured. | `src/transport_ipc.cc` |

### Tests

| # | Issue | Where |
|---|---|---|
| 12 | The Abort test only covers idle peers. Spec 7 asks for Abort to return while peers are blocked in another collective. The implementation makes no MPI calls, so it satisfies the requirement, but no test shows it. | `tests/test_comm.cc` |

### Unexplained observations

| # | Observation | Status |
|---|---|---|
| 13 | At N = 8, allreduce of 16 MB and above varies a lot from run to run, for both hxccl and OpenMPI (README). | Cause not confirmed. One candidate is 8 ranks on 4 performance + 4 efficiency cores; it was not tested. |
| 14 | The 64 KB and 1 MB rows of the A2 benchmark differ a lot from OpenMPI (Reduce 1.59×, Scatter 0.45× and 1.48×). | Not investigated. OpenMPI's algorithm choice was not inspected. |
| 15 | A Send→Recv demo program outside the test suite segfaulted (address 0x0) 4 times in a row on 2026-11-08, right after hung runs were killed. 35 later runs (20 alone, 15 alongside ctest) did not reproduce it. | Cause not identified |
| 24 | The X experiment (`experimental/device_allreduce.cu`, a spinning cross-process kernel) hung on its 2nd np4 run on 2026-11-11. After that the GPU stopped answering in WSL and on Windows until `wsl --shutdown`. | Cause not confirmed (experimental/README.md) |
| 25 | Nsight Compute's multi-pass SpeedOfLight replay of the reduction kernel ran at a DRAM clock of 0.81 GHz (299 µs). A single-pass run with metrics only ran at 6.79 GHz (31 µs). | Cause of the lowered memory clock not confirmed |

## Resolved

| Issue | Fixed in |
|---|---|
| A PreMulSum op from another communicator was accepted when that communicator had an op at the same index | c566754 |
| In-place AlltoAll / AllToAllv corrupted data | c566754 |
| AllToAllv ignored Finalize | c566754 |
| Integer Avg narrowed the divisor to the element type (uint8 at 256 ranks divided by zero) | c566754 |
| The return code of the AllToAllv agreement `MPI_Allreduce` was ignored | 4deac21 |
| `cancel_all` used `MPI_Cancel` on send requests, which MPI-4 deprecates | 4deac21 |
| The fp8 comment said "via half" while the code computes in float (its later "both exact" justification was also wrong; an exhaustive check now backs the claim); spec 6 promised InvalidUsage after Abort although Abort frees the handle | 4deac21 |
| #26: the extra GPU memory (about 115 MiB per rank) in host-only runs of the `HXCCL_CUDA` build came from `hxccl_perf` calling `cudaSetDevice(0)` unconditionally, not from the library as first written. Measured: `cudaPointerGetAttributes` on a host pointer leaves the primary context inactive (CUDA 12.2), and with `hxccl_perf` fixed, host np4 runs stay at idle GPU memory (441 MiB) with the library before and after. The new test `gpu_host_no_context` checks that host-only calls create no context. | this commit |
