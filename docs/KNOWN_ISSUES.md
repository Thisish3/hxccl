# Known issues

Open problems, deferred review findings and unexplained observations. Update this file whenever something is deferred or resolved; move resolved items to the bottom with the commit that fixed them.

## Open

### Behavior

| # | Issue | Where | Why deferred | Revisit |
|---|---|---|---|---|
| 1 | **AllToAllv's argument check runs at call time, even inside a group.** Ranks first agree on argument validity with one `MPI_Allreduce`. If ranks put AllToAllv calls on two communicators into a group in different orders, hxccl deadlocks where NCCL would not. | `src/api_alltoall.cc` | Moving the agreement to GroupEnd changes the group design | When groups get async execution (B) |
| 2 | **Argument errors that only some ranks see leave the others waiting.** This covers a bad color or config passed to `CommSplit` or `InitRankConfig` on some ranks only, and a bad `root` on some ranks only. | `src/comm.cc`, `src/api*.cc` | User error, and NCCL behaves the same. An agreement round on every call would cost one extra message. | If a real user hits it |
| 3 | **`async_error` stays set after a user-caused failure**, such as a Recv buffer smaller than the message (`MPI_ERR_TRUNCATE`), even though the communicator still works. | `src/comm.cc` (`comm_fail`) | Matches the spec text ("last failure") | C: decide what PyTorch's watchdog should treat as fatal |
| 4 | **fp8 overflow is undefined.** e4m3 overflow becomes NaN and IEEE-style types become inf. NCCL's mode (`__nv_fp8_e4m3(__half)`) has not been checked. | `src/fp.h` | The CUDA header is only in the toolkit | B: run the same inputs through NCCL on the 3060 Ti |
| 5 | **RCCL `AllReduceWithBias` is not implemented.** The kernel stores `op(result, acc)` when it writes the output, but how often `acc` is added along the ring could not be pinned down from the source alone. | — | Needs RCCL's actual output | D (MI300X) |
| 6 | **The PreMulSum handle tag wraps after 32766 communicators in one process**, after which a handle from another communicator could be accepted again. | `src/comm.cc` (`make_comm`) | Needs that many communicators | — |
| 7 | **Out-of-range enum values are cast in tests** (for example `static_cast<hxcclDataType_t>(42)`), which UBSan reports as undefined behavior. | `tests/test_errors.cc`, `src/reduce.cc` | The library rejects them correctly, and only the cast itself is UB | When adding a UBSan CI build |

### Performance and implementation

| # | Issue | Where |
|---|---|---|
| 8 | `ring_reducescatter` copies the whole input even when the call is not in place | `src/algo_ring.cc` |
| 9 | Ring sends zero-byte messages when count < N | `src/algo_ring.cc` |
| 10 | All collectives share tag 0 and rely on MPI's per-(source, tag, comm) ordering. Send/Recv use tag 1. | `src/transport.h` |
| 11 | `hxccl_perf` hard-codes the 64 KB rd/ring boundary in its "algo" column and ignores return codes | `bench/hxccl_perf.cc` |

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

## Resolved

| Issue | Fixed in |
|---|---|
| A PreMulSum op from another communicator was accepted when that communicator had an op at the same index | c566754 |
| In-place AlltoAll / AllToAllv corrupted data | c566754 |
| AllToAllv ignored Finalize | c566754 |
| Integer Avg narrowed the divisor to the element type (uint8 at 256 ranks divided by zero) | c566754 |
| The return code of the AllToAllv agreement `MPI_Allreduce` was ignored | 4deac21 |
| `cancel_all` used `MPI_Cancel` on send requests, which MPI-4 deprecates | 4deac21 |
| The fp8 comment said "via half" while the code computes in float; spec 6 promised InvalidUsage after Abort although Abort frees the handle | 4deac21 |
