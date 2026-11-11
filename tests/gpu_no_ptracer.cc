// torchrun processes do not call MPI_Init, and OpenMPI's MPI_Init is what allows other
// processes to ptrace this one (PR_SET_PTRACER). With Yama ptrace_scope=1, CUDA IPC open
// then fails with "invalid resource handle" (measured 2026-11-11, LEDGER E4). This test clears
// the setting after MPI_Init to recreate the torchrun condition: the device path must still
// work, by every rank falling back to the staged (through-host) transport together.
#include <sys/prctl.h>

#include <cstdio>

#include "check.h"
#include "comm.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  prctl(PR_SET_PTRACER, 0, 0, 0, 0);  // undo what MPI_Init did
  MPI_Barrier(MPI_COMM_WORLD);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  float* d = dev_copy(std::vector<float>(4096, float(rank + 1)));
  const hxcclResult_t r = hxcclAllReduce(d, d, 4096, hxcclFloat32, hxcclSum, c, nullptr);
  if (r != hxcclSuccess) std::fprintf(stderr, "[rank %d] %s\n", rank, hxcclGetLastError(c));
  CHECK(r == hxcclSuccess);
  CHECK(host_copy(d, 4096) == std::vector<float>(4096, float(n * (n + 1) / 2)));

  // where ptrace is restricted, the ranks must have agreed on the staged path
  int scope = 0;
  if (FILE* f = std::fopen("/proc/sys/kernel/yama/ptrace_scope", "r")) {
    if (std::fscanf(f, "%d", &scope) != 1) scope = 0;
    std::fclose(f);
  }
  const bool staged = c->dev_ops && static_cast<DeviceOps*>(c->dev_ops.get())->staged();
  if (rank == 0) std::printf("ptrace_scope=%d nranks=%d staged=%d\n", scope, n, int(staged));
  if (scope == 1 && n > 1) CHECK(staged);

  // a message larger than one IPC chunk, and a short receive, on whichever path is active
  const size_t big = (size_t(5) << 20) / 4;
  float* b = dev_copy(std::vector<float>(big, 1.f));
  CHECK(hxcclAllReduce(b, b, big, hxcclFloat32, hxcclSum, c, nullptr) == hxcclSuccess);
  CHECK(host_copy(b, big) == std::vector<float>(big, float(n)));
  cudaFree(b);
  cudaFree(d);
  hxcclCommDestroy(c);
  return finish();
}
