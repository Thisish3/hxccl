// I4 (Mac review): the first device Send/Recv made by only some ranks must not hang.
// np4: only ranks 0 and 1 use device buffers, and do so before any other device call.
// Every rank has a CUDA context before the communicator is created.
#include <cstdlib>

#include "check.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  // REPRO_NO_CONTEXT_RANK=k: rank k creates no CUDA context before the communicator (E-I3)
  const char* nc = std::getenv("REPRO_NO_CONTEXT_RANK");
  if (!nc || std::atoi(nc) != rank) {
    cudaSetDevice(0);
    cudaFree(nullptr);  // make sure the CUDA context exists
  }
  hxcclComm_t c = world_comm();
  if (n >= 2 && rank < 2) {
    std::vector<int> mine(1000, rank == 0 ? 7 : -1);
    int* d = dev_copy(mine);
    const hxcclResult_t r = rank == 0 ? hxcclSend(d, 1000, hxcclInt32, 1, c, nullptr)
                                      : hxcclRecv(d, 1000, hxcclInt32, 0, c, nullptr);
    if (r != hxcclSuccess) std::fprintf(stderr, "[rank %d] %s\n", rank, hxcclGetLastError(c));
    CHECK(r == hxcclSuccess);
    CHECK(host_copy(d, 1000) == std::vector<int>(1000, 7));
    cudaFree(d);
  }
  hxcclCommDestroy(c);
  return finish();
}
