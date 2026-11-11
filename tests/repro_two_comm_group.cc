// I1 (Mac review): one group holding device Send/Recv on two communicators, in crossed order:
//   rank 0: Send c1 -> 1, Recv c2 <- 1      rank 1: Send c2 -> 0, Recv c1 <- 0
// Must finish (np2). Both communicators' device paths are set up first.
#include "check.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  cudaSetDevice(0);
  hxcclComm_t c1 = world_comm(), c2 = world_comm();
  float* warm = dev_copy(std::vector<float>(4, 1.f));
  CHECK(hxcclAllReduce(warm, warm, 4, hxcclFloat32, hxcclSum, c1, nullptr) == hxcclSuccess);
  CHECK(hxcclAllReduce(warm, warm, 4, hxcclFloat32, hxcclSum, c2, nullptr) == hxcclSuccess);
  if (n == 2) {
    const size_t count = 3000;
    int* s = dev_copy(std::vector<int>(count, 10 + rank));
    int* r = dev_copy(std::vector<int>(count, -1));
    const int peer = 1 - rank;
    hxcclGroupStart();
    if (rank == 0) {
      hxcclSend(s, count, hxcclInt32, peer, c1, nullptr);
      hxcclRecv(r, count, hxcclInt32, peer, c2, nullptr);
    } else {
      hxcclSend(s, count, hxcclInt32, peer, c2, nullptr);
      hxcclRecv(r, count, hxcclInt32, peer, c1, nullptr);
    }
    CHECK(hxcclGroupEnd() == hxcclSuccess);
    CHECK(host_copy(r, count) == std::vector<int>(count, 10 + peer));
    cudaFree(s);
    cudaFree(r);
  }
  cudaFree(warm);
  hxcclCommDestroy(c1);
  hxcclCommDestroy(c2);
  return finish();
}
