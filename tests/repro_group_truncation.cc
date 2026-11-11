// E-I2 (Mac review of E): a plain truncation in one communicator of a group must not cancel
// the device transfers of another communicator in the same group. Afterwards that other
// communicator must still deliver the right data (no message left over, no hang).
//   rank 1: Send 2000 ints on c1, Send 1000 ints on c2      (one group)
//   rank 0: Recv 1000 ints on c1 (too small), Recv 1000 on c2 (one group)
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
    for (int round = 0; round < 3; ++round) {
      int* a = dev_copy(std::vector<int>(2000, 10 + round));
      int* b = dev_copy(std::vector<int>(1000, 20 + round));
      hxcclGroupStart();
      if (rank == 1) {
        hxcclSend(a, 2000, hxcclInt32, 0, c1, nullptr);
        hxcclSend(b, 1000, hxcclInt32, 0, c2, nullptr);
      } else {
        hxcclRecv(a, 1000, hxcclInt32, 1, c1, nullptr);  // too small: truncation
        hxcclRecv(b, 1000, hxcclInt32, 1, c2, nullptr);
      }
      const hxcclResult_t r = hxcclGroupEnd();
      CHECK(rank == 1 ? r == hxcclSuccess : r == hxcclSystemError);
      if (rank == 0) CHECK(host_copy(b, 1000) == std::vector<int>(1000, 20 + round));
      // the next transfer on c2 must line up
      int* d = dev_copy(std::vector<int>(1000, rank == 1 ? 30 + round : -1));
      const hxcclResult_t r2 = rank == 1 ? hxcclSend(d, 1000, hxcclInt32, 0, c2, nullptr)
                                         : hxcclRecv(d, 1000, hxcclInt32, 1, c2, nullptr);
      CHECK(r2 == hxcclSuccess);
      if (rank == 0) {
        const bool same = host_copy(d, 1000) == std::vector<int>(1000, 30 + round);
        if (!same) std::fprintf(stderr, "[rank 0] round %d: next c2 transfer is wrong\n", round);
        CHECK(same);
      }
      for (int* p : {a, b, d}) cudaFree(p);
    }
  }
  cudaFree(warm);
  hxcclCommDestroy(c1);
  hxcclCommDestroy(c2);
  return finish();
}
