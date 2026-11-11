// I2 (Mac review): when device-path setup fails on one rank only, the other ranks must not
// hang, and later device calls must not drift apart. Run as MPMD with a bad
// HXCCL_IPC_CHUNK on rank 0 only (its cudaMalloc of the staging buffer fails).
// Every rank must get the same result code; if it is success, the data must be right.
#include "check.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  for (int round = 0; round < 2; ++round) {
    float* d = dev_copy(std::vector<float>(4096, 1.f));
    int r = hxcclAllReduce(d, d, 4096, hxcclFloat32, hxcclSum, c, nullptr);
    int rmin = r, rmax = r;
    MPI_Allreduce(MPI_IN_PLACE, &rmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) std::printf("round %d: result codes %d..%d\n", round, rmin, rmax);
    CHECK(rmin == rmax);
    if (r == hxcclSuccess) CHECK(host_copy(d, 4096) == std::vector<float>(4096, float(n)));
    cudaFree(d);
  }
  hxcclCommDestroy(c);
  return finish();
}
