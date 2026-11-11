// I5 (Mac review): a device-transfer failure on one rank must not leave its peers waiting
// forever. Run as MPMD with HXCCL_TEST_FAIL_DEVICE_COPY=1 on rank 0 only, which makes that
// rank's device copies fail. Every rank must return (the test's TIMEOUT catches a hang),
// and every rank must report a failure.
#include "check.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  float* d = dev_copy(std::vector<float>(1 << 16, 1.f));
  const hxcclResult_t r = hxcclAllReduce(d, d, 1 << 16, hxcclFloat32, hxcclSum, c, nullptr);
  std::printf("rank %d: result %d (%s)\n", rank, int(r), hxcclGetLastError(c));
  if (n > 1) CHECK(r != hxcclSuccess);
  cudaFree(d);
  hxcclCommDestroy(c);
  return finish();
}
