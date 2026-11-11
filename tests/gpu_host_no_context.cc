// KNOWN_ISSUES #26: in a HXCCL_CUDA build, a program that only passes host buffers must not
// create a CUDA context. Checked with the driver API, which creates none itself.
// The program then switches to device buffers, and the device path must still work.
#include <cuda.h>

#include <vector>

#include "check.h"
#include "gpu_util.h"

namespace {
bool any_primary_context() {
  if (cuInit(0) != CUDA_SUCCESS) return false;
  int n = 0;
  cuDeviceGetCount(&n);
  for (int i = 0; i < n; ++i) {
    CUdevice d;
    unsigned flags = 0;
    int active = 0;
    if (cuDeviceGet(&d, i) == CUDA_SUCCESS &&
        cuDevicePrimaryCtxGetState(d, &flags, &active) == CUDA_SUCCESS && active)
      return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  hxcclComm_t c = world_comm();

  // host-only calls, including a group and a rooted collective
  std::vector<int> h(1000, rank + 1), want(1000, n * (n + 1) / 2), g(1000 * n);
  CHECK(hxcclAllReduce(h.data(), h.data(), h.size(), hxcclInt32, hxcclSum, c, nullptr) == 0);
  CHECK(h == want);
  CHECK(hxcclAllGather(h.data(), g.data(), h.size(), hxcclInt32, c, nullptr) == 0);
  int out = rank, in = -1;
  hxcclGroupStart();
  hxcclSend(&out, 1, hxcclInt32, (rank + 1) % n, c, nullptr);
  hxcclRecv(&in, 1, hxcclInt32, (rank - 1 + n) % n, c, nullptr);
  CHECK(hxcclGroupEnd() == 0);
  CHECK(in == (rank - 1 + n) % n);
  CHECK(!any_primary_context());

  // the device path still starts once the program uses the GPU
  cudaSetDevice(0);
  float* d = dev_copy(std::vector<float>(4096, 1.f));
  CHECK(hxcclAllReduce(d, d, 4096, hxcclFloat32, hxcclSum, c, nullptr) == hxcclSuccess);
  CHECK(host_copy(d, 4096) == std::vector<float>(4096, float(n)));
  cudaFree(d);
  hxcclCommDestroy(c);
  return finish();
}
