// I3 (Mac review): PreMulSum with hxcclScalarDevice and a real device pointer (cudaMalloc'd
// scalar), the normal NCCL usage. Must not crash, and must scale by the scalar's value at
// the time of the call.
#include "check.h"
#include "gpu_util.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  float* scalar = dev_copy(std::vector<float>(1, 0.5f));
  hxcclRedOp_t op;
  CHECK(hxcclRedOpCreatePreMulSum(&op, scalar, hxcclFloat32, hxcclScalarDevice, c) ==
        hxcclSuccess);
  float* d = dev_copy(std::vector<float>(1000, 4.f));
  CHECK(hxcclAllReduce(d, d, 1000, hxcclFloat32, op, c, nullptr) == hxcclSuccess);
  CHECK(host_copy(d, 1000) == std::vector<float>(1000, 2.f * n));
  // the value is read at call time: change it and call again
  cudaMemcpy(scalar, std::vector<float>(1, 0.25f).data(), sizeof(float), cudaMemcpyHostToDevice);
  float* e = dev_copy(std::vector<float>(1000, 4.f));
  CHECK(hxcclAllReduce(e, e, 1000, hxcclFloat32, op, c, nullptr) == hxcclSuccess);
  CHECK(host_copy(e, 1000) == std::vector<float>(1000, 1.f * n));
  // host buffers with the same device scalar
  std::vector<float> h(1000, 4.f);
  CHECK(hxcclAllReduce(h.data(), h.data(), 1000, hxcclFloat32, op, c, nullptr) == hxcclSuccess);
  CHECK(h == std::vector<float>(1000, 1.f * n));
  hxcclRedOpDestroy(op, c);
  for (float* p : {scalar, d, e}) cudaFree(p);
  hxcclCommDestroy(c);
  return finish();
}
