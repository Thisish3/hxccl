// Device reductions must match the CPU path bit for bit (spec 3.3). Each rank runs the
// whole check on its own; MPI only gathers the verdict.
#include <cuda_runtime.h>

#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "gpu_reduce.h"
#include "reduce.h"

namespace {
const hxcclDataType_t kTypes[] = {hxcclInt8,     hxcclUint8,      hxcclInt32,     hxcclUint32,
                                  hxcclInt64,    hxcclUint64,     hxcclFloat16,   hxcclFloat32,
                                  hxcclFloat64,  hxcclBfloat16,   hxcclFloat8e4m3, hxcclFloat8e5m2};
std::vector<unsigned char> random_bytes(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::vector<unsigned char> v(n);
  for (auto& b : v) b = static_cast<unsigned char>(g());
  return v;
}
void* to_dev(const std::vector<unsigned char>& h) {
  void* d;
  cudaMalloc(&d, h.size() ? h.size() : 1);
  cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
  return d;
}
std::vector<unsigned char> from_dev(const void* d, size_t n) {
  std::vector<unsigned char> h(n);
  cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
  return h;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  cudaSetDevice(0);
  for (size_t count : {size_t(1), size_t(7), size_t(4097), size_t(1) << 20})
    for (hxcclDataType_t dt : kTypes) {
      const size_t bytes = count * dtype_size(dt);
      auto a = random_bytes(bytes, 1 + dt), b = random_bytes(bytes, 100 + dt);
      for (hxcclRedOp_t op : {hxcclSum, hxcclProd, hxcclMax, hxcclMin}) {
        auto want = a;
        reduce_into(want.data(), b.data(), count, dt, op);
        void *da = to_dev(a), *db = to_dev(b);
        CHECK(gpu_reduce(da, db, count, dt, op, nullptr) == cudaSuccess);
        const bool same = from_dev(da, bytes) == want;
        if (!same) std::fprintf(stderr, "reduce mismatch dt=%d op=%d count=%zu\n", dt, op, count);
        CHECK(same);
        cudaFree(da);
        cudaFree(db);
      }
      if (!is_integer(dt)) {  // PreMulSum / float Avg path
        unsigned char scalar[8];
        store_scalar(0.25, dt, scalar);
        std::vector<unsigned char> want(bytes);
        scale_into(want.data(), a.data(), count, dt, scalar);
        void *da = to_dev(a), *dd = to_dev(std::vector<unsigned char>(bytes));
        CHECK(gpu_scale(dd, da, count, dt, scalar, nullptr) == cudaSuccess);
        const bool same = from_dev(dd, bytes) == want;
        if (!same) std::fprintf(stderr, "scale mismatch dt=%d count=%zu\n", dt, count);
        CHECK(same);
        cudaFree(da);
        cudaFree(dd);
      } else {  // integer Avg path
        auto want = a;
        divide_trunc(want.data(), count, dt, 3);
        void* da = to_dev(a);
        CHECK(gpu_divide_trunc(da, count, dt, 3, nullptr) == cudaSuccess);
        const bool same = from_dev(da, bytes) == want;
        if (!same) std::fprintf(stderr, "divide mismatch dt=%d count=%zu\n", dt, count);
        CHECK(same);
        cudaFree(da);
      }
    }
  // dtype/op combinations the kernels do not take
  CHECK(gpu_scale(nullptr, nullptr, 1, hxcclInt32, "\0\0\0\0", nullptr) == cudaErrorInvalidValue);
  CHECK(gpu_divide_trunc(nullptr, 1, hxcclFloat32, 3, nullptr) == cudaErrorInvalidValue);
  CHECK(gpu_reduce(nullptr, nullptr, 0, hxcclFloat32, hxcclSum, nullptr) == cudaSuccess);
  return finish();
}
