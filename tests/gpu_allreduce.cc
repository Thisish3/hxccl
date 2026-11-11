// AllReduce on device buffers must give the host path's bits and traffic counts.
#include <cstring>
#include <random>

#include "check.h"
#include "gpu_util.h"
#include "reduce.h"

namespace {
int rank_, n_;
std::vector<unsigned char> input(size_t bytes, unsigned salt) {
  std::mt19937 g(1000 * rank_ + salt);
  std::vector<unsigned char> v(bytes);
  for (auto& b : v) b = static_cast<unsigned char>(g() & 0x3f);  // small values, no NaN
  return v;
}
void same_as_host(hxcclComm_t c, size_t count, hxcclDataType_t dt, hxcclRedOp_t op,
                  bool in_place) {
  const size_t bytes = count * dtype_size(dt);
  auto in = input(bytes, dt * 16 + op);
  std::vector<unsigned char> hout(bytes);
  CHECK(hxcclAllReduce(in.data(), in_place ? (void*)in.data() : hout.data(), count, dt, op, c,
                       nullptr) == hxcclSuccess);
  if (in_place) hout = in;
  auto in2 = input(bytes, dt * 16 + op);
  unsigned char* din = dev_copy(in2);
  unsigned char* dout = in_place ? din : dev_copy(std::vector<unsigned char>(bytes));
  CHECK(hxcclAllReduce(din, dout, count, dt, op, c, nullptr) == hxcclSuccess);
  const bool same = host_copy(dout, bytes) == hout;
  if (!same)
    std::fprintf(stderr, "[rank %d] mismatch count=%zu dt=%d op=%d in_place=%d\n", rank_, count,
                 dt, op, in_place);
  CHECK(same);
  cudaFree(din);
  if (!in_place) cudaFree(dout);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  for (size_t count : {size_t(0), size_t(1), size_t(7), size_t(1000), size_t(4097),
                       size_t(1) << 20, (size_t(4) << 20) / 4 + 3})  // last: > 4 MiB of fp32
    for (hxcclDataType_t dt : {hxcclInt32, hxcclFloat32, hxcclFloat16, hxcclBfloat16,
                               hxcclFloat8e4m3, hxcclFloat64, hxcclUint8})
      for (hxcclRedOp_t op : {hxcclSum, hxcclMax, hxcclAvg})
        same_as_host(c, count, dt, op, false);
  same_as_host(c, 4097, hxcclFloat32, hxcclSum, true);  // in-place
  {  // traffic counters match the host path for the same call
    uint64_t hb0, hm0, hb1, hm1, db0, dm0, db1, dm1;
    std::vector<float> h(1 << 16, 1.f);
    hxcclCommGetStats(c, &hb0, &hm0);
    hxcclAllReduce(h.data(), h.data(), h.size(), hxcclFloat32, hxcclSum, c, nullptr);
    hxcclCommGetStats(c, &hb1, &hm1);
    float* d = dev_copy(std::vector<float>(1 << 16, 1.f));
    hxcclCommGetStats(c, &db0, &dm0);
    hxcclAllReduce(d, d, 1 << 16, hxcclFloat32, hxcclSum, c, nullptr);
    hxcclCommGetStats(c, &db1, &dm1);
    CHECK(hb1 - hb0 == db1 - db0);
    CHECK(hm1 - hm0 == dm1 - dm0);
    cudaFree(d);
  }
  {  // mixed host/device buffers are rejected on every rank
    std::vector<float> h(16, 1.f);
    float* d = dev_copy(h);
    CHECK(hxcclAllReduce(h.data(), d, 16, hxcclFloat32, hxcclSum, c, nullptr) ==
          hxcclInvalidArgument);
    cudaFree(d);
  }
  {  // host_device_host: the lazy device init must not break later host calls
    std::vector<int> h(64, rank_ + 1), want(64, n_ * (n_ + 1) / 2);
    CHECK(hxcclAllReduce(h.data(), h.data(), 64, hxcclInt32, hxcclSum, c, nullptr) == 0);
    CHECK(h == want);
  }
  hxcclCommDestroy(c);
  return finish();
}
