#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "fp.h"
#include "hxccl.h"
#include "reduce.h"
#include "select.h"

namespace {
int rank_, n_;

// Every partial sum/product stays exactly representable in every dtype, so the
// reduction order (ring vs rd) cannot change the bits.
double input(int r, size_t i, hxcclRedOp_t op, bool is_signed) {
  if (op == hxcclSum) return static_cast<double>((r + i) % 2);
  if (op == hxcclProd) return 1.0 + static_cast<double>((r + i) % 2);
  const int v = static_cast<int>((r * 3 + i) % 7);
  return is_signed ? v - 3 : v;
}

template <typename T>
void put(T x, void* out) {
  std::memcpy(out, &x, sizeof x);
}

// integer results wrap modulo the width, like the reduction does
void store(double v, hxcclDataType_t dt, void* out) {
  const int64_t i = static_cast<int64_t>(v);
  const float f = static_cast<float>(v);
  switch (dt) {
    case hxcclInt8: put(static_cast<int8_t>(i), out); break;
    case hxcclUint8: put(static_cast<uint8_t>(i), out); break;
    case hxcclInt32: put(static_cast<int32_t>(i), out); break;
    case hxcclUint32: put(static_cast<uint32_t>(i), out); break;
    case hxcclInt64: put(i, out); break;
    case hxcclUint64: put(static_cast<uint64_t>(i), out); break;
    case hxcclFloat16: put(fp::float_to_half(f), out); break;
    case hxcclBfloat16: put(fp::float_to_bf16(f), out); break;
    case hxcclFloat8e4m3: put(fp::float_to_e4m3(f), out); break;
    case hxcclFloat8e5m2: put(fp::float_to_e5m2(f), out); break;
    case hxcclFloat32: put(f, out); break;
    case hxcclFloat64: put(v, out); break;
    default: break;
  }
}

void check(hxcclComm_t c, Algo algo, hxcclDataType_t dt, hxcclRedOp_t op, size_t count) {
  const size_t es = dtype_size(dt);
  CHECK(es > 0);
  if (es == 0) return;
  const bool sgn = dt != hxcclUint8 && dt != hxcclUint32 && dt != hxcclUint64;
  std::vector<char> in(count * es), got(count * es), want(count * es);
  for (size_t i = 0; i < count; ++i) {
    store(input(rank_, i, op, sgn), dt, in.data() + i * es);
    double acc = input(0, i, op, sgn);
    for (int r = 1; r < n_; ++r) {
      const double x = input(r, i, op, sgn);
      acc = op == hxcclSum ? acc + x
            : op == hxcclProd ? acc * x
            : op == hxcclMax ? std::max(acc, x)
                             : std::min(acc, x);
    }
    store(acc, dt, want.data() + i * es);
  }
  CHECK(allreduce_with(c, algo, in.data(), got.data(), count, dt, op) == hxcclSuccess);
  if (got != want)
    std::fprintf(stderr, "dtype %d op %d algo %d count %zu\n", dt, op,
                 static_cast<int>(algo), count);
  CHECK(got == want);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  for (int d = 0; d < hxcclNumTypes; ++d)
    for (hxcclRedOp_t op : {hxcclSum, hxcclProd, hxcclMax, hxcclMin})
      for (Algo a : {Algo::Ring, Algo::Rd})
        for (size_t count : {size_t{1}, size_t{7}, size_t{4099}})
          check(c, a, static_cast<hxcclDataType_t>(d), op, count);
  hxcclCommDestroy(c);
  return finish();
}
