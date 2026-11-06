#include <cstdint>
#include <vector>

#include "check.h"
#include "chunks.h"
#include "hxccl.h"

namespace {

int rank_, nranks_;

MPI_Datatype mpi_type(hxcclDataType_t dt) {
  return dt == hxcclInt32 ? MPI_INT : dt == hxcclFloat32 ? MPI_FLOAT : MPI_DOUBLE;
}
MPI_Op mpi_op(hxcclRedOp_t op) {
  return op == hxcclSum ? MPI_SUM : op == hxcclProd ? MPI_PROD : op == hxcclMax ? MPI_MAX
                                                                                  : MPI_MIN;
}

// Exactly representable inputs so every reduction order gives the same bits.
template <typename T>
std::vector<T> input(size_t count, hxcclRedOp_t op) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i)
    v[i] = op == hxcclProd ? T((i + rank_) % 3 == 0 ? 2 : 1)
                           : T((rank_ + 1) * static_cast<int>(i % 5 + 1));
  return v;
}

template <typename T>
void check_one(hxcclComm_t c, size_t count, hxcclDataType_t dt, hxcclRedOp_t op,
               bool in_place) {
  std::vector<T> in = input<T>(count, op), want(count), got(count);
  MPI_Allreduce(in.data(), want.data(), static_cast<int>(count), mpi_type(dt), mpi_op(op),
                MPI_COMM_WORLD);
  hxcclResult_t r;
  if (in_place) {
    got = in;
    r = hxcclAllReduce(got.data(), got.data(), count, dt, op, c, nullptr);
  } else {
    r = hxcclAllReduce(in.data(), got.data(), count, dt, op, c, nullptr);
  }
  CHECK(r == hxcclSuccess);
  CHECK(got == want);
}

void ring_traffic(hxcclComm_t c, size_t count) {
  std::vector<float> in(count, 1.0f), out(count);
  hxcclCommResetStats(c);
  CHECK(hxcclAllReduce(in.data(), out.data(), count, hxcclFloat32, hxcclSum, c, nullptr) ==
        hxcclSuccess);
  uint64_t bytes = 0, msgs = 0;
  hxcclCommGetStats(c, &bytes, &msgs);
  uint64_t want = 0;
  for (int s = 0; s < nranks_ - 1; ++s) {
    want += chunk_of(count, nranks_, ((rank_ - s) % nranks_ + nranks_) % nranks_).len;
    want += chunk_of(count, nranks_, ((rank_ + 1 - s) % nranks_ + nranks_) % nranks_).len;
  }
  CHECK(bytes == want * sizeof(float));
  // count == 0 returns before any communication (spec 5)
  CHECK(msgs == (count == 0 ? 0u : static_cast<uint64_t>(2 * (nranks_ - 1))));
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks_);
  setenv("HXCCL_ALGO", "ring", 1);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  const size_t counts[] = {0, 1, 3, 7, 1000, 4099};
  const hxcclRedOp_t ops[] = {hxcclSum, hxcclProd, hxcclMax, hxcclMin};
  for (size_t n : counts)
    for (hxcclRedOp_t op : ops) {
      check_one<int32_t>(c, n, hxcclInt32, op, false);
      check_one<float>(c, n, hxcclFloat32, op, false);
      check_one<double>(c, n, hxcclFloat64, op, n == 1000);
    }
  for (size_t n : counts) ring_traffic(c, n);
  hxcclCommDestroy(c);
  return finish();
}
