#include <cstdint>
#include <vector>

#include "check.h"
#include "chunks.h"
#include "hxccl.h"
#include "select.h"

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

Algo g_algo = Algo::Ring;
bool g_auto = false;

hxcclResult_t run(hxcclComm_t c, const void* s, void* r, size_t n, hxcclDataType_t dt,
                  hxcclRedOp_t op) {
  return g_auto ? hxcclAllReduce(s, r, n, dt, op, c, nullptr)
                : allreduce_with(c, g_algo, s, r, n, dt, op);
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
    r = run(c, got.data(), got.data(), count, dt, op);
  } else {
    r = run(c, in.data(), got.data(), count, dt, op);
  }
  CHECK(r == hxcclSuccess);
  CHECK(got == want);
}

void ring_traffic(hxcclComm_t c, size_t count) {
  std::vector<float> in(count, 1.0f), out(count);
  hxcclCommResetStats(c);
  CHECK(allreduce_with(c, Algo::Ring, in.data(), out.data(), count, hxcclFloat32,
                       hxcclSum) == hxcclSuccess);
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

void rd_traffic(hxcclComm_t c, size_t count) {
  if (nranks_ & (nranks_ - 1)) return;  // the log2 formula holds for powers of two
  std::vector<float> in(count, 1.0f), out(count);
  hxcclCommResetStats(c);
  CHECK(allreduce_with(c, Algo::Rd, in.data(), out.data(), count, hxcclFloat32, hxcclSum) ==
        hxcclSuccess);
  uint64_t bytes = 0, msgs = 0, steps = 0;
  hxcclCommGetStats(c, &bytes, &msgs);
  for (int m = 1; m < nranks_; m <<= 1) ++steps;
  if (count == 0) steps = 0;
  CHECK(msgs == steps);
  CHECK(bytes == steps * count * sizeof(float));
}

void env_forces_rd(hxcclComm_t c) {
  setenv("HXCCL_ALGO", "rd", 1);
  CHECK(choose_allreduce(64 << 20) == Algo::Rd);
  setenv("HXCCL_ALGO", "ring", 1);
  CHECK(choose_allreduce(8) == Algo::Ring);
  unsetenv("HXCCL_ALGO");
  CHECK(choose_allreduce(8) == Algo::Rd && choose_allreduce(64 << 20) == Algo::Ring);
  (void)c;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  const size_t counts[] = {0, 1, 3, 7, 1000, 4099};
  const hxcclRedOp_t ops[] = {hxcclSum, hxcclProd, hxcclMax, hxcclMin};
  for (int mode = 0; mode < 3; ++mode) {
    g_auto = mode == 2;
    g_algo = mode == 0 ? Algo::Ring : Algo::Rd;
    for (size_t n : counts)
      for (hxcclRedOp_t op : ops) {
        check_one<int32_t>(c, n, hxcclInt32, op, false);
        check_one<float>(c, n, hxcclFloat32, op, false);
        check_one<double>(c, n, hxcclFloat64, op, n == 1000);
      }
  }
  for (size_t n : counts) {
    ring_traffic(c, n);
    rd_traffic(c, n);
  }
  env_forces_rd(c);
  hxcclCommDestroy(c);
  return finish();
}
