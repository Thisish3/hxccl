// Every collective on device buffers must give the host path's bits for the same input.
// Each helper runs the call on host vectors and on device copies, then compares bytes.
#include <random>

#include "check.h"
#include "gpu_util.h"
#include "reduce.h"

namespace {
int rank_, n_;
using Bytes = std::vector<unsigned char>;

Bytes input(size_t bytes, unsigned salt) {
  std::mt19937 g(1000 * rank_ + salt);
  Bytes v(bytes);
  for (auto& b : v) b = static_cast<unsigned char>(g() & 0x3f);  // small values, no NaN
  return v;
}
void expect_same(const unsigned char* dev, const Bytes& want, const char* what, size_t a,
                 int dt) {
  const bool same = host_copy(dev, want.size()) == want;
  if (!same)
    std::fprintf(stderr, "[rank %d] %s mismatch (size %zu, dt %d)\n", rank_, what, a, dt);
  CHECK(same);
}
void free_all(std::initializer_list<void*> ps) {
  for (void* p : ps)
    if (p) cudaFree(p);
}

void allgather(hxcclComm_t c, size_t per, hxcclDataType_t dt, bool in_place) {
  const size_t es = dtype_size(dt), block = per * es;
  Bytes in = input(block, 1 + dt), hout(block * n_);
  if (in_place) std::copy(in.begin(), in.end(), hout.begin() + rank_ * block);
  CHECK(hxcclAllGather(in_place ? hout.data() + rank_ * block : in.data(), hout.data(), per, dt,
                       c, nullptr) == hxcclSuccess);
  Bytes start(block * n_);
  if (in_place) std::copy(in.begin(), in.end(), start.begin() + rank_ * block);
  unsigned char* dout = dev_copy(start);
  unsigned char* din = in_place ? nullptr : dev_copy(in);
  CHECK(hxcclAllGather(in_place ? dout + rank_ * block : din, dout, per, dt, c, nullptr) ==
        hxcclSuccess);
  expect_same(dout, hout, "allgather", per, dt);
  free_all({din, dout});
}

void reducescatter(hxcclComm_t c, size_t per, hxcclDataType_t dt, hxcclRedOp_t op,
                   bool in_place) {
  const size_t es = dtype_size(dt), block = per * es;
  Bytes in = input(block * n_, 2 + dt + 16 * op), hin = in, hout(block);
  CHECK(hxcclReduceScatter(hin.data(), in_place ? hin.data() + rank_ * block : hout.data(), per,
                           dt, op, c, nullptr) == hxcclSuccess);
  if (in_place) hout.assign(hin.begin() + rank_ * block, hin.begin() + (rank_ + 1) * block);
  unsigned char* din = dev_copy(in);
  unsigned char* dout = in_place ? nullptr : dev_copy(Bytes(block));
  CHECK(hxcclReduceScatter(din, in_place ? din + rank_ * block : dout, per, dt, op, c,
                           nullptr) == hxcclSuccess);
  expect_same(in_place ? din + rank_ * block : dout, hout, "reducescatter", per, dt);
  free_all({din, dout});
}

void broadcast(hxcclComm_t c, size_t count, hxcclDataType_t dt, int root, bool in_place) {
  const size_t bytes = count * dtype_size(dt);
  Bytes in = input(bytes, 3 + dt), hout(bytes, 0xee);
  if (in_place && rank_ == root) hout = in;
  CHECK(hxcclBroadcast(in_place ? hout.data() : in.data(), hout.data(), count, dt, root, c,
                       nullptr) == hxcclSuccess);
  unsigned char* dout = dev_copy(in_place && rank_ == root ? in : Bytes(bytes, 0xee));
  unsigned char* din = in_place ? nullptr : dev_copy(in);
  if (in_place)
    CHECK(hxcclBcast(dout, count, dt, root, c, nullptr) == hxcclSuccess);
  else
    CHECK(hxcclBroadcast(din, dout, count, dt, root, c, nullptr) == hxcclSuccess);
  expect_same(dout, hout, "broadcast", count, dt);
  free_all({din, dout});
}

void reduce(hxcclComm_t c, size_t count, hxcclDataType_t dt, hxcclRedOp_t op, int root,
            bool null_nonroot) {
  const size_t bytes = count * dtype_size(dt);
  const bool is_root = rank_ == root, give_recv = is_root || !null_nonroot;
  Bytes in = input(bytes, 4 + dt + 16 * op), hout(bytes);
  CHECK(hxcclReduce(in.data(), give_recv ? hout.data() : nullptr, count, dt, op, root, c,
                    nullptr) == hxcclSuccess);
  unsigned char* din = dev_copy(in);
  unsigned char* dout = give_recv ? dev_copy(Bytes(bytes)) : nullptr;
  CHECK(hxcclReduce(din, dout, count, dt, op, root, c, nullptr) == hxcclSuccess);
  if (is_root) expect_same(dout, hout, "reduce", count, dt);
  free_all({din, dout});
}

void gather(hxcclComm_t c, size_t per, hxcclDataType_t dt, int root, bool null_nonroot) {
  const size_t block = per * dtype_size(dt);
  const bool is_root = rank_ == root, give_recv = is_root || !null_nonroot;
  Bytes in = input(block, 5 + dt), hout(block * n_);
  CHECK(hxcclGather(in.data(), give_recv ? hout.data() : nullptr, per, dt, root, c, nullptr) ==
        hxcclSuccess);
  unsigned char* din = dev_copy(in);
  unsigned char* dout = give_recv ? dev_copy(Bytes(block * n_)) : nullptr;
  CHECK(hxcclGather(din, dout, per, dt, root, c, nullptr) == hxcclSuccess);
  if (is_root) expect_same(dout, hout, "gather", per, dt);
  free_all({din, dout});
}

void scatter(hxcclComm_t c, size_t per, hxcclDataType_t dt, int root, bool null_nonroot) {
  const size_t block = per * dtype_size(dt);
  const bool is_root = rank_ == root, give_send = is_root || !null_nonroot;
  Bytes in = input(block * n_, 6 + dt), hout(block);
  CHECK(hxcclScatter(give_send ? in.data() : nullptr, hout.data(), per, dt, root, c, nullptr) ==
        hxcclSuccess);
  unsigned char* din = give_send ? dev_copy(in) : nullptr;
  unsigned char* dout = dev_copy(Bytes(block));
  CHECK(hxcclScatter(din, dout, per, dt, root, c, nullptr) == hxcclSuccess);
  expect_same(dout, hout, "scatter", per, dt);
  free_all({din, dout});
}

void alltoall(hxcclComm_t c, size_t per, hxcclDataType_t dt, bool in_place) {
  const size_t block = per * dtype_size(dt);
  Bytes in = input(block * n_, 7 + dt), hin = in, hout(block * n_);
  CHECK(hxcclAlltoAll(hin.data(), in_place ? hin.data() : hout.data(), per, dt, c, nullptr) ==
        hxcclSuccess);
  if (in_place) hout = hin;
  unsigned char* din = dev_copy(in);
  unsigned char* dout = in_place ? nullptr : dev_copy(Bytes(block * n_));
  CHECK(hxcclAllToAll(din, in_place ? din : dout, per, dt, c, nullptr) == hxcclSuccess);
  expect_same(in_place ? din : dout, hout, "alltoall", per, dt);
  free_all({din, dout});
}

void alltoallv(hxcclComm_t c) {  // counts (rank + peer) % 3 * 1000
  std::vector<size_t> sc(n_), sd(n_), rc(n_), rd(n_);
  size_t st = 0, rt = 0;
  for (int j = 0; j < n_; ++j) {
    sc[j] = (rank_ + j) % 3 * 1000;
    rc[j] = (j + rank_) % 3 * 1000;
    sd[j] = st;
    rd[j] = rt;
    st += sc[j];
    rt += rc[j];
  }
  Bytes in = input(st * 4, 8), hout(rt * 4);
  CHECK(hxcclAllToAllv(in.data(), sc.data(), sd.data(), hout.data(), rc.data(), rd.data(),
                       hxcclInt32, c, nullptr) == hxcclSuccess);
  unsigned char* din = dev_copy(in);
  unsigned char* dout = dev_copy(Bytes(rt * 4));
  CHECK(hxcclAllToAllv(din, sc.data(), sd.data(), dout, rc.data(), rd.data(), hxcclInt32, c,
                       nullptr) == hxcclSuccess);
  expect_same(dout, hout, "alltoallv", rt, hxcclInt32);
  free_all({din, dout});
}

void group_of_two_allreduces(hxcclComm_t c) {
  std::vector<float> a(5000, float(rank_ + 1)), b(3, float(2 * rank_));
  float* da = dev_copy(a);
  float* db = dev_copy(b);
  hxcclGroupStart();
  hxcclAllReduce(da, da, a.size(), hxcclFloat32, hxcclSum, c, nullptr);
  hxcclAllReduce(db, db, b.size(), hxcclFloat32, hxcclMax, c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(host_copy(da, a.size()) == std::vector<float>(a.size(), float(n_ * (n_ + 1) / 2)));
  CHECK(host_copy(db, b.size()) == std::vector<float>(b.size(), float(2 * (n_ - 1))));
  free_all({da, db});
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  const hxcclDataType_t types[] = {hxcclFloat32, hxcclInt32, hxcclBfloat16};
  for (size_t count : {size_t(0), size_t(1), size_t(1000), size_t(4097), size_t(1) << 20})
    for (hxcclDataType_t dt : types) {
      const bool big = count == (size_t(1) << 20);
      for (bool ip : {false, true}) {
        allgather(c, big ? count / n_ : count, dt, ip);
        reducescatter(c, big ? count / n_ : count, dt, hxcclSum, ip);
        alltoall(c, big ? count / n_ : count, dt, ip);
      }
      reducescatter(c, count / (big ? n_ : 1), dt, hxcclAvg, false);
      for (int root : {0, n_ - 1}) {
        broadcast(c, count, dt, root, false);
        broadcast(c, count, dt, root, true);
        for (bool null_nonroot : {false, true}) {
          reduce(c, count, dt, hxcclSum, root, null_nonroot);
          gather(c, big ? count / n_ : count, dt, root, null_nonroot);
          scatter(c, big ? count / n_ : count, dt, root, null_nonroot);
        }
        reduce(c, count, dt, hxcclAvg, root, false);
      }
    }
  broadcast(c, 600000, hxcclFloat32, 0, false);  // > kChainMinBytes: the chain schedule
  reduce(c, 600000, hxcclFloat32, hxcclSum, n_ - 1, true);
  alltoallv(c);
  group_of_two_allreduces(c);
  hxcclCommDestroy(c);
  return finish();
}
