#include <cstdint>
#include <cstring>
#include <vector>

#include "check.h"
#include "hxccl.h"

namespace {
int rank_, n_;
const size_t kCounts[] = {1, 7, 4099, 300000};  // 300000 x 4 B = 1.2 MB: chain path

void reduce(hxcclComm_t c, int root, size_t count, bool in_place) {
  std::vector<int32_t> in(count), want(count), got(count, -1);
  for (size_t i = 0; i < count; ++i) in[i] = static_cast<int32_t>((rank_ + 1) * (i % 9 + 1));
  MPI_Reduce(in.data(), want.data(), static_cast<int>(count), MPI_INT, MPI_SUM, root,
             MPI_COMM_WORLD);
  hxcclResult_t r;
  if (rank_ == root && in_place) {
    got = in;
    r = hxcclReduce(got.data(), got.data(), count, hxcclInt32, hxcclSum, root, c, nullptr);
  } else {
    r = hxcclReduce(in.data(), rank_ == root ? got.data() : nullptr, count, hxcclInt32,
                    hxcclSum, root, c, nullptr);
  }
  CHECK(r == hxcclSuccess);
  if (rank_ == root) CHECK(got == want);
}

void reduce_avg_int(hxcclComm_t c) {  // the integer divide happens on the root only
  int32_t x = rank_ * 2 - 5, got = 99;
  CHECK(hxcclReduce(&x, &got, 1, hxcclInt32, hxcclAvg, 0, c, nullptr) == hxcclSuccess);
  int sum = 0;
  for (int r = 0; r < n_; ++r) sum += r * 2 - 5;
  if (rank_ == 0)
    CHECK(got == sum / n_);
  else
    CHECK(got == 99);
}

void bcast(hxcclComm_t c, int root, size_t count) {
  std::vector<float> buf(count, rank_ == root ? 3.5f : 0.0f);
  CHECK(hxcclBcast(buf.data(), count, hxcclFloat32, root, c, nullptr) == hxcclSuccess);
  CHECK(buf.front() == 3.5f && buf.back() == 3.5f);
}

void gather(hxcclComm_t c, int root, size_t count, bool in_place) {
  std::vector<int32_t> mine(count), want(count * n_), got(count * n_, -1);
  for (size_t i = 0; i < count; ++i) mine[i] = rank_ * 100000 + static_cast<int32_t>(i);
  MPI_Gather(mine.data(), static_cast<int>(count), MPI_INT, want.data(),
             static_cast<int>(count), MPI_INT, root, MPI_COMM_WORLD);
  const void* send = mine.data();
  if (rank_ == root && in_place) {
    std::memcpy(got.data() + root * count, mine.data(), count * 4);
    send = got.data() + root * count;
  }
  CHECK(hxcclGather(send, rank_ == root ? got.data() : nullptr, count, hxcclInt32, root, c,
                    nullptr) == hxcclSuccess);
  if (rank_ == root) CHECK(got == want);
}

void scatter(hxcclComm_t c, int root, size_t count, bool in_place) {
  std::vector<int32_t> all(count * n_), want(count), got(count, -1);
  for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int32_t>(i * 3 + 1);
  MPI_Scatter(all.data(), static_cast<int>(count), MPI_INT, want.data(),
              static_cast<int>(count), MPI_INT, root, MPI_COMM_WORLD);
  int32_t* recv = rank_ == root && in_place ? all.data() + root * count : got.data();
  CHECK(hxcclScatter(rank_ == root ? all.data() : nullptr, recv, count, hxcclInt32, root, c,
                     nullptr) == hxcclSuccess);
  CHECK(std::vector<int32_t>(recv, recv + count) == want);
}

void reduce_traffic(hxcclComm_t c) {
  const size_t count = 7;
  std::vector<int32_t> in(count, 1), out(count);
  hxcclCommResetStats(c);
  hxcclReduce(in.data(), out.data(), count, hxcclInt32, hxcclSum, 0, c, nullptr);
  uint64_t b = 0, m = 0;
  hxcclCommGetStats(c, &b, &m);
  // binomial tree: every non-root sends its partial result exactly once
  CHECK(rank_ == 0 ? (b == 0 && m == 0) : (b == count * 4 && m == 1));
}

void zero_count_no_traffic(hxcclComm_t c) {
  hxcclCommResetStats(c);
  CHECK(hxcclReduce(nullptr, nullptr, 0, hxcclInt32, hxcclSum, 0, c, nullptr) ==
        hxcclSuccess);
  CHECK(hxcclGather(nullptr, nullptr, 0, hxcclInt32, 0, c, nullptr) == hxcclSuccess);
  CHECK(hxcclScatter(nullptr, nullptr, 0, hxcclInt32, 0, c, nullptr) == hxcclSuccess);
  uint64_t b = 9, m = 9;
  hxcclCommGetStats(c, &b, &m);
  CHECK(b == 0 && m == 0);
}

void errors(hxcclComm_t c) {
  int32_t x = 0;
  CHECK(hxcclReduce(&x, &x, 1, hxcclInt32, hxcclSum, n_, c, nullptr) == hxcclInvalidArgument);
  CHECK(hxcclGather(&x, &x, 1, hxcclInt32, -1, c, nullptr) == hxcclInvalidArgument);
  CHECK(hxcclScatter(&x, &x, 1, hxcclInt32, n_, c, nullptr) == hxcclInvalidArgument);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  for (int root : {0, n_ - 1})
    for (size_t count : kCounts)
      for (bool ip : {false, true}) {
        reduce(c, root, count, ip);
        gather(c, root, count, ip);
        scatter(c, root, count, ip);
        if (!ip) bcast(c, root, count);
      }
  reduce_avg_int(c);
  reduce_traffic(c);
  zero_count_no_traffic(c);
  errors(c);
  hxcclCommDestroy(c);
  return finish();
}
