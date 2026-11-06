#include <cstdint>
#include <cstring>
#include <vector>

#include "check.h"
#include "hxccl.h"

namespace {
int rank_, n_;

void allgather(hxcclComm_t c, size_t per, bool in_place) {
  std::vector<int32_t> mine(per), want(per * n_), got(per * n_, -1);
  for (size_t i = 0; i < per; ++i) mine[i] = rank_ * 1000 + static_cast<int>(i);
  MPI_Allgather(mine.data(), static_cast<int>(per), MPI_INT, want.data(),
                static_cast<int>(per), MPI_INT, MPI_COMM_WORLD);
  const void* send = mine.data();
  if (in_place) {
    std::memcpy(got.data() + rank_ * per, mine.data(), per * sizeof(int32_t));
    send = got.data() + rank_ * per;
  }
  CHECK(hxcclAllGather(send, got.data(), per, hxcclInt32, c, nullptr) == hxcclSuccess);
  CHECK(got == want);
}

void reducescatter(hxcclComm_t c, size_t per, bool in_place) {
  std::vector<float> in(per * n_), want(per), got(per, -1.0f);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>((rank_ + 1) * (i % 7 + 1));
  MPI_Reduce_scatter_block(in.data(), want.data(), static_cast<int>(per), MPI_FLOAT, MPI_SUM,
                           MPI_COMM_WORLD);
  if (in_place) {
    CHECK(hxcclReduceScatter(in.data(), in.data() + rank_ * per, per, hxcclFloat32, hxcclSum,
                             c, nullptr) == hxcclSuccess);
    CHECK(std::vector<float>(in.begin() + rank_ * per, in.begin() + (rank_ + 1) * per) ==
          want);
  } else {
    CHECK(hxcclReduceScatter(in.data(), got.data(), per, hxcclFloat32, hxcclSum, c, nullptr) ==
          hxcclSuccess);
    CHECK(got == want);
  }
}

void broadcast(hxcclComm_t c, size_t count, int root, bool in_place) {
  std::vector<double> want(count), got(count, -1.0), src(count);
  for (size_t i = 0; i < count; ++i) want[i] = static_cast<double>(root * 7 + i);
  if (rank_ == root) src = want;
  const void* send = src.data();
  if (in_place && rank_ == root) {
    got = want;
    send = got.data();
  }
  CHECK(hxcclBroadcast(send, got.data(), count, hxcclFloat64, root, c, nullptr) ==
        hxcclSuccess);
  CHECK(got == want);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  for (size_t per : {0, 1, 3, 7, 1000, 4099}) {
    allgather(c, per, false);
    allgather(c, per, true);
    reducescatter(c, per, false);
    reducescatter(c, per, true);
  }
  // 200000 doubles = 1.6 MB, above the 512 KB chain threshold
  for (size_t count : {0, 1, 3, 1000, 200000})
    for (int root : {0, n_ - 1}) {
      broadcast(c, count, root, false);
      broadcast(c, count, root, true);
    }
  hxcclCommDestroy(c);
  return finish();
}
