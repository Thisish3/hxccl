#include <cstdint>
#include <vector>

#include "check.h"
#include "hxccl.h"

namespace {
int rank_, n_;
int right() { return (rank_ + 1) % n_; }
int left() { return (rank_ + n_ - 1) % n_; }

// 4 MB each way: above the eager limit, so Send-then-Recv on both sides needs the group
void pair_exchange(hxcclComm_t c) {
  const int peer = rank_ ^ 1;
  if (peer >= n_) return;
  const size_t count = 1 << 20;
  std::vector<float> out(count, static_cast<float>(rank_)), in(count, -1.0f);
  CHECK(hxcclGroupStart() == hxcclSuccess);
  CHECK(hxcclSend(out.data(), count, hxcclFloat32, peer, c, nullptr) == hxcclSuccess);
  CHECK(hxcclRecv(in.data(), count, hxcclFloat32, peer, c, nullptr) == hxcclSuccess);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(in.front() == static_cast<float>(peer) && in.back() == static_cast<float>(peer));
}

void ring_shift_traffic(hxcclComm_t c) {
  std::vector<int32_t> out(100, rank_), in(100, -1);
  hxcclCommResetStats(c);
  hxcclGroupStart();
  hxcclSend(out.data(), 100, hxcclInt32, right(), c, nullptr);
  hxcclRecv(in.data(), 100, hxcclInt32, left(), c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(in[0] == left() && in[99] == left());
  uint64_t b = 0, m = 0;
  hxcclCommGetStats(c, &b, &m);
  CHECK(b == 400 && m == 1);
}

void same_peer_order(hxcclComm_t c) {
  int32_t a[3] = {rank_ * 10 + 1, rank_ * 10 + 2, rank_ * 10 + 3}, got[3] = {0, 0, 0};
  hxcclGroupStart();
  for (int i = 0; i < 3; ++i) hxcclSend(&a[i], 1, hxcclInt32, right(), c, nullptr);
  for (int i = 0; i < 3; ++i) hxcclRecv(&got[i], 1, hxcclInt32, left(), c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  for (int i = 0; i < 3; ++i) CHECK(got[i] == left() * 10 + i + 1);
}

void nested_and_mixed(hxcclComm_t c) {
  int32_t x = rank_ + 1, sum = 0, out = rank_, in = -1;
  hxcclCommResetStats(c);
  hxcclGroupStart();
  hxcclGroupStart();
  hxcclAllReduce(&x, &sum, 1, hxcclInt32, hxcclSum, c, nullptr);
  hxcclSend(&out, 1, hxcclInt32, right(), c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  uint64_t b = 0, m = 0;
  hxcclCommGetStats(c, &b, &m);
  CHECK(m == 0 && sum == 0);  // inner End runs nothing
  hxcclRecv(&in, 1, hxcclInt32, left(), c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(sum == n_ * (n_ + 1) / 2 && in == left());
}

void group_skips_rejected_call(hxcclComm_t c) {
  int32_t out = rank_, in = -1;
  hxcclGroupStart();
  CHECK(hxcclSend(&out, 1, hxcclInt32, n_, c, nullptr) == hxcclInvalidArgument);
  hxcclSend(&out, 1, hxcclInt32, right(), c, nullptr);
  hxcclRecv(&in, 1, hxcclInt32, left(), c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(in == left());
  CHECK(hxcclGroupEnd() == hxcclInvalidUsage);  // depth 0
}

void recv_too_small(hxcclComm_t c) {
  if (n_ < 2 || rank_ > 1) return;
  int32_t buf[8] = {0};
  hxcclResult_t r = rank_ == 0 ? hxcclSend(buf, 8, hxcclInt32, 1, c, nullptr)
                               : hxcclRecv(buf, 4, hxcclInt32, 0, c, nullptr);
  if (rank_ == 1) CHECK(r == hxcclSystemError);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  pair_exchange(c);
  ring_shift_traffic(c);
  same_peer_order(c);
  nested_and_mixed(c);
  group_skips_rejected_call(c);
  MPI_Barrier(MPI_COMM_WORLD);
  recv_too_small(c);  // last: nothing else runs on c after a truncated message
  hxcclCommDestroy(c);
  return finish();
}
