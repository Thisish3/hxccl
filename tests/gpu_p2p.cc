// Send/Recv and groups on device buffers (Review Focus 2), and the host/device mix rule.
#include "check.h"
#include "gpu_util.h"

namespace {
int rank_, n_;

void ring_shift(hxcclComm_t c, size_t count) {
  std::vector<int> mine(count, rank_), want(count, (rank_ - 1 + n_) % n_);
  int* s = dev_copy(mine);
  int* r = dev_copy(std::vector<int>(count, -1));
  hxcclGroupStart();
  hxcclSend(s, count, hxcclInt32, (rank_ + 1) % n_, c, nullptr);
  hxcclRecv(r, count, hxcclInt32, (rank_ - 1 + n_) % n_, c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(host_copy(r, count) == want);
  cudaFree(s);
  cudaFree(r);
}

void group_two_sends_same_peer(hxcclComm_t c) {  // rank 0 <-> 1 only
  if (n_ < 2 || rank_ > 1) return;
  const int peer = 1 - rank_;
  const size_t count = 3000;  // > 4096-byte chunk in the chunk4k run
  std::vector<int> a(count, 10 * rank_ + 1), b(count, 10 * rank_ + 2);
  int *da = dev_copy(a), *db = dev_copy(b);
  int *ra = dev_copy(std::vector<int>(count)), *rb = dev_copy(std::vector<int>(count));
  hxcclGroupStart();
  hxcclSend(da, count, hxcclInt32, peer, c, nullptr);
  hxcclSend(db, count, hxcclInt32, peer, c, nullptr);
  hxcclRecv(ra, count, hxcclInt32, peer, c, nullptr);
  hxcclRecv(rb, count, hxcclInt32, peer, c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  CHECK(host_copy(ra, count) == std::vector<int>(count, 10 * peer + 1));
  CHECK(host_copy(rb, count) == std::vector<int>(count, 10 * peer + 2));
  for (int* p : {da, db, ra, rb}) cudaFree(p);
}

void blocking_pair(hxcclComm_t c) {  // outside a group: rank 0 sends, rank 1 receives
  if (n_ < 2 || rank_ > 1) return;
  const size_t count = 5000;
  int* d = dev_copy(std::vector<int>(count, rank_ == 0 ? 77 : -1));
  if (rank_ == 0)
    CHECK(hxcclSend(d, count, hxcclInt32, 1, c, nullptr) == hxcclSuccess);
  else
    CHECK(hxcclRecv(d, count, hxcclInt32, 0, c, nullptr) == hxcclSuccess);
  CHECK(host_copy(d, count) == std::vector<int>(count, 77));
  cudaFree(d);
}

void recv_too_small(hxcclComm_t c) {  // like MPI_ERR_TRUNCATE on the host path
  if (n_ < 2 || rank_ > 1) return;
  int* d = dev_copy(std::vector<int>(8, 5));
  hxcclResult_t r = rank_ == 0 ? hxcclSend(d, 8, hxcclInt32, 1, c, nullptr)
                               : hxcclRecv(d, 4, hxcclInt32, 0, c, nullptr);
  CHECK(r == (rank_ == 1 ? hxcclSystemError : hxcclSuccess));
  cudaFree(d);
}

void mixed_group_rejected(hxcclComm_t c) {
  std::vector<int> h(4);
  int* d = dev_copy(h);
  hxcclGroupStart();
  hxcclSend(h.data(), 4, hxcclInt32, (rank_ + 1) % n_, c, nullptr);
  hxcclRecv(d, 4, hxcclInt32, (rank_ - 1 + n_) % n_, c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclInvalidUsage);
  cudaFree(d);
}

void stats_match_host(hxcclComm_t c) {
  const size_t count = 1000;
  std::vector<int> hs(count, rank_), hr(count);
  int *ds = dev_copy(hs), *dr = dev_copy(hr);
  uint64_t b0, m0, b1, m1, b2, m2;
  hxcclCommGetStats(c, &b0, &m0);
  hxcclGroupStart();
  hxcclSend(hs.data(), count, hxcclInt32, (rank_ + 1) % n_, c, nullptr);
  hxcclRecv(hr.data(), count, hxcclInt32, (rank_ - 1 + n_) % n_, c, nullptr);
  hxcclGroupEnd();
  hxcclCommGetStats(c, &b1, &m1);
  hxcclGroupStart();
  hxcclSend(ds, count, hxcclInt32, (rank_ + 1) % n_, c, nullptr);
  hxcclRecv(dr, count, hxcclInt32, (rank_ - 1 + n_) % n_, c, nullptr);
  hxcclGroupEnd();
  hxcclCommGetStats(c, &b2, &m2);
  CHECK(b1 - b0 == b2 - b1);
  CHECK(m1 - m0 == m2 - m1);
  CHECK(host_copy(dr, count) == hr);
  cudaFree(ds);
  cudaFree(dr);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  cudaSetDevice(0);
  hxcclComm_t c = world_comm();
  for (size_t count : {size_t(0), size_t(1), size_t(1000), size_t(1) << 20}) ring_shift(c, count);
  group_two_sends_same_peer(c);
  blocking_pair(c);
  recv_too_small(c);
  mixed_group_rejected(c);
  stats_match_host(c);
  hxcclCommDestroy(c);
  return finish();
}
