#include <cstdint>
#include <memory>
#include <vector>

#include "check.h"
#include "comm.h"
#include "hxccl.h"

namespace {
int rank_, n_;

struct FailingTransport : Transport {
  hxcclResult_t fail() {
    error = "injected";
    return hxcclSystemError;
  }
  hxcclResult_t sendrecv(int, const void*, size_t, int, void*, size_t) override { return fail(); }
  hxcclResult_t send(int, const void*, size_t) override { return fail(); }
  hxcclResult_t recv(int, void*, size_t) override { return fail(); }
  hxcclResult_t isend(int, const void*, size_t, int) override { return fail(); }
  hxcclResult_t irecv(int, void*, size_t, int) override { return fail(); }
  hxcclResult_t waitall() override { return hxcclSuccess; }
  void cancel_all() override {}
};

void split_parity(hxcclComm_t c) {
  hxcclComm_t sub = nullptr;
  CHECK(hxcclCommSplit(c, rank_ % 2, -rank_, &sub, nullptr) == hxcclSuccess);
  int sn = 0, sr = 0;
  hxcclCommCount(sub, &sn);
  hxcclCommUserRank(sub, &sr);
  CHECK(sn == (n_ + 1 - rank_ % 2) / 2);
  CHECK(sr == sn - 1 - rank_ / 2);  // key = -rank reverses the order
  int32_t x = rank_, sum = 0, want = 0;
  for (int r = rank_ % 2; r < n_; r += 2) want += r;
  CHECK(hxcclAllReduce(&x, &sum, 1, hxcclInt32, hxcclSum, sub, nullptr) == hxcclSuccess);
  CHECK(sum == want);
  hxcclCommDestroy(sub);
}

void split_nocolor(hxcclComm_t c) {
  hxcclComm_t sub = reinterpret_cast<hxcclComm_t>(1);
  CHECK(hxcclCommSplit(c, rank_ == 0 ? HXCCL_SPLIT_NOCOLOR : 0, rank_, &sub, nullptr) ==
        hxcclSuccess);
  if (rank_ == 0) {
    CHECK(sub == nullptr);
  } else {
    int sn = 0;
    hxcclCommCount(sub, &sn);
    CHECK(sn == n_ - 1);
    hxcclCommDestroy(sub);
  }
}

void group_two_comms(hxcclComm_t c) {
  hxcclComm_t sub = nullptr;
  hxcclCommSplit(c, rank_ % 2, rank_, &sub, nullptr);
  int32_t x = 1, all = 0, part = 0;
  hxcclGroupStart();
  hxcclAllReduce(&x, &part, 1, hxcclInt32, hxcclSum, sub, nullptr);
  hxcclAllReduce(&x, &all, 1, hxcclInt32, hxcclSum, c, nullptr);
  CHECK(hxcclGroupEnd() == hxcclSuccess);
  int sn = 0;
  hxcclCommCount(sub, &sn);
  CHECK(all == n_ && part == sn);
  hxcclCommDestroy(sub);
}

void shrink(hxcclComm_t c) {
  if (n_ < 3) return;
  int excl[2] = {0, n_ - 1};
  if (rank_ == 0 || rank_ == n_ - 1) return;  // excluded ranks do not call (NCCL)
  hxcclComm_t s = nullptr;
  CHECK(hxcclCommShrink(c, excl, 2, &s, nullptr, 0) == hxcclSuccess);
  int sn = 0, sr = 0;
  hxcclCommCount(s, &sn);
  hxcclCommUserRank(s, &sr);
  CHECK(sn == n_ - 2 && sr == rank_ - 1);
  int32_t x = 1, sum = 0;
  CHECK(hxcclAllReduce(&x, &sum, 1, hxcclInt32, hxcclSum, s, nullptr) == hxcclSuccess);
  CHECK(sum == n_ - 2);
  hxcclCommDestroy(s);
}

void shrink_errors(hxcclComm_t c) {
  hxcclComm_t s = nullptr;
  int self = rank_, out = n_, dup[2] = {0, 0};
  CHECK(hxcclCommShrink(c, &self, 1, &s, nullptr, 0) == hxcclInvalidUsage);
  CHECK(hxcclCommShrink(c, &out, 1, &s, nullptr, 0) == hxcclInvalidArgument);
  CHECK(hxcclCommShrink(c, dup, 2, &s, nullptr, 0) == hxcclInvalidArgument);
  CHECK(hxcclCommShrink(c, nullptr, 0, &s, nullptr, 1) == hxcclInvalidArgument);
}

// Abort is local: odd ranks abort, even ranks never touch the comm, the barrier still passes
void abort_is_local() {
  hxcclComm_t a;
  CHECK(hxcclCommInitMPI(&a, MPI_COMM_WORLD) == hxcclSuccess);
  if (rank_ % 2) CHECK(hxcclCommAbort(a) == hxcclSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
}

void finalize_then_use() {
  hxcclComm_t f;
  CHECK(hxcclCommInitMPI(&f, MPI_COMM_WORLD) == hxcclSuccess);
  CHECK(hxcclCommFinalize(f) == hxcclSuccess);
  int32_t x = 1;
  CHECK(hxcclAllReduce(&x, &x, 1, hxcclInt32, hxcclSum, f, nullptr) == hxcclInvalidUsage);
  CHECK(hxcclSend(&x, 1, hxcclInt32, 0, f, nullptr) == hxcclInvalidUsage);
  std::vector<size_t> one(n_, 1), dsp(n_);
  for (int j = 0; j < n_; ++j) dsp[j] = static_cast<size_t>(j);
  std::vector<int32_t> a(n_), b(n_);
  CHECK(hxcclAllToAllv(a.data(), one.data(), dsp.data(), b.data(), one.data(), dsp.data(),
                       hxcclInt32, f, nullptr) == hxcclInvalidUsage);
  CHECK(hxcclCommDestroy(f) == hxcclSuccess);
}

void async_error() {
  hxcclComm_t f;
  CHECK(hxcclCommInitMPI(&f, MPI_COMM_WORLD) == hxcclSuccess);
  hxcclResult_t e = hxcclInternalError;
  CHECK(hxcclCommGetAsyncError(f, &e) == hxcclSuccess && e == hxcclSuccess);
  float x = 0;
  hxcclAllReduce(&x, &x, 1, static_cast<hxcclDataType_t>(42), hxcclSum, f, nullptr);
  hxcclCommGetAsyncError(f, &e);
  CHECK(e == hxcclSuccess);  // argument errors are not communicator state
  if (n_ > 1) {
    f->tr = std::make_unique<FailingTransport>();
    hxcclAllReduce(&x, &x, 1, hxcclFloat32, hxcclSum, f, nullptr);
    hxcclCommGetAsyncError(f, &e);
    CHECK(e == hxcclSystemError);
  }
  hxcclCommDestroy(f);
}

void config() {
  hxcclUniqueId id;
  if (rank_ == 0) hxcclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
  hxcclConfig_t cfg = HXCCL_CONFIG_INITIALIZER;
  hxcclComm_t k = nullptr;
  CHECK(hxcclCommInitRankConfig(&k, n_, id, rank_, &cfg) == hxcclSuccess);
  hxcclConfig_t nb = HXCCL_CONFIG_INITIALIZER;
  nb.blocking = 0;
  hxcclComm_t s = nullptr;
  CHECK(hxcclCommSplit(k, 0, rank_, &s, &nb) == hxcclInvalidArgument);
  hxcclConfig_t bad = HXCCL_CONFIG_INITIALIZER;
  bad.magic = 0;
  CHECK(hxcclCommSplit(k, 0, rank_, &s, &bad) == hxcclInvalidArgument);
  CHECK(hxcclCommInitRankConfig(&s, n_, id, rank_, &nb) == hxcclInvalidArgument);
  hxcclCommDestroy(k);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  split_parity(c);
  split_nocolor(c);
  group_two_comms(c);
  shrink(c);
  shrink_errors(c);
  abort_is_local();
  finalize_then_use();
  async_error();
  config();
  hxcclCommDestroy(c);
  return finish();
}
