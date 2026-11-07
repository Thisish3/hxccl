#include <climits>
#include <cstdint>
#include <vector>

#include "check.h"
#include "hxccl.h"

namespace {
int rank_, n_;

void alltoall(hxcclComm_t c, size_t count, bool rccl_name) {
  std::vector<int32_t> in(count * n_), want(count * n_), got(count * n_, -1);
  for (size_t i = 0; i < in.size(); ++i) in[i] = rank_ * 1000000 + static_cast<int32_t>(i);
  MPI_Alltoall(in.data(), static_cast<int>(count), MPI_INT, want.data(),
               static_cast<int>(count), MPI_INT, MPI_COMM_WORLD);
  hxcclResult_t r = rccl_name
                        ? hxcclAllToAll(in.data(), got.data(), count, hxcclInt32, c, nullptr)
                        : hxcclAlltoAll(in.data(), got.data(), count, hxcclInt32, c, nullptr);
  CHECK(r == hxcclSuccess);
  CHECK(got == want);
}

size_t sendcount(int from, int to) { return static_cast<size_t>((from + 2 * to) % 4); }

void alltoallv(hxcclComm_t c) {
  std::vector<size_t> sc(n_), sd(n_), rc(n_), rd(n_);
  std::vector<int> isc(n_), isd(n_), irc(n_), ird(n_);
  size_t so = 0, ro = 0;
  for (int j = 0; j < n_; ++j) {  // a one-element gap after each block checks displs are used
    sc[j] = sendcount(rank_, j);
    sd[j] = so;
    so += sc[j] + 1;
    rc[j] = sendcount(j, rank_);
    rd[j] = ro;
    ro += rc[j] + 1;
    isc[j] = static_cast<int>(sc[j]);
    isd[j] = static_cast<int>(sd[j]);
    irc[j] = static_cast<int>(rc[j]);
    ird[j] = static_cast<int>(rd[j]);
  }
  std::vector<int32_t> in(so), want(ro, -7), got(ro, -7);
  for (size_t i = 0; i < so; ++i) in[i] = rank_ * 1000 + static_cast<int32_t>(i);
  MPI_Alltoallv(in.data(), isc.data(), isd.data(), MPI_INT, want.data(), irc.data(),
                ird.data(), MPI_INT, MPI_COMM_WORLD);
  CHECK(hxcclAllToAllv(in.data(), sc.data(), sd.data(), got.data(), rc.data(), rd.data(),
                       hxcclInt32, c, nullptr) == hxcclSuccess);
  CHECK(got == want);
}

// MPI_Alltoall supports in-place, so a user may pass send == recv
void alltoall_in_place(hxcclComm_t c, size_t count) {
  std::vector<int32_t> buf(count * n_), want(count * n_);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = rank_ * 1000000 + static_cast<int32_t>(i);
  MPI_Alltoall(buf.data(), static_cast<int>(count), MPI_INT, want.data(),
               static_cast<int>(count), MPI_INT, MPI_COMM_WORLD);
  CHECK(hxcclAlltoAll(buf.data(), buf.data(), count, hxcclInt32, c, nullptr) == hxcclSuccess);
  CHECK(buf == want);
}

// symmetric counts so one layout serves as both send and receive side
void alltoallv_in_place(hxcclComm_t c) {
  std::vector<size_t> cnt(n_), dsp(n_);
  std::vector<int> icnt(n_), idsp(n_);
  size_t off = 0;
  for (int j = 0; j < n_; ++j) {
    cnt[j] = static_cast<size_t>((rank_ + j) % 3 + 1);
    dsp[j] = off;
    off += cnt[j];
    icnt[j] = static_cast<int>(cnt[j]);
    idsp[j] = static_cast<int>(dsp[j]);
  }
  std::vector<int32_t> buf(off), want(off);
  for (size_t i = 0; i < off; ++i) buf[i] = rank_ * 1000 + static_cast<int32_t>(i);
  MPI_Alltoallv(buf.data(), icnt.data(), idsp.data(), MPI_INT, want.data(), icnt.data(),
                idsp.data(), MPI_INT, MPI_COMM_WORLD);
  CHECK(hxcclAllToAllv(buf.data(), cnt.data(), dsp.data(), buf.data(), cnt.data(), dsp.data(),
                       hxcclInt32, c, nullptr) == hxcclSuccess);
  CHECK(buf == want);
}

void traffic(hxcclComm_t c) {
  std::vector<float> in(5 * n_, 1.0f), out(5 * n_);
  hxcclCommResetStats(c);
  hxcclAlltoAll(in.data(), out.data(), 5, hxcclFloat32, c, nullptr);
  uint64_t b = 0, m = 0;
  hxcclCommGetStats(c, &b, &m);
  CHECK(b == static_cast<uint64_t>(n_ - 1) * 5 * 4 && m == static_cast<uint64_t>(n_ - 1));
}

void zero_count_no_traffic(hxcclComm_t c) {
  hxcclCommResetStats(c);
  CHECK(hxcclAlltoAll(nullptr, nullptr, 0, hxcclInt32, c, nullptr) == hxcclSuccess);
  uint64_t b = 9, m = 9;
  hxcclCommGetStats(c, &b, &m);
  CHECK(b == 0);
}

// one rank's oversized block or bad argument makes every rank reject; nobody waits
void agreed_rejection(hxcclComm_t c) {
  if (n_ < 2) return;
  std::vector<size_t> cnt(n_, 1), dsp(n_);
  for (int j = 0; j < n_; ++j) dsp[j] = static_cast<size_t>(j);
  int32_t buf[64] = {0};
  std::vector<size_t> big = cnt;
  if (rank_ == 0) big[1] = static_cast<size_t>(INT_MAX) / 4 + 1;
  CHECK(hxcclAllToAllv(buf, big.data(), dsp.data(), buf, cnt.data(), dsp.data(), hxcclInt32, c,
                       nullptr) == hxcclInvalidArgument);
  const size_t* sc = rank_ == 0 ? nullptr : cnt.data();
  CHECK(hxcclAllToAllv(buf, sc, dsp.data(), buf, cnt.data(), dsp.data(), hxcclInt32, c,
                       nullptr) == hxcclInvalidArgument);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  for (size_t count : {size_t{1}, size_t{7}, size_t{4099}}) {
    alltoall(c, count, false);
    alltoall(c, count, true);
  }
  alltoallv(c);
  for (size_t count : {size_t{1}, size_t{7}}) alltoall_in_place(c, count);
  alltoallv_in_place(c);
  traffic(c);
  zero_count_no_traffic(c);
  agreed_rejection(c);
  hxcclCommDestroy(c);
  return finish();
}
