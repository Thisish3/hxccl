// E-I1 (Mac review of E): a cancelled send must not keep reading the caller's memory.
// Rank 0 starts a 32 MiB send to rank 1 (larger than the socket buffers, so only part of it
// goes out at once), cancels it, then overwrites its buffer. Rank 1 then receives:
// whatever arrives must be the original bytes, and a following message must still match.
// Socket communicators only: an MPI communicator cannot give the caller its buffer back
// early (MPI may still read a cancelled send's buffer; KNOWN_ISSUES).
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "check.h"
#include "comm.h"
#include "hxccl.h"

namespace {
uint32_t pattern(size_t i) { return static_cast<uint32_t>(i * 2654435761u + 12345u); }
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  if (n >= 2 && c->mpi == MPI_COMM_NULL && rank < 2) {
    Net& net = *c->net;
    const size_t words = (size_t(32) << 20) / 4;
    std::vector<uint32_t> buf(words);
    uint32_t tail = 0;
    if (rank == 0) {
      for (size_t i = 0; i < words; ++i) buf[i] = pattern(i);
      int req;
      CHECK(net.isend(1, 5, buf.data(), words * 4, &req) == hxcclSuccess);
      bool done = false;
      for (int k = 0; k < 10 && !done; ++k) net.test(req, &done);  // push out what fits
      CHECK(!done);  // 32 MiB cannot all fit in the socket buffers yet
      if (!done) net.cancel(req);
      std::memset(buf.data(), 0xEE, words * 4);  // the caller reuses its memory
      tail = 0xC0FFEE;
      CHECK(net.send(1, 6, &tail, 4) == hxcclSuccess);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));  // let rank 0 cancel first
      int big, small;
      CHECK(net.irecv(0, 5, buf.data(), words * 4, &big) == hxcclSuccess);
      CHECK(net.irecv(0, 6, &tail, 4, &small) == hxcclSuccess);
      CHECK(net.wait(small) == hxcclSuccess);
      CHECK(tail == 0xC0FFEE);
      bool done = false;
      for (int k = 0; k < 2000 && !done; ++k) {
        net.test(big, &done);
        if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (done) {  // the cancelled message was finished from a private copy: all original
        size_t bad = 0;
        for (size_t i = 0; i < words; ++i) bad += buf[i] != pattern(i);
        if (bad) std::fprintf(stderr, "[rank 1] %zu of %zu words differ\n", bad, words);
        CHECK(bad == 0);
      } else {
        net.cancel(big);  // dropped before any byte went out: also fine
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  hxcclCommDestroy(c);
  return finish();
}
