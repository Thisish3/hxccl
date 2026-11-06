#include <cstdint>
#include <cstring>
#include <vector>

#include "check.h"
#include "hxccl.h"
#include "comm.h"
#include "transport.h"

namespace {
struct FailingTransport : Transport {
  hxcclResult_t fail() {
    error = "injected MPI failure";
    return hxcclSystemError;
  }
  hxcclResult_t sendrecv(int, const void*, size_t, int, void*, size_t) override { return fail(); }
  hxcclResult_t send(int, const void*, size_t) override { return fail(); }
  hxcclResult_t recv(int, void*, size_t) override { return fail(); }
};
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int n;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  float buf[4] = {0};
  auto bad = static_cast<hxcclDataType_t>(42);
  auto badop = static_cast<hxcclRedOp_t>(42);

  CHECK(hxcclAllReduce(buf, buf, 4, hxcclFloat32, hxcclSum, nullptr, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclAllReduce(buf, buf, 4, bad, hxcclSum, c, nullptr) == hxcclInvalidArgument);
  CHECK(std::strlen(hxcclGetLastError(c)) > 0);
  CHECK(hxcclAllReduce(buf, buf, 4, hxcclFloat32, badop, c, nullptr) == hxcclInvalidArgument);
  CHECK(hxcclAllReduce(nullptr, buf, 4, hxcclFloat32, hxcclSum, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclAllReduce(nullptr, nullptr, 0, hxcclFloat32, hxcclSum, c, nullptr) ==
        hxcclSuccess);
  CHECK(hxcclAllReduce(buf, buf, SIZE_MAX / 2, hxcclFloat32, hxcclSum, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclAllGather(buf, buf, SIZE_MAX / 3, hxcclInt32, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclReduceScatter(buf, nullptr, 1, hxcclFloat32, hxcclSum, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclBroadcast(buf, buf, 4, hxcclFloat32, n, c, nullptr) == hxcclInvalidArgument);
  CHECK(hxcclBroadcast(buf, buf, 4, hxcclFloat32, -1, c, nullptr) == hxcclInvalidArgument);
  uint64_t b = 9, m = 9;
  hxcclCommGetStats(c, &b, &m);
  CHECK(b == 0 && m == 0);

  // MPI failure is returned, not fatal (communicator uses MPI_ERRORS_RETURN)
  MPI_Comm dup;
  MPI_Comm_dup(MPI_COMM_WORLD, &dup);
  MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN);
  MpiTransport tr(dup);
  CHECK(tr.send(n + 5, buf, sizeof buf) == hxcclSystemError);
  CHECK(!tr.error.empty());
  MPI_Comm_free(&dup);

  // the communicator itself returns MPI errors instead of aborting
  MPI_Errhandler eh;
  MPI_Comm_get_errhandler(c->mpi, &eh);
  CHECK(eh == MPI_ERRORS_RETURN);
  MPI_Errhandler_free(&eh);

  // a transport failure surfaces through the public API with its message
  if (n > 1) {
    hxcclComm_t f;
    CHECK(hxcclCommInitMPI(&f, MPI_COMM_WORLD) == hxcclSuccess);
    f->tr = std::make_unique<FailingTransport>();
    CHECK(hxcclAllReduce(buf, buf, 4, hxcclFloat32, hxcclSum, f, nullptr) == hxcclSystemError);
    CHECK(std::strstr(hxcclGetLastError(f), "injected MPI failure") != nullptr);
    CHECK(hxcclBroadcast(buf, buf, 4, hxcclFloat32, 0, f, nullptr) == hxcclSystemError);
    hxcclCommDestroy(f);
  }

  // every rank rejects a message the MPI transport cannot carry, before sending
  if (n > 1) {
    size_t huge = (static_cast<size_t>(INT32_MAX) / sizeof(float)) * n + 1;
    hxcclCommResetStats(c);
    setenv("HXCCL_ALGO", "ring", 1);
    CHECK(hxcclAllReduce(buf, buf, huge, hxcclFloat32, hxcclSum, c, nullptr) ==
          hxcclInvalidArgument);
    setenv("HXCCL_ALGO", "rd", 1);
    // rd sends the whole buffer in one message
    const size_t over = static_cast<size_t>(INT32_MAX) / sizeof(float) + 1;
    CHECK(hxcclAllReduce(buf, buf, over, hxcclFloat32, hxcclSum, c, nullptr) ==
          hxcclInvalidArgument);
    unsetenv("HXCCL_ALGO");
    CHECK(hxcclAllGather(buf, buf, over, hxcclFloat32, c, nullptr) ==
          hxcclInvalidArgument);
    uint64_t b2 = 9, m2 = 9;
    hxcclCommGetStats(c, &b2, &m2);
    CHECK(b2 == 0 && m2 == 0);
  }

  // InitRank checks that the given rank is the rank the caller really gets
  if (n > 1) {
    int wr;
    MPI_Comm_rank(MPI_COMM_WORLD, &wr);
    hxcclUniqueId id;
    if (wr == 0) hxcclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
    hxcclComm_t w = nullptr;
    hxcclResult_t r = hxcclCommInitRank(&w, n, id, 0);  // everyone claims rank 0
    CHECK(wr == 0 ? r == hxcclSuccess : r == hxcclInvalidUsage);
    if (r == hxcclSuccess) hxcclCommDestroy(w);
  }

  hxcclCommDestroy(c);
  return finish();
}
