#include <cstdint>
#include <cstring>
#include <vector>

#include "check.h"
#include "hxccl.h"
#include "transport.h"

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

  hxcclCommDestroy(c);
  return finish();
}
