#include <cstring>

#include "check.h"
#include "hxccl.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int wr, wn;
  MPI_Comm_rank(MPI_COMM_WORLD, &wr);
  MPI_Comm_size(MPI_COMM_WORLD, &wn);

  hxcclComm_t c = nullptr;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  int n = -1, r = -1;
  CHECK(hxcclCommCount(c, &n) == hxcclSuccess && n == wn);
  CHECK(hxcclCommUserRank(c, &r) == hxcclSuccess && r == wr);
  uint64_t b = 1, m = 1;
  CHECK(hxcclCommGetStats(c, &b, &m) == hxcclSuccess && b == 0 && m == 0);
  CHECK(hxcclCommDestroy(c) == hxcclSuccess);

  hxcclUniqueId id;
  if (wr == 0) CHECK(hxcclGetUniqueId(&id) == hxcclSuccess);
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
  hxcclComm_t c2 = nullptr;
  CHECK(hxcclCommInitRank(&c2, wn, id, wr) == hxcclSuccess);
  CHECK(hxcclCommUserRank(c2, &r) == hxcclSuccess && r == wr);
  CHECK(hxcclCommDestroy(c2) == hxcclSuccess);

  CHECK(hxcclCommInitMPI(nullptr, MPI_COMM_WORLD) == hxcclInvalidArgument);
  CHECK(hxcclCommInitRank(&c2, wn, id, wn) == hxcclInvalidArgument);
  CHECK(std::strlen(hxcclGetErrorString(hxcclSystemError)) > 0);
  return finish();
}
