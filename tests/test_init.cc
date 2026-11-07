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
  // numbering matches nccl.h v2.32.3 so a torch backend can pass values through
  CHECK(hxcclInvalidArgument == 4 && hxcclInvalidUsage == 5 && hxcclTimeout == 8);
  CHECK(hxcclInt8 == 0 && hxcclInt32 == 2 && hxcclFloat16 == 6 && hxcclFloat32 == 7 &&
        hxcclBfloat16 == 9 && hxcclFloat8e5m2 == 11 && hxcclNumTypes == 12);
  CHECK(hxcclAvg == 4 && hxcclNumOps == 5);
  int v = 0;
  CHECK(hxcclGetVersion(&v) == hxcclSuccess && v == 200 && v == HXCCL_VERSION_CODE);
  CHECK(HXCCL_VERSION(2, 32, 3) == 23203 && HXCCL_VERSION(2, 8, 4) == 2804);
  CHECK(hxcclGetVersion(nullptr) == hxcclInvalidArgument);
  CHECK(std::strcmp(hxcclGetErrorString(hxcclTimeout), "unknown result") != 0);
  hxcclConfig_t cfg = HXCCL_CONFIG_INITIALIZER;
  CHECK(cfg.size == sizeof(hxcclConfig_t) && cfg.magic == HXCCL_CONFIG_MAGIC &&
        cfg.blocking == HXCCL_CONFIG_UNDEF_INT);
  return finish();
}
