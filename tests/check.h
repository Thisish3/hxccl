#pragma once
#include <mpi.h>

#include <cstdio>

inline int g_fail = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      int r_;                                                                  \
      MPI_Comm_rank(MPI_COMM_WORLD, &r_);                                      \
      std::fprintf(stderr, "[rank %d] %s:%d CHECK(%s) failed\n", r_, __FILE__, \
                   __LINE__, #cond);                                           \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)

// every rank returns the same exit code: nonzero if any rank failed
inline int finish() {
  int any = 0;
  MPI_Allreduce(&g_fail, &any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Finalize();
  return any ? 1 : 0;
}
