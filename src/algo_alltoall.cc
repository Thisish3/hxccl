#include "algo.h"

// Pairwise exchange: at step s every rank sends to r+s and receives from r-s.
hxcclResult_t alltoallv(MemOps& m, int rank, int n, const char* send, const size_t* sbytes,
                        const size_t* soff, char* recv, const size_t* rbytes,
                        const size_t* roff) {
  if (rbytes[rank])
    if (hxcclResult_t r = m.copy(recv + roff[rank], send + soff[rank], rbytes[rank])) return r;
  for (int s = 1; s < n; ++s) {
    const int to = (rank + s) % n, from = (rank - s + n) % n;
    if (hxcclResult_t r = m.tr().sendrecv(to, send + soff[to], sbytes[to], from,
                                          recv + roff[from], rbytes[from]))
      return r;
  }
  return hxcclSuccess;
}
