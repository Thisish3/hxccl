#include <cstring>
#include <vector>

#include "algo.h"
#include "reduce.h"

// Recursive doubling: log2(p) full-buffer exchanges among p = 2^k ranks. With
// n = p + rem, ranks < 2*rem pair up first: the even one hands its data to the
// odd one, sits out, and gets the result back at the end.
hxcclResult_t rd_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                           size_t count, hxcclDataType_t dt, hxcclRedOp_t op) {
  const size_t bytes = count * dtype_size(dt);
  if (send != recv) std::memcpy(recv, send, bytes);
  if (n == 1 || count == 0) return hxcclSuccess;
  int p = 1;
  while (p * 2 <= n) p *= 2;
  const int rem = n - p;
  std::vector<char> tmp(bytes);
  hxcclResult_t r;

  if (rank < 2 * rem) {
    if (rank % 2 == 0) {
      if ((r = tr.send(rank + 1, recv, bytes)) != hxcclSuccess) return r;
      return tr.recv(rank + 1, recv, bytes);
    }
    if ((r = tr.recv(rank - 1, tmp.data(), bytes)) != hxcclSuccess) return r;
    reduce_into(recv, tmp.data(), count, dt, op);
  }
  const int nr = rank < 2 * rem ? rank / 2 : rank - rem;
  auto real = [rem](int q) { return q < rem ? 2 * q + 1 : q + rem; };
  for (int mask = 1; mask < p; mask <<= 1) {
    const int peer = real(nr ^ mask);
    if ((r = tr.sendrecv(peer, recv, bytes, peer, tmp.data(), bytes)) != hxcclSuccess)
      return r;
    reduce_into(recv, tmp.data(), count, dt, op);
  }
  if (rank < 2 * rem) return tr.send(rank - 1, recv, bytes);
  return hxcclSuccess;
}
