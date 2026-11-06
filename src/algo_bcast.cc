#include <algorithm>
#include <cstring>

#include "algo.h"

hxcclResult_t bcast(Transport& tr, int rank, int n, const void* send, void* recv,
                    size_t bytes, int root) {
  char* buf = static_cast<char*>(recv);
  if (rank == root && send != recv) std::memcpy(buf, send, bytes);
  if (n == 1 || bytes == 0) return hxcclSuccess;
  const int v = (rank - root + n) % n;  // position relative to root
  auto real = [=](int x) { return (x + root) % n; };
  hxcclResult_t r;

  if (bytes < kChainMinBytes) {
    int mask = 1;
    while (mask < n) {
      if (v & mask) {
        if ((r = tr.recv(real(v - mask), buf, bytes)) != hxcclSuccess) return r;
        break;
      }
      mask <<= 1;
    }
    for (mask >>= 1; mask > 0; mask >>= 1)
      if (v + mask < n && (r = tr.send(real(v + mask), buf, bytes)) != hxcclSuccess) return r;
    return hxcclSuccess;
  }
  for (size_t off = 0; off < bytes; off += kChainSegment) {
    const size_t len = std::min(kChainSegment, bytes - off);
    if (v > 0 && (r = tr.recv(real(v - 1), buf + off, len)) != hxcclSuccess) return r;
    if (v < n - 1 && (r = tr.send(real(v + 1), buf + off, len)) != hxcclSuccess) return r;
  }
  return hxcclSuccess;
}
