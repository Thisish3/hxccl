#include <algorithm>
#include <cstring>
#include <vector>

#include "algo.h"
#include "reduce.h"

hxcclResult_t reduce_root(Transport& tr, int rank, int n, const void* send, void* recv,
                          size_t count, hxcclDataType_t dt, hxcclRedOp_t op, int root) {
  const size_t es = dtype_size(dt), bytes = count * es;
  std::vector<char> own;
  char* buf;
  if (rank == root) {
    buf = static_cast<char*>(recv);
    if (send != recv && bytes) std::memcpy(buf, send, bytes);
  } else {  // non-root recv may be NULL (NCCL), so partial results live in a private copy
    own.resize(bytes);
    if (bytes) std::memcpy(own.data(), send, bytes);
    buf = own.data();
  }
  if (n == 1 || count == 0) return hxcclSuccess;
  const int v = (rank - root + n) % n;
  auto real = [=](int x) { return (x + root) % n; };
  hxcclResult_t r;
  if (bytes < kChainMinBytes) {  // binomial tree, the mirror of bcast
    std::vector<char> tmp(bytes);
    for (int mask = 1; mask < n; mask <<= 1) {
      if (v & mask) return tr.send(real(v - mask), buf, bytes);
      if (v + mask < n) {
        if ((r = tr.recv(real(v + mask), tmp.data(), bytes)) != hxcclSuccess) return r;
        reduce_into(buf, tmp.data(), count, dt, op);
      }
    }
    return hxcclSuccess;
  }
  // chain toward the root in kChainSegment pieces (a multiple of every dtype size)
  std::vector<char> tmp(kChainSegment);
  for (size_t off = 0; off < bytes; off += kChainSegment) {
    const size_t len = std::min(kChainSegment, bytes - off);
    if (v < n - 1) {
      if ((r = tr.recv(real(v + 1), tmp.data(), len)) != hxcclSuccess) return r;
      reduce_into(buf + off, tmp.data(), len / es, dt, op);
    }
    if (v > 0 && (r = tr.send(real(v - 1), buf + off, len)) != hxcclSuccess) return r;
  }
  return hxcclSuccess;
}

hxcclResult_t gather(Transport& tr, int rank, int n, const void* send, void* recv,
                     size_t per, int root) {
  if (per == 0) return hxcclSuccess;
  if (rank != root) return tr.send(root, send, per);
  char* out = static_cast<char*>(recv);
  if (send != out + root * per) std::memcpy(out + root * per, send, per);
  for (int i = 0; i < n; ++i)
    if (i != root)
      if (hxcclResult_t r = tr.irecv(i, out + i * per, per, kCollTag)) {
        tr.cancel_all();
        return r;
      }
  return tr.waitall();
}

hxcclResult_t scatter(Transport& tr, int rank, int n, const void* send, void* recv,
                      size_t per, int root) {
  if (per == 0) return hxcclSuccess;
  if (rank != root) return tr.recv(root, recv, per);
  const char* in = static_cast<const char*>(send);
  for (int i = 0; i < n; ++i)
    if (i != root)
      if (hxcclResult_t r = tr.isend(i, in + i * per, per, kCollTag)) {
        tr.cancel_all();
        return r;
      }
  if (recv != in + root * per) std::memcpy(recv, in + root * per, per);
  return tr.waitall();
}
