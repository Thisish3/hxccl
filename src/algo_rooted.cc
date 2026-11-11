#include <algorithm>

#include "algo.h"
#include "reduce.h"

hxcclResult_t reduce_root(MemOps& m, int rank, int n, const void* send, void* recv,
                          size_t count, hxcclDataType_t dt, hxcclRedOp_t op, int root) {
  const size_t es = dtype_size(dt), bytes = count * es;
  hxcclResult_t r;
  // non-root recv may be NULL (NCCL), so partial results live in a private copy
  Scratch own(m, rank == root ? 0 : bytes);
  if (own.rc) return own.rc;
  char* buf;
  if (rank == root) {
    buf = static_cast<char*>(recv);
    if (send != recv && (r = m.copy(buf, send, bytes)) != hxcclSuccess) return r;
  } else {
    buf = own.get();
    if ((r = m.copy(buf, send, bytes)) != hxcclSuccess) return r;
  }
  if (n == 1 || count == 0) return hxcclSuccess;
  const int v = (rank - root + n) % n;
  auto real = [=](int x) { return (x + root) % n; };
  Transport& tr = m.tr();
  if (bytes < kChainMinBytes) {  // binomial tree, the mirror of bcast
    Scratch tmp(m, bytes);
    if (tmp.rc) return tmp.rc;
    for (int mask = 1; mask < n; mask <<= 1) {
      if (v & mask) return tr.send(real(v - mask), buf, bytes);
      if (v + mask < n) {
        if ((r = tr.recv(real(v + mask), tmp.get(), bytes)) != hxcclSuccess) return r;
        if ((r = m.reduce(buf, tmp.get(), count, dt, op)) != hxcclSuccess) return r;
      }
    }
    return hxcclSuccess;
  }
  // chain toward the root in kChainSegment pieces (a multiple of every dtype size)
  Scratch tmp(m, kChainSegment);
  if (tmp.rc) return tmp.rc;
  for (size_t off = 0; off < bytes; off += kChainSegment) {
    const size_t len = std::min(kChainSegment, bytes - off);
    if (v < n - 1) {
      if ((r = tr.recv(real(v + 1), tmp.get(), len)) != hxcclSuccess) return r;
      if ((r = m.reduce(buf + off, tmp.get(), len / es, dt, op)) != hxcclSuccess) return r;
    }
    if (v > 0 && (r = tr.send(real(v - 1), buf + off, len)) != hxcclSuccess) return r;
  }
  return hxcclSuccess;
}

hxcclResult_t gather(MemOps& m, int rank, int n, const void* send, void* recv, size_t per,
                     int root) {
  if (per == 0) return hxcclSuccess;
  Transport& tr = m.tr();
  if (rank != root) return tr.send(root, send, per);
  char* out = static_cast<char*>(recv);
  if (send != out + root * per)
    if (hxcclResult_t r = m.copy(out + root * per, send, per)) return r;
  for (int i = 0; i < n; ++i)
    if (i != root)
      if (hxcclResult_t r = tr.irecv(i, out + i * per, per, kCollTag)) {
        tr.cancel_all();
        return r;
      }
  return tr.waitall();
}

hxcclResult_t scatter(MemOps& m, int rank, int n, const void* send, void* recv, size_t per,
                      int root) {
  if (per == 0) return hxcclSuccess;
  Transport& tr = m.tr();
  if (rank != root) return tr.recv(root, recv, per);
  const char* in = static_cast<const char*>(send);
  for (int i = 0; i < n; ++i)
    if (i != root)
      if (hxcclResult_t r = tr.isend(i, in + i * per, per, kCollTag)) {
        tr.cancel_all();
        return r;
      }
  if (recv != in + root * per)
    if (hxcclResult_t r = m.copy(recv, in + root * per, per)) {
      tr.cancel_all();
      return r;
    }
  return tr.waitall();
}
