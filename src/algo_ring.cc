#include "algo.h"
#include "chunks.h"
#include "reduce.h"

namespace {

int mod(int a, int n) { return ((a % n) + n) % n; }

}  // namespace

// reduce-scatter then all-gather; each rank sends 2(n-1) chunks of ~count/n
hxcclResult_t ring_allreduce(MemOps& m, int rank, int n, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op) {
  const size_t es = dtype_size(dt);
  char* buf = static_cast<char*>(recv);
  hxcclResult_t r;
  if (send != recv && (r = m.copy(buf, send, count * es)) != hxcclSuccess) return r;
  if (n == 1 || count == 0) return hxcclSuccess;
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  Scratch tmp(m, chunk_of(count, n, 0).len * es);
  if (tmp.rc) return tmp.rc;

  for (int s = 0; s < n - 1; ++s) {
    Chunk out = chunk_of(count, n, mod(rank - s, n));
    Chunk in = chunk_of(count, n, mod(rank - s - 1, n));
    if ((r = m.tr().sendrecv(right, buf + out.off * es, out.len * es, left, tmp.get(),
                             in.len * es)) != hxcclSuccess)
      return r;
    if ((r = m.reduce(buf + in.off * es, tmp.get(), in.len, dt, op)) != hxcclSuccess) return r;
  }
  for (int s = 0; s < n - 1; ++s) {
    Chunk out = chunk_of(count, n, mod(rank + 1 - s, n));
    Chunk in = chunk_of(count, n, mod(rank - s, n));
    if ((r = m.tr().sendrecv(right, buf + out.off * es, out.len * es, left, buf + in.off * es,
                             in.len * es)) != hxcclSuccess)
      return r;
  }
  return hxcclSuccess;
}

hxcclResult_t ring_allgather(MemOps& m, int rank, int n, const void* send, void* recv,
                             size_t per, size_t es) {
  char* out = static_cast<char*>(recv);
  const size_t block = per * es;
  hxcclResult_t r;
  if (send != out + rank * block && (r = m.copy(out + rank * block, send, block)) != hxcclSuccess)
    return r;
  if (n == 1 || per == 0) return hxcclSuccess;
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  for (int s = 0; s < n - 1; ++s) {
    const int sb = mod(rank - s, n), rb = mod(rank - s - 1, n);
    if ((r = m.tr().sendrecv(right, out + sb * block, block, left, out + rb * block, block)) !=
        hxcclSuccess)
      return r;
  }
  return hxcclSuccess;
}

hxcclResult_t ring_reducescatter(MemOps& m, int rank, int n, const void* send, void* recv,
                                 size_t per, hxcclDataType_t dt, hxcclRedOp_t op) {
  const size_t es = dtype_size(dt), block = per * es;
  if (per == 0) return hxcclSuccess;
  hxcclResult_t r;
  // work on a copy: with in-place, recv points inside send
  Scratch work(m, block * n);
  if (work.rc) return work.rc;
  if ((r = m.copy(work.get(), send, block * n)) != hxcclSuccess) return r;
  Scratch tmp(m, block);
  if (tmp.rc) return tmp.rc;
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  // shifted by one from the allreduce schedule so rank r ends owning block r
  for (int s = 0; s < n - 1; ++s) {
    const int sb = mod(rank - s - 1, n), rb = mod(rank - s - 2, n);
    if ((r = m.tr().sendrecv(right, work.get() + sb * block, block, left, tmp.get(), block)) !=
        hxcclSuccess)
      return r;
    if ((r = m.reduce(work.get() + rb * block, tmp.get(), per, dt, op)) != hxcclSuccess) return r;
  }
  return m.copy(recv, work.get() + rank * block, block);
}
