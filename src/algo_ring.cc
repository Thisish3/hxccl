#include <cstring>
#include <vector>

#include "algo.h"
#include "chunks.h"
#include "reduce.h"

namespace {

int mod(int a, int n) { return ((a % n) + n) % n; }

}  // namespace

// reduce-scatter then all-gather; each rank sends 2(n-1) chunks of ~count/n
hxcclResult_t ring_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op) {
  const size_t es = dtype_size(dt);
  char* buf = static_cast<char*>(recv);
  if (send != recv) std::memcpy(buf, send, count * es);
  if (n == 1 || count == 0) return hxcclSuccess;
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  std::vector<char> tmp(chunk_of(count, n, 0).len * es);

  for (int s = 0; s < n - 1; ++s) {
    Chunk out = chunk_of(count, n, mod(rank - s, n));
    Chunk in = chunk_of(count, n, mod(rank - s - 1, n));
    hxcclResult_t r = tr.sendrecv(right, buf + out.off * es, out.len * es, left, tmp.data(),
                                  in.len * es);
    if (r != hxcclSuccess) return r;
    reduce_into(buf + in.off * es, tmp.data(), in.len, dt, op);
  }
  for (int s = 0; s < n - 1; ++s) {
    Chunk out = chunk_of(count, n, mod(rank + 1 - s, n));
    Chunk in = chunk_of(count, n, mod(rank - s, n));
    hxcclResult_t r =
        tr.sendrecv(right, buf + out.off * es, out.len * es, left, buf + in.off * es,
                    in.len * es);
    if (r != hxcclSuccess) return r;
  }
  return hxcclSuccess;
}

hxcclResult_t ring_allgather(Transport& tr, int rank, int n, const void* send, void* recv,
                             size_t per, size_t es) {
  char* out = static_cast<char*>(recv);
  const size_t block = per * es;
  if (send != out + rank * block) std::memcpy(out + rank * block, send, block);
  if (n == 1 || per == 0) return hxcclSuccess;
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  for (int s = 0; s < n - 1; ++s) {
    const int sb = mod(rank - s, n), rb = mod(rank - s - 1, n);
    hxcclResult_t r = tr.sendrecv(right, out + sb * block, block, left, out + rb * block, block);
    if (r != hxcclSuccess) return r;
  }
  return hxcclSuccess;
}

hxcclResult_t ring_reducescatter(Transport& tr, int rank, int n, const void* send,
                                 void* recv, size_t per, hxcclDataType_t dt, hxcclRedOp_t op) {
  const size_t es = dtype_size(dt), block = per * es;
  if (per == 0) return hxcclSuccess;
  // work on a copy: with in-place, recv points inside send
  std::vector<char> work(static_cast<const char*>(send),
                         static_cast<const char*>(send) + block * n);
  std::vector<char> tmp(block);
  const int right = mod(rank + 1, n), left = mod(rank - 1, n);
  // shifted by one from the allreduce schedule so rank r ends owning block r
  for (int s = 0; s < n - 1; ++s) {
    const int sb = mod(rank - s - 1, n), rb = mod(rank - s - 2, n);
    hxcclResult_t r =
        tr.sendrecv(right, work.data() + sb * block, block, left, tmp.data(), block);
    if (r != hxcclSuccess) return r;
    reduce_into(work.data() + rb * block, tmp.data(), per, dt, op);
  }
  std::memcpy(recv, work.data() + rank * block, block);
  return hxcclSuccess;
}
