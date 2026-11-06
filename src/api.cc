#include "algo.h"
#include "comm.h"
#include "reduce.h"
#include "select.h"

#include <climits>
#include <cstdint>
#include <string>

#include "chunks.h"

namespace {

// Shared front-end checks; nothing is sent when they fail (spec 5).
hxcclResult_t check_args(hxcclComm* c, const void* send, const void* recv, size_t count,
                         size_t multiplier, hxcclDataType_t dt) {
  if (!c) return hxcclInvalidArgument;
  const size_t es = dtype_size(dt);
  if (es == 0) return comm_fail(c, hxcclInvalidArgument, "unknown datatype");
  if (count != 0 && count > SIZE_MAX / es / multiplier)
    return comm_fail(c, hxcclInvalidArgument, "count * sizeof(datatype) overflows size_t");
  if (count != 0 && (!send || !recv))
    return comm_fail(c, hxcclInvalidArgument, "null buffer with count > 0");
  return hxcclSuccess;
}

// The MPI transport carries at most INT_MAX bytes per message. Every rank
// computes the same largest message, so all of them reject together.
hxcclResult_t check_message(hxcclComm* c, size_t largest_bytes) {
  if (largest_bytes <= static_cast<size_t>(INT_MAX)) return hxcclSuccess;
  return comm_fail(c, hxcclInvalidArgument,
                   "a single message would exceed INT_MAX bytes (MPI transport limit)");
}

hxcclResult_t check_op(hxcclComm* c, hxcclRedOp_t op) {
  return valid_op(op) ? hxcclSuccess : comm_fail(c, hxcclInvalidArgument, "unknown op");
}

}  // namespace

extern "C" {

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, count, 1, dt)) return r;
  if (hxcclResult_t r = check_op(c, op)) return r;
  const size_t es = dtype_size(dt);
  const Algo algo = choose_allreduce(count * es);
  const size_t largest =
      algo == Algo::Ring ? chunk_of(count, c->nranks, 0).len * es : count * es;
  if (hxcclResult_t r = check_message(c, largest)) return r;
  return allreduce_with(c, algo, send, recv, count, dt, op);
}
hxcclResult_t hxcclAllGather(const void* send, void* recv, size_t sendcount,
                             hxcclDataType_t dt, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, sendcount, c ? c->nranks : 1, dt)) return r;
  if (hxcclResult_t r = check_message(c, sendcount * dtype_size(dt))) return r;
  hxcclResult_t r =
      ring_allgather(*c->tr, c->rank, c->nranks, send, recv, sendcount, dtype_size(dt));
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
hxcclResult_t hxcclReduceScatter(const void* send, void* recv, size_t recvcount,
                                 hxcclDataType_t dt, hxcclRedOp_t op, hxcclComm_t c,
                                 hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, recvcount, c ? c->nranks : 1, dt)) return r;
  if (hxcclResult_t r = check_op(c, op)) return r;
  if (hxcclResult_t r = check_message(c, recvcount * dtype_size(dt))) return r;
  hxcclResult_t r =
      ring_reducescatter(*c->tr, c->rank, c->nranks, send, recv, recvcount, dt, op);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
hxcclResult_t hxcclBroadcast(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             int root, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, c && c->rank == root ? send : recv, recv, count, 1, dt))
    return r;
  if (root < 0 || root >= c->nranks)
    return comm_fail(c, hxcclInvalidArgument, "root out of range");
  hxcclResult_t r =
      bcast(*c->tr, c->rank, c->nranks, send, recv, count * dtype_size(dt), root);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}

}  // extern "C"
