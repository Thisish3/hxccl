#include "algo.h"
#include "checks.h"
#include "comm.h"
#include "group.h"
#include "reduce.h"
#include "run.h"

namespace {
hxcclResult_t check_root(hxcclComm* c, int root) {
  return root >= 0 && root < c->nranks
             ? hxcclSuccess
             : comm_fail(c, hxcclInvalidArgument, "root out of range");
}
}  // namespace

// Only the buffers a rank actually uses pick its memory path: non-root ranks pass no
// Reduce/Gather recv and no Scatter send (they may be NULL, Review Focus 1).

extern "C" {

hxcclResult_t hxcclReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                          hxcclRedOp_t op, int root, hxcclComm_t c, hxcclStream_t stream) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r = check_args(c, send, c->rank == root ? recv : send, count, 1, dt))
    return r;
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p, stream)) return r;
  const size_t bytes = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, bytes < kChainMinBytes ? bytes : kChainSegment))
    return r;
  return dispatch(c, [=] {
    const bool is_root = c->rank == root;
    return run_on(c, stream, send, is_root ? recv : nullptr, "hxccl:reduce", [&](MemOps& m) {
      hxcclResult_t r;
      Scratch tmp(m, p.premul ? bytes : 0);
      const void* in = premul_input(m, p, send, count, dt, tmp, &r);
      if (r == hxcclSuccess)
        r = reduce_root(m, c->rank, c->nranks, in, recv, count, dt, p.base, root);
      if (r == hxcclSuccess && is_root) r = finish_op(m, p, recv, count, dt);
      return r;
    });
  });
}

hxcclResult_t hxcclBcast(void* buff, size_t count, hxcclDataType_t dt, int root, hxcclComm_t c,
                         hxcclStream_t s) {
  return hxcclBroadcast(buff, buff, count, dt, root, c, s);
}

hxcclResult_t hxcclGather(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                          int root, hxcclComm_t c, hxcclStream_t stream) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r =
          check_args(c, send, c->rank == root ? recv : send, count, c->nranks, dt))
    return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  return dispatch(c, [=] {
    return run_on(c, stream, send, c->rank == root ? recv : nullptr, "hxccl:gather",
                  [&](MemOps& m) { return gather(m, c->rank, c->nranks, send, recv, per, root); });
  });
}

hxcclResult_t hxcclScatter(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                           int root, hxcclComm_t c, hxcclStream_t stream) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r =
          check_args(c, c->rank == root ? send : recv, recv, count, c->nranks, dt))
    return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  return dispatch(c, [=] {
    return run_on(c, stream, c->rank == root ? send : nullptr, recv, "hxccl:scatter",
                  [&](MemOps& m) { return scatter(m, c->rank, c->nranks, send, recv, per, root); });
  });
}

}  // extern "C"
