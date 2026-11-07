#include <vector>

#include "algo.h"
#include "checks.h"
#include "comm.h"
#include "group.h"
#include "reduce.h"

namespace {
hxcclResult_t check_root(hxcclComm* c, int root) {
  return root >= 0 && root < c->nranks
             ? hxcclSuccess
             : comm_fail(c, hxcclInvalidArgument, "root out of range");
}
hxcclResult_t fail_if(hxcclComm* c, hxcclResult_t r) {
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
}  // namespace

extern "C" {

hxcclResult_t hxcclReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                          hxcclRedOp_t op, int root, hxcclComm_t c, hxcclStream_t) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r = check_args(c, send, c->rank == root ? recv : send, count, 1, dt))
    return r;
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p)) return r;
  const size_t bytes = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, bytes < kChainMinBytes ? bytes : kChainSegment))
    return r;
  return dispatch(c, [=] {
    std::vector<char> tmp;
    const void* in = premul_input(p, send, count, dt, tmp);
    hxcclResult_t r = fail_if(
        c, reduce_root(*c->tr, c->rank, c->nranks, in, recv, count, dt, p.base, root));
    if (r == hxcclSuccess && c->rank == root) finish_op(p, recv, count, dt);
    return r;
  });
}

hxcclResult_t hxcclBcast(void* buff, size_t count, hxcclDataType_t dt, int root, hxcclComm_t c,
                         hxcclStream_t s) {
  return hxcclBroadcast(buff, buff, count, dt, root, c, s);
}

hxcclResult_t hxcclGather(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                          int root, hxcclComm_t c, hxcclStream_t) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r =
          check_args(c, send, c->rank == root ? recv : send, count, c->nranks, dt))
    return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  return dispatch(c, [=] {
    return fail_if(c, gather(*c->tr, c->rank, c->nranks, send, recv, per, root));
  });
}

hxcclResult_t hxcclScatter(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                           int root, hxcclComm_t c, hxcclStream_t) {
  if (!c) return hxcclInvalidArgument;
  if (hxcclResult_t r = check_root(c, root)) return r;
  if (hxcclResult_t r =
          check_args(c, c->rank == root ? send : recv, recv, count, c->nranks, dt))
    return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  return dispatch(c, [=] {
    return fail_if(c, scatter(*c->tr, c->rank, c->nranks, send, recv, per, root));
  });
}

}  // extern "C"
