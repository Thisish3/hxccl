#include "algo.h"
#include "checks.h"
#include "chunks.h"
#include "comm.h"
#include "group.h"
#include "memspace.h"
#include "reduce.h"
#include "select.h"
#include "trace.h"

#include <cstring>
#include <vector>

extern "C" {

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t c, hxcclStream_t stream) {
  if (hxcclResult_t r = check_args(c, send, recv, count, 1, dt)) return r;
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p)) return r;
  const size_t es = dtype_size(dt);
  const Algo algo = choose_allreduce(count * es);
  const size_t largest =
      algo == Algo::Ring ? chunk_of(count, c->nranks, 0).len * es : count * es;
  if (hxcclResult_t r = check_message(c, largest)) return r;
  return dispatch(c, [=] {
    hxcclResult_t r;
    MemOps* m = ops_for(c, stream, send, recv, &r);
    if (!m) return r;
    HX_RANGE_PUSH("hxccl:allreduce");
    {
      Scratch tmp(*m, p.premul ? count * es : 0);
      const void* in = premul_input(*m, p, send, count, dt, tmp, &r);
      if (r == hxcclSuccess) r = allreduce_with(*m, c, algo, in, recv, count, dt, p.base);
      if (r == hxcclSuccess) r = finish_op(*m, p, recv, count, dt);
    }
    if (r == hxcclSuccess) r = sync_ops(*m);
    HX_RANGE_POP();
    return ops_fail(c, *m, r);
  });
}
hxcclResult_t hxcclAllGather(const void* send, void* recv, size_t sendcount,
                             hxcclDataType_t dt, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, sendcount, c ? c->nranks : 1, dt)) return r;
  if (hxcclResult_t r = check_message(c, sendcount * dtype_size(dt))) return r;
  return dispatch(c, [=] {
    MemOps& m = *c->host_ops;
    return ops_fail(
        c, m, ring_allgather(m, c->rank, c->nranks, send, recv, sendcount, dtype_size(dt)));
  });
}
hxcclResult_t hxcclReduceScatter(const void* send, void* recv, size_t recvcount,
                                 hxcclDataType_t dt, hxcclRedOp_t op, hxcclComm_t c,
                                 hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, recvcount, c ? c->nranks : 1, dt)) return r;
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p)) return r;
  if (hxcclResult_t r = check_message(c, recvcount * dtype_size(dt))) return r;
  return dispatch(c, [=] {
    MemOps& m = *c->host_ops;
    const size_t total = recvcount * c->nranks;
    Scratch tmp(m, p.premul ? total * dtype_size(dt) : 0);
    hxcclResult_t r;
    const void* in = premul_input(m, p, send, total, dt, tmp, &r);
    if (r != hxcclSuccess) return ops_fail(c, m, r);
    r = ring_reducescatter(m, c->rank, c->nranks, in, recv, recvcount, dt, p.base);
    if (r != hxcclSuccess) return ops_fail(c, m, r);
    return ops_fail(c, m, finish_op(m, p, recv, recvcount, dt));
  });
}

hxcclResult_t hxcclBroadcast(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             int root, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, c && c->rank == root ? send : recv, recv, count, 1, dt))
    return r;
  if (root < 0 || root >= c->nranks)
    return comm_fail(c, hxcclInvalidArgument, "root out of range");
  return dispatch(c, [=] {
    MemOps& m = *c->host_ops;
    return ops_fail(c, m,
                    bcast(m, c->rank, c->nranks, send, recv, count * dtype_size(dt), root));
  });
}

hxcclResult_t hxcclRedOpCreatePreMulSum(hxcclRedOp_t* op, void* scalar, hxcclDataType_t dt,
                                        hxcclScalarResidence_t res, hxcclComm_t c) {
  if (!c || !op || !scalar) return hxcclInvalidArgument;
  if (dtype_size(dt) == 0) return comm_fail(c, hxcclInvalidArgument, "unknown datatype");
  if (res != hxcclScalarDevice && res != hxcclScalarHostImmediate)
    return comm_fail(c, hxcclInvalidArgument, "unknown scalar residence");
  size_t ix = 0;
  while (ix < c->user_ops.size() && c->user_ops[ix].live) ++ix;
  if (ix >> kOpIndexBits)
    return comm_fail(c, hxcclInternalError, "too many live PreMulSum ops on this communicator");
  if (ix == c->user_ops.size()) c->user_ops.emplace_back();
  UserOp& u = c->user_ops[ix];
  u = UserOp{};
  u.live = true;
  u.dt = dt;
  u.by_ptr = res == hxcclScalarDevice;  // CPU buffers: "device" means read at call time
  if (u.by_ptr)
    u.ptr = scalar;
  else
    std::memcpy(u.value, scalar, dtype_size(dt));
  *op = static_cast<hxcclRedOp_t>(hxcclNumOps + (c->op_tag << kOpIndexBits) +
                                  static_cast<int>(ix));
  return hxcclSuccess;
}

hxcclResult_t hxcclRedOpDestroy(hxcclRedOp_t op, hxcclComm_t c) {
  if (!c) return hxcclInvalidArgument;
  const long ix = user_op_index(c, op);
  if (ix < 0)
    return comm_fail(c, hxcclInvalidArgument, "not a live PreMulSum op of this communicator");
  c->user_ops[ix].live = false;
  return hxcclSuccess;
}

}  // extern "C"
