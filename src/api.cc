#include "algo.h"
#include "checks.h"
#include "chunks.h"
#include "comm.h"
#include "reduce.h"
#include "select.h"

#include <cstring>
#include <vector>

extern "C" {

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, count, 1, dt)) return r;
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p)) return r;
  const size_t es = dtype_size(dt);
  const Algo algo = choose_allreduce(count * es);
  const size_t largest =
      algo == Algo::Ring ? chunk_of(count, c->nranks, 0).len * es : count * es;
  if (hxcclResult_t r = check_message(c, largest)) return r;
  std::vector<char> tmp;
  const void* in = premul_input(p, send, count, dt, tmp);
  hxcclResult_t r = allreduce_with(c, algo, in, recv, count, dt, p.base);
  if (r == hxcclSuccess) finish_op(p, recv, count, dt);
  return r;
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
  OpPlan p;
  if (hxcclResult_t r = plan_op(c, op, dt, &p)) return r;
  if (hxcclResult_t r = check_message(c, recvcount * dtype_size(dt))) return r;
  std::vector<char> tmp;
  const void* in = premul_input(p, send, recvcount * c->nranks, dt, tmp);
  hxcclResult_t r =
      ring_reducescatter(*c->tr, c->rank, c->nranks, in, recv, recvcount, dt, p.base);
  if (r != hxcclSuccess) return comm_fail(c, r, c->tr->error);
  finish_op(p, recv, recvcount, dt);
  return r;
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

hxcclResult_t hxcclRedOpCreatePreMulSum(hxcclRedOp_t* op, void* scalar, hxcclDataType_t dt,
                                        hxcclScalarResidence_t res, hxcclComm_t c) {
  if (!c || !op || !scalar) return hxcclInvalidArgument;
  if (dtype_size(dt) == 0) return comm_fail(c, hxcclInvalidArgument, "unknown datatype");
  if (res != hxcclScalarDevice && res != hxcclScalarHostImmediate)
    return comm_fail(c, hxcclInvalidArgument, "unknown scalar residence");
  size_t ix = 0;
  while (ix < c->user_ops.size() && c->user_ops[ix].live) ++ix;
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
  *op = static_cast<hxcclRedOp_t>(hxcclNumOps + static_cast<int>(ix));
  return hxcclSuccess;
}

hxcclResult_t hxcclRedOpDestroy(hxcclRedOp_t op, hxcclComm_t c) {
  if (!c) return hxcclInvalidArgument;
  const long ix = static_cast<long>(op) - hxcclNumOps;
  if (ix < 0 || ix >= static_cast<long>(c->user_ops.size()) || !c->user_ops[ix].live)
    return comm_fail(c, hxcclInvalidArgument, "not a live PreMulSum op of this communicator");
  c->user_ops[ix].live = false;
  return hxcclSuccess;
}

}  // extern "C"
