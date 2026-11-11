#include "checks.h"

#include <climits>
#include <cstdint>
#include <cstring>

#include "reduce.h"

hxcclResult_t check_args(hxcclComm* c, const void* send, const void* recv, size_t count,
                         size_t multiplier, hxcclDataType_t dt) {
  if (!c) return hxcclInvalidArgument;
  if (c->finalized) return comm_fail(c, hxcclInvalidUsage, "communicator was finalized");
  const size_t es = dtype_size(dt);
  if (es == 0) return comm_fail(c, hxcclInvalidArgument, "unknown datatype");
  if (count != 0 && count > SIZE_MAX / es / multiplier)
    return comm_fail(c, hxcclInvalidArgument, "count * sizeof(datatype) overflows size_t");
  if (count != 0 && (!send || !recv))
    return comm_fail(c, hxcclInvalidArgument, "null buffer with count > 0");
  return hxcclSuccess;
}

hxcclResult_t check_message(hxcclComm* c, size_t largest_bytes) {
  if (largest_bytes <= static_cast<size_t>(INT_MAX)) return hxcclSuccess;
  return comm_fail(c, hxcclInvalidArgument,
                   "a single message would exceed INT_MAX bytes (MPI transport limit)");
}

hxcclResult_t plan_op(hxcclComm* c, hxcclRedOp_t op, hxcclDataType_t dt, OpPlan* p) {
  *p = OpPlan{};
  if (valid_op(op)) {
    p->base = op;
    return hxcclSuccess;
  }
  if (op == hxcclAvg) {  // NCCL: integers sum then divide, floats premultiply by 1/n
    if (is_integer(dt)) {
      p->postdiv = true;
      p->divisor = c->nranks;
    } else {
      p->premul = true;
      store_scalar(1.0 / c->nranks, dt, p->scalar);
    }
    return hxcclSuccess;
  }
  const long ix = user_op_index(c, op);
  if (ix < 0) return comm_fail(c, hxcclInvalidArgument, "unknown op for this communicator");
  const UserOp& u = c->user_ops[ix];
  if (u.dt != dt)
    return comm_fail(c, hxcclInvalidArgument, "PreMulSum op was created for another datatype");
  p->premul = true;
  std::memcpy(p->scalar, u.by_ptr ? u.ptr : u.value, dtype_size(dt));
  return hxcclSuccess;
}

const void* premul_input(MemOps& m, const OpPlan& p, const void* send, size_t count,
                         hxcclDataType_t dt, Scratch& tmp, hxcclResult_t* rc) {
  *rc = hxcclSuccess;
  if (!p.premul || count == 0) return send;
  if ((*rc = tmp.rc) != hxcclSuccess) return nullptr;
  *rc = m.scale(tmp.get(), send, count, dt, p.scalar);
  return tmp.get();
}

hxcclResult_t finish_op(MemOps& m, const OpPlan& p, void* buf, size_t count,
                        hxcclDataType_t dt) {
  return p.postdiv && count ? m.divide_trunc(buf, count, dt, p.divisor) : hxcclSuccess;
}

hxcclResult_t ops_fail(hxcclComm* c, MemOps& m, hxcclResult_t r) {
  if (r == hxcclSuccess) return r;
  std::string msg = m.error.empty() ? m.tr().error : m.error;
  m.error.clear();
  return comm_fail(c, r, msg);
}
