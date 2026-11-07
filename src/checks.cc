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
  const long ix = static_cast<long>(op) - hxcclNumOps;
  if (ix < 0 || ix >= static_cast<long>(c->user_ops.size()) || !c->user_ops[ix].live)
    return comm_fail(c, hxcclInvalidArgument, "unknown op");
  const UserOp& u = c->user_ops[ix];
  if (u.dt != dt)
    return comm_fail(c, hxcclInvalidArgument, "PreMulSum op was created for another datatype");
  p->premul = true;
  std::memcpy(p->scalar, u.by_ptr ? u.ptr : u.value, dtype_size(dt));
  return hxcclSuccess;
}

const void* premul_input(const OpPlan& p, const void* send, size_t count, hxcclDataType_t dt,
                         std::vector<char>& tmp) {
  if (!p.premul || count == 0) return send;
  tmp.resize(count * dtype_size(dt));
  scale_into(tmp.data(), send, count, dt, p.scalar);
  return tmp.data();
}

void finish_op(const OpPlan& p, void* buf, size_t count, hxcclDataType_t dt) {
  if (p.postdiv && count) divide_trunc(buf, count, dt, p.divisor);
}
