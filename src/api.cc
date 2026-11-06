#include "algo.h"
#include "comm.h"

extern "C" {

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t c, hxcclStream_t) {
  hxcclResult_t r = ring_allreduce(*c->tr, c->rank, c->nranks, send, recv, count, dt, op);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
hxcclResult_t hxcclAllGather(const void*, void*, size_t, hxcclDataType_t, hxcclComm_t c,
                             hxcclStream_t) {
  return comm_fail(c, hxcclInternalError, "not implemented");
}
hxcclResult_t hxcclReduceScatter(const void*, void*, size_t, hxcclDataType_t, hxcclRedOp_t,
                                 hxcclComm_t c, hxcclStream_t) {
  return comm_fail(c, hxcclInternalError, "not implemented");
}
hxcclResult_t hxcclBroadcast(const void*, void*, size_t, hxcclDataType_t, int,
                             hxcclComm_t c, hxcclStream_t) {
  return comm_fail(c, hxcclInternalError, "not implemented");
}

}  // extern "C"
