#include "algo.h"
#include "comm.h"
#include "reduce.h"
#include "select.h"

extern "C" {

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t c, hxcclStream_t) {
  return allreduce_with(c, choose_allreduce(count * dtype_size(dt)), send, recv, count, dt,
                        op);
}
hxcclResult_t hxcclAllGather(const void* send, void* recv, size_t sendcount,
                             hxcclDataType_t dt, hxcclComm_t c, hxcclStream_t) {
  hxcclResult_t r =
      ring_allgather(*c->tr, c->rank, c->nranks, send, recv, sendcount, dtype_size(dt));
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
hxcclResult_t hxcclReduceScatter(const void* send, void* recv, size_t recvcount,
                                 hxcclDataType_t dt, hxcclRedOp_t op, hxcclComm_t c,
                                 hxcclStream_t) {
  hxcclResult_t r =
      ring_reducescatter(*c->tr, c->rank, c->nranks, send, recv, recvcount, dt, op);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
hxcclResult_t hxcclBroadcast(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             int root, hxcclComm_t c, hxcclStream_t) {
  hxcclResult_t r =
      bcast(*c->tr, c->rank, c->nranks, send, recv, count * dtype_size(dt), root);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}

}  // extern "C"
