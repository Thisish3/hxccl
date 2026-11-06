#include "comm.h"

extern "C" {

hxcclResult_t hxcclAllReduce(const void*, void*, size_t, hxcclDataType_t, hxcclRedOp_t,
                             hxcclComm_t c, hxcclStream_t) {
  return comm_fail(c, hxcclInternalError, "not implemented");
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
