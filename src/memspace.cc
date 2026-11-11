// CPU build (HXCCL_CUDA off): every buffer is host memory.
#include "memspace.h"

MemOps* ops_for(hxcclComm* c, hxcclStream_t, const void*, const void*, hxcclResult_t* rc) {
  *rc = hxcclSuccess;
  return c->host_ops.get();
}

hxcclResult_t sync_ops(MemOps&) { return hxcclSuccess; }
