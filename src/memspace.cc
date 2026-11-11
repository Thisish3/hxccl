// CPU build (HXCCL_CUDA off): every buffer is host memory.
#include "memspace.h"

#include <cstring>

MemOps* ops_for(hxcclComm* c, hxcclStream_t, const void*, const void*, hxcclResult_t* rc) {
  *rc = hxcclSuccess;
  return c->host_ops.get();
}

hxcclResult_t sync_ops(MemOps&) { return hxcclSuccess; }

void device_eager_init(hxcclComm*) {}

hxcclResult_t read_scalar(void* dst, const void* src, size_t bytes, hxcclStream_t,
                          std::string*) {
  std::memcpy(dst, src, bytes);
  return hxcclSuccess;
}
