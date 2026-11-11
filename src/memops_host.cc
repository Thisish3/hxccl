#include <cstdlib>
#include <cstring>

#include "memops.h"
#include "reduce.h"

hxcclResult_t HostOps::copy(void* dst, const void* src, size_t bytes) {
  if (bytes && dst != src) std::memmove(dst, src, bytes);
  return hxcclSuccess;
}
hxcclResult_t HostOps::reduce(void* inout, const void* in, size_t count, hxcclDataType_t dt,
                              hxcclRedOp_t op) {
  reduce_into(inout, in, count, dt, op);
  return hxcclSuccess;
}
hxcclResult_t HostOps::scale(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                             const void* scalar) {
  scale_into(dst, src, count, dt, scalar);
  return hxcclSuccess;
}
hxcclResult_t HostOps::divide_trunc(void* buf, size_t count, hxcclDataType_t dt, int divisor) {
  ::divide_trunc(buf, count, dt, divisor);
  return hxcclSuccess;
}
hxcclResult_t HostOps::alloc(size_t bytes, void** p) {
  *p = std::malloc(bytes);
  if (!*p) {
    error = "host allocation failed";
    return hxcclSystemError;
  }
  return hxcclSuccess;
}
void HostOps::release(void* p) { std::free(p); }
