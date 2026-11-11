#include "checks.h"
#include "comm.h"
#include "group.h"
#include "memspace.h"
#include "reduce.h"

namespace {
hxcclResult_t check_p2p(hxcclComm* c, const void* buf, size_t count, hxcclDataType_t dt,
                        int peer) {
  if (hxcclResult_t r = check_args(c, buf, buf, count, 1, dt)) return r;
  if (peer < 0 || peer >= c->nranks)
    return comm_fail(c, hxcclInvalidArgument, "peer out of range");
  return check_message(c, count * dtype_size(dt));
}
}  // namespace

extern "C" {

// The memory path is chosen at call time (the first device call starts the device path on
// every rank that makes one); the transfer itself runs now or at GroupEnd.
hxcclResult_t hxcclSend(const void* buf, size_t count, hxcclDataType_t dt, int peer,
                        hxcclComm_t c, hxcclStream_t stream) {
  if (hxcclResult_t r = check_p2p(c, buf, count, dt, peer)) return r;
  hxcclResult_t r;
  MemOps* m = ops_for(c, stream, buf, nullptr, &r);
  if (!m) return r;
  return p2p(c, true, peer, const_cast<void*>(buf), count * dtype_size(dt), m);
}

hxcclResult_t hxcclRecv(void* buf, size_t count, hxcclDataType_t dt, int peer, hxcclComm_t c,
                        hxcclStream_t stream) {
  if (hxcclResult_t r = check_p2p(c, buf, count, dt, peer)) return r;
  hxcclResult_t r;
  MemOps* m = ops_for(c, stream, buf, nullptr, &r);
  if (!m) return r;
  return p2p(c, false, peer, buf, count * dtype_size(dt), m);
}

}  // extern "C"
