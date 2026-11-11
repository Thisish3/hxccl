#pragma once
#include "checks.h"
#include "comm.h"
#include "memops.h"
#include "memspace.h"
#include "trace.h"

// The body of every collective: pick the memory path from the buffers this rank uses
// (a, b; NULL is ignored), run `f(MemOps&)`, wait for device work, record a failure.
// The call returns when the work is done (spec 3.5).
template <typename F>
hxcclResult_t run_on(hxcclComm* c, hxcclStream_t stream, const void* a, const void* b,
                     const char* range, F f) {
  hxcclResult_t r;
  MemOps* m = ops_for(c, stream, a, b, &r);
  if (!m) return r;
  HX_RANGE_PUSH(range);
  r = f(*m);
  if (r == hxcclSuccess) r = sync_ops(*m);
  HX_RANGE_POP();
  (void)range;
  return ops_fail(c, *m, r);
}
