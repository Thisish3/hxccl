#pragma once
#include <cstddef>

#include "comm.h"
#include "memops.h"

// Shared front-end checks; nothing is sent when they fail.
hxcclResult_t check_args(hxcclComm* c, const void* send, const void* recv, size_t count,
                         size_t multiplier, hxcclDataType_t dt);
// The MPI transport carries at most INT_MAX bytes per message. Every rank
// computes the same largest message, so all of them reject together.
hxcclResult_t check_message(hxcclComm* c, size_t largest_bytes);

// A reduction op resolved at call time: the algorithms only see `base`.
struct OpPlan {
  hxcclRedOp_t base = hxcclSum;
  bool premul = false;  // multiply the local input by `scalar` first
  unsigned char scalar[8] = {};
  bool postdiv = false;  // integer Avg: divide the result by `divisor`
  int divisor = 1;
};
// `stream`: a PreMulSum scalar in device memory (hxcclScalarDevice) is read on it at call time.
hxcclResult_t plan_op(hxcclComm* c, hxcclRedOp_t op, hxcclDataType_t dt, OpPlan* p,
                      hxcclStream_t stream);
// Returns the buffer the algorithm should read: `send`, or `tmp` holding send * scalar.
// `tmp` must hold count elements when p.premul.
const void* premul_input(MemOps& m, const OpPlan& p, const void* send, size_t count,
                         hxcclDataType_t dt, Scratch& tmp, hxcclResult_t* rc);
hxcclResult_t finish_op(MemOps& m, const OpPlan& p, void* buf, size_t count,
                        hxcclDataType_t dt);

// Records a failure of `m` (its own error, else its transport's) on the communicator.
hxcclResult_t ops_fail(hxcclComm* c, MemOps& m, hxcclResult_t r);
