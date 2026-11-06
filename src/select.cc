#include "select.h"

#include <cstdlib>
#include <cstring>

#include "algo.h"
#include "reduce.h"

Algo choose_allreduce(size_t bytes) {
  if (const char* e = std::getenv("HXCCL_ALGO")) {
    if (std::strcmp(e, "ring") == 0) return Algo::Ring;
    if (std::strcmp(e, "rd") == 0) return Algo::Rd;
  }
  return bytes < kRdMaxBytes ? Algo::Rd : Algo::Ring;
}

hxcclResult_t allreduce_with(hxcclComm* c, Algo algo, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op) {
  hxcclResult_t r = algo == Algo::Ring
                        ? ring_allreduce(*c->tr, c->rank, c->nranks, send, recv, count, dt, op)
                        : rd_allreduce(*c->tr, c->rank, c->nranks, send, recv, count, dt, op);
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}
