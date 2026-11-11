#pragma once
#include <cstddef>
#include <functional>

#include "comm.h"
#include "memops.h"

// Group semantics (spec 5.2): outside a group these run now; inside, collectives
// queue in call order and Send/Recv are posted together at the outermost GroupEnd.
hxcclResult_t dispatch(hxcclComm* c, std::function<hxcclResult_t()> run);
// `m` is the memory path of `buf` (ops_for). Host and device Send/Recv cannot share one
// group: GroupEnd rejects that with hxcclInvalidUsage.
hxcclResult_t p2p(hxcclComm* c, bool is_send, int peer, void* buf, size_t bytes, MemOps* m);
