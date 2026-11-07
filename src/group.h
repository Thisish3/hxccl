#pragma once
#include <cstddef>
#include <functional>

#include "comm.h"

// Group semantics (spec 5.2): outside a group these run now; inside, collectives
// queue in call order and Send/Recv are posted together at the outermost GroupEnd.
hxcclResult_t dispatch(hxcclComm* c, std::function<hxcclResult_t()> run);
hxcclResult_t p2p(hxcclComm* c, bool is_send, int peer, void* buf, size_t bytes);
