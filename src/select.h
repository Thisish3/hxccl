#pragma once
#include <cstddef>

#include "comm.h"

enum class Algo { Ring, Rd };

// HXCCL_ALGO=ring|rd wins; otherwise rd below kRdMaxBytes (M2 measurements in
// allreduce-practice crossed over between 64 and 256 KB).
constexpr size_t kRdMaxBytes = 64 * 1024;
Algo choose_allreduce(size_t bytes);

hxcclResult_t allreduce_with(hxcclComm* c, Algo algo, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op);
