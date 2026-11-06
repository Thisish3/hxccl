#pragma once
#include <cstddef>

#include "hxccl.h"
#include "transport.h"

hxcclResult_t ring_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op);

hxcclResult_t rd_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                           size_t count, hxcclDataType_t dt, hxcclRedOp_t op);
