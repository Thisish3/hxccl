#pragma once
#include <cstddef>

#include "hxccl.h"
#include "transport.h"

hxcclResult_t ring_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op);

hxcclResult_t rd_allreduce(Transport& tr, int rank, int n, const void* send, void* recv,
                           size_t count, hxcclDataType_t dt, hxcclRedOp_t op);

// rank r contributes `per` elements and receives all n*per (NCCL sendcount semantics)
hxcclResult_t ring_allgather(Transport& tr, int rank, int n, const void* send, void* recv,
                             size_t per, size_t elem_size);
// `send` holds n*per elements, rank r receives reduced block r (NCCL recvcount semantics)
hxcclResult_t ring_reducescatter(Transport& tr, int rank, int n, const void* send,
                                 void* recv, size_t per, hxcclDataType_t dt, hxcclRedOp_t op);
// binomial tree below kChainMinBytes, else a chain in kChainSegment pieces
constexpr size_t kChainMinBytes = 512 * 1024;
constexpr size_t kChainSegment = 512 * 1024;
hxcclResult_t bcast(Transport& tr, int rank, int n, const void* send, void* recv,
                    size_t bytes, int root);

// Rooted collectives. Non-root `recv` (reduce, gather) and `send` (scatter) may be NULL.
hxcclResult_t reduce_root(Transport& tr, int rank, int n, const void* send, void* recv,
                          size_t count, hxcclDataType_t dt, hxcclRedOp_t op, int root);
hxcclResult_t gather(Transport& tr, int rank, int n, const void* send, void* recv,
                     size_t bytes_per, int root);
hxcclResult_t scatter(Transport& tr, int rank, int n, const void* send, void* recv,
                      size_t bytes_per, int root);
