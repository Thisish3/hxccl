#pragma once
#include <cstddef>

#include "hxccl.h"
#include "memops.h"

// Every algorithm does its memory work through `m` and its messages through `m.tr()`.

hxcclResult_t ring_allreduce(MemOps& m, int rank, int n, const void* send, void* recv,
                             size_t count, hxcclDataType_t dt, hxcclRedOp_t op);

hxcclResult_t rd_allreduce(MemOps& m, int rank, int n, const void* send, void* recv,
                           size_t count, hxcclDataType_t dt, hxcclRedOp_t op);

// rank r contributes `per` elements and receives all n*per (NCCL sendcount semantics)
hxcclResult_t ring_allgather(MemOps& m, int rank, int n, const void* send, void* recv,
                             size_t per, size_t elem_size);
// `send` holds n*per elements, rank r receives reduced block r (NCCL recvcount semantics)
hxcclResult_t ring_reducescatter(MemOps& m, int rank, int n, const void* send, void* recv,
                                 size_t per, hxcclDataType_t dt, hxcclRedOp_t op);
// binomial tree below kChainMinBytes, else a chain in kChainSegment pieces
constexpr size_t kChainMinBytes = 512 * 1024;
constexpr size_t kChainSegment = 512 * 1024;
hxcclResult_t bcast(MemOps& m, int rank, int n, const void* send, void* recv, size_t bytes,
                    int root);

// Rooted collectives. Non-root `recv` (reduce, gather) and `send` (scatter) may be NULL.
hxcclResult_t reduce_root(MemOps& m, int rank, int n, const void* send, void* recv,
                          size_t count, hxcclDataType_t dt, hxcclRedOp_t op, int root);
hxcclResult_t gather(MemOps& m, int rank, int n, const void* send, void* recv,
                     size_t bytes_per, int root);
hxcclResult_t scatter(MemOps& m, int rank, int n, const void* send, void* recv,
                      size_t bytes_per, int root);
// byte counts and byte offsets per peer
hxcclResult_t alltoallv(MemOps& m, int rank, int n, const char* send, const size_t* sbytes,
                        const size_t* soff, char* recv, const size_t* rbytes,
                        const size_t* roff);
