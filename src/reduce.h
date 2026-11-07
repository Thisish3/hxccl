#pragma once
#include <cstddef>

#include "hxccl.h"

size_t dtype_size(hxcclDataType_t dt);  // 0 for an unknown dtype
bool valid_op(hxcclRedOp_t op);  // the four builtin elementwise ops
bool is_integer(hxcclDataType_t dt);
// dst[i] = dst[i] op src[i]
void reduce_into(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                 hxcclRedOp_t op);

// Avg and PreMulSum support (NCCL src/enqueue/enqueue.cc translates them the same way).
void store_scalar(double v, hxcclDataType_t dt, void* out);  // floating dtypes only
// dst[i] = src[i] * scalar, rounded to dt
void scale_into(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                const void* scalar);
// integer dtypes only, truncating toward zero
void divide_trunc(void* buf, size_t count, hxcclDataType_t dt, int divisor);
