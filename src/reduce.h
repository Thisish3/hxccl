#pragma once
#include <cstddef>

#include "hxccl.h"

size_t dtype_size(hxcclDataType_t dt);  // 0 for an unknown dtype
bool valid_op(hxcclRedOp_t op);  // the four builtin elementwise ops
bool is_integer(hxcclDataType_t dt);
// dst[i] = dst[i] op src[i]
void reduce_into(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                 hxcclRedOp_t op);
