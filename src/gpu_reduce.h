#pragma once
// Device reductions (HXCCL_CUDA only). Same element rules as reduce.cc, so the results
// match the CPU path bit for bit. Kernels are queued on `s`; the caller synchronizes.
#include <cuda_runtime.h>

#include <cstddef>

#include "hxccl.h"

// inout[i] = inout[i] op in[i] for the four builtin ops
cudaError_t gpu_reduce(void* inout, const void* in, size_t count, hxcclDataType_t dt,
                       hxcclRedOp_t op, cudaStream_t s);
// dst[i] = src[i] * scalar; floating dtypes only, host_scalar is one dt element in host memory
cudaError_t gpu_scale(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                      const void* host_scalar, cudaStream_t s);
// buf[i] /= divisor truncating toward zero; integer dtypes only
cudaError_t gpu_divide_trunc(void* buf, size_t count, hxcclDataType_t dt, int divisor,
                             cudaStream_t s);
