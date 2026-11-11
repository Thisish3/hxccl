#include <cstdint>
#include <cstring>

#include "elem.h"
#include "gpu_reduce.h"
#include "reduce.h"

namespace {

constexpr int kThreads = 256;
inline int blocks(size_t n) {
  size_t b = (n + kThreads - 1) / kThreads;
  return static_cast<int>(b < 65535 ? (b ? b : 1) : 65535);
}

#define HX_GRID_LOOP(i, n)                                                  \
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < (n); \
       i += size_t(gridDim.x) * blockDim.x)

template <typename T>
__global__ void k_reduce(T* d, const T* s, size_t n, hxcclRedOp_t op) {
  HX_GRID_LOOP(i, n) d[i] = elem_apply<T>(d[i], s[i], op);
}
template <typename S, float (*ToF)(S), S (*FromF)(float)>
__global__ void k_reduce_small(S* d, const S* s, size_t n, hxcclRedOp_t op) {
  HX_GRID_LOOP(i, n) d[i] = elem_apply_small<S, ToF, FromF>(d[i], s[i], op);
}
template <typename T>
__global__ void k_scale(T* d, const T* s, size_t n, T k) {
  HX_GRID_LOOP(i, n) d[i] = elem_apply<T>(s[i], k, hxcclProd);
}
template <typename S, float (*ToF)(S), S (*FromF)(float)>
__global__ void k_scale_small(S* d, const S* s, size_t n, S k) {
  HX_GRID_LOOP(i, n) d[i] = elem_apply_small<S, ToF, FromF>(s[i], k, hxcclProd);
}
template <typename T, typename W>
__global__ void k_div(T* d, size_t n, int div) {
  HX_GRID_LOOP(i, n) d[i] = elem_div<T, W>(d[i], div);
}

template <typename T>
T scalar_as(const void* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

}  // namespace

// Each switch lists the same (dtype, type, converters) rows as reduce_into in reduce.cc.
#define HX_SMALL_TYPES(X)                                                 \
  X(hxcclFloat16, uint16_t, fp::half_to_float, fp::float_to_half)        \
  X(hxcclBfloat16, uint16_t, fp::bf16_to_float, fp::float_to_bf16)       \
  X(hxcclFloat8e4m3, uint8_t, fp::e4m3_to_float, fp::float_to_e4m3)      \
  X(hxcclFloat8e5m2, uint8_t, fp::e5m2_to_float, fp::float_to_e5m2)

cudaError_t gpu_reduce(void* inout, const void* in, size_t n, hxcclDataType_t dt,
                       hxcclRedOp_t op, cudaStream_t st) {
  if (!n) return cudaSuccess;
#define HX_K(DT, T)                                                                    \
  case DT:                                                                             \
    k_reduce<T><<<blocks(n), kThreads, 0, st>>>(static_cast<T*>(inout),                \
                                                static_cast<const T*>(in), n, op);     \
    break;
#define HX_S(DT, S, TO, FROM)                                                          \
  case DT:                                                                             \
    k_reduce_small<S, TO, FROM><<<blocks(n), kThreads, 0, st>>>(                       \
        static_cast<S*>(inout), static_cast<const S*>(in), n, op);                     \
    break;
  switch (dt) {
    HX_K(hxcclInt8, int8_t)
    HX_K(hxcclUint8, uint8_t)
    HX_K(hxcclInt32, int32_t)
    HX_K(hxcclUint32, uint32_t)
    HX_K(hxcclInt64, int64_t)
    HX_K(hxcclUint64, uint64_t)
    HX_K(hxcclFloat32, float)
    HX_K(hxcclFloat64, double)
    HX_SMALL_TYPES(HX_S)
    default: return cudaErrorInvalidValue;
  }
#undef HX_K
#undef HX_S
  return cudaGetLastError();
}

cudaError_t gpu_scale(void* dst, const void* src, size_t n, hxcclDataType_t dt,
                      const void* host_scalar, cudaStream_t st) {
  if (is_integer(dt) || dtype_size(dt) == 0) return cudaErrorInvalidValue;
  if (!n) return cudaSuccess;
#define HX_K(DT, T)                                                                    \
  case DT:                                                                             \
    k_scale<T><<<blocks(n), kThreads, 0, st>>>(static_cast<T*>(dst),                  \
                                               static_cast<const T*>(src), n,          \
                                               scalar_as<T>(host_scalar));             \
    break;
#define HX_S(DT, S, TO, FROM)                                                          \
  case DT:                                                                             \
    k_scale_small<S, TO, FROM><<<blocks(n), kThreads, 0, st>>>(                        \
        static_cast<S*>(dst), static_cast<const S*>(src), n, scalar_as<S>(host_scalar)); \
    break;
  switch (dt) {
    HX_K(hxcclFloat32, float)
    HX_K(hxcclFloat64, double)
    HX_SMALL_TYPES(HX_S)
    default: return cudaErrorInvalidValue;
  }
#undef HX_K
#undef HX_S
  return cudaGetLastError();
}

cudaError_t gpu_divide_trunc(void* buf, size_t n, hxcclDataType_t dt, int divisor,
                             cudaStream_t st) {
  if (!is_integer(dt)) return cudaErrorInvalidValue;
  if (!n) return cudaSuccess;
#define HX_D(DT, T, W)                                                                    \
  case DT:                                                                                \
    k_div<T, W><<<blocks(n), kThreads, 0, st>>>(static_cast<T*>(buf), n, divisor);        \
    break;
  switch (dt) {  // the same (T, W) pairs as HX_DIV in reduce.cc
    HX_D(hxcclInt8, int8_t, int64_t)
    HX_D(hxcclUint8, uint8_t, uint64_t)
    HX_D(hxcclInt32, int32_t, int64_t)
    HX_D(hxcclUint32, uint32_t, uint64_t)
    HX_D(hxcclInt64, int64_t, int64_t)
    HX_D(hxcclUint64, uint64_t, uint64_t)
    default: return cudaErrorInvalidValue;
  }
#undef HX_D
  return cudaGetLastError();
}
