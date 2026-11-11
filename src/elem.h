#pragma once
#include <math.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "fp.h"
#include "gpu.h"
#include "hxccl.h"

// One element of the reductions, shared by the CPU loops (reduce.cc) and the GPU
// kernels (gpu_reduce.cu) so both give the same bits. Two hardware differences are
// spelled out here instead of left to the FPU (measured 2026-11-11, docs/LEDGER.md):
//  - NaN in float/double add and mul: x86 SSE returns the first operand if it is NaN,
//    else the second, quieted; sm_86 fp32 returns the canonical NaN 0x7fffffff.
//  - fmaxf/fminf on +0 vs -0: glibc (maxss/minss) returns the second operand, CUDA
//    orders -0 below +0.

template <typename T>
HX_HD inline T quiet_nan(T x) {
  if constexpr (sizeof(T) == 4) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u |= 0x00400000u;
    memcpy(&x, &u, 4);
  } else {
    uint64_t u;
    memcpy(&u, &x, 8);
    u |= 0x0008000000000000ull;
    memcpy(&x, &u, 8);
  }
  return x;
}

// fmax/fmin as glibc does them on x86-64: a NaN operand yields the other one (two NaNs
// yield the first), equal values (including +0 vs -0) yield the second operand.
HX_HD inline float hx_fmax(float a, float b) {
  return isnan(b) ? a : isnan(a) ? b : (a > b ? a : b);
}
HX_HD inline float hx_fmin(float a, float b) {
  return isnan(b) ? a : isnan(a) ? b : (a < b ? a : b);
}

// one element of dst op src; integer sum/prod wrap like the GPU (unsigned math)
template <typename T>
HX_HD inline T elem_apply(T a, T b, hxcclRedOp_t op) {
  using U = typename std::conditional_t<std::is_integral_v<T>, std::make_unsigned<T>,
                                        std::common_type<T>>::type;
#ifdef __CUDA_ARCH__
  if constexpr (std::is_floating_point_v<T>)
    if ((op == hxcclSum || op == hxcclProd) && (isnan(a) || isnan(b)))
      return quiet_nan(isnan(a) ? a : b);  // the x86 SSE rule, see above
#endif
  switch (op) {
    case hxcclSum: return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
    case hxcclProd: return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
    case hxcclMax: return a < b ? b : a;  // std::max(a, b)
    case hxcclMin: return b < a ? b : a;  // std::min(a, b)
    default: return a;
  }
}

// fp16/bf16/fp8: widen to float, reduce, round back
template <typename S, float (*ToF)(S), S (*FromF)(float)>
HX_HD inline S elem_apply_small(S d, S s, hxcclRedOp_t op) {
  const float a = ToF(d), b = ToF(s);
  const float r = op == hxcclSum    ? a + b
                  : op == hxcclProd ? a * b
                  : op == hxcclMax  ? hx_fmax(a, b)
                                    : hx_fmin(a, b);
  return FromF(r);
}

// integer Avg: divide in the wide type W, truncating toward zero
template <typename T, typename W>
HX_HD inline T elem_div(T v, int d) {
  return static_cast<T>(static_cast<W>(v) / static_cast<W>(d));
}
