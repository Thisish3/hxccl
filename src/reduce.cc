#include "reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "fp.h"

namespace {

template <typename T>
void apply(T* d, const T* s, size_t n, hxcclRedOp_t op) {
  // integer sum/prod wrap like the GPU does; unsigned math keeps signed overflow defined
  using U = typename std::conditional_t<std::is_integral_v<T>, std::make_unsigned<T>,
                                        std::common_type<T>>::type;
  switch (op) {
    case hxcclSum:
      for (size_t i = 0; i < n; ++i)
        d[i] = static_cast<T>(static_cast<U>(d[i]) + static_cast<U>(s[i]));
      break;
    case hxcclProd:
      for (size_t i = 0; i < n; ++i)
        d[i] = static_cast<T>(static_cast<U>(d[i]) * static_cast<U>(s[i]));
      break;
    case hxcclMax:
      for (size_t i = 0; i < n; ++i) d[i] = std::max(d[i], s[i]);
      break;
    case hxcclMin:
      for (size_t i = 0; i < n; ++i) d[i] = std::min(d[i], s[i]);
      break;
    default:
      break;
  }
}

// fp16/bf16/fp8: widen to float, reduce, round back (NCCL does the same, fp8 via half).
template <typename S, float (*ToF)(S), S (*FromF)(float)>
void apply_small(S* d, const S* s, size_t n, hxcclRedOp_t op) {
  for (size_t i = 0; i < n; ++i) {
    const float a = ToF(d[i]), b = ToF(s[i]);
    const float r = op == hxcclSum    ? a + b
                    : op == hxcclProd ? a * b
                    : op == hxcclMax  ? std::fmax(a, b)
                                      : std::fmin(a, b);
    d[i] = FromF(r);
  }
}

}  // namespace

size_t dtype_size(hxcclDataType_t dt) {
  switch (dt) {
    case hxcclInt8:
    case hxcclUint8:
    case hxcclFloat8e4m3:
    case hxcclFloat8e5m2:
      return 1;
    case hxcclFloat16:
    case hxcclBfloat16:
      return 2;
    case hxcclInt32:
    case hxcclUint32:
    case hxcclFloat32:
      return 4;
    case hxcclInt64:
    case hxcclUint64:
    case hxcclFloat64:
      return 8;
    default:
      return 0;
  }
}

bool is_integer(hxcclDataType_t dt) { return dt >= hxcclInt8 && dt <= hxcclUint64; }

bool valid_op(hxcclRedOp_t op) { return op >= hxcclSum && op <= hxcclMin; }

void reduce_into(void* dst, const void* src, size_t n, hxcclDataType_t dt, hxcclRedOp_t op) {
#define HX_CASE(DT, T) \
  case DT:             \
    apply(static_cast<T*>(dst), static_cast<const T*>(src), n, op); \
    break;
#define HX_SMALL(DT, S, TO, FROM)                                                       \
  case DT:                                                                              \
    apply_small<S, TO, FROM>(static_cast<S*>(dst), static_cast<const S*>(src), n, op); \
    break;
  switch (dt) {
    HX_CASE(hxcclInt8, int8_t)
    HX_CASE(hxcclUint8, uint8_t)
    HX_CASE(hxcclInt32, int32_t)
    HX_CASE(hxcclUint32, uint32_t)
    HX_CASE(hxcclInt64, int64_t)
    HX_CASE(hxcclUint64, uint64_t)
    HX_CASE(hxcclFloat32, float)
    HX_CASE(hxcclFloat64, double)
    HX_SMALL(hxcclFloat16, uint16_t, fp::half_to_float, fp::float_to_half)
    HX_SMALL(hxcclBfloat16, uint16_t, fp::bf16_to_float, fp::float_to_bf16)
    HX_SMALL(hxcclFloat8e4m3, uint8_t, fp::e4m3_to_float, fp::float_to_e4m3)
    HX_SMALL(hxcclFloat8e5m2, uint8_t, fp::e5m2_to_float, fp::float_to_e5m2)
    default:
      break;
  }
#undef HX_CASE
#undef HX_SMALL
}
