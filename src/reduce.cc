#include "reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

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

// fp16/bf16/fp8: widen to float, reduce, round back. NCCL computes fp8 in half; for one
// binary op on fp8 inputs both are exact before the final rounding, so results match.
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

void store_scalar(double v, hxcclDataType_t dt, void* out) {
  const float f = static_cast<float>(v);
  uint16_t h;
  uint8_t b;
  switch (dt) {
    case hxcclFloat64: std::memcpy(out, &v, 8); break;
    case hxcclFloat32: std::memcpy(out, &f, 4); break;
    case hxcclFloat16: h = fp::float_to_half(f); std::memcpy(out, &h, 2); break;
    case hxcclBfloat16: h = fp::float_to_bf16(f); std::memcpy(out, &h, 2); break;
    case hxcclFloat8e4m3: b = fp::float_to_e4m3(f); std::memcpy(out, &b, 1); break;
    case hxcclFloat8e5m2: b = fp::float_to_e5m2(f); std::memcpy(out, &b, 1); break;
    default: break;
  }
}

void scale_into(void* dst, const void* src, size_t n, hxcclDataType_t dt, const void* scalar) {
  const size_t es = dtype_size(dt);
  std::memcpy(dst, src, n * es);
  std::vector<char> k(n * es);
  for (size_t i = 0; i < n; ++i) std::memcpy(k.data() + i * es, scalar, es);
  reduce_into(dst, k.data(), n, dt, hxcclProd);
}

void divide_trunc(void* buf, size_t n, hxcclDataType_t dt, int divisor) {
  // divide in 64 bits: nranks need not fit the element type (int8 at 200 ranks)
#define HX_DIV(DT, T, W)                                                               \
  case DT:                                                                             \
    for (size_t i = 0; i < n; ++i)                                                     \
      static_cast<T*>(buf)[i] =                                                        \
          static_cast<T>(static_cast<W>(static_cast<T*>(buf)[i]) / static_cast<W>(divisor)); \
    break;
  switch (dt) {
    HX_DIV(hxcclInt8, int8_t, int64_t)
    HX_DIV(hxcclUint8, uint8_t, uint64_t)
    HX_DIV(hxcclInt32, int32_t, int64_t)
    HX_DIV(hxcclUint32, uint32_t, uint64_t)
    HX_DIV(hxcclInt64, int64_t, int64_t)
    HX_DIV(hxcclUint64, uint64_t, uint64_t)
    default: break;
  }
#undef HX_DIV
}
