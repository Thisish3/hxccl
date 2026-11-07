#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// Software float16 / bfloat16 / fp8 (OCP) conversions, round-to-nearest-even.
// IEEE: all-ones exponent is inf/NaN. Otherwise (e4m3fn) only S.1111.111 is NaN.
// Out-of-range values become inf (IEEE) or NaN (e4m3); A2 leaves overflow undefined.
namespace fp {

template <int E, int M, bool IEEE>
float decode(uint32_t v) {
  const int bias = (1 << (E - 1)) - 1;
  const uint32_t emask = (1u << E) - 1, mmask = (1u << M) - 1;
  const uint32_t e = (v >> M) & emask, m = v & mmask;
  const float sign = ((v >> (E + M)) & 1) ? -1.0f : 1.0f;
  if (IEEE && e == emask) return m ? NAN : sign * INFINITY;
  if (!IEEE && e == emask && m == mmask) return NAN;
  if (e == 0) return sign * std::ldexp(static_cast<float>(m), 1 - bias - M);
  return sign * std::ldexp(static_cast<float>(m | (1u << M)), static_cast<int>(e) - bias - M);
}

template <int E, int M, bool IEEE>
uint32_t encode(float x) {
  const int bias = (1 << (E - 1)) - 1, emin = 1 - bias;
  const uint32_t emask = (1u << E) - 1, sbit = 1u << (E + M);
  const uint32_t nan = IEEE ? (emask << M) | (1u << (M - 1)) : (emask << M) | ((1u << M) - 1);
  if (std::isnan(x)) return nan;
  const uint32_t s = std::signbit(x) ? sbit : 0;
  const uint32_t overflow = IEEE ? s | (emask << M) : s | nan;
  const float a = std::fabs(x);
  if (std::isinf(a)) return overflow;
  if (a == 0.0f) return s;
  const int emax = IEEE ? static_cast<int>(emask) - 1 - bias : static_cast<int>(emask) - bias;
  const float maxv =
      std::ldexp(static_cast<float>(IEEE ? (2u << M) - 1 : (2u << M) - 2), emax - M);
  const int scale = std::max(std::ilogb(a), emin) - M;
  const float v = std::ldexp(std::nearbyint(std::ldexp(a, -scale)), scale);
  if (v > maxv) return overflow;
  if (v == 0.0f) return s;
  const int ve = std::ilogb(v);
  if (ve < emin) return s | static_cast<uint32_t>(std::ldexp(v, M - emin));
  const uint32_t mant = static_cast<uint32_t>(std::ldexp(v, M - ve)) - (1u << M);
  return s | (static_cast<uint32_t>(ve + bias) << M) | mant;
}

inline float half_to_float(uint16_t v) { return decode<5, 10, true>(v); }
inline uint16_t float_to_half(float f) { return static_cast<uint16_t>(encode<5, 10, true>(f)); }
inline float bf16_to_float(uint16_t v) { return decode<8, 7, true>(v); }
inline uint16_t float_to_bf16(float f) { return static_cast<uint16_t>(encode<8, 7, true>(f)); }
inline float e4m3_to_float(uint8_t v) { return decode<4, 3, false>(v); }
inline uint8_t float_to_e4m3(float f) { return static_cast<uint8_t>(encode<4, 3, false>(f)); }
inline float e5m2_to_float(uint8_t v) { return decode<5, 2, true>(v); }
inline uint8_t float_to_e5m2(float f) { return static_cast<uint8_t>(encode<5, 2, true>(f)); }

}  // namespace fp
