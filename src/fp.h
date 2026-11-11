#pragma once
#include <math.h>

#include <cmath>
#include <cstdint>

#include "gpu.h"

// Software float16 / bfloat16 / fp8 (OCP) conversions, round-to-nearest-even.
// IEEE: all-ones exponent is inf/NaN. Otherwise (e4m3fn) only S.1111.111 is NaN.
// Out-of-range values become inf (IEEE) or NaN (e4m3); A2 leaves overflow undefined.
namespace fp {

template <int E, int M, bool IEEE>
HX_HD float decode(uint32_t v) {
  const int bias = (1 << (E - 1)) - 1;
  const uint32_t emask = (1u << E) - 1, mmask = (1u << M) - 1;
  const uint32_t e = (v >> M) & emask, m = v & mmask;
  const float sign = ((v >> (E + M)) & 1) ? -1.0f : 1.0f;
  if (IEEE && e == emask) return m ? NAN : sign * INFINITY;
  if (!IEEE && e == emask && m == mmask) return NAN;
  if (e == 0) return sign * ldexpf(static_cast<float>(m), 1 - bias - M);
  return sign * ldexpf(static_cast<float>(m | (1u << M)), static_cast<int>(e) - bias - M);
}

template <int E, int M, bool IEEE>
HX_HD uint32_t encode(float x) {
  const int bias = (1 << (E - 1)) - 1, emin = 1 - bias;
  const uint32_t emask = (1u << E) - 1, sbit = 1u << (E + M);
  const uint32_t nan = IEEE ? (emask << M) | (1u << (M - 1)) : (emask << M) | ((1u << M) - 1);
  if (isnan(x)) return nan;
  const uint32_t s = signbit(x) ? sbit : 0;
  const uint32_t overflow = IEEE ? s | (emask << M) : s | nan;
  const float a = fabsf(x);
  if (isinf(a)) return overflow;
  if (a == 0.0f) return s;
  const int emax = IEEE ? static_cast<int>(emask) - 1 - bias : static_cast<int>(emask) - bias;
  const float maxv =
      ldexpf(static_cast<float>(IEEE ? (2u << M) - 1 : (2u << M) - 2), emax - M);
  const int scale = (ilogbf(a) < emin ? emin : ilogbf(a)) - M;
  const float v = ldexpf(nearbyintf(ldexpf(a, -scale)), scale);
  if (v > maxv) return overflow;
  if (v == 0.0f) return s;
  const int ve = ilogbf(v);
  if (ve < emin) return s | static_cast<uint32_t>(ldexpf(v, M - emin));
  const uint32_t mant = static_cast<uint32_t>(ldexpf(v, M - ve)) - (1u << M);
  return s | (static_cast<uint32_t>(ve + bias) << M) | mant;
}

HX_HD inline float half_to_float(uint16_t v) { return decode<5, 10, true>(v); }
HX_HD inline uint16_t float_to_half(float f) { return static_cast<uint16_t>(encode<5, 10, true>(f)); }
HX_HD inline float bf16_to_float(uint16_t v) { return decode<8, 7, true>(v); }
HX_HD inline uint16_t float_to_bf16(float f) { return static_cast<uint16_t>(encode<8, 7, true>(f)); }
HX_HD inline float e4m3_to_float(uint8_t v) { return decode<4, 3, false>(v); }
HX_HD inline uint8_t float_to_e4m3(float f) { return static_cast<uint8_t>(encode<4, 3, false>(f)); }
HX_HD inline float e5m2_to_float(uint8_t v) { return decode<5, 2, true>(v); }
HX_HD inline uint8_t float_to_e5m2(float f) { return static_cast<uint8_t>(encode<5, 2, true>(f)); }

}  // namespace fp
