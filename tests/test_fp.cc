#include <cmath>
#include <cstdint>

#include "check.h"
#include "fp.h"

using namespace fp;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  if (r == 0) {
    // fp16 (IEEE binary16)
    CHECK(float_to_half(1.0f) == 0x3c00 && float_to_half(-0.0f) == 0x8000);
    CHECK(float_to_half(65504.0f) == 0x7bff && float_to_half(65520.0f) == 0x7c00);
    CHECK(float_to_half(std::ldexp(1.0f, -24)) == 0x0001);  // smallest subnormal
    CHECK(float_to_half(std::ldexp(1.0f, -25)) == 0x0000);  // tie -> even (0)
    CHECK(float_to_half(std::ldexp(3.0f, -25)) == 0x0002);  // 1.5 ulp -> 2 (even)
    CHECK(float_to_half(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);  // tie -> even
    CHECK(float_to_half(1.0f + std::ldexp(3.0f, -11)) == 0x3c02);
    CHECK(std::isinf(half_to_float(0x7c00)) && std::isnan(half_to_float(0x7e00)));
    CHECK(std::isnan(half_to_float(float_to_half(NAN))));
    // bf16
    CHECK(float_to_bf16(1.0f) == 0x3f80 &&
          float_to_bf16(1.0f + std::ldexp(1.0f, -8)) == 0x3f80);
    CHECK(float_to_bf16(1.0f + std::ldexp(3.0f, -8)) == 0x3f82);
    CHECK(float_to_bf16(bf16_to_float(0x7f7f)) == 0x7f7f);  // largest finite
    // fp8 e4m3 (OCP E4M3FN: bias 7, max 448, no inf, S.1111.111 = NaN)
    CHECK(float_to_e4m3(1.0f) == 0x38 && float_to_e4m3(448.0f) == 0x7e);
    CHECK(float_to_e4m3(464.0f) == 0x7e);  // tie -> even mantissa 110
    CHECK(float_to_e4m3(std::ldexp(1.0f, -9)) == 0x01);
    CHECK(e4m3_to_float(0x78) == 256.0f && std::isnan(e4m3_to_float(0x7f)));
    // fp8 e5m2 (IEEE-like: bias 15, max 57344, inf)
    CHECK(float_to_e5m2(1.0f) == 0x3c && float_to_e5m2(57344.0f) == 0x7b);
    CHECK(std::isinf(e5m2_to_float(0x7c)) && float_to_e5m2(std::ldexp(1.0f, -16)) == 0x01);
    // every finite code round-trips
    for (uint32_t v = 0; v < 256; ++v) {
      const uint8_t b = static_cast<uint8_t>(v);
      if (!std::isnan(e4m3_to_float(b))) CHECK(float_to_e4m3(e4m3_to_float(b)) == b);
      if (!std::isnan(e5m2_to_float(b))) CHECK(float_to_e5m2(e5m2_to_float(b)) == b);
    }
    int bad = 0;
    for (uint32_t v = 0; v < 65536; ++v) {
      const uint16_t h = static_cast<uint16_t>(v);
      if (!std::isnan(half_to_float(h)) && float_to_half(half_to_float(h)) != h) ++bad;
      if (!std::isnan(bf16_to_float(h)) && float_to_bf16(bf16_to_float(h)) != h) ++bad;
    }
    CHECK(bad == 0);
  }
  return finish();
}
