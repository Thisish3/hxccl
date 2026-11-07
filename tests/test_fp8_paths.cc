// hxccl reduces fp8 through float; NCCL reduces through half (__hadd/__hmul on half inputs).
// For every pair of finite fp8 inputs, both paths must give the same fp8 sum and product.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "check.h"
#include "fp.h"

namespace {
double round_half(double a) {  // IEEE binary16 round-to-nearest-even from an exact double
  if (std::isnan(a) || std::isinf(a) || a == 0) return a;
  const double m = std::fabs(a);
  const int e = std::max(std::ilogb(m), -14) - 10;
  double v = std::ldexp(std::nearbyint(std::ldexp(m, -e)), e);
  if (v > 65504.0) v = INFINITY;
  return std::copysign(v, a);
}

template <float (*Dec)(uint8_t), uint8_t (*Enc)(float)>
long mismatches() {
  long bad = 0;
  for (int a = 0; a < 256; ++a)
    for (int b = 0; b < 256; ++b) {
      const double x = Dec(static_cast<uint8_t>(a)), y = Dec(static_cast<uint8_t>(b));
      if (std::isnan(x) || std::isnan(y)) continue;
      for (double exact : {x + y, x * y}) {  // exact in double for fp8 inputs
        const uint8_t via_float = Enc(static_cast<float>(exact));
        const uint8_t via_half = Enc(static_cast<float>(round_half(exact)));
        if (via_float != via_half && !(std::isnan(Dec(via_float)) && std::isnan(Dec(via_half))))
          ++bad;
      }
    }
  return bad;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  if (r == 0) {
    CHECK((mismatches<fp::e4m3_to_float, fp::float_to_e4m3>() == 0));
    CHECK((mismatches<fp::e5m2_to_float, fp::float_to_e5m2>() == 0));
  }
  return finish();
}
