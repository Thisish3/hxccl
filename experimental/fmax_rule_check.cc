// Build: mpicxx -O3 -std=c++17 -Iinclude -Isrc experimental/fmax_rule_check.cc -o build/fmax_rule_check
// Host-only: hx_fmax/hx_fmin must equal glibc fmaxf/fminf bit for bit on every value the
// small-type path can feed them (all fp8 pairs, 50M random fp16/bf16 pairs, signed zeros, NaN).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "elem.h"

static unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
static long diff = 0, total = 0;
static void cmp(float a, float b) {
  ++total;
  if (bits(hx_fmax(a, b)) != bits(std::fmax(a, b)) || bits(hx_fmin(a, b)) != bits(std::fmin(a, b))) {
    if (diff < 5) std::printf("DIFF a=%08x b=%08x\n", bits(a), bits(b));
    ++diff;
  }
}
int main() {
  for (int x = 0; x < 256; ++x)
    for (int y = 0; y < 256; ++y) {
      cmp(fp::e4m3_to_float(x), fp::e4m3_to_float(y));
      cmp(fp::e5m2_to_float(x), fp::e5m2_to_float(y));
    }
  std::mt19937 g(7);
  for (int i = 0; i < 25000000; ++i) {
    uint16_t x = g(), y = g();
    cmp(fp::half_to_float(x), fp::half_to_float(y));
    cmp(fp::bf16_to_float(x), fp::bf16_to_float(y));
  }
  const float sp[] = {0.0f, -0.0f, NAN, -NAN, INFINITY, -INFINITY, 1.0f};
  for (float a : sp) for (float b : sp) cmp(a, b);
  std::printf("fmax/fmin host check: %ld diffs of %ld pairs\n", diff, total);
  return diff != 0;
}
