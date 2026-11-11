// Compare hxccl's float -> fp8 conversion with CUDA's __nv_fp8 types on overflow and NaN.
// Build: /usr/local/cuda/bin/nvcc -O2 -arch=sm_86 -std=c++17 experimental/fp8_overflow.cu -o build/fp8_overflow
#include <cuda_fp8.h>

#include <cstdio>

#include "../src/fp.h"

int main() {
  const float xs[] = {448.f, 449.f, 464.f, 480.f, 1000.f, 1e30f, -1e30f, INFINITY, -INFINITY, NAN,
                      57344.f, 61440.f, 65536.f};
  for (float x : xs) {
    unsigned a = fp::float_to_e4m3(x), b = __nv_fp8_e4m3(x).__x;
    unsigned c = fp::float_to_e5m2(x), d = __nv_fp8_e5m2(x).__x;
    std::printf("%-12g e4m3 hxccl=0x%02x cuda=0x%02x %s | e5m2 hxccl=0x%02x cuda=0x%02x %s\n", x,
                a, b, a == b ? "same" : "DIFF", c, d, c == d ? "same" : "DIFF");
  }
}
