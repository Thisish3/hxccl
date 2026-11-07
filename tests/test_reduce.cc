#include "check.h"
#include "chunks.h"
#include "reduce.h"

#include <cstdint>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  if (r == 0) {
    CHECK(dtype_size(hxcclInt32) == 4 && dtype_size(hxcclFloat64) == 8);
    CHECK(dtype_size(static_cast<hxcclDataType_t>(99)) == 0);
    CHECK(valid_op(hxcclMin) && !valid_op(static_cast<hxcclRedOp_t>(9)));

    int a[3] = {1, 5, -2}, b[3] = {4, 2, 3};
    int s[3] = {1, 5, -2};
    reduce_into(s, b, 3, hxcclInt32, hxcclSum);
    CHECK(s[0] == 5 && s[1] == 7 && s[2] == 1);
    int p[3] = {1, 5, -2};
    reduce_into(p, b, 3, hxcclInt32, hxcclProd);
    CHECK(p[0] == 4 && p[1] == 10 && p[2] == -6);
    int mx[3] = {1, 5, -2}, mn[3] = {1, 5, -2};
    reduce_into(mx, b, 3, hxcclInt32, hxcclMax);
    reduce_into(mn, a, 3, hxcclInt32, hxcclMin);
    CHECK(mx[0] == 4 && mx[1] == 5 && mx[2] == 3);
    CHECK(mn[0] == 1 && mn[1] == 5 && mn[2] == -2);

    float f[2] = {1.5f, 2.0f}, g[2] = {0.5f, -4.0f};
    reduce_into(f, g, 2, hxcclFloat32, hxcclSum);
    CHECK(f[0] == 2.0f && f[1] == -2.0f);
    double d[2] = {3.0, -1.0}, e[2] = {2.0, -7.0};
    reduce_into(d, e, 2, hxcclFloat64, hxcclMin);
    CHECK(d[0] == 2.0 && d[1] == -7.0);

    // integer Avg divides by nranks, which may not fit the element type (int8 at 200 ranks)
    int8_t i8 = -100;
    divide_trunc(&i8, 1, hxcclInt8, 200);
    CHECK(i8 == 0);
    uint8_t u8 = 200;
    divide_trunc(&u8, 1, hxcclUint8, 256);
    CHECK(u8 == 0);
    int64_t i64 = -7;
    divide_trunc(&i64, 1, hxcclInt64, 2);
    CHECK(i64 == -3);

    Chunk c0 = chunk_of(7, 3, 0), c1 = chunk_of(7, 3, 1), c2 = chunk_of(7, 3, 2);
    CHECK(c0.off == 0 && c0.len == 3 && c1.off == 3 && c1.len == 2 && c2.off == 5 &&
          c2.len == 2);
    size_t total = 0;
    for (int i = 0; i < 5; ++i) total += chunk_of(2, 5, i).len;
    CHECK(total == 2 && chunk_of(2, 5, 4).len == 0 && chunk_of(2, 5, 4).off == 2);
  }
  return finish();
}
