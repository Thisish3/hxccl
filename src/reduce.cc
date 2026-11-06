#include "reduce.h"

#include <algorithm>
#include <cstdint>

namespace {

template <typename T>
void apply(T* d, const T* s, size_t n, hxcclRedOp_t op) {
  switch (op) {
    case hxcclSum:
      for (size_t i = 0; i < n; ++i) d[i] += s[i];
      break;
    case hxcclProd:
      for (size_t i = 0; i < n; ++i) d[i] *= s[i];
      break;
    case hxcclMax:
      for (size_t i = 0; i < n; ++i) d[i] = std::max(d[i], s[i]);
      break;
    case hxcclMin:
      for (size_t i = 0; i < n; ++i) d[i] = std::min(d[i], s[i]);
      break;
  }
}

}  // namespace

size_t dtype_size(hxcclDataType_t dt) {
  switch (dt) {
    case hxcclInt32: return sizeof(int32_t);
    case hxcclFloat32: return sizeof(float);
    case hxcclFloat64: return sizeof(double);
  }
  return 0;
}

bool valid_op(hxcclRedOp_t op) { return op >= hxcclSum && op <= hxcclMin; }

void reduce_into(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                 hxcclRedOp_t op) {
  switch (dt) {
    case hxcclInt32:
      apply(static_cast<int32_t*>(dst), static_cast<const int32_t*>(src), count, op);
      break;
    case hxcclFloat32:
      apply(static_cast<float*>(dst), static_cast<const float*>(src), count, op);
      break;
    case hxcclFloat64:
      apply(static_cast<double*>(dst), static_cast<const double*>(src), count, op);
      break;
  }
}
