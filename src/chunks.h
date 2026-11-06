#pragma once
#include <cstddef>

// Ring chunk i of a buffer with `count` elements split over n ranks: the first
// count % n chunks hold one extra element. Empty chunks sit at the end.
struct Chunk {
  size_t off;
  size_t len;
};

inline Chunk chunk_of(size_t count, int n, int i) {
  size_t base = count / static_cast<size_t>(n), extra = count % static_cast<size_t>(n);
  size_t ui = static_cast<size_t>(i);
  size_t len = base + (ui < extra ? 1 : 0);
  size_t off = ui * base + (ui < extra ? ui : extra);
  return {off, len};
}
