#pragma once
#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

#include "hxccl.h"

inline hxcclComm_t world_comm() {
  hxcclComm_t c;
  hxcclCommInitMPI(&c, MPI_COMM_WORLD);
  return c;
}
template <typename T>
T* dev_copy(const std::vector<T>& h) {
  T* d;
  cudaMalloc(&d, h.size() ? h.size() * sizeof(T) : 1);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <typename T>
std::vector<T> host_copy(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
