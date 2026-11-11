#include <cuda_runtime.h>

#include <string>

#include "gpu_reduce.h"
#include "memops.h"
#include "trace.h"
#include "transport.h"

// Reductions run on the same stream as the IPC copies, so received data is reduced in
// order without extra synchronization. Each API call ends with sync().

DeviceOps::DeviceOps(Net& net, size_t chunk)
    : ipc_(std::make_unique<IpcTransport>(net, chunk)) {}
DeviceOps::~DeviceOps() = default;

hxcclResult_t DeviceOps::init() {
  hxcclResult_t r = ipc_->init();
  if (r) error = ipc_->error;
  return r;
}
void DeviceOps::set_stream(cudaStream_t s) {
  stream_ = s;
  ipc_->set_stream(s);
}
Transport& DeviceOps::tr() { return *ipc_; }

hxcclResult_t DeviceOps::cuda(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return hxcclSuccess;
  error = std::string(what) + ": " + cudaGetErrorString(e);
  return hxcclUnhandledCudaError;
}
hxcclResult_t DeviceOps::copy(void* d, const void* s, size_t b) {
  if (!b || d == s) return hxcclSuccess;
  return cuda(cudaMemcpyAsync(d, s, b, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync");
}
hxcclResult_t DeviceOps::reduce(void* io, const void* in, size_t n, hxcclDataType_t dt,
                                hxcclRedOp_t op) {
  HX_RANGE_PUSH("reduce");
  hxcclResult_t r = cuda(gpu_reduce(io, in, n, dt, op, stream_), "gpu_reduce");
  HX_RANGE_POP();
  return r;
}
hxcclResult_t DeviceOps::scale(void* d, const void* s, size_t n, hxcclDataType_t dt,
                               const void* k) {
  return cuda(gpu_scale(d, s, n, dt, k, stream_), "gpu_scale");
}
hxcclResult_t DeviceOps::divide_trunc(void* b, size_t n, hxcclDataType_t dt, int div) {
  return cuda(gpu_divide_trunc(b, n, dt, div, stream_), "gpu_divide_trunc");
}
hxcclResult_t DeviceOps::alloc(size_t b, void** p) {
  return cuda(cudaMallocAsync(p, b, stream_), "cudaMallocAsync");
}
void DeviceOps::release(void* p) { cudaFreeAsync(p, stream_); }
hxcclResult_t DeviceOps::sync() {
  return cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}
