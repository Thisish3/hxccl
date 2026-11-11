#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include "hxccl.h"
#include "transport.h"

// Memory work the algorithms need, separated from where the memory lives:
// HostOps for CPU buffers, DeviceOps (HXCCL_CUDA) for GPU buffers.
class MemOps {
 public:
  virtual ~MemOps() = default;
  virtual hxcclResult_t copy(void* dst, const void* src, size_t bytes) = 0;
  // inout[i] = inout[i] op in[i]
  virtual hxcclResult_t reduce(void* inout, const void* in, size_t count, hxcclDataType_t dt,
                               hxcclRedOp_t op) = 0;
  // dst[i] = src[i] * scalar; scalar is host memory holding one dt element
  virtual hxcclResult_t scale(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                              const void* scalar) = 0;
  virtual hxcclResult_t divide_trunc(void* buf, size_t count, hxcclDataType_t dt,
                                     int divisor) = 0;
  virtual hxcclResult_t alloc(size_t bytes, void** p) = 0;
  virtual void release(void* p) = 0;
  virtual Transport& tr() = 0;
  std::string error;  // set when a non-transport step fails
};

// Scratch buffer in the same memory space as the user buffers, released on scope exit.
class Scratch {
 public:
  Scratch(MemOps& m, size_t bytes) : m_(m) {
    if (bytes) rc = m_.alloc(bytes, &p_);
  }
  ~Scratch() {
    if (p_) m_.release(p_);
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  char* get() const { return static_cast<char*>(p_); }
  hxcclResult_t rc = hxcclSuccess;

 private:
  MemOps& m_;
  void* p_ = nullptr;
};

// Holds the communicator's owning pointer, not the Transport itself, so replacing
// `comm->tr` (tests inject failing transports this way) keeps HostOps valid.
class HostOps : public MemOps {
 public:
  explicit HostOps(std::unique_ptr<Transport>& t) : t_(t) {}
  hxcclResult_t copy(void* dst, const void* src, size_t bytes) override;
  hxcclResult_t reduce(void* inout, const void* in, size_t count, hxcclDataType_t dt,
                       hxcclRedOp_t op) override;
  hxcclResult_t scale(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                      const void* scalar) override;
  hxcclResult_t divide_trunc(void* buf, size_t count, hxcclDataType_t dt, int divisor) override;
  hxcclResult_t alloc(size_t bytes, void** p) override;
  void release(void* p) override;
  Transport& tr() override { return *t_; }

 private:
  std::unique_ptr<Transport>& t_;
};

#ifdef HXCCL_CUDA
#include <cuda_runtime.h>

// Device buffers: work is queued on the caller's stream; sync() waits for it.
class DeviceOps : public MemOps {
 public:
  DeviceOps(Net& net, size_t chunk);
  ~DeviceOps() override;
  hxcclResult_t init();  // collective over comm: allocate and exchange
  void set_stream(cudaStream_t s);
  hxcclResult_t copy(void* dst, const void* src, size_t bytes) override;
  hxcclResult_t reduce(void* inout, const void* in, size_t count, hxcclDataType_t dt,
                       hxcclRedOp_t op) override;
  hxcclResult_t scale(void* dst, const void* src, size_t count, hxcclDataType_t dt,
                      const void* scalar) override;
  hxcclResult_t divide_trunc(void* buf, size_t count, hxcclDataType_t dt, int divisor) override;
  hxcclResult_t alloc(size_t bytes, void** p) override;
  void release(void* p) override;
  Transport& tr() override;
  hxcclResult_t sync();  // wait for the stream
  bool staged() const;   // device data goes through host memory, not CUDA IPC
  hxcclResult_t broken(std::string* why) const;  // a failed transfer, sticky (I5)

 private:
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<IpcTransport> ipc_;
  hxcclResult_t cuda(cudaError_t e, const char* what);
};
#endif
