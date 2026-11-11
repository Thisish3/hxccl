// CUDA build: choose host or device per call from the pointers.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "memspace.h"

namespace {
// 1 = device, 0 = host, -1 = ignore (null)
int space(const void* p) {
  if (!p) return -1;
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged ? 1 : 0;
}
// HXCCL_IPC_CHUNK in bytes, 64 B .. 1 GiB; anything else (not a number, out of range) gives
// the default 4 MiB (I2: a bad value used to make one rank's cudaMalloc fail). The ranks then
// agree on the smallest value in IpcTransport::init.
size_t chunk_bytes() {
  constexpr size_t kDefault = size_t(4) << 20, kMax = size_t(1) << 30;
  const char* e = std::getenv("HXCCL_IPC_CHUNK");
  if (!e || !*e) return kDefault;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(e, &end, 10);
  if (errno || *end || e[0] == '-' || v < 64 || v > kMax) return kDefault;
  return static_cast<size_t>(v);
}

// Is there a CUDA context in this process already? Asked through the driver API, which does
// not create one (asking the runtime could).
bool cuda_context_exists() {
  if (cuInit(0) != CUDA_SUCCESS) return false;
  CUcontext cur = nullptr;
  if (cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur) return true;
  int n = 0;
  if (cuDeviceGetCount(&n) != CUDA_SUCCESS) return false;
  for (int i = 0; i < n; ++i) {
    CUdevice d;
    unsigned flags = 0;
    int active = 0;
    if (cuDeviceGet(&d, i) == CUDA_SUCCESS &&
        cuDevicePrimaryCtxGetState(d, &flags, &active) == CUDA_SUCCESS && active)
      return true;
  }
  return false;
}

// Collective over the communicator. A failure is kept on the communicator (sticky), so
// every later device call returns it instead of trying again on some ranks only (I2).
void start_device(hxcclComm* c) {
  auto d = std::make_unique<DeviceOps>(*c->net, chunk_bytes());
  const hxcclResult_t r = d->init();
  if (r != hxcclSuccess) {
    c->dev_error = r;
    c->dev_error_msg = "device path setup failed: " + d->error;
    return;
  }
  c->dev_ops = std::move(d);
}
}  // namespace

// I4: the first device call starts the device path with a collective step, so a device
// Send/Recv between two ranks hung when the other ranks never made a device call. So the
// device path starts when the communicator is created, if every rank already has a CUDA
// context. A program with no CUDA context keeps creating none (#26) and starts lazily.
// HXCCL_DEVICE=1 makes every rank start it (creating a context), HXCCL_DEVICE=0 never does.
void device_eager_init(hxcclComm* c) {
  const char* e = std::getenv("HXCCL_DEVICE");
  int want;
  if (e && std::strcmp(e, "0") == 0) {
    want = 0;
  } else if (e && *e) {
    want = 1;
    cudaFree(nullptr);  // creates the context on the current device
  } else {
    want = cuda_context_exists() ? 1 : 0;
  }
  int no = want ? 0 : 1;
  if (c->net->allreduce_max(&no) != hxcclSuccess) {
    c->dev_error = hxcclSystemError;
    c->dev_error_msg = "agreement on starting the device path failed: " + c->net->error;
    return;
  }
  if (!no) start_device(c);
}

MemOps* ops_for(hxcclComm* c, hxcclStream_t stream, const void* a, const void* b,
                hxcclResult_t* rc) {
  *rc = hxcclSuccess;
  const int sa = space(a), sb = space(b);
  if (sa >= 0 && sb >= 0 && sa != sb) {
    *rc = comm_fail(c, hxcclInvalidArgument,
                    "send and recv buffers must both be host or both be device memory");
    return nullptr;
  }
  if (sa != 1 && sb != 1) return c->host_ops.get();
  if (!c->dev_ops && c->dev_error == hxcclSuccess) start_device(c);
  if (c->dev_error == hxcclSuccess && c->dev_ops) {
    std::string why;
    if (hxcclResult_t r = static_cast<DeviceOps*>(c->dev_ops.get())->broken(&why)) {
      c->dev_error = r;  // a failed transfer earlier (I5)
      c->dev_error_msg = why;
    }
  }
  if (c->dev_error != hxcclSuccess) {
    *rc = comm_fail(c, c->dev_error, c->dev_error_msg);
    return nullptr;
  }
  auto* d = static_cast<DeviceOps*>(c->dev_ops.get());
  d->set_stream(static_cast<cudaStream_t>(stream));
  return d;
}

hxcclResult_t read_scalar(void* dst, const void* src, size_t bytes, hxcclStream_t stream,
                          std::string* err) {
  if (space(src) != 1) {
    std::memcpy(dst, src, bytes);
    return hxcclSuccess;
  }
  cudaError_t e = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  if (e == cudaSuccess) e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  if (e == cudaSuccess) return hxcclSuccess;
  *err = cudaGetErrorString(e);
  return hxcclUnhandledCudaError;
}

hxcclResult_t sync_ops(MemOps& m) {
  if (auto* d = dynamic_cast<DeviceOps*>(&m)) return d->sync();
  return hxcclSuccess;
}
