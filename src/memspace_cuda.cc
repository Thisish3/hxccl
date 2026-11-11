// CUDA build: choose host or device per call from the pointers.
#include <cuda_runtime.h>

#include <cstdlib>

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
size_t chunk_bytes() {
  const char* e = std::getenv("HXCCL_IPC_CHUNK");
  size_t v = e ? std::strtoull(e, nullptr, 10) : 0;
  return v >= 64 ? v : size_t(4) << 20;
}
}  // namespace

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
  if (!c->dev_ops) {
    auto d = std::make_unique<DeviceOps>(*c->net, chunk_bytes());
    if ((*rc = d->init()) != hxcclSuccess) {
      comm_fail(c, *rc, d->error);
      return nullptr;
    }
    c->dev_ops = std::move(d);
  }
  auto* d = static_cast<DeviceOps*>(c->dev_ops.get());
  d->set_stream(static_cast<cudaStream_t>(stream));
  return d;
}

hxcclResult_t sync_ops(MemOps& m) {
  if (auto* d = dynamic_cast<DeviceOps*>(&m)) return d->sync();
  return hxcclSuccess;
}
