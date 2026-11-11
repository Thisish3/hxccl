#pragma once
#include "comm.h"
#include "memops.h"

// Picks the memory path for one call. Null pointers are ignored. Returns nullptr and sets *rc
// (hxcclInvalidArgument, with c->last_error) when host and device pointers are mixed, or the
// device path fails to start. The first device call on a communicator starts the device path
// with a collective step, so every rank must pass the same kind of buffer (README).
MemOps* ops_for(hxcclComm* c, hxcclStream_t stream, const void* a, const void* b,
                hxcclResult_t* rc);

// Copies `bytes` from `src` (host or device memory) to host `dst`. A device `src` is read after
// the work already queued on `stream`.
hxcclResult_t read_scalar(void* dst, const void* src, size_t bytes, hxcclStream_t stream,
                          std::string* err);

// Called by every rank creating a communicator: starts the device path now when every rank
// already has a CUDA context (I4). A no-op in CPU builds.
void device_eager_init(hxcclComm* c);

// Waits for the device work of a call; a no-op for host buffers.
hxcclResult_t sync_ops(MemOps& m);
