#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hxccl.h"
#include "net.h"

// collectives and Send/Recv use separate tags so their messages never match each other
constexpr int kCollTag = 0, kP2pTag = 1;

// Point-to-point layer the algorithms use. A: MPI. B adds an IPC transport.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from,
                                 void* rbuf, size_t rbytes) = 0;
  virtual hxcclResult_t send(int to, const void* buf, size_t bytes) = 0;
  virtual hxcclResult_t recv(int from, void* buf, size_t bytes) = 0;
  // non-blocking: post with isend/irecv, then complete everything posted with waitall
  virtual hxcclResult_t isend(int to, const void* buf, size_t bytes, int tag) = 0;
  virtual hxcclResult_t irecv(int from, void* buf, size_t bytes, int tag) = 0;
  virtual hxcclResult_t waitall() = 0;
  virtual void cancel_all() = 0;
  // One non-blocking pass over everything posted; *done once nothing is left (then returns
  // what waitall would). A group spanning several transports drives them all through step in
  // one loop, so none waits on another (I1). The default just blocks in waitall: MPI
  // progresses every request of the process whichever one it waits on.
  virtual hxcclResult_t step(bool* done) {
    *done = true;
    return waitall();
  }

  uint64_t bytes_sent = 0;
  uint64_t msgs_sent = 0;
  std::string error;
};

class MpiTransport : public Transport {
 public:
  explicit MpiTransport(MPI_Comm comm) : comm_(comm) {}
  hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from, void* rbuf,
                         size_t rbytes) override;
  hxcclResult_t send(int to, const void* buf, size_t bytes) override;
  hxcclResult_t recv(int from, void* buf, size_t bytes) override;
  hxcclResult_t isend(int to, const void* buf, size_t bytes, int tag) override;
  hxcclResult_t irecv(int from, void* buf, size_t bytes, int tag) override;
  hxcclResult_t waitall() override;
  void cancel_all() override;

 private:
  hxcclResult_t check(int rc, const char* what);
  MPI_Comm comm_;
  std::vector<MPI_Request> pending_;
  std::vector<bool> pending_is_send_;
};

// Host buffers over a Net (socket communicators, spec E). Counts traffic like MpiTransport.
class NetTransport : public Transport {
 public:
  explicit NetTransport(std::shared_ptr<Net> net) : net_(std::move(net)) {}
  hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from, void* rbuf,
                         size_t rbytes) override;
  hxcclResult_t send(int to, const void* buf, size_t bytes) override;
  hxcclResult_t recv(int from, void* buf, size_t bytes) override;
  hxcclResult_t isend(int to, const void* buf, size_t bytes, int tag) override;
  hxcclResult_t irecv(int from, void* buf, size_t bytes, int tag) override;
  hxcclResult_t waitall() override;
  void cancel_all() override;
  hxcclResult_t step(bool* done) override;

 private:
  hxcclResult_t fail(hxcclResult_t r);
  std::shared_ptr<Net> net_;
  std::vector<int> pending_;
};

#ifdef HXCCL_CUDA
#include <cuda_runtime.h>

#include <deque>
#include <list>

// Device buffers over CUDA IPC (spec 3.4). Every rank keeps one staging buffer per sender;
// data moves with cudaMemcpyAsync into the receiver's staging buffer and is handed over with
// READY/DONE control messages on the communicator's Net. Every call finishes before it returns.
class IpcTransport : public Transport {
 public:
  IpcTransport(Net& net, size_t chunk);
  ~IpcTransport() override;
  // Collective over comm: set up CUDA IPC, or agree on the staged path if any rank cannot.
  hxcclResult_t init();
  void set_stream(cudaStream_t s) { stream_ = s; }
  // true: device data goes through host memory and the Net instead of CUDA IPC
  bool staged() const { return staged_; }
  const std::string& staged_reason() const { return staged_reason_; }
  hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from, void* rbuf,
                         size_t rbytes) override;
  hxcclResult_t send(int to, const void* buf, size_t bytes) override;
  hxcclResult_t recv(int from, void* buf, size_t bytes) override;
  hxcclResult_t isend(int to, const void* buf, size_t bytes, int tag) override;
  hxcclResult_t irecv(int from, void* buf, size_t bytes, int tag) override;
  hxcclResult_t waitall() override;
  void cancel_all() override;
  hxcclResult_t step(bool* done) override;
  // after a failed transfer (here or on a peer) the device path stays failed (I5)
  hxcclResult_t broken() const { return broken_; }

 private:
  struct Op;
  struct StagedOp;
  hxcclResult_t progress(Op& o, bool* finished);
  hxcclResult_t progress_self(hxcclResult_t* first);
  hxcclResult_t staged_post(bool is_send, int peer, char* dev, size_t bytes, int tag);
  hxcclResult_t staged_step(bool* done, hxcclResult_t* first);
  hxcclResult_t fail(hxcclResult_t r);
  hxcclResult_t check_abort();
  void release_ipc();
  hxcclResult_t cuda(cudaError_t e, const char* what);
  hxcclResult_t ctl(hxcclResult_t r, const char* what);
  Net& net_;
  int rank_, n_;
  size_t chunk_;
  cudaStream_t stream_ = nullptr;
  std::vector<void*> mine_;  // mine_[s]: my staging buffer that sender s writes into
  std::vector<void*> peer_;  // peer_[p]: rank p's staging buffer reserved for me
  std::vector<std::deque<Op>> sends_, recvs_;  // per peer, in posting order
  std::vector<int> abort_req_;        // posted receive of ABORT from each peer
  std::vector<uint64_t> abort_in_;
  bool staged_ = false;
  std::string staged_reason_;
  std::list<StagedOp> staged_ops_;  // list: the Net holds pointers into each op
  hxcclResult_t broken_ = hxcclSuccess;
  std::string broken_msg_;
  hxcclResult_t pending_rc_ = hxcclSuccess;  // a truncation, reported when all is done
  std::string trunc_msg_;
};
#endif
