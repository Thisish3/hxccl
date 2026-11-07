#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hxccl.h"

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
