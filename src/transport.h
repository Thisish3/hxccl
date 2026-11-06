#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "hxccl.h"

// Point-to-point layer the algorithms use. A: MPI. B adds an IPC transport.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from,
                                 void* rbuf, size_t rbytes) = 0;
  virtual hxcclResult_t send(int to, const void* buf, size_t bytes) = 0;
  virtual hxcclResult_t recv(int from, void* buf, size_t bytes) = 0;

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

 private:
  hxcclResult_t check(int rc, const char* what);
  MPI_Comm comm_;
};
