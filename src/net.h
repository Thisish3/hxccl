#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "hxccl.h"

// Control traffic of one communicator: small point-to-point messages matched by
// (peer, tag) in posting order, plus the two collectives the library needs.
// MpiNet wraps an MPI communicator; SockNet (E) runs over TCP sockets.
// Requests are small integers; test() frees a request once it reports done.
class Net {
 public:
  virtual ~Net() = default;
  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual hxcclResult_t isend(int peer, int tag, const void* buf, size_t bytes, int* req) = 0;
  virtual hxcclResult_t irecv(int peer, int tag, void* buf, size_t bytes, int* req) = 0;
  virtual hxcclResult_t test(int req, bool* done) = 0;
  virtual void cancel(int req) = 0;
  virtual hxcclResult_t allgather(const void* mine, void* all, size_t bytes) = 0;
  virtual hxcclResult_t allreduce_max(int* v) = 0;
  hxcclResult_t wait(int req) {
    for (bool done = false; !done;)
      if (hxcclResult_t r = test(req, &done)) return r;
    return hxcclSuccess;
  }
  hxcclResult_t send(int peer, int tag, const void* buf, size_t bytes) {
    int req;
    if (hxcclResult_t r = isend(peer, tag, buf, bytes, &req)) return r;
    return wait(req);
  }
  std::string error;
};

class MpiNet : public Net {
 public:
  explicit MpiNet(MPI_Comm comm);
  int rank() const override { return rank_; }
  int size() const override { return size_; }
  hxcclResult_t isend(int peer, int tag, const void* buf, size_t bytes, int* req) override;
  hxcclResult_t irecv(int peer, int tag, void* buf, size_t bytes, int* req) override;
  hxcclResult_t test(int req, bool* done) override;
  void cancel(int req) override;
  hxcclResult_t allgather(const void* mine, void* all, size_t bytes) override;
  hxcclResult_t allreduce_max(int* v) override;

 private:
  hxcclResult_t check(int rc, const char* what);
  int slot(MPI_Request r, bool is_send);
  MPI_Comm comm_;
  int rank_ = 0, size_ = 1;
  std::vector<MPI_Request> reqs_;
  std::vector<bool> is_send_;
  std::vector<int> free_;
};
