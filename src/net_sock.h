#pragma once
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "net.h"

// TCP connections between every pair of ranks of a root communicator, driven by one poll()
// loop in the calling thread (spec E 3.2). Messages carry {comm id, tag, bytes}; receives
// match per (sender, comm, tag) in posting order, and messages nobody has asked for yet are
// copied to an "unexpected" list. Sub-communicators share the mesh and differ by comm id.
class SockMesh {
 public:
  SockMesh(int rank, int size, std::vector<int> fds);  // fds[rank] == -1
  ~SockMesh();
  SockMesh(const SockMesh&) = delete;
  SockMesh& operator=(const SockMesh&) = delete;
  int rank() const { return rank_; }
  int size() const { return size_; }
  hxcclResult_t isend(int peer, uint64_t comm, int tag, const void* buf, size_t bytes, int* req);
  hxcclResult_t irecv(int peer, uint64_t comm, int tag, void* buf, size_t bytes, int* req);
  // Completes the request when done; *err gets the failure message.
  hxcclResult_t test(int req, bool* done, std::string* err);
  void cancel(int req);

  struct Header {
    uint64_t comm;
    int32_t tag;
    uint32_t pad;
    uint64_t bytes;
  };

 private:
  struct Req {
    bool used = false, done = false;
    hxcclResult_t rc = hxcclSuccess;
    std::string err;
  };
  struct SendOp {
    int req;
    Header h;
    const char* buf;
    size_t off = 0;  // bytes of header + body written so far
  };
  struct RecvOp {
    int req;
    uint64_t comm;
    int tag;
    char* buf;
    size_t cap;
  };
  struct Unexpected {
    uint64_t comm;
    int tag;
    std::vector<char> data;
    bool complete = false;
    int claim = -1;  // a receive posted while the body was still arriving
    char* claim_buf = nullptr;
    size_t claim_cap = 0;
  };
  struct Peer {
    int fd = -1;
    bool dead = false;
    std::deque<SendOp> sendq;
    std::list<RecvOp> posted;
    std::list<Unexpected> unexpected;
    // incoming message being read
    Header hdr{};
    size_t hdr_got = 0;
    bool in_body = false;
    size_t body_got = 0;
    int dest_req = -1;  // matched posted receive, or -1
    bool dest_cancelled = false;
    char* dest_buf = nullptr;
    size_t dest_cap = 0;
    Unexpected* dest_unexp = nullptr;
  };

  int new_req();
  void finish(int req, hxcclResult_t rc, std::string err = {});
  void deliver(Peer& p);  // the current incoming message is complete
  void fail_peer(int peer, const std::string& why);
  void progress(int timeout_ms);
  bool write_some(int peer);
  bool read_some(int peer);
  void begin_body(Peer& p);

  int rank_, size_;
  std::vector<Peer> peers_;
  std::vector<Req> reqs_;
  std::vector<int> free_;
  std::vector<char> scratch_;  // discarded bytes of truncated messages
};

class SockNet : public Net {
 public:
  SockNet(std::shared_ptr<SockMesh> mesh, uint64_t id, std::vector<int> members, int rank);
  int rank() const override { return rank_; }
  int size() const override { return static_cast<int>(members_.size()); }
  hxcclResult_t isend(int peer, int tag, const void* buf, size_t bytes, int* req) override;
  hxcclResult_t irecv(int peer, int tag, void* buf, size_t bytes, int* req) override;
  hxcclResult_t test(int req, bool* done) override;
  void cancel(int req) override;
  hxcclResult_t allgather(const void* mine, void* all, size_t bytes) override;
  hxcclResult_t allreduce_max(int* v) override;

  uint64_t id() const { return id_; }
  // A sub-communicator over `local` (ranks of this one, in their new order).
  std::shared_ptr<SockNet> sub(const std::vector<int>& local, int new_rank, uint64_t id) const;

 private:
  std::shared_ptr<SockMesh> mesh_;
  uint64_t id_;
  std::vector<int> members_;  // local rank -> mesh rank
  int rank_;
};

// Bootstrap (spec E 3.3).
hxcclResult_t sock_unique_id(hxcclUniqueId* id, std::string* err);
// Returns hxcclInvalidArgument for an id that is not a socket id and hxcclInvalidUsage when the
// ranks disagree (duplicate or missing rank, different nranks).
hxcclResult_t sock_init_rank(const hxcclUniqueId& id, int nranks, int rank,
                             std::shared_ptr<SockNet>* out, std::string* err);
// Deterministic id for a communicator derived from `parent` (Split, Shrink).
uint64_t derive_comm_id(uint64_t parent, uint64_t seq, uint64_t salt);
