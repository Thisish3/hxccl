#include "net_sock.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <thread>

namespace {

constexpr int kAllgatherTag = -1;  // internal; library and user tags are >= 0
constexpr uint32_t kIdMagic = 0x48585342;  // "HXSB"
const char* const kTruncated = "message truncated: the receive buffer is smaller than the message";

std::string sys_error(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

}  // namespace

// ---------------------------------------------------------------- SockMesh

SockMesh::SockMesh(int rank, int size, std::vector<int> fds)
    : rank_(rank), size_(size), peers_(size), scratch_(1 << 16) {
  for (int i = 0; i < size; ++i) peers_[i].fd = fds[i];
}

SockMesh::~SockMesh() {
  for (Peer& p : peers_)
    if (p.fd >= 0) close(p.fd);
}

int SockMesh::new_req() {
  int i;
  if (!free_.empty()) {
    i = free_.back();
    free_.pop_back();
  } else {
    i = static_cast<int>(reqs_.size());
    reqs_.emplace_back();
  }
  reqs_[i] = Req{};
  reqs_[i].used = true;
  return i;
}

void SockMesh::finish(int req, hxcclResult_t rc, std::string err) {
  if (req < 0 || !reqs_[req].used) return;
  reqs_[req].done = true;
  reqs_[req].rc = rc;
  reqs_[req].err = std::move(err);
}

hxcclResult_t SockMesh::isend(int peer, uint64_t comm, int tag, const void* buf, size_t bytes,
                              int* req) {
  *req = new_req();
  const char* src = static_cast<const char*>(buf);
  if (peer == rank_) {  // to myself: hand over now, no socket
    Peer& p = peers_[rank_];
    auto it = std::find_if(p.posted.begin(), p.posted.end(),
                           [&](const RecvOp& r) { return r.comm == comm && r.tag == tag; });
    if (it != p.posted.end()) {
      if (bytes) std::memcpy(it->buf, src, std::min(bytes, it->cap));
      finish(it->req, bytes > it->cap ? hxcclSystemError : hxcclSuccess,
             bytes > it->cap ? kTruncated : "");
      p.posted.erase(it);
    } else {
      Unexpected u;
      u.comm = comm;
      u.tag = tag;
      u.data.assign(src, src + bytes);
      u.complete = true;
      p.unexpected.push_back(std::move(u));
    }
    finish(*req, hxcclSuccess);
    return hxcclSuccess;
  }
  Peer& p = peers_[peer];
  if (p.dead) {
    finish(*req, hxcclSystemError, "rank " + std::to_string(peer) + " closed the connection");
    return hxcclSuccess;
  }
  SendOp op{*req, Header{comm, tag, 0, bytes}, src};
  p.sendq.push_back(op);
  if (p.sendq.size() == 1) write_some(peer);  // usually goes out right away
  return hxcclSuccess;
}

hxcclResult_t SockMesh::irecv(int peer, uint64_t comm, int tag, void* buf, size_t cap, int* req) {
  *req = new_req();
  Peer& p = peers_[peer];
  for (auto it = p.unexpected.begin(); it != p.unexpected.end(); ++it) {
    if (it->comm != comm || it->tag != tag || it->claim >= 0) continue;
    if (!it->complete) {  // still arriving: take it when the body is in
      it->claim = *req;
      it->claim_buf = static_cast<char*>(buf);
      it->claim_cap = cap;
      return hxcclSuccess;
    }
    const size_t n = it->data.size();
    if (n) std::memcpy(buf, it->data.data(), std::min(n, cap));
    finish(*req, n > cap ? hxcclSystemError : hxcclSuccess, n > cap ? kTruncated : "");
    p.unexpected.erase(it);
    return hxcclSuccess;
  }
  if (p.dead) {
    finish(*req, hxcclSystemError, "rank " + std::to_string(peer) + " closed the connection");
    return hxcclSuccess;
  }
  p.posted.push_back(RecvOp{*req, comm, tag, static_cast<char*>(buf), cap});
  return hxcclSuccess;
}

hxcclResult_t SockMesh::test(int req, bool* done, std::string* err) {
  if (!reqs_[req].done) progress(0);
  *done = reqs_[req].done;
  if (!*done) return hxcclSuccess;
  const hxcclResult_t rc = reqs_[req].rc;
  if (rc != hxcclSuccess) *err = reqs_[req].err;
  reqs_[req].used = false;
  free_.push_back(req);
  return rc;
}

void SockMesh::cancel(int req) {
  for (Peer& p : peers_) {
    p.posted.remove_if([&](const RecvOp& r) { return r.req == req; });
    if (p.dest_req == req) p.dest_cancelled = true;
    for (Unexpected& u : p.unexpected)
      if (u.claim == req) u.claim = -1;
    for (SendOp& s : p.sendq)  // a send in flight finishes in the background
      if (s.req == req) s.req = -1;
  }
  if (reqs_[req].used) {
    reqs_[req].used = false;
    free_.push_back(req);
  }
}

void SockMesh::fail_peer(int peer, const std::string& why) {
  Peer& p = peers_[peer];
  if (p.dead) return;
  p.dead = true;
  if (p.fd >= 0) close(p.fd);
  p.fd = -1;
  for (SendOp& s : p.sendq) finish(s.req, hxcclSystemError, why);
  p.sendq.clear();
  for (RecvOp& r : p.posted) finish(r.req, hxcclSystemError, why);
  p.posted.clear();
  if (p.in_body && p.dest_req >= 0 && !p.dest_cancelled)
    finish(p.dest_req, hxcclSystemError, why);
  for (Unexpected& u : p.unexpected)
    if (!u.complete && u.claim >= 0) finish(u.claim, hxcclSystemError, why);
  p.unexpected.remove_if([](const Unexpected& u) { return !u.complete; });
  p.in_body = false;
  p.dest_req = -1;
}

bool SockMesh::write_some(int peer) {
  Peer& p = peers_[peer];
  while (!p.sendq.empty()) {
    SendOp& op = p.sendq.front();
    const size_t hs = sizeof(Header), total = hs + op.h.bytes;
    iovec iov[2];
    int cnt = 0;
    if (op.off < hs) {
      iov[cnt].iov_base = reinterpret_cast<char*>(&op.h) + op.off;
      iov[cnt++].iov_len = hs - op.off;
    }
    const size_t body_off = op.off > hs ? op.off - hs : 0;
    if (op.h.bytes > body_off) {
      iov[cnt].iov_base = const_cast<char*>(op.buf) + body_off;
      iov[cnt++].iov_len = op.h.bytes - body_off;
    }
    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = cnt;
    const ssize_t n = sendmsg(p.fd, &msg, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
      fail_peer(peer, sys_error("send"));
      return false;
    }
    op.off += static_cast<size_t>(n);
    if (op.off == total) {
      finish(op.req, hxcclSuccess);
      p.sendq.pop_front();
    }
  }
  return true;
}

void SockMesh::begin_body(Peer& p) {
  p.in_body = true;
  p.body_got = 0;
  p.dest_req = -1;
  p.dest_cancelled = false;
  p.dest_buf = nullptr;
  p.dest_cap = 0;
  p.dest_unexp = nullptr;
  auto it = std::find_if(p.posted.begin(), p.posted.end(), [&](const RecvOp& r) {
    return r.comm == p.hdr.comm && r.tag == p.hdr.tag;
  });
  if (it != p.posted.end()) {
    p.dest_req = it->req;
    p.dest_buf = it->buf;
    p.dest_cap = it->cap;
    p.posted.erase(it);
    return;
  }
  Unexpected u;
  u.comm = p.hdr.comm;
  u.tag = p.hdr.tag;
  u.data.resize(p.hdr.bytes);
  p.unexpected.push_back(std::move(u));
  p.dest_unexp = &p.unexpected.back();
}

void SockMesh::deliver(Peer& p) {
  const size_t n = p.hdr.bytes;
  if (p.dest_unexp) {
    Unexpected* u = p.dest_unexp;
    u->complete = true;
    if (u->claim >= 0) {
      if (n) std::memcpy(u->claim_buf, u->data.data(), std::min(n, u->claim_cap));
      finish(u->claim, n > u->claim_cap ? hxcclSystemError : hxcclSuccess,
             n > u->claim_cap ? kTruncated : "");
      p.unexpected.remove_if([u](const Unexpected& x) { return &x == u; });
    }
  } else if (p.dest_req >= 0 && !p.dest_cancelled) {
    finish(p.dest_req, n > p.dest_cap ? hxcclSystemError : hxcclSuccess,
           n > p.dest_cap ? kTruncated : "");
  }
  p.in_body = false;
  p.hdr_got = 0;
  p.dest_req = -1;
  p.dest_unexp = nullptr;
}

bool SockMesh::read_some(int peer) {
  Peer& p = peers_[peer];
  for (;;) {
    char* dst;
    size_t len;
    if (!p.in_body) {
      dst = reinterpret_cast<char*>(&p.hdr) + p.hdr_got;
      len = sizeof(Header) - p.hdr_got;
    } else if (p.dest_unexp) {
      dst = p.dest_unexp->data.data() + p.body_got;
      len = p.hdr.bytes - p.body_got;
    } else if (p.dest_req >= 0 && !p.dest_cancelled && p.body_got < p.dest_cap) {
      dst = p.dest_buf + p.body_got;
      len = std::min<size_t>(p.hdr.bytes, p.dest_cap) - p.body_got;
    } else {  // truncated tail or cancelled receive: drop the bytes
      dst = scratch_.data();
      len = std::min<size_t>(scratch_.size(), p.hdr.bytes - p.body_got);
    }
    const ssize_t n = recv(p.fd, dst, len, 0);
    if (n == 0) {
      fail_peer(peer, "rank " + std::to_string(peer) + " closed the connection");
      return false;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
      fail_peer(peer, sys_error("recv"));
      return false;
    }
    if (!p.in_body) {
      p.hdr_got += static_cast<size_t>(n);
      if (p.hdr_got < sizeof(Header)) continue;
      begin_body(p);
      if (p.hdr.bytes == 0) deliver(p);
      continue;
    }
    p.body_got += static_cast<size_t>(n);
    if (p.body_got == p.hdr.bytes) deliver(p);
  }
}

void SockMesh::progress(int timeout_ms) {
  std::vector<pollfd> fds;
  std::vector<int> who;
  for (int i = 0; i < size_; ++i) {
    const Peer& p = peers_[i];
    if (i == rank_ || p.dead || p.fd < 0) continue;
    pollfd f{p.fd, static_cast<short>(POLLIN | (p.sendq.empty() ? 0 : POLLOUT)), 0};
    fds.push_back(f);
    who.push_back(i);
  }
  if (fds.empty()) return;
  if (poll(fds.data(), fds.size(), timeout_ms) <= 0) return;
  for (size_t k = 0; k < fds.size(); ++k) {
    const short ev = fds[k].revents;
    if (!ev) continue;
    if ((ev & POLLOUT) && !write_some(who[k])) continue;
    if (ev & (POLLIN | POLLHUP | POLLERR)) read_some(who[k]);
  }
}

// ---------------------------------------------------------------- SockNet

SockNet::SockNet(std::shared_ptr<SockMesh> mesh, uint64_t id, std::vector<int> members, int rank)
    : mesh_(std::move(mesh)), id_(id), members_(std::move(members)), rank_(rank) {}

hxcclResult_t SockNet::isend(int peer, int tag, const void* buf, size_t bytes, int* req) {
  return mesh_->isend(members_[peer], id_, tag, buf, bytes, req);
}
hxcclResult_t SockNet::irecv(int peer, int tag, void* buf, size_t bytes, int* req) {
  return mesh_->irecv(members_[peer], id_, tag, buf, bytes, req);
}
hxcclResult_t SockNet::test(int req, bool* done) { return mesh_->test(req, done, &error); }
void SockNet::cancel(int req) { mesh_->cancel(req); }

hxcclResult_t SockNet::allgather(const void* mine, void* all, size_t bytes) {
  const int n = size();
  char* out = static_cast<char*>(all);
  std::memmove(out + size_t(rank_) * bytes, mine, bytes);
  std::vector<int> reqs;
  for (int p = 0; p < n; ++p) {
    if (p == rank_) continue;
    int r;
    irecv(p, kAllgatherTag, out + size_t(p) * bytes, bytes, &r);
    reqs.push_back(r);
  }
  for (int p = 0; p < n; ++p) {
    if (p == rank_) continue;
    int r;
    isend(p, kAllgatherTag, out + size_t(rank_) * bytes, bytes, &r);
    reqs.push_back(r);
  }
  hxcclResult_t first = hxcclSuccess;
  for (int r : reqs) {
    hxcclResult_t e = wait(r);
    if (e != hxcclSuccess && first == hxcclSuccess) first = e;
  }
  return first;
}

hxcclResult_t SockNet::allreduce_max(int* v) {
  std::vector<int> all(size());
  if (hxcclResult_t r = allgather(v, all.data(), sizeof(int))) return r;
  *v = *std::max_element(all.begin(), all.end());
  return hxcclSuccess;
}

std::shared_ptr<SockNet> SockNet::sub(const std::vector<int>& local, int new_rank,
                                      uint64_t id) const {
  std::vector<int> m(local.size());
  for (size_t i = 0; i < local.size(); ++i) m[i] = members_[local[i]];
  return std::make_shared<SockNet>(mesh_, id, std::move(m), new_rank);
}

// ---------------------------------------------------------------- bootstrap

namespace {

struct IdData {
  uint32_t magic;
  uint32_t addr;  // IPv4, network order
  uint16_t port;  // network order
  uint16_t pad;
  uint64_t nonce;
};
static_assert(sizeof(IdData) <= sizeof(hxcclUniqueId), "unique id too small");

struct Info {  // what each rank tells the root
  int32_t rank, nranks;
  uint32_t addr;
  uint16_t port;
  uint16_t pad;
};

std::mutex g_mu;
std::map<uint64_t, int>& listeners() {  // nonce -> listening socket made by GetUniqueId
  static std::map<uint64_t, int> m;
  return m;
}

uint32_t local_addr() {
  in_addr a{};
  const char* e = std::getenv("HXCCL_SOCKET_ADDR");
  if (!e || inet_pton(AF_INET, e, &a) != 1) inet_pton(AF_INET, "127.0.0.1", &a);
  return a.s_addr;
}

// SOCK_CLOEXEC and accept4 are Linux-only; macOS sets FD_CLOEXEC afterwards.
int cloexec_socket() {
#ifdef SOCK_CLOEXEC
  return socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd >= 0) fcntl(fd, F_SETFD, FD_CLOEXEC);
  return fd;
#endif
}

hxcclResult_t listen_on(uint32_t addr, int* fd, uint16_t* port, std::string* err) {
  *fd = cloexec_socket();
  if (*fd < 0) return *err = sys_error("socket"), hxcclSystemError;
  int one = 1;
  setsockopt(*fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = addr;
  sa.sin_port = 0;
  socklen_t len = sizeof sa;
  if (bind(*fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0 || listen(*fd, 128) != 0 ||
      getsockname(*fd, reinterpret_cast<sockaddr*>(&sa), &len) != 0) {
    *err = sys_error("bind/listen");
    close(*fd);
    return hxcclSystemError;
  }
  *port = sa.sin_port;
  return hxcclSuccess;
}

hxcclResult_t connect_to(uint32_t addr, uint16_t port, int* fd, std::string* err) {
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (;;) {
    *fd = cloexec_socket();
    if (*fd < 0) return *err = sys_error("socket"), hxcclSystemError;
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = addr;
    sa.sin_port = port;
    if (connect(*fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0) return hxcclSuccess;
    const int e = errno;
    close(*fd);
    if ((e != ECONNREFUSED && e != EINTR && e != ETIMEDOUT) ||
        std::chrono::steady_clock::now() > give_up) {
      errno = e;
      *err = sys_error("connect");
      return hxcclSystemError;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool write_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n) {
    const ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return false;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

bool read_all(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n) {
    const ssize_t k = recv(fd, p, n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return false;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

int accept_one(int lfd) {
  for (;;) {
#ifdef SOCK_CLOEXEC
    const int c = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
#else
    const int c = accept(lfd, nullptr, nullptr);
    if (c >= 0) fcntl(c, F_SETFD, FD_CLOEXEC);
#endif
    if (c >= 0 || errno != EINTR) return c;
  }
}

void tune(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

}  // namespace

uint64_t derive_comm_id(uint64_t parent, uint64_t seq, uint64_t salt) {
  auto mix = [](uint64_t x) {  // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  };
  return mix(mix(mix(parent) ^ seq) ^ salt);
}

hxcclResult_t sock_unique_id(hxcclUniqueId* id, std::string* err) {
  IdData d{};
  d.magic = kIdMagic;
  d.addr = local_addr();
  int fd;
  if (hxcclResult_t r = listen_on(d.addr, &fd, &d.port, err)) return r;
  std::random_device rd;
  do {
    d.nonce = (uint64_t(rd()) << 32) ^ rd();
  } while (d.nonce == 0);
  std::memset(id, 0, sizeof *id);
  std::memcpy(id->internal, &d, sizeof d);
  std::lock_guard<std::mutex> lock(g_mu);
  listeners()[d.nonce] = fd;
  return hxcclSuccess;
}

hxcclResult_t sock_init_rank(const hxcclUniqueId& id, int nranks, int rank,
                             std::shared_ptr<SockNet>* out, std::string* err) {
  IdData d;
  std::memcpy(&d, id.internal, sizeof d);
  if (d.magic != kIdMagic) {
    *err = "not a unique id from hxcclGetUniqueId";
    return hxcclInvalidArgument;
  }
  int lfd;  // this rank's mesh listener
  Info me{rank, nranks, local_addr(), 0, 0};
  if (hxcclResult_t r = listen_on(me.addr, &lfd, &me.port, err)) return r;

  int root_fd = -1;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = listeners().find(d.nonce);
    if (it != listeners().end()) {
      root_fd = it->second;
      listeners().erase(it);
    }
  }
  std::vector<Info> table(nranks);
  int32_t status = 0;
  if (root_fd >= 0) {  // root: collect every rank's address, check them, send the table back
    std::vector<Info> got{me};
    std::vector<int> conns;
    bool io_ok = true;
    for (int i = 1; i < nranks && io_ok; ++i) {
      const int c = accept_one(root_fd);
      Info info;
      io_ok = c >= 0 && read_all(c, &info, sizeof info);
      if (c >= 0) conns.push_back(c);
      if (io_ok) got.push_back(info);
    }
    close(root_fd);
    std::vector<bool> seen(nranks, false);
    for (const Info& i : got) {
      if (i.nranks != nranks || i.rank < 0 || i.rank >= nranks || seen[i.rank]) status = 1;
      else seen[i.rank] = true;
    }
    if (!io_ok) status = 2;
    if (status == 0)
      for (const Info& i : got) table[i.rank] = i;
    for (int c : conns) {
      write_all(c, &status, sizeof status);
      if (status == 0) write_all(c, table.data(), sizeof(Info) * nranks);
      close(c);
    }
  } else {
    int c;
    if (hxcclResult_t r = connect_to(d.addr, d.port, &c, err)) {
      close(lfd);
      return r;
    }
    const bool ok = write_all(c, &me, sizeof me) && read_all(c, &status, sizeof status) &&
                    (status != 0 || read_all(c, table.data(), sizeof(Info) * nranks));
    close(c);
    if (!ok) status = 2;
  }
  if (status != 0) {
    close(lfd);
    *err = status == 1 ? "ranks disagree: a rank is repeated or missing, or nranks differs"
                       : "bootstrap connection to the root failed";
    return status == 1 ? hxcclInvalidUsage : hxcclSystemError;
  }

  // full mesh: connect to every lower rank, accept every higher rank
  std::vector<int> fds(nranks, -1);
  auto bail = [&](const std::string& why) {
    for (int f : fds)
      if (f >= 0) close(f);
    close(lfd);
    *err = why;
    return hxcclSystemError;
  };
  for (int p = 0; p < rank; ++p) {
    int c;
    if (connect_to(table[p].addr, table[p].port, &c, err)) return bail(*err);
    fds[p] = c;
    const int32_t r32 = rank;
    if (!write_all(c, &r32, sizeof r32) || !write_all(c, &d.nonce, sizeof d.nonce))
      return bail(sys_error("mesh handshake"));
  }
  for (int k = rank + 1; k < nranks; ++k) {
    const int c = accept_one(lfd);
    int32_t r32 = -1;
    uint64_t nonce = 0;
    if (c < 0) return bail(sys_error("accept"));
    if (!read_all(c, &r32, sizeof r32) || !read_all(c, &nonce, sizeof nonce) ||
        nonce != d.nonce || r32 <= rank || r32 >= nranks || fds[r32] >= 0) {
      close(c);
      return bail("mesh handshake from an unexpected peer");
    }
    fds[r32] = c;
  }
  close(lfd);
  for (int f : fds)
    if (f >= 0) tune(f);
  auto mesh = std::make_shared<SockMesh>(rank, nranks, fds);
  std::vector<int> members(nranks);
  for (int i = 0; i < nranks; ++i) members[i] = i;
  *out = std::make_shared<SockNet>(mesh, d.nonce, members, rank);
  return hxcclSuccess;
}
