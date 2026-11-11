#include "group.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace {

struct P2p {
  hxcclComm* c;
  Transport* tr;  // the transport for this buffer's memory space
  bool send;
  int peer;
  void* buf;
  size_t bytes;
};

struct Group {
  int depth = 0;
  std::vector<std::pair<hxcclComm*, std::function<hxcclResult_t()>>> colls;
  std::vector<P2p> p2p;
};
thread_local Group g;

// Post every send/recv before waiting on any, so pairs that send to each other finish.
hxcclResult_t run_p2p(const std::vector<P2p>& ops) {
  std::vector<std::pair<hxcclComm*, Transport*>> used;  // one entry per distinct transport
  for (const P2p& o : ops) {
    if (std::none_of(used.begin(), used.end(), [&](const auto& u) { return u.second == o.tr; }))
      used.emplace_back(o.c, o.tr);
    hxcclResult_t r = o.send ? o.tr->isend(o.peer, o.buf, o.bytes, kP2pTag)
                             : o.tr->irecv(o.peer, o.buf, o.bytes, kP2pTag);
    if (r != hxcclSuccess) {
      for (const auto& u : used) u.second->cancel_all();
      return comm_fail(o.c, r, o.tr->error);
    }
  }
  hxcclResult_t first = hxcclSuccess;
  for (const auto& u : used) {
    hxcclResult_t r = u.second->waitall();
    if (r != hxcclSuccess && first == hxcclSuccess) first = comm_fail(u.first, r, u.second->error);
  }
  return first;
}

}  // namespace

hxcclResult_t dispatch(hxcclComm* c, std::function<hxcclResult_t()> run) {
  if (g.depth == 0) return run();
  g.colls.emplace_back(c, std::move(run));
  return hxcclSuccess;
}

hxcclResult_t p2p(hxcclComm* c, bool is_send, int peer, void* buf, size_t bytes) {
  P2p op{c, &c->host_ops->tr(), is_send, peer, buf, bytes};
  if (g.depth == 0) return run_p2p({op});
  g.p2p.push_back(op);
  return hxcclSuccess;
}

extern "C" {

hxcclResult_t hxcclGroupStart(void) {
  ++g.depth;
  return hxcclSuccess;
}

hxcclResult_t hxcclGroupEnd(void) {
  if (g.depth == 0) return hxcclInvalidUsage;
  if (--g.depth > 0) return hxcclSuccess;
  auto colls = std::move(g.colls);
  auto ops = std::move(g.p2p);
  g.colls.clear();
  g.p2p.clear();
  for (auto& entry : colls)  // collectives first, in call order
    if (hxcclResult_t r = entry.second()) return r;
  return ops.empty() ? hxcclSuccess : run_p2p(ops);
}

}  // extern "C"
