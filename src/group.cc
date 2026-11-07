#include "group.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace {

struct P2p {
  hxcclComm* c;
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
  std::vector<hxcclComm*> comms;
  for (const P2p& o : ops) {
    if (std::find(comms.begin(), comms.end(), o.c) == comms.end()) comms.push_back(o.c);
    Transport& tr = *o.c->tr;
    hxcclResult_t r = o.send ? tr.isend(o.peer, o.buf, o.bytes, kP2pTag)
                             : tr.irecv(o.peer, o.buf, o.bytes, kP2pTag);
    if (r != hxcclSuccess) {
      for (hxcclComm* c : comms) c->tr->cancel_all();
      return comm_fail(o.c, r, tr.error);
    }
  }
  hxcclResult_t first = hxcclSuccess;
  for (hxcclComm* c : comms) {
    hxcclResult_t r = c->tr->waitall();
    if (r != hxcclSuccess && first == hxcclSuccess) first = comm_fail(c, r, c->tr->error);
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
  P2p op{c, is_send, peer, buf, bytes};
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
