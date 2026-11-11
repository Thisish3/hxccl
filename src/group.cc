#include "group.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "checks.h"
#include "memspace.h"
#include "trace.h"

namespace {

struct P2p {
  hxcclComm* c;
  MemOps* m;  // host_ops or dev_ops of c, chosen from the buffer
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

bool is_host(const P2p& o) { return o.m == o.c->host_ops.get(); }

// Post every send/recv before waiting on any, so pairs that send to each other finish.
hxcclResult_t run_p2p(const std::vector<P2p>& ops) {
  // The host transport blocks (MPI_Waitall) while the device one polls, so a rank waiting
  // on host traffic could stall a peer's device traffic. Mixing is rejected for now.
  const bool any_host = std::any_of(ops.begin(), ops.end(), is_host);
  const bool any_dev = !std::all_of(ops.begin(), ops.end(), is_host);
  if (any_host && any_dev)
    return comm_fail(ops.front().c, hxcclInvalidUsage,
                     "a group cannot mix host and device Send/Recv buffers");
  HX_RANGE_PUSH("hxccl:p2p");
  std::vector<std::pair<hxcclComm*, MemOps*>> used;  // one entry per distinct memory path
  for (const P2p& o : ops) {
    if (std::none_of(used.begin(), used.end(), [&](const auto& u) { return u.second == o.m; }))
      used.emplace_back(o.c, o.m);
    Transport& tr = o.m->tr();
    hxcclResult_t r = o.send ? tr.isend(o.peer, o.buf, o.bytes, kP2pTag)
                             : tr.irecv(o.peer, o.buf, o.bytes, kP2pTag);
    if (r != hxcclSuccess) {
      for (const auto& u : used) u.second->tr().cancel_all();
      HX_RANGE_POP();
      return comm_fail(o.c, r, tr.error);
    }
  }
  // Drive every transport in one loop (I1): waiting on one communicator's transport while
  // a peer waits on another's would deadlock (crossed Send/Recv over two communicators).
  hxcclResult_t first = hxcclSuccess;
  std::vector<bool> fin(used.size(), false);
  for (size_t left = used.size(); left > 0;) {
    for (size_t i = 0; i < used.size(); ++i) {
      if (fin[i]) continue;
      bool done = false;
      hxcclResult_t r = used[i].second->tr().step(&done);
      if (!done) continue;
      fin[i] = true;
      --left;
      if (r == hxcclSuccess) r = sync_ops(*used[i].second);
      if (r != hxcclSuccess && first == hxcclSuccess) {
        first = ops_fail(used[i].first, *used[i].second, r);
        // the others may wait for a peer that now fails: stop them too
        for (size_t j = 0; j < used.size(); ++j)
          if (!fin[j]) used[j].second->tr().cancel_all(), fin[j] = true, --left;
      }
    }
  }
  HX_RANGE_POP();
  return first;
}

}  // namespace

hxcclResult_t dispatch(hxcclComm* c, std::function<hxcclResult_t()> run) {
  if (g.depth == 0) return run();
  g.colls.emplace_back(c, std::move(run));
  return hxcclSuccess;
}

hxcclResult_t p2p(hxcclComm* c, bool is_send, int peer, void* buf, size_t bytes, MemOps* m) {
  P2p op{c, m, is_send, peer, buf, bytes};
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
