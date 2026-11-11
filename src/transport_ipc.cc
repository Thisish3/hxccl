#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "trace.h"
#include "transport.h"

// One staging buffer per (receiver, sender) pair, owned by the receiver and opened by the
// sender over CUDA IPC. Per chunk: the receiver sends READY, the sender copies into the
// staging buffer, synchronizes its stream and sends DONE, the receiver copies out.
//
// No deadlock: READY is a small control message, and a sender makes no blocking call
// before its READY arrives. Ops to the same peer in the same direction advance one at a time
// from the front of their deque, so the single staging buffer is reused in posting order.
//
// Sizes travel with the control messages so a short receive behaves like MPI: READY carries
// the receive size R, DONE carries the send size S and the chunk length. min(S, R) bytes
// move; the receiver fails with "message truncated" when S > R. Zero-byte messages still
// do one READY/DONE round, so message counts match the host transport.
//
// Control messages go through the communicator's Net (MPI or sockets) with their own tags,
// so they never match user data that the host transport of the same communicator sends.
namespace {
constexpr int kReadyBase = 1000, kDoneBase = 2000;
}  // namespace

struct IpcTransport::Op {
  bool is_send;
  int peer;
  char* buf;
  size_t bytes;
  int tag;
  size_t off = 0;
  size_t total = SIZE_MAX;  // min(S, R), known after the first control message
  bool waiting = false;
  int req = -1;  // pending control receive on net_
  uint64_t ctrl_out[2] = {0, 0};
  uint64_t ctrl_in[2] = {0, 0};
};

IpcTransport::IpcTransport(Net& net, size_t chunk)
    : net_(net),
      rank_(net.rank()),
      n_(net.size()),
      chunk_(chunk),
      mine_(n_, nullptr),
      peer_(n_, nullptr),
      sends_(n_),
      recvs_(n_) {}

hxcclResult_t IpcTransport::init() {
  std::vector<cudaIpcMemHandle_t> mine_h(n_), all(size_t(n_) * n_);
  for (int s = 0; s < n_; ++s) {
    if (s == rank_) continue;
    if (hxcclResult_t r = cuda(cudaMalloc(&mine_[s], chunk_), "cudaMalloc staging")) return r;
    if (hxcclResult_t r =
            cuda(cudaIpcGetMemHandle(&mine_h[s], mine_[s]), "cudaIpcGetMemHandle"))
      return r;
  }
  if (hxcclResult_t r = net_.allgather(mine_h.data(), all.data(),
                                       sizeof(cudaIpcMemHandle_t) * n_)) {
    error = "allgather of ipc handles: " + net_.error;
    return r;
  }
  for (int p = 0; p < n_; ++p) {
    if (p == rank_) continue;
    // all[p * n + rank_] is the buffer rank p keeps for data from me
    if (hxcclResult_t r = cuda(cudaIpcOpenMemHandle(&peer_[p], all[size_t(p) * n_ + rank_],
                                                    cudaIpcMemLazyEnablePeerAccess),
                               "cudaIpcOpenMemHandle"))
      return r;
  }
  return hxcclSuccess;
}

IpcTransport::~IpcTransport() {
  for (void* p : peer_)
    if (p) cudaIpcCloseMemHandle(p);
  for (void* p : mine_)
    if (p) cudaFree(p);
}

hxcclResult_t IpcTransport::isend(int to, const void* buf, size_t bytes, int tag) {
  Op o{true, to, const_cast<char*>(static_cast<const char*>(buf)), bytes, tag};
  sends_[to].push_back(o);
  return hxcclSuccess;
}
hxcclResult_t IpcTransport::irecv(int from, void* buf, size_t bytes, int tag) {
  Op o{false, from, static_cast<char*>(buf), bytes, tag};
  recvs_[from].push_back(o);
  return hxcclSuccess;
}

hxcclResult_t IpcTransport::progress(Op& o, bool* finished) {
  *finished = false;
  bool arrived = false;
  hxcclResult_t r;
  if (o.is_send) {
    if (!o.waiting) {  // wait for READY {R}
      if ((r = ctl(net_.irecv(o.peer, kReadyBase + o.tag, o.ctrl_in, 8, &o.req), "recv ready")))
        return r;
      o.waiting = true;
    }
    r = ctl(net_.test(o.req, &arrived), "test ready");
    if (arrived) o.req = -1;
    if (r) return r;
    if (!arrived) return hxcclSuccess;
    o.waiting = false;
    o.total = std::min<size_t>(o.bytes, o.ctrl_in[0]);
    const size_t len = std::min(chunk_, o.total - o.off);
    if (len) {
      HX_RANGE_PUSH("ipc:copy");
      r = cuda(cudaMemcpyAsync(peer_[o.peer], o.buf + o.off, len, cudaMemcpyDeviceToDevice,
                               stream_),
               "cudaMemcpyAsync to peer");
      if (!r) r = cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
      HX_RANGE_POP();
      if (r) return r;
    }
    o.ctrl_out[0] = o.bytes;
    o.ctrl_out[1] = len;
    if ((r = ctl(net_.send(o.peer, kDoneBase + o.tag, o.ctrl_out, 16), "send done"))) return r;
    o.off += len;
    if (o.off >= o.total) {
      *finished = true;
      bytes_sent += o.bytes;
      ++msgs_sent;
    }
    return hxcclSuccess;
  }
  if (!o.waiting) {  // announce the free buffer, then wait for DONE {S, len}
    o.ctrl_out[0] = o.bytes;
    if ((r = ctl(net_.send(o.peer, kReadyBase + o.tag, o.ctrl_out, 8), "send ready"))) return r;
    if ((r = ctl(net_.irecv(o.peer, kDoneBase + o.tag, o.ctrl_in, 16, &o.req), "recv done")))
      return r;
    o.waiting = true;
  }
  r = ctl(net_.test(o.req, &arrived), "test done");
  if (arrived) o.req = -1;
  if (r) return r;
  if (!arrived) return hxcclSuccess;
  o.waiting = false;
  const size_t sbytes = o.ctrl_in[0], len = o.ctrl_in[1];
  o.total = std::min(sbytes, o.bytes);
  if (len) {
    HX_RANGE_PUSH("ipc:copy");
    r = cuda(cudaMemcpyAsync(o.buf + o.off, mine_[o.peer], len, cudaMemcpyDeviceToDevice,
                             stream_),
             "cudaMemcpyAsync from staging");
    if (!r) r = cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    HX_RANGE_POP();
    if (r) return r;
  }
  o.off += len;
  if (o.off >= o.total) {
    *finished = true;
    if (sbytes > o.bytes) {
      error = "message truncated: the receive buffer is smaller than the message";
      return hxcclSystemError;
    }
  }
  return hxcclSuccess;
}

// Sends to myself pair up with receives from myself in posting order: one device copy.
hxcclResult_t IpcTransport::progress_self() {
  auto& sq = sends_[rank_];
  auto& rq = recvs_[rank_];
  while (!sq.empty() && !rq.empty()) {
    Op s = sq.front(), d = rq.front();
    sq.pop_front();
    rq.pop_front();
    const size_t len = std::min(s.bytes, d.bytes);
    if (len && s.buf != d.buf)
      if (hxcclResult_t r = cuda(cudaMemcpyAsync(d.buf, s.buf, len, cudaMemcpyDeviceToDevice,
                                                 stream_),
                                 "cudaMemcpyAsync to self"))
        return r;
    bytes_sent += s.bytes;
    ++msgs_sent;
    if (s.bytes > d.bytes) {
      error = "message truncated: the receive buffer is smaller than the message";
      return hxcclSystemError;
    }
  }
  return cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

hxcclResult_t IpcTransport::waitall() {
  hxcclResult_t first = hxcclSuccess;
  for (;;) {
    bool any = false;
    if (!sends_[rank_].empty() || !recvs_[rank_].empty()) {
      any = true;  // an unmatched self op waits forever, as MPI would
      if (!sends_[rank_].empty() && !recvs_[rank_].empty())
        if (hxcclResult_t r = progress_self()) {
          cancel_all();
          return r;
        }
    }
    for (int p = 0; p < n_; ++p) {
      if (p == rank_) continue;
      for (auto* q : {&sends_[p], &recvs_[p]}) {
        if (q->empty()) continue;
        any = true;
        bool done = false;
        hxcclResult_t r = progress(q->front(), &done);
        if (r != hxcclSuccess && !done) {
          cancel_all();
          return r;
        }
        if (r != hxcclSuccess && first == hxcclSuccess) first = r;  // truncation: keep going
        if (done) q->pop_front();
      }
    }
    if (!any) return first;
  }
}

hxcclResult_t IpcTransport::sendrecv(int to, const void* sb, size_t sn, int from, void* rb,
                                     size_t rn) {
  if (hxcclResult_t r = irecv(from, rb, rn, kCollTag)) return r;
  if (hxcclResult_t r = isend(to, sb, sn, kCollTag)) return r;
  return waitall();
}
hxcclResult_t IpcTransport::send(int to, const void* b, size_t n) {
  if (hxcclResult_t r = isend(to, b, n, kCollTag)) return r;
  return waitall();
}
hxcclResult_t IpcTransport::recv(int from, void* b, size_t n) {
  if (hxcclResult_t r = irecv(from, b, n, kCollTag)) return r;
  return waitall();
}

void IpcTransport::cancel_all() {
  for (auto* qs : {&sends_, &recvs_})
    for (auto& q : *qs) {
      for (Op& o : q)
        if (o.req >= 0) net_.cancel(o.req);
      q.clear();
    }
}

hxcclResult_t IpcTransport::cuda(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return hxcclSuccess;
  error = std::string(what) + ": " + cudaGetErrorString(e);
  return hxcclUnhandledCudaError;
}
hxcclResult_t IpcTransport::ctl(hxcclResult_t r, const char* what) {
  if (r != hxcclSuccess) error = std::string("ipc control ") + what + ": " + net_.error;
  return r;
}
