#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Failures (I5): every rank keeps one receive posted for an ABORT message from each peer.
// A rank whose device transfer fails sends ABORT to every peer, so ranks waiting on it
// return an error instead of polling forever. After that the device path of the
// communicator stays failed (every later device call returns the error).
//
// Control messages go through the communicator's Net (MPI or sockets) with their own tags,
// so they never match user data that the host transport of the same communicator sends.
namespace {
constexpr int kReadyBase = 1000, kDoneBase = 2000, kDataBase = 3000, kAbortTag = 4000;

bool env_set(const char* name) {
  const char* e = std::getenv(name);
  return e && *e && std::strcmp(e, "0") != 0;
}
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

// Staged path, one message: the sender copies the device buffer to host memory and sends
// {size} then the bytes; the receiver takes both and copies min(S, R) bytes to its device
// buffer. A receive buffer that is too small fails in the Net, like on host.
struct IpcTransport::StagedOp {
  bool is_send;
  char* dev;
  size_t bytes;
  std::vector<char> host;
  uint64_t size_msg = 0;
  int req_size = -1, req_data = -1;
  bool size_done = false, data_done = false;
};

IpcTransport::IpcTransport(Net& net, size_t chunk)
    : net_(net),
      rank_(net.rank()),
      n_(net.size()),
      chunk_(chunk),
      mine_(n_, nullptr),
      peer_(n_, nullptr),
      sends_(n_),
      recvs_(n_),
      abort_req_(n_, -1),
      abort_in_(n_, 0) {}

// Collective. Every rank tries to set up CUDA IPC, then all ranks agree on the result: if any
// rank could not export or open a handle, every rank drops IPC and uses the staged path
// (device -> host -> Net -> host -> device). One known cause: with Linux Yama ptrace_scope=1,
// cudaIpcOpenMemHandle fails with "invalid resource handle" unless the exporting process
// allows ptrace from its peers. OpenMPI's MPI_Init sets that; torchrun processes do not
// (measured 2026-11-11, LEDGER E4). HXCCL_DEVICE_TRANSPORT=staged forces the staged path.
// Every step below runs on every rank whatever happened before it (I2), so the ranks stay in
// step; only a failure of the Net itself returns an error.
hxcclResult_t IpcTransport::init() {
  // the staging buffers must have the same size everywhere: take the smallest chunk
  int neg_chunk = -static_cast<int>(chunk_);
  if (hxcclResult_t r = net_.allreduce_max(&neg_chunk)) {
    error = "agreement on the IPC chunk size: " + net_.error;
    return r;
  }
  chunk_ = static_cast<size_t>(-neg_chunk);

  std::string why;
  const char* force = std::getenv("HXCCL_DEVICE_TRANSPORT");
  if (force && std::strcmp(force, "staged") == 0) why = "HXCCL_DEVICE_TRANSPORT=staged";
  if (why.empty() && env_set("HXCCL_TEST_FAIL_DEVICE_INIT"))  // test hook (repro_init_failure)
    why = "injected device init failure (HXCCL_TEST_FAIL_DEVICE_INIT)";
  std::vector<cudaIpcMemHandle_t> mine_h(n_), all(size_t(n_) * n_);
  for (int s = 0; s < n_ && why.empty(); ++s) {
    if (s == rank_) continue;
    if (cuda(cudaMalloc(&mine_[s], chunk_), "cudaMalloc staging") ||
        cuda(cudaIpcGetMemHandle(&mine_h[s], mine_[s]), "cudaIpcGetMemHandle"))
      why = error;
  }
  // always exchanged, so the ranks stay in step even when some of them failed above
  if (hxcclResult_t r = net_.allgather(mine_h.data(), all.data(),
                                       sizeof(cudaIpcMemHandle_t) * n_)) {
    error = "allgather of ipc handles: " + net_.error;
    return r;
  }
  int bad = why.empty() ? 0 : 1;  // do not open peers' buffers if any export failed
  if (hxcclResult_t r = net_.allreduce_max(&bad)) {
    error = "agreement on the device transport: " + net_.error;
    return r;
  }
  for (int p = 0; p < n_ && !bad && why.empty(); ++p) {
    if (p == rank_) continue;
    // all[p * n + rank_] is the buffer rank p keeps for data from me
    if (cuda(cudaIpcOpenMemHandle(&peer_[p], all[size_t(p) * n_ + rank_],
                                  cudaIpcMemLazyEnablePeerAccess),
             "cudaIpcOpenMemHandle")) {
      peer_[p] = nullptr;
      why = error;
    }
  }
  int bad_open = why.empty() ? 0 : 1;
  if (hxcclResult_t r = net_.allreduce_max(&bad_open)) {
    error = "agreement on the device transport: " + net_.error;
    return r;
  }
  error.clear();
  cudaGetLastError();  // a failed IPC call is handled here, not an error of the user's call
  if (bad || bad_open) {
    // close the handles of other ranks' buffers everywhere before any rank frees its own
    for (void*& p : peer_) {
      if (p) cudaIpcCloseMemHandle(p);
      p = nullptr;
    }
    int dummy = 0;
    if (hxcclResult_t r = net_.allreduce_max(&dummy)) {
      error = "agreement on the device transport: " + net_.error;
      return r;
    }
    release_ipc();
    staged_ = true;
    staged_reason_ = why.empty() ? "another rank could not set up CUDA IPC" : why;
  }
  for (int p = 0; p < n_; ++p)  // ABORT from any peer, see the top of this file
    if (p != rank_)
      if (hxcclResult_t r = net_.irecv(p, kAbortTag, &abort_in_[p], 8, &abort_req_[p])) {
        error = "posting the abort receive: " + net_.error;
        return r;
      }
  if (env_set("HXCCL_DEBUG") && rank_ == 0)
    std::fprintf(stderr, "hxccl: device transport = %s (chunk %zu B)%s%s\n",
                 staged_ ? "staged" : "ipc", chunk_, staged_ ? ": " : "",
                 staged_reason_.c_str());
  return hxcclSuccess;
}

void IpcTransport::release_ipc() {
  for (void*& p : peer_) {
    if (p) cudaIpcCloseMemHandle(p);
    p = nullptr;
  }
  for (void*& p : mine_) {
    if (p) cudaFree(p);
    p = nullptr;
  }
}

IpcTransport::~IpcTransport() {
  for (int& r : abort_req_)
    if (r >= 0) net_.cancel(r);
  release_ipc();
}

// A fatal error on this rank: remember it, tell every peer, drop everything posted.
hxcclResult_t IpcTransport::fail(hxcclResult_t r) {
  if (broken_ == hxcclSuccess) {
    broken_ = r;
    broken_msg_ = error;
    const uint64_t code = 1;
    for (int p = 0; p < n_; ++p)
      if (p != rank_) net_.send(p, kAbortTag, &code, 8);
  }
  cancel_all();
  error = broken_msg_;
  return broken_;
}

// Has any peer sent ABORT? Then this rank fails too (without echoing it back).
hxcclResult_t IpcTransport::check_abort() {
  for (int p = 0; p < n_; ++p) {
    if (p == rank_ || abort_req_[p] < 0) continue;
    bool done = false;
    const hxcclResult_t got = net_.test(abort_req_[p], &done);
    if (!done) continue;
    abort_req_[p] = -1;
    // a receive that failed (the peer closed its sockets after destroying its communicator)
    // is not an ABORT; a transfer with that peer would fail on its own control messages
    if (got != hxcclSuccess) continue;
    if (broken_ == hxcclSuccess) {
      broken_ = hxcclRemoteError;
      broken_msg_ = "rank " + std::to_string(p) + " failed during a device transfer";
    }
    cancel_all();
    error = broken_msg_;
    return broken_;
  }
  return hxcclSuccess;
}

hxcclResult_t IpcTransport::staged_post(bool is_send, int peer, char* dev, size_t bytes,
                                        int tag) {
  staged_ops_.emplace_back();
  StagedOp& o = staged_ops_.back();
  o.is_send = is_send;
  o.dev = dev;
  o.bytes = bytes;
  o.host.resize(bytes);
  hxcclResult_t r = hxcclSuccess;
  if (is_send) {
    if (bytes) {
      HX_RANGE_PUSH("staged:d2h");
      if (env_set("HXCCL_TEST_FAIL_DEVICE_COPY")) {  // test hook (repro_peer_failure)
        error = "injected device copy failure (HXCCL_TEST_FAIL_DEVICE_COPY)";
        r = hxcclUnhandledCudaError;
      } else {
        r = cuda(cudaMemcpyAsync(o.host.data(), dev, bytes, cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync device to host");
        if (!r) r = cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
      }
      HX_RANGE_POP();
      if (r) return fail(r);
    }
    o.size_msg = bytes;
    if (!r) r = ctl(net_.isend(peer, kDoneBase + tag, &o.size_msg, 8, &o.req_size), "send size");
    if (!r) r = ctl(net_.isend(peer, kDataBase + tag, o.host.data(), bytes, &o.req_data),
                    "send data");
    if (!r) {
      bytes_sent += bytes;
      ++msgs_sent;
    }
  } else {
    r = ctl(net_.irecv(peer, kDoneBase + tag, &o.size_msg, 8, &o.req_size), "recv size");
    if (!r) r = ctl(net_.irecv(peer, kDataBase + tag, o.host.data(), bytes, &o.req_data),
                    "recv data");
  }
  return r ? fail(r) : r;
}

// One non-blocking pass over the staged ops. Received data goes to the device as soon as it
// is in; when everything is in, the stream is synchronized (the host buffers must outlive the
// copies) and the ops are dropped. *first collects a truncation error.
hxcclResult_t IpcTransport::staged_step(bool* done, hxcclResult_t* first) {
  *done = true;
  for (StagedOp& o : staged_ops_) {
    for (auto [req, flag] : {std::pair<int*, bool*>{&o.req_size, &o.size_done},
                             std::pair<int*, bool*>{&o.req_data, &o.data_done}}) {
      if (*flag || *req < 0) continue;
      bool d = false;
      const hxcclResult_t r = net_.test(*req, &d);
      if (!d) {
        *done = false;
        continue;
      }
      *req = -1;
      *flag = true;
      if (r != hxcclSuccess) {
        if (std::strstr(net_.error.c_str(), "truncated")) {  // like the host path: keep going
          if (*first == hxcclSuccess) {
            *first = r;
            error = "staged device transfer: " + net_.error;
          }
        } else {
          error = "staged device transfer: " + net_.error;
          return fail(r);
        }
      }
      if (!o.is_send && o.size_done && o.data_done && r == hxcclSuccess) {
        const size_t len = std::min<size_t>(o.size_msg, o.bytes);
        if (len) {
          HX_RANGE_PUSH("staged:h2d");
          const hxcclResult_t c = cuda(
              cudaMemcpyAsync(o.dev, o.host.data(), len, cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync host to device");
          HX_RANGE_POP();
          if (c) return fail(c);
        }
      }
    }
  }
  if (!*done) return hxcclSuccess;
  if (hxcclResult_t r = cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"))
    return fail(r);
  staged_ops_.clear();
  return hxcclSuccess;
}

hxcclResult_t IpcTransport::isend(int to, const void* buf, size_t bytes, int tag) {
  if (broken_) return error = broken_msg_, broken_;
  char* p = const_cast<char*>(static_cast<const char*>(buf));
  if (staged_ && to != rank_) return staged_post(true, to, p, bytes, tag);
  Op o{true, to, p, bytes, tag};
  sends_[to].push_back(o);
  return hxcclSuccess;
}
hxcclResult_t IpcTransport::irecv(int from, void* buf, size_t bytes, int tag) {
  if (broken_) return error = broken_msg_, broken_;
  if (staged_ && from != rank_)
    return staged_post(false, from, static_cast<char*>(buf), bytes, tag);
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
      if (env_set("HXCCL_TEST_FAIL_DEVICE_COPY")) {  // test hook (repro_peer_failure)
        error = "injected device copy failure (HXCCL_TEST_FAIL_DEVICE_COPY)";
        r = hxcclUnhandledCudaError;
      } else {
        r = cuda(cudaMemcpyAsync(peer_[o.peer], o.buf + o.off, len, cudaMemcpyDeviceToDevice,
                                 stream_),
                 "cudaMemcpyAsync to peer");
        if (!r) r = cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
      }
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
// A short receive is reported like on the other paths, without cancelling anything (M3).
hxcclResult_t IpcTransport::progress_self(hxcclResult_t* first) {
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
    if (s.bytes > d.bytes && *first == hxcclSuccess) {
      *first = hxcclSystemError;
      trunc_msg_ = "message truncated: the receive buffer is smaller than the message";
    }
  }
  return cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

// One non-blocking pass over everything posted (I1: a group drives several transports
// through this in one loop). *done when nothing is left.
hxcclResult_t IpcTransport::step(bool* done) {
  *done = false;
  if (broken_) {
    cancel_all();
    *done = true;
    error = broken_msg_;
    return broken_;
  }
  if (hxcclResult_t r = check_abort()) {
    *done = true;
    return r;
  }
  bool any = false;
  if (!sends_[rank_].empty() || !recvs_[rank_].empty()) {
    any = true;  // an unmatched self op waits forever, as MPI would
    if (!sends_[rank_].empty() && !recvs_[rank_].empty())
      if (hxcclResult_t r = progress_self(&pending_rc_)) {
        *done = true;
        return fail(r);
      }
  }
  for (int p = 0; p < n_; ++p) {
    if (p == rank_) continue;
    for (auto* q : {&sends_[p], &recvs_[p]}) {
      if (q->empty()) continue;
      any = true;
      bool fin = false;
      hxcclResult_t r = progress(q->front(), &fin);
      if (r != hxcclSuccess && !fin) {
        *done = true;
        return fail(r);
      }
      if (r != hxcclSuccess && pending_rc_ == hxcclSuccess) {  // truncation: keep going
        pending_rc_ = r;
        trunc_msg_ = error;
      }
      if (fin) q->pop_front();
    }
  }
  if (!staged_ops_.empty()) {
    bool sdone = false;
    if (hxcclResult_t r = staged_step(&sdone, &pending_rc_)) {
      *done = true;
      return r;
    }
    if (!sdone) any = true;
    else if (pending_rc_ != hxcclSuccess && trunc_msg_.empty()) trunc_msg_ = error;
  }
  if (any) return hxcclSuccess;
  *done = true;
  const hxcclResult_t r = pending_rc_;
  pending_rc_ = hxcclSuccess;
  if (r != hxcclSuccess) error = trunc_msg_;
  trunc_msg_.clear();
  return r;
}

hxcclResult_t IpcTransport::waitall() {
  for (;;) {
    bool done = false;
    const hxcclResult_t r = step(&done);
    if (done) return r;
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
  for (StagedOp& o : staged_ops_) {
    if (o.req_size >= 0) net_.cancel(o.req_size);
    if (o.req_data >= 0) net_.cancel(o.req_data);
  }
  cudaStreamSynchronize(stream_);  // nothing may still read a staged host buffer
  staged_ops_.clear();
  pending_rc_ = hxcclSuccess;
  trunc_msg_.clear();
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
