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
// No deadlock: READY is a small eager MPI_Send, and a sender makes no blocking call before
// its READY arrives. Ops to the same peer in the same direction advance one at a time from
// the front of their deque, so the single staging buffer is reused in posting order.
//
// Sizes travel with the control messages so a short receive behaves like MPI: READY carries
// the receive size R, DONE carries the send size S and the chunk length. min(S, R) bytes
// move; the receiver fails with "message truncated" when S > R. Zero-byte messages still
// do one READY/DONE round, so message counts match the MPI transport.
//
// Control messages use their own tags, so they never match user data that the MPI
// transport of the same communicator sends.
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
  MPI_Request req = MPI_REQUEST_NULL;
  uint64_t ctrl_out[2] = {0, 0};
  uint64_t ctrl_in[2] = {0, 0};
};

IpcTransport::IpcTransport(MPI_Comm comm, int rank, int n, size_t chunk)
    : comm_(comm),
      rank_(rank),
      n_(n),
      chunk_(chunk),
      mine_(n, nullptr),
      peer_(n, nullptr),
      sends_(n),
      recvs_(n) {}

hxcclResult_t IpcTransport::init() {
  std::vector<cudaIpcMemHandle_t> mine_h(n_), all(size_t(n_) * n_);
  for (int s = 0; s < n_; ++s) {
    if (s == rank_) continue;
    if (hxcclResult_t r = cuda(cudaMalloc(&mine_[s], chunk_), "cudaMalloc staging")) return r;
    if (hxcclResult_t r =
            cuda(cudaIpcGetMemHandle(&mine_h[s], mine_[s]), "cudaIpcGetMemHandle"))
      return r;
  }
  const int hb = static_cast<int>(sizeof(cudaIpcMemHandle_t)) * n_;
  if (hxcclResult_t r = mpi(MPI_Allgather(mine_h.data(), hb, MPI_BYTE, all.data(), hb,
                                          MPI_BYTE, comm_),
                            "MPI_Allgather ipc handles"))
    return r;
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
  int flag = 0;
  hxcclResult_t r;
  if (o.is_send) {
    if (!o.waiting) {  // wait for READY {R}
      if ((r = mpi(MPI_Irecv(o.ctrl_in, 1, MPI_UINT64_T, o.peer, kReadyBase + o.tag, comm_,
                             &o.req),
                   "MPI_Irecv ready")))
        return r;
      o.waiting = true;
    }
    if ((r = mpi(MPI_Test(&o.req, &flag, MPI_STATUS_IGNORE), "MPI_Test ready"))) return r;
    if (!flag) return hxcclSuccess;
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
    if ((r = mpi(MPI_Send(o.ctrl_out, 2, MPI_UINT64_T, o.peer, kDoneBase + o.tag, comm_),
                 "MPI_Send done")))
      return r;
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
    if ((r = mpi(MPI_Send(o.ctrl_out, 1, MPI_UINT64_T, o.peer, kReadyBase + o.tag, comm_),
                 "MPI_Send ready")))
      return r;
    if ((r = mpi(MPI_Irecv(o.ctrl_in, 2, MPI_UINT64_T, o.peer, kDoneBase + o.tag, comm_,
                           &o.req),
                 "MPI_Irecv done")))
      return r;
    o.waiting = true;
  }
  if ((r = mpi(MPI_Test(&o.req, &flag, MPI_STATUS_IGNORE), "MPI_Test done"))) return r;
  if (!flag) return hxcclSuccess;
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
      const bool can = !sends_[rank_].empty() && !recvs_[rank_].empty();
      if (can) {
        any = true;
        if (hxcclResult_t r = progress_self()) {
          cancel_all();
          return r;
        }
      } else {
        any = true;  // an unmatched self op waits forever, as MPI would
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
        if (o.req != MPI_REQUEST_NULL) {
          MPI_Cancel(&o.req);
          MPI_Request_free(&o.req);
        }
      q.clear();
    }
}

hxcclResult_t IpcTransport::cuda(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return hxcclSuccess;
  error = std::string(what) + ": " + cudaGetErrorString(e);
  return hxcclUnhandledCudaError;
}
hxcclResult_t IpcTransport::mpi(int rc, const char* what) {
  if (rc == MPI_SUCCESS) return hxcclSuccess;
  char msg[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, msg, &len);
  error = std::string(what) + ": " + std::string(msg, len);
  return hxcclSystemError;
}
