#include "transport.h"

hxcclResult_t NetTransport::fail(hxcclResult_t r) {
  error = net_->error;
  return r;
}

hxcclResult_t NetTransport::sendrecv(int to, const void* sbuf, size_t sbytes, int from,
                                     void* rbuf, size_t rbytes) {
  int rr, sr;
  if (hxcclResult_t r = net_->irecv(from, kCollTag, rbuf, rbytes, &rr)) return fail(r);
  if (hxcclResult_t r = net_->isend(to, kCollTag, sbuf, sbytes, &sr)) {
    net_->cancel(rr);
    return fail(r);
  }
  const hxcclResult_t a = net_->wait(sr);
  const hxcclResult_t b = net_->wait(rr);
  if (a != hxcclSuccess) return fail(a);
  if (b != hxcclSuccess) return fail(b);
  bytes_sent += sbytes;
  ++msgs_sent;
  return hxcclSuccess;
}

hxcclResult_t NetTransport::send(int to, const void* buf, size_t bytes) {
  int req;
  if (hxcclResult_t r = net_->isend(to, kCollTag, buf, bytes, &req)) return fail(r);
  if (hxcclResult_t r = net_->wait(req)) return fail(r);
  bytes_sent += bytes;
  ++msgs_sent;
  return hxcclSuccess;
}

hxcclResult_t NetTransport::recv(int from, void* buf, size_t bytes) {
  int req;
  if (hxcclResult_t r = net_->irecv(from, kCollTag, buf, bytes, &req)) return fail(r);
  if (hxcclResult_t r = net_->wait(req)) return fail(r);
  return hxcclSuccess;
}

hxcclResult_t NetTransport::isend(int to, const void* buf, size_t bytes, int tag) {
  int req;
  if (hxcclResult_t r = net_->isend(to, tag, buf, bytes, &req)) return fail(r);
  pending_.push_back(req);
  bytes_sent += bytes;
  ++msgs_sent;
  return hxcclSuccess;
}

hxcclResult_t NetTransport::irecv(int from, void* buf, size_t bytes, int tag) {
  int req;
  if (hxcclResult_t r = net_->irecv(from, tag, buf, bytes, &req)) return fail(r);
  pending_.push_back(req);
  return hxcclSuccess;
}

hxcclResult_t NetTransport::waitall() {
  hxcclResult_t first = hxcclSuccess;
  std::string msg;
  for (int req : pending_) {
    const hxcclResult_t r = net_->wait(req);
    if (r != hxcclSuccess && first == hxcclSuccess) {
      first = r;
      msg = net_->error;
    }
  }
  pending_.clear();
  if (first != hxcclSuccess) error = msg;
  return first;
}

void NetTransport::cancel_all() {
  for (int req : pending_) net_->cancel(req);
  pending_.clear();
}
