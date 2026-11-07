#include <climits>

#include "transport.h"

namespace {
bool too_big(size_t n) { return n > static_cast<size_t>(INT_MAX); }
}  // namespace

hxcclResult_t MpiTransport::check(int rc, const char* what) {
  if (rc == MPI_SUCCESS) return hxcclSuccess;
  char msg[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, msg, &len);
  error = std::string(what) + ": " + std::string(msg, len);
  return hxcclSystemError;
}

hxcclResult_t MpiTransport::sendrecv(int to, const void* sbuf, size_t sbytes, int from,
                                     void* rbuf, size_t rbytes) {
  if (too_big(sbytes) || too_big(rbytes)) {
    error = "message larger than INT_MAX bytes is not supported by the MPI transport";
    return hxcclSystemError;
  }
  hxcclResult_t r = check(
      MPI_Sendrecv(sbuf, static_cast<int>(sbytes), MPI_BYTE, to, kCollTag, rbuf,
                   static_cast<int>(rbytes), MPI_BYTE, from, kCollTag, comm_, MPI_STATUS_IGNORE),
      "MPI_Sendrecv");
  if (r == hxcclSuccess) {
    bytes_sent += sbytes;
    ++msgs_sent;
  }
  return r;
}

hxcclResult_t MpiTransport::send(int to, const void* buf, size_t bytes) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported by the MPI transport";
    return hxcclSystemError;
  }
  hxcclResult_t r =
      check(MPI_Send(buf, static_cast<int>(bytes), MPI_BYTE, to, kCollTag, comm_), "MPI_Send");
  if (r == hxcclSuccess) {
    bytes_sent += bytes;
    ++msgs_sent;
  }
  return r;
}

hxcclResult_t MpiTransport::recv(int from, void* buf, size_t bytes) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported by the MPI transport";
    return hxcclSystemError;
  }
  return check(MPI_Recv(buf, static_cast<int>(bytes), MPI_BYTE, from, kCollTag, comm_,
                        MPI_STATUS_IGNORE),
               "MPI_Recv");
}

hxcclResult_t MpiTransport::isend(int to, const void* buf, size_t bytes, int tag) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported by the MPI transport";
    return hxcclSystemError;
  }
  MPI_Request req;
  hxcclResult_t r = check(
      MPI_Isend(buf, static_cast<int>(bytes), MPI_BYTE, to, tag, comm_, &req), "MPI_Isend");
  if (r != hxcclSuccess) return r;
  pending_.push_back(req);
  pending_is_send_.push_back(true);
  bytes_sent += bytes;
  ++msgs_sent;
  return hxcclSuccess;
}

hxcclResult_t MpiTransport::irecv(int from, void* buf, size_t bytes, int tag) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported by the MPI transport";
    return hxcclSystemError;
  }
  MPI_Request req;
  hxcclResult_t r = check(
      MPI_Irecv(buf, static_cast<int>(bytes), MPI_BYTE, from, tag, comm_, &req), "MPI_Irecv");
  if (r == hxcclSuccess) {
    pending_.push_back(req);
    pending_is_send_.push_back(false);
  }
  return r;
}

hxcclResult_t MpiTransport::waitall() {
  if (pending_.empty()) return hxcclSuccess;
  std::vector<MPI_Status> st(pending_.size());
  const int rc = MPI_Waitall(static_cast<int>(pending_.size()), pending_.data(), st.data());
  pending_.clear();
  pending_is_send_.clear();
  if (rc == MPI_ERR_IN_STATUS)
    for (const MPI_Status& s : st)
      if (s.MPI_ERROR != MPI_SUCCESS) return check(s.MPI_ERROR, "MPI_Waitall");
  return check(rc, "MPI_Waitall");
}

void MpiTransport::cancel_all() {
  // MPI-4 deprecates cancelling sends: let them finish in the background instead
  for (size_t i = 0; i < pending_.size(); ++i) {
    if (!pending_is_send_[i]) MPI_Cancel(&pending_[i]);
    MPI_Request_free(&pending_[i]);
  }
  pending_.clear();
  pending_is_send_.clear();
}
