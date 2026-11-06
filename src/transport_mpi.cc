#include <climits>

#include "transport.h"

namespace {
constexpr int kTag = 0;  // collectives run in order on a private dup'ed communicator

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
    error = "message larger than INT_MAX bytes is not supported in A";
    return hxcclSystemError;
  }
  hxcclResult_t r = check(
      MPI_Sendrecv(sbuf, static_cast<int>(sbytes), MPI_BYTE, to, kTag, rbuf,
                   static_cast<int>(rbytes), MPI_BYTE, from, kTag, comm_, MPI_STATUS_IGNORE),
      "MPI_Sendrecv");
  if (r == hxcclSuccess) {
    bytes_sent += sbytes;
    ++msgs_sent;
  }
  return r;
}

hxcclResult_t MpiTransport::send(int to, const void* buf, size_t bytes) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported in A";
    return hxcclSystemError;
  }
  hxcclResult_t r =
      check(MPI_Send(buf, static_cast<int>(bytes), MPI_BYTE, to, kTag, comm_), "MPI_Send");
  if (r == hxcclSuccess) {
    bytes_sent += bytes;
    ++msgs_sent;
  }
  return r;
}

hxcclResult_t MpiTransport::recv(int from, void* buf, size_t bytes) {
  if (too_big(bytes)) {
    error = "message larger than INT_MAX bytes is not supported in A";
    return hxcclSystemError;
  }
  return check(MPI_Recv(buf, static_cast<int>(bytes), MPI_BYTE, from, kTag, comm_,
                        MPI_STATUS_IGNORE),
               "MPI_Recv");
}
