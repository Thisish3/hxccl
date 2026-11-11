#include <climits>

#include "net.h"

MpiNet::MpiNet(MPI_Comm comm) : comm_(comm) {
  MPI_Comm_rank(comm, &rank_);
  MPI_Comm_size(comm, &size_);
}

hxcclResult_t MpiNet::check(int rc, const char* what) {
  if (rc == MPI_SUCCESS) return hxcclSuccess;
  char msg[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, msg, &len);
  error = std::string(what) + ": " + std::string(msg, len);
  return hxcclSystemError;
}

int MpiNet::slot(MPI_Request r, bool is_send) {
  if (!free_.empty()) {
    const int i = free_.back();
    free_.pop_back();
    reqs_[i] = r;
    is_send_[i] = is_send;
    return i;
  }
  reqs_.push_back(r);
  is_send_.push_back(is_send);
  return static_cast<int>(reqs_.size()) - 1;
}

hxcclResult_t MpiNet::isend(int peer, int tag, const void* buf, size_t bytes, int* req) {
  if (bytes > static_cast<size_t>(INT_MAX)) {
    error = "control message larger than INT_MAX bytes";
    return hxcclSystemError;
  }
  MPI_Request r;
  if (hxcclResult_t e = check(
          MPI_Isend(buf, static_cast<int>(bytes), MPI_BYTE, peer, tag, comm_, &r), "MPI_Isend"))
    return e;
  *req = slot(r, true);
  return hxcclSuccess;
}

hxcclResult_t MpiNet::irecv(int peer, int tag, void* buf, size_t bytes, int* req) {
  if (bytes > static_cast<size_t>(INT_MAX)) {
    error = "control message larger than INT_MAX bytes";
    return hxcclSystemError;
  }
  MPI_Request r;
  if (hxcclResult_t e = check(
          MPI_Irecv(buf, static_cast<int>(bytes), MPI_BYTE, peer, tag, comm_, &r), "MPI_Irecv"))
    return e;
  *req = slot(r, false);
  return hxcclSuccess;
}

hxcclResult_t MpiNet::test(int req, bool* done) {
  int flag = 0;
  MPI_Status st;
  hxcclResult_t r = check(MPI_Test(&reqs_[req], &flag, &st), "MPI_Test");
  *done = flag != 0 || r != hxcclSuccess;
  if (*done) free_.push_back(req);
  return r;
}

void MpiNet::cancel(int req) {
  // MPI-4 deprecates cancelling sends: let them finish in the background instead
  if (reqs_[req] != MPI_REQUEST_NULL) {
    if (!is_send_[req]) MPI_Cancel(&reqs_[req]);
    MPI_Request_free(&reqs_[req]);
  }
  free_.push_back(req);
}

hxcclResult_t MpiNet::allgather(const void* mine, void* all, size_t bytes) {
  return check(MPI_Allgather(mine, static_cast<int>(bytes), MPI_BYTE, all,
                             static_cast<int>(bytes), MPI_BYTE, comm_),
               "MPI_Allgather");
}

hxcclResult_t MpiNet::allreduce_max(int* v) {
  return check(MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_INT, MPI_MAX, comm_), "MPI_Allreduce");
}
