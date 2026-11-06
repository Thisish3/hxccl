#include "comm.h"

#include <cstdint>
#include <cstring>
#include <random>

namespace {

hxcclResult_t make_comm(hxcclComm_t* out, MPI_Comm owned) {
  MPI_Comm_set_errhandler(owned, MPI_ERRORS_RETURN);
  auto* c = new hxcclComm;
  c->mpi = owned;
  MPI_Comm_rank(owned, &c->rank);
  MPI_Comm_size(owned, &c->nranks);
  c->tr = std::make_unique<MpiTransport>(owned);
  *out = c;
  return hxcclSuccess;
}

uint32_t fnv1a(const hxcclUniqueId& id) {
  uint32_t h = 2166136261u;
  for (char ch : id.internal) {
    h ^= static_cast<unsigned char>(ch);
    h *= 16777619u;
  }
  return h & 0x7fffffffu;
}

}  // namespace

hxcclResult_t comm_fail(hxcclComm* c, hxcclResult_t r, const std::string& msg) {
  if (c) c->last_error = msg;
  return r;
}

extern "C" {

hxcclResult_t hxcclGetUniqueId(hxcclUniqueId* id) {
  if (!id) return hxcclInvalidArgument;
  std::random_device rd;
  for (char& ch : id->internal) ch = static_cast<char>(rd() & 0xff);
  return hxcclSuccess;
}

hxcclResult_t hxcclCommInitMPI(hxcclComm_t* comm, MPI_Comm mpi) {
  if (!comm || mpi == MPI_COMM_NULL) return hxcclInvalidArgument;
  MPI_Comm dup;
  if (MPI_Comm_dup(mpi, &dup) != MPI_SUCCESS) return hxcclSystemError;
  return make_comm(comm, dup);
}

hxcclResult_t hxcclCommInitRank(hxcclComm_t* comm, int nranks, hxcclUniqueId id, int rank) {
  int world = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (!comm || nranks < 1 || nranks > world || rank < 0 || rank >= nranks)
    return hxcclInvalidArgument;
  MPI_Comm sub;
  if (MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(fnv1a(id)), rank, &sub) != MPI_SUCCESS)
    return hxcclSystemError;
  int size = 0;
  MPI_Comm_size(sub, &size);
  if (size != nranks) {
    MPI_Comm_free(&sub);
    return hxcclInvalidUsage;
  }
  return make_comm(comm, sub);
}

hxcclResult_t hxcclCommDestroy(hxcclComm_t comm) {
  if (!comm) return hxcclInvalidArgument;
  comm->tr.reset();
  if (comm->mpi != MPI_COMM_NULL) MPI_Comm_free(&comm->mpi);
  delete comm;
  return hxcclSuccess;
}

hxcclResult_t hxcclCommCount(hxcclComm_t comm, int* count) {
  if (!comm || !count) return hxcclInvalidArgument;
  *count = comm->nranks;
  return hxcclSuccess;
}

hxcclResult_t hxcclCommUserRank(hxcclComm_t comm, int* rank) {
  if (!comm || !rank) return hxcclInvalidArgument;
  *rank = comm->rank;
  return hxcclSuccess;
}

const char* hxcclGetErrorString(hxcclResult_t r) {
  switch (r) {
    case hxcclSuccess: return "no error";
    case hxcclInvalidArgument: return "invalid argument";
    case hxcclSystemError: return "system (MPI) error";
    case hxcclInternalError: return "internal error";
    case hxcclInvalidUsage: return "invalid usage";
  }
  return "unknown result";
}

const char* hxcclGetLastError(hxcclComm_t comm) {
  return comm ? comm->last_error.c_str() : "";
}

hxcclResult_t hxcclCommGetStats(hxcclComm_t comm, uint64_t* bytes, uint64_t* msgs) {
  if (!comm || !bytes || !msgs) return hxcclInvalidArgument;
  *bytes = comm->tr->bytes_sent;
  *msgs = comm->tr->msgs_sent;
  return hxcclSuccess;
}

hxcclResult_t hxcclCommResetStats(hxcclComm_t comm) {
  if (!comm) return hxcclInvalidArgument;
  comm->tr->bytes_sent = comm->tr->msgs_sent = 0;
  return hxcclSuccess;
}

}  // extern "C"
