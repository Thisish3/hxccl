#include "comm.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

hxcclResult_t make_comm(hxcclComm_t* out, MPI_Comm owned) {
  MPI_Comm_set_errhandler(owned, MPI_ERRORS_RETURN);
  auto* c = new hxcclComm;
  c->mpi = owned;
  MPI_Comm_rank(owned, &c->rank);
  MPI_Comm_size(owned, &c->nranks);
  c->tr = std::make_unique<MpiTransport>(owned);
  static int next_tag = 0;  // tags wrap after 32766 communicators; a collision needs that many
  next_tag = next_tag % 0x7ffe + 1;
  c->op_tag = next_tag;
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

hxcclResult_t check_config(const hxcclConfig_t* cfg) {
  if (!cfg) return hxcclSuccess;
  if (cfg->size != sizeof(hxcclConfig_t) || cfg->magic != HXCCL_CONFIG_MAGIC)
    return hxcclInvalidArgument;  // not built with HXCCL_CONFIG_INITIALIZER
  if (cfg->blocking == 0) return hxcclInvalidArgument;  // non-blocking arrives in B
  return hxcclSuccess;
}

}  // namespace

long user_op_index(const hxcclComm* c, hxcclRedOp_t op) {
  const long v = static_cast<long>(op) - hxcclNumOps;
  if (v < 0 || (v >> kOpIndexBits) != c->op_tag) return -1;
  const long ix = v & ((1L << kOpIndexBits) - 1);
  if (ix >= static_cast<long>(c->user_ops.size()) || !c->user_ops[ix].live) return -1;
  return ix;
}

hxcclResult_t comm_fail(hxcclComm* c, hxcclResult_t r, const std::string& msg) {
  if (c) {
    c->last_error = msg;
    if (r == hxcclSystemError || r == hxcclInternalError || r == hxcclRemoteError)
      c->async_error = r;
  }
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
  int got = -1;
  MPI_Comm_rank(sub, &got);
  if (size != nranks || got != rank) {  // ranks disagree about who is who
    MPI_Comm_free(&sub);
    return hxcclInvalidUsage;
  }
  return make_comm(comm, sub);
}

hxcclResult_t hxcclCommInitRankConfig(hxcclComm_t* comm, int nranks, hxcclUniqueId id, int rank,
                                      hxcclConfig_t* config) {
  if (hxcclResult_t r = check_config(config)) return r;
  return hxcclCommInitRank(comm, nranks, id, rank);
}

hxcclResult_t hxcclCommSplit(hxcclComm_t comm, int color, int key, hxcclComm_t* newcomm,
                             hxcclConfig_t* config) {
  if (!comm || !newcomm || (color < 0 && color != HXCCL_SPLIT_NOCOLOR))
    return hxcclInvalidArgument;
  if (hxcclResult_t r = check_config(config)) return comm_fail(comm, r, "unsupported config");
  MPI_Comm sub;
  if (MPI_Comm_split(comm->mpi, color == HXCCL_SPLIT_NOCOLOR ? MPI_UNDEFINED : color, key,
                     &sub) != MPI_SUCCESS)
    return comm_fail(comm, hxcclSystemError, "MPI_Comm_split failed");
  if (sub == MPI_COMM_NULL) {
    *newcomm = nullptr;
    return hxcclSuccess;
  }
  return make_comm(newcomm, sub);
}

hxcclResult_t hxcclCommShrink(hxcclComm_t comm, int* excl, int nexcl, hxcclComm_t* newcomm,
                              hxcclConfig_t* config, int flags) {
  if (!comm || !newcomm || nexcl < 0 || (nexcl > 0 && !excl)) return hxcclInvalidArgument;
  if (flags != 0)
    return comm_fail(comm, hxcclInvalidArgument, "shrinkFlags other than 0 arrive in E");
  if (hxcclResult_t r = check_config(config)) return comm_fail(comm, r, "unsupported config");
  std::vector<bool> seen(comm->nranks, false);
  for (int i = 0; i < nexcl; ++i) {
    if (excl[i] < 0 || excl[i] >= comm->nranks || seen[excl[i]])
      return comm_fail(comm, hxcclInvalidArgument, "excluded rank out of range or repeated");
    seen[excl[i]] = true;
  }
  if (seen[comm->rank])
    return comm_fail(comm, hxcclInvalidUsage, "excluded ranks must not call Shrink");
  MPI_Group all, kept;
  MPI_Comm sub;
  MPI_Comm_group(comm->mpi, &all);
  MPI_Group_excl(all, nexcl, excl, &kept);
  const int rc = MPI_Comm_create_group(comm->mpi, kept, /*tag=*/7, &sub);
  MPI_Group_free(&kept);
  MPI_Group_free(&all);
  if (rc != MPI_SUCCESS) return comm_fail(comm, hxcclSystemError, "MPI_Comm_create_group failed");
  return make_comm(newcomm, sub);
}

hxcclResult_t hxcclCommAbort(hxcclComm_t comm) {
  if (!comm) return hxcclInvalidArgument;
  // local teardown only: MPI_Comm_free may need peers, so the MPI handle is leaked on purpose
  comm->tr.reset();
  delete comm;
  return hxcclSuccess;
}

hxcclResult_t hxcclCommFinalize(hxcclComm_t comm) {
  if (!comm) return hxcclInvalidArgument;
  comm->finalized = true;  // every call runs to completion, so nothing is left in flight
  return hxcclSuccess;
}

hxcclResult_t hxcclCommGetAsyncError(hxcclComm_t comm, hxcclResult_t* err) {
  if (!comm || !err) return hxcclInvalidArgument;
  *err = comm->async_error;
  return hxcclSuccess;
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
    case hxcclUnhandledCudaError: return "unhandled device error";
    case hxcclSystemError: return "system (MPI) error";
    case hxcclInternalError: return "internal error";
    case hxcclInvalidArgument: return "invalid argument";
    case hxcclInvalidUsage: return "invalid usage";
    case hxcclRemoteError: return "remote process error";
    case hxcclInProgress: return "operation in progress";
    case hxcclTimeout: return "timeout";
    case hxcclNumResults: break;
  }
  return "unknown result";
}

hxcclResult_t hxcclGetVersion(int* version) {
  if (!version) return hxcclInvalidArgument;
  *version = HXCCL_VERSION_CODE;
  return hxcclSuccess;
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
