#include "comm.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "memspace.h"
#include "net_sock.h"

namespace {

void set_op_tag(hxcclComm* c) {
  static int next_tag = 0;  // tags wrap after 32766 communicators; a collision needs that many
  next_tag = next_tag % 0x7ffe + 1;
  c->op_tag = next_tag;
}

// MPI communicator: MPI transport and MPI control traffic.
hxcclResult_t make_comm(hxcclComm_t* out, MPI_Comm owned) {
  MPI_Comm_set_errhandler(owned, MPI_ERRORS_RETURN);
  auto* c = new hxcclComm;
  c->mpi = owned;
  MPI_Comm_rank(owned, &c->rank);
  MPI_Comm_size(owned, &c->nranks);
  c->tr = std::make_unique<MpiTransport>(owned);
  c->net = std::make_shared<MpiNet>(owned);
  c->host_ops = std::make_unique<HostOps>(c->tr);
  set_op_tag(c);
  device_eager_init(c);
  *out = c;
  return hxcclSuccess;
}

// Socket communicator (spec E): no MPI at all.
hxcclResult_t make_sock_comm(hxcclComm_t* out, std::shared_ptr<SockNet> net) {
  auto* c = new hxcclComm;
  c->rank = net->rank();
  c->nranks = net->size();
  c->net = net;
  c->tr = std::make_unique<NetTransport>(net);
  c->host_ops = std::make_unique<HostOps>(c->tr);
  set_op_tag(c);
  device_eager_init(c);
  *out = c;
  return hxcclSuccess;
}

SockNet* sock_net(hxcclComm* c) { return c->mpi == MPI_COMM_NULL ? static_cast<SockNet*>(c->net.get()) : nullptr; }

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

// The id holds the address of a listening socket in this process; the process that makes the
// id must take part in hxcclCommInitRank (spec E 3.3).
hxcclResult_t hxcclGetUniqueId(hxcclUniqueId* id) {
  if (!id) return hxcclInvalidArgument;
  std::string err;
  return sock_unique_id(id, &err);
}

hxcclResult_t hxcclCommInitMPI(hxcclComm_t* comm, MPI_Comm mpi) {
  if (!comm || mpi == MPI_COMM_NULL) return hxcclInvalidArgument;
  const char* boot = std::getenv("HXCCL_BOOTSTRAP");
  if (boot && std::strcmp(boot, "socket") == 0) {  // same ranks, socket transport
    int rank = 0, size = 0;
    MPI_Comm_rank(mpi, &rank);
    MPI_Comm_size(mpi, &size);
    hxcclUniqueId id;
    int ok = 1;
    if (rank == 0) ok = hxcclGetUniqueId(&id) == hxcclSuccess;
    MPI_Bcast(&ok, 1, MPI_INT, 0, mpi);
    if (!ok) return hxcclSystemError;
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, mpi);
    return hxcclCommInitRank(comm, size, id, rank);
  }
  MPI_Comm dup;
  if (MPI_Comm_dup(mpi, &dup) != MPI_SUCCESS) return hxcclSystemError;
  return make_comm(comm, dup);
}

hxcclResult_t hxcclCommInitRank(hxcclComm_t* comm, int nranks, hxcclUniqueId id, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return hxcclInvalidArgument;
  std::shared_ptr<SockNet> net;
  std::string err;
  if (hxcclResult_t r = sock_init_rank(id, nranks, rank, &net, &err)) return r;
  return make_sock_comm(comm, net);
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
  if (SockNet* net = sock_net(comm)) {  // agree on members through the parent, share its mesh
    const uint64_t seq = comm->derived++;
    const int mine[2] = {color, key};
    std::vector<int> all(2 * size_t(comm->nranks));
    if (hxcclResult_t r = net->allgather(mine, all.data(), sizeof mine))
      return comm_fail(comm, r, "Split: " + net->error);
    if (color == HXCCL_SPLIT_NOCOLOR) {
      *newcomm = nullptr;
      return hxcclSuccess;
    }
    std::vector<int> members;  // parent ranks with my color, ordered by (key, parent rank)
    for (int i = 0; i < comm->nranks; ++i)
      if (all[2 * i] == color) members.push_back(i);
    std::stable_sort(members.begin(), members.end(),
                     [&](int a, int b) { return all[2 * a + 1] < all[2 * b + 1]; });
    const int me = static_cast<int>(std::find(members.begin(), members.end(), comm->rank) -
                                    members.begin());
    return make_sock_comm(newcomm,
                          net->sub(members, me, derive_comm_id(net->id(), seq, uint32_t(color))));
  }
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
  if (SockNet* net = sock_net(comm)) {  // only survivors call: no messages, same list everywhere
    uint64_t salt = 0x5348524eull;  // "SHRN"
    std::vector<int> members;
    for (int i = 0; i < comm->nranks; ++i) {
      if (seen[i]) salt = derive_comm_id(salt, uint64_t(i), 1);
      else members.push_back(i);
    }
    const int me = static_cast<int>(std::find(members.begin(), members.end(), comm->rank) -
                                    members.begin());
    return make_sock_comm(newcomm, net->sub(members, me, derive_comm_id(net->id(),
                                                                        comm->derived++, salt)));
  }
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
  comm->dev_ops.reset();
  comm->host_ops.reset();
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
  comm->dev_ops.reset();
  comm->host_ops.reset();
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
  // user data only, summed over the host (MPI) and device (IPC) transports
  *bytes = comm->tr->bytes_sent;
  *msgs = comm->tr->msgs_sent;
  if (comm->dev_ops) {
    *bytes += comm->dev_ops->tr().bytes_sent;
    *msgs += comm->dev_ops->tr().msgs_sent;
  }
  return hxcclSuccess;
}

hxcclResult_t hxcclCommResetStats(hxcclComm_t comm) {
  if (!comm) return hxcclInvalidArgument;
  comm->tr->bytes_sent = comm->tr->msgs_sent = 0;
  if (comm->dev_ops) comm->dev_ops->tr().bytes_sent = comm->dev_ops->tr().msgs_sent = 0;
  return hxcclSuccess;
}

}  // extern "C"
