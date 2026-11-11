#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#include "algo.h"
#include "checks.h"
#include "comm.h"
#include "group.h"
#include "reduce.h"
#include "run.h"

namespace {
// 0 ok, 1 invalid argument, 2 a block exceeds INT_MAX bytes
int local_v_status(const hxcclComm* c, const void* send, const size_t* sc, const size_t* sd,
                   const void* recv, const size_t* rc, const size_t* rd, size_t es) {
  if (es == 0 || !sc || !sd || !rc || !rd) return 1;
  int status = 0;
  for (int j = 0; j < c->nranks; ++j) {
    for (size_t cnt : {sc[j], rc[j]}) {
      if (cnt > SIZE_MAX / es) return 1;
      if (cnt * es > static_cast<size_t>(INT_MAX)) status = 2;
    }
    if (sd[j] > SIZE_MAX / es - sc[j] || rd[j] > SIZE_MAX / es - rc[j]) return 1;
    if ((sc[j] && !send) || (rc[j] && !recv)) return 1;
  }
  return status;
}

// One past the last byte a rank touches on one side of an exchange.
size_t extent(const std::vector<size_t>& bytes, const std::vector<size_t>& off) {
  size_t end = 0;
  for (size_t j = 0; j < bytes.size(); ++j)
    if (bytes[j]) end = std::max(end, off[j] + bytes[j]);
  return end;
}

// In-place (or otherwise overlapping) buffers: the pairwise exchange would overwrite
// blocks before sending them, so send from a private copy instead.
hxcclResult_t run_alltoallv(hxcclComm* c, hxcclStream_t stream, const char* send,
                            const std::vector<size_t>& sb, const std::vector<size_t>& so,
                            char* recv, const std::vector<size_t>& rb,
                            const std::vector<size_t>& ro) {
  return run_on(c, stream, send, recv, "hxccl:alltoall", [&](MemOps& m) {
    const size_t send_end = extent(sb, so), recv_end = extent(rb, ro);
    const bool overlap =
        send_end && recv_end && send < recv + recv_end && recv < send + send_end;
    Scratch stage(m, overlap ? send_end : 0);
    if (stage.rc) return stage.rc;
    const char* from = send;
    if (overlap) {
      if (hxcclResult_t r = m.copy(stage.get(), send, send_end)) return r;
      from = stage.get();
    }
    return alltoallv(m, c->rank, c->nranks, from, sb.data(), so.data(), recv, rb.data(),
                     ro.data());
  });
}
}  // namespace

extern "C" {

// Block sizes differ per rank, so a rank cannot see another rank's bad block.
// Every rank shares its local verdict first and all of them reject together.
hxcclResult_t hxcclAllToAllv(const void* send, const size_t sc[], const size_t sd[], void* recv,
                             const size_t rc[], const size_t rd[], hxcclDataType_t dt,
                             hxcclComm_t c, hxcclStream_t stream) {
  if (!c) return hxcclInvalidArgument;
  // every rank sees the same flag, so rejecting before the agreement step is safe
  if (c->finalized) return comm_fail(c, hxcclInvalidUsage, "communicator was finalized");
  const size_t es = dtype_size(dt);
  int status = local_v_status(c, send, sc, sd, recv, rc, rd, es);
  // runs at call time, even inside a group (see docs/KNOWN_ISSUES.md)
  if (c->net->allreduce_max(&status) != hxcclSuccess)
    return comm_fail(c, hxcclSystemError,
                     "the AllToAllv argument check failed: " + c->net->error);
  if (status == 1)
    return comm_fail(c, hxcclInvalidArgument, "invalid AllToAllv arguments on some rank");
  if (status == 2)
    return comm_fail(c, hxcclInvalidArgument,
                     "a single message would exceed INT_MAX bytes (MPI transport limit)");
  std::vector<size_t> sb(c->nranks), so(c->nranks), rb(c->nranks), ro(c->nranks);
  for (int j = 0; j < c->nranks; ++j) {
    sb[j] = sc[j] * es;
    so[j] = sd[j] * es;
    rb[j] = rc[j] * es;
    ro[j] = rd[j] * es;
  }
  return dispatch(c, [=] {
    return run_alltoallv(c, stream, static_cast<const char*>(send), sb, so,
                         static_cast<char*>(recv), rb, ro);
  });
}

hxcclResult_t hxcclAlltoAll(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                            hxcclComm_t c, hxcclStream_t stream) {
  if (hxcclResult_t r = check_args(c, send, recv, count, c ? c->nranks : 1, dt)) return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  if (count == 0) return hxcclSuccess;
  std::vector<size_t> b(c->nranks, per), off(c->nranks);
  for (int j = 0; j < c->nranks; ++j) off[j] = static_cast<size_t>(j) * per;
  return dispatch(c, [=] {
    return run_alltoallv(c, stream, static_cast<const char*>(send), b, off,
                         static_cast<char*>(recv), b, off);
  });
}

hxcclResult_t hxcclAllToAll(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                            hxcclComm_t c, hxcclStream_t s) {
  return hxcclAlltoAll(send, recv, count, dt, c, s);
}

}  // extern "C"
