#include <climits>
#include <cstdint>
#include <vector>

#include "algo.h"
#include "checks.h"
#include "comm.h"
#include "group.h"
#include "reduce.h"

namespace {
hxcclResult_t fail_if(hxcclComm* c, hxcclResult_t r) {
  return r == hxcclSuccess ? r : comm_fail(c, r, c->tr->error);
}

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
}  // namespace

extern "C" {

// Block sizes differ per rank, so a rank cannot see another rank's bad block.
// Every rank shares its local verdict first and all of them reject together.
hxcclResult_t hxcclAllToAllv(const void* send, const size_t sc[], const size_t sd[], void* recv,
                             const size_t rc[], const size_t rd[], hxcclDataType_t dt,
                             hxcclComm_t c, hxcclStream_t) {
  if (!c) return hxcclInvalidArgument;
  const size_t es = dtype_size(dt);
  int status = local_v_status(c, send, sc, sd, recv, rc, rd, es);
  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, c->mpi);
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
    return fail_if(c, alltoallv(*c->tr, c->rank, c->nranks, static_cast<const char*>(send),
                                sb.data(), so.data(), static_cast<char*>(recv), rb.data(),
                                ro.data()));
  });
}

hxcclResult_t hxcclAlltoAll(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                            hxcclComm_t c, hxcclStream_t) {
  if (hxcclResult_t r = check_args(c, send, recv, count, c ? c->nranks : 1, dt)) return r;
  const size_t per = count * dtype_size(dt);
  if (hxcclResult_t r = check_message(c, per)) return r;
  if (count == 0) return hxcclSuccess;
  std::vector<size_t> b(c->nranks, per), off(c->nranks);
  for (int j = 0; j < c->nranks; ++j) off[j] = static_cast<size_t>(j) * per;
  return dispatch(c, [=] {
    return fail_if(c, alltoallv(*c->tr, c->rank, c->nranks, static_cast<const char*>(send),
                                b.data(), off.data(), static_cast<char*>(recv), b.data(),
                                off.data()));
  });
}

hxcclResult_t hxcclAllToAll(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                            hxcclComm_t c, hxcclStream_t s) {
  return hxcclAlltoAll(send, recv, count, dt, c, s);
}

}  // extern "C"
