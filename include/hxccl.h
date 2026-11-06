/* hxccl: a small NCCL-style collective library (A: CPU buffers over MPI). */
#ifndef HXCCL_H_
#define HXCCL_H_

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hxcclComm* hxcclComm_t;
typedef struct {
  char internal[128];
} hxcclUniqueId;
typedef void* hxcclStream_t; /* ignored in A; cudaStream_t / hipStream_t in B */

typedef enum {
  hxcclSuccess = 0,
  hxcclInvalidArgument = 1,
  hxcclSystemError = 2,
  hxcclInternalError = 3,
  hxcclInvalidUsage = 4,
} hxcclResult_t;

typedef enum { hxcclInt32 = 0, hxcclFloat32 = 1, hxcclFloat64 = 2 } hxcclDataType_t;
typedef enum { hxcclSum = 0, hxcclProd = 1, hxcclMax = 2, hxcclMin = 3 } hxcclRedOp_t;

hxcclResult_t hxcclGetUniqueId(hxcclUniqueId* id);
/* All ranks of MPI_COMM_WORLD must call this together, in the same order. */
hxcclResult_t hxcclCommInitRank(hxcclComm_t* comm, int nranks, hxcclUniqueId id, int rank);
hxcclResult_t hxcclCommInitMPI(hxcclComm_t* comm, MPI_Comm mpi);
hxcclResult_t hxcclCommDestroy(hxcclComm_t comm);
hxcclResult_t hxcclCommCount(hxcclComm_t comm, int* count);
hxcclResult_t hxcclCommUserRank(hxcclComm_t comm, int* rank);

hxcclResult_t hxcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                             hxcclDataType_t datatype, hxcclRedOp_t op, hxcclComm_t comm,
                             hxcclStream_t stream);
hxcclResult_t hxcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                             hxcclDataType_t datatype, hxcclComm_t comm,
                             hxcclStream_t stream);
hxcclResult_t hxcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                 hxcclDataType_t datatype, hxcclRedOp_t op,
                                 hxcclComm_t comm, hxcclStream_t stream);
hxcclResult_t hxcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                             hxcclDataType_t datatype, int root, hxcclComm_t comm,
                             hxcclStream_t stream);

const char* hxcclGetErrorString(hxcclResult_t result);
const char* hxcclGetLastError(hxcclComm_t comm);
hxcclResult_t hxcclCommGetStats(hxcclComm_t comm, uint64_t* bytes_sent, uint64_t* msgs_sent);
hxcclResult_t hxcclCommResetStats(hxcclComm_t comm);

#ifdef __cplusplus
}
#endif
#endif /* HXCCL_H_ */
