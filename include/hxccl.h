/* hxccl: a small NCCL-style collective library (CPU buffers over MPI). */
#ifndef HXCCL_H_
#define HXCCL_H_

#include <mpi.h>
#include <limits.h>
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
  hxcclUnhandledCudaError = 1,
  hxcclSystemError = 2,
  hxcclInternalError = 3,
  hxcclInvalidArgument = 4,
  hxcclInvalidUsage = 5,
  hxcclRemoteError = 6,
  hxcclInProgress = 7,
  hxcclTimeout = 8,
  hxcclNumResults = 9
} hxcclResult_t;

/* Same numbering as nccl.h v2.32.3. */
typedef enum {
  hxcclInt8 = 0, hxcclChar = 0,
  hxcclUint8 = 1,
  hxcclInt32 = 2, hxcclInt = 2,
  hxcclUint32 = 3,
  hxcclInt64 = 4,
  hxcclUint64 = 5,
  hxcclFloat16 = 6, hxcclHalf = 6,
  hxcclFloat32 = 7, hxcclFloat = 7,
  hxcclFloat64 = 8, hxcclDouble = 8,
  hxcclBfloat16 = 9,
  hxcclFloat8e4m3 = 10,
  hxcclFloat8e5m2 = 11,
  hxcclNumTypes = 12
} hxcclDataType_t;

typedef enum {
  hxcclSum = 0,
  hxcclProd = 1,
  hxcclMax = 2,
  hxcclMin = 3,
  hxcclAvg = 4,
  hxcclNumOps = 5, /* user-created ops are hxcclNumOps + index */
  hxcclMaxRedOp = 0x7fffffff
} hxcclRedOp_t;

typedef enum { hxcclScalarDevice = 0, hxcclScalarHostImmediate = 1 } hxcclScalarResidence_t;

#define HXCCL_SPLIT_NOCOLOR -1
#define HXCCL_VERSION(X, Y, Z) \
  (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))
#define HXCCL_VERSION_CODE HXCCL_VERSION(0, 2, 0)

#define HXCCL_CONFIG_UNDEF_INT INT_MIN
#define HXCCL_CONFIG_MAGIC 0x6878636cu
typedef struct {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int blocking; /* only 1 (or unset) in A2; non-blocking arrives in B */
  const char* commName;
} hxcclConfig_t;
#define HXCCL_CONFIG_INITIALIZER \
  {sizeof(hxcclConfig_t), HXCCL_CONFIG_MAGIC, HXCCL_VERSION_CODE, HXCCL_CONFIG_UNDEF_INT, NULL}

hxcclResult_t hxcclGetVersion(int* version);
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

/* Rooted collectives: non-root recvbuff (Reduce, Gather) and sendbuff (Scatter) may be NULL. */
hxcclResult_t hxcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                          hxcclDataType_t datatype, hxcclRedOp_t op, int root, hxcclComm_t comm,
                          hxcclStream_t stream);
hxcclResult_t hxcclBcast(void* buff, size_t count, hxcclDataType_t datatype, int root,
                         hxcclComm_t comm, hxcclStream_t stream);
hxcclResult_t hxcclGather(const void* sendbuff, void* recvbuff, size_t count,
                          hxcclDataType_t datatype, int root, hxcclComm_t comm,
                          hxcclStream_t stream);
hxcclResult_t hxcclScatter(const void* sendbuff, void* recvbuff, size_t count,
                           hxcclDataType_t datatype, int root, hxcclComm_t comm,
                           hxcclStream_t stream);

/* Send/Recv block until done; pairs that must progress together go in a group. */
hxcclResult_t hxcclSend(const void* sendbuff, size_t count, hxcclDataType_t datatype, int peer,
                        hxcclComm_t comm, hxcclStream_t stream);
hxcclResult_t hxcclRecv(void* recvbuff, size_t count, hxcclDataType_t datatype, int peer,
                        hxcclComm_t comm, hxcclStream_t stream);
/* Calls between GroupStart and the outermost GroupEnd run at that GroupEnd. */
hxcclResult_t hxcclGroupStart(void);
hxcclResult_t hxcclGroupEnd(void);

/* PreMulSum: sum of (input * scalar); usable only with `comm` and `datatype`. */
hxcclResult_t hxcclRedOpCreatePreMulSum(hxcclRedOp_t* op, void* scalar,
                                        hxcclDataType_t datatype,
                                        hxcclScalarResidence_t residence, hxcclComm_t comm);
hxcclResult_t hxcclRedOpDestroy(hxcclRedOp_t op, hxcclComm_t comm);

const char* hxcclGetErrorString(hxcclResult_t result);
const char* hxcclGetLastError(hxcclComm_t comm);
hxcclResult_t hxcclCommGetStats(hxcclComm_t comm, uint64_t* bytes_sent, uint64_t* msgs_sent);
hxcclResult_t hxcclCommResetStats(hxcclComm_t comm);

#ifdef __cplusplus
}
#endif
#endif /* HXCCL_H_ */
