#pragma once
#include <memory>
#include <string>
#include <vector>

#include "hxccl.h"
#include "transport.h"

// A PreMulSum op made by hxcclRedOpCreatePreMulSum; its handle is hxcclNumOps + index.
struct UserOp {
  bool live = false;
  hxcclDataType_t dt = hxcclFloat32;
  bool by_ptr = false;  // ScalarDevice: read the scalar when the collective is called
  unsigned char value[8] = {};
  const void* ptr = nullptr;
};

struct hxcclComm {
  int rank = 0;
  int nranks = 1;
  MPI_Comm mpi = MPI_COMM_NULL;
  std::unique_ptr<Transport> tr;
  std::string last_error;
  std::vector<UserOp> user_ops;
  bool finalized = false;
  hxcclResult_t async_error = hxcclSuccess;  // last transport-level failure
};

hxcclResult_t comm_fail(hxcclComm* c, hxcclResult_t r, const std::string& msg);
