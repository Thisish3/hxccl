#pragma once
#include <memory>
#include <string>
#include <vector>

#include "hxccl.h"
#include "transport.h"

// A PreMulSum op made by hxcclRedOpCreatePreMulSum.
// Handle = hxcclNumOps + (comm op_tag << 16) + index, so another comm's handle is rejected.
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
  int op_tag = 0;  // 1..0x7ffe, assigned per communicator in this process
  bool finalized = false;
  hxcclResult_t async_error = hxcclSuccess;  // last transport-level failure
};

hxcclResult_t comm_fail(hxcclComm* c, hxcclResult_t r, const std::string& msg);

constexpr int kOpIndexBits = 16;
// Returns the user_ops index of `op` on `c`, or -1 when it is not a live op of `c`.
long user_op_index(const hxcclComm* c, hxcclRedOp_t op);
