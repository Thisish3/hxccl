#pragma once
#include <memory>
#include <string>

#include "hxccl.h"
#include "transport.h"

struct hxcclComm {
  int rank = 0;
  int nranks = 1;
  MPI_Comm mpi = MPI_COMM_NULL;
  std::unique_ptr<Transport> tr;
  std::string last_error;
};

hxcclResult_t comm_fail(hxcclComm* c, hxcclResult_t r, const std::string& msg);
