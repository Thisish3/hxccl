#include <cmath>
#include <cstdint>
#include <vector>

#include "check.h"
#include "fp.h"
#include "hxccl.h"

namespace {
int rank_, n_;
bool pow2() { return (n_ & (n_ - 1)) == 0; }

void avg_int(hxcclComm_t c) {
  std::vector<int32_t> in(5), got(5);
  for (int i = 0; i < 5; ++i) in[i] = (rank_ - 3) * (i + 1);
  CHECK(hxcclAllReduce(in.data(), got.data(), 5, hxcclInt32, hxcclAvg, c, nullptr) ==
        hxcclSuccess);
  for (int i = 0; i < 5; ++i) {
    int64_t sum = 0;
    for (int r = 0; r < n_; ++r) sum += (r - 3) * (i + 1);
    CHECK(got[i] == static_cast<int32_t>(sum / n_));  // C++ '/' truncates toward zero
  }
}

void avg_float(hxcclComm_t c) {
  std::vector<float> in(9, static_cast<float>(rank_ + 1)), got(9);
  CHECK(hxcclAllReduce(in.data(), got.data(), 9, hxcclFloat32, hxcclAvg, c, nullptr) ==
        hxcclSuccess);
  const double want = (n_ + 1) / 2.0;
  // powers of two make 1/n exact, so every order gives the same bits
  for (float g : got)
    CHECK(pow2() ? g == static_cast<float>(want) : std::fabs(g - want) <= 1e-6 * want);
}

// NCCL multiplies by 1/n before summing: two fp16 60000s average to 60000, not inf
void avg_half_premul(hxcclComm_t c) {
  if (n_ != 2) return;
  uint16_t in = fp::float_to_half(60000.0f), got = 0;
  CHECK(hxcclAllReduce(&in, &got, 1, hxcclFloat16, hxcclAvg, c, nullptr) == hxcclSuccess);
  CHECK(fp::half_to_float(got) == 60000.0f);
}

void premul_host_and_device(hxcclComm_t c) {
  float s = 2.0f;
  hxcclRedOp_t host, dev;
  CHECK(hxcclRedOpCreatePreMulSum(&host, &s, hxcclFloat32, hxcclScalarHostImmediate, c) ==
        hxcclSuccess);
  CHECK(hxcclRedOpCreatePreMulSum(&dev, &s, hxcclFloat32, hxcclScalarDevice, c) ==
        hxcclSuccess);
  CHECK(host >= hxcclNumOps && dev != host);
  s = 5.0f;  // host-immediate copied 2.0 at creation; device reads 5.0 at call time
  std::vector<float> one(3, 1.0f), got(3);
  CHECK(hxcclAllReduce(one.data(), got.data(), 3, hxcclFloat32, host, c, nullptr) ==
        hxcclSuccess);
  for (float g : got) CHECK(g == 2.0f * n_);
  CHECK(hxcclAllReduce(one.data(), got.data(), 3, hxcclFloat32, dev, c, nullptr) ==
        hxcclSuccess);
  for (float g : got) CHECK(g == 5.0f * n_);
  // wrong datatype, destroyed op, builtin op destroy
  CHECK(hxcclAllReduce(one.data(), got.data(), 3, hxcclFloat64, host, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclRedOpDestroy(host, c) == hxcclSuccess);
  CHECK(hxcclAllReduce(one.data(), got.data(), 3, hxcclFloat32, host, c, nullptr) ==
        hxcclInvalidArgument);
  CHECK(hxcclRedOpDestroy(host, c) == hxcclInvalidArgument);
  CHECK(hxcclRedOpDestroy(hxcclSum, c) == hxcclInvalidArgument);
  hxcclRedOp_t again;  // a freed slot is reused
  CHECK(hxcclRedOpCreatePreMulSum(&again, &s, hxcclFloat32, hxcclScalarHostImmediate, c) ==
        hxcclSuccess);
  CHECK(again == host);
  hxcclRedOpDestroy(again, c);
  hxcclRedOpDestroy(dev, c);
}

void premul_int_reducescatter(hxcclComm_t c) {
  int32_t k = 3;
  hxcclRedOp_t op;
  CHECK(hxcclRedOpCreatePreMulSum(&op, &k, hxcclInt32, hxcclScalarHostImmediate, c) ==
        hxcclSuccess);
  std::vector<int32_t> in(2 * n_), got(2);
  for (int i = 0; i < 2 * n_; ++i) in[i] = rank_ + i;
  CHECK(hxcclReduceScatter(in.data(), got.data(), 2, hxcclInt32, op, c, nullptr) ==
        hxcclSuccess);
  for (int j = 0; j < 2; ++j) {
    int sum = 0;
    for (int r = 0; r < n_; ++r) sum += r + rank_ * 2 + j;
    CHECK(got[j] == 3 * sum);
  }
  hxcclRedOpDestroy(op, c);
}

void op_from_other_comm(hxcclComm_t c) {
  hxcclComm_t other;
  CHECK(hxcclCommInitMPI(&other, MPI_COMM_WORLD) == hxcclSuccess);
  float s = 1.0f;
  hxcclRedOp_t op;
  CHECK(hxcclRedOpCreatePreMulSum(&op, &s, hxcclFloat32, hxcclScalarHostImmediate, other) ==
        hxcclSuccess);
  float x = 1.0f, y = 0.0f;
  CHECK(hxcclAllReduce(&x, &y, 1, hxcclFloat32, op, c, nullptr) == hxcclInvalidArgument);
  hxcclRedOpDestroy(op, other);
  hxcclCommDestroy(other);
}
// handles are per comm: index 0 on one comm must not be accepted as index 0 on another
void op_from_other_comm_with_ops(hxcclComm_t c) {
  hxcclComm_t other;
  CHECK(hxcclCommInitMPI(&other, MPI_COMM_WORLD) == hxcclSuccess);
  float mine = 5.0f, theirs = 2.0f;
  hxcclRedOp_t op_c, op_other;
  CHECK(hxcclRedOpCreatePreMulSum(&op_c, &mine, hxcclFloat32, hxcclScalarHostImmediate, c) ==
        hxcclSuccess);
  CHECK(hxcclRedOpCreatePreMulSum(&op_other, &theirs, hxcclFloat32, hxcclScalarHostImmediate,
                                  other) == hxcclSuccess);
  float x = 1.0f, y = 0.0f;
  CHECK(hxcclAllReduce(&x, &y, 1, hxcclFloat32, op_other, c, nullptr) == hxcclInvalidArgument);
  CHECK(hxcclRedOpDestroy(op_other, c) == hxcclInvalidArgument);
  hxcclRedOpDestroy(op_c, c);
  hxcclRedOpDestroy(op_other, other);
  hxcclCommDestroy(other);
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_);
  hxcclComm_t c;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  avg_int(c);
  avg_float(c);
  avg_half_premul(c);
  premul_host_and_device(c);
  premul_int_reducescatter(c);
  op_from_other_comm(c);
  op_from_other_comm_with_ops(c);
  hxcclCommDestroy(c);
  return finish();
}
