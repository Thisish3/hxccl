// torch.distributed backend over the hxccl C API. Every call finishes before it returns,
// so each Work is already complete.
#include <ATen/cuda/CUDAContext.h>
#include <mpi.h>
#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/python.h>

#include <cstdlib>
#include <cstring>

#include "hxccl.h"

namespace {
using namespace c10d;

hxcclDataType_t to_dt(at::ScalarType t) {
  switch (t) {
    case at::kChar: return hxcclInt8;
    case at::kByte: return hxcclUint8;
    case at::kInt: return hxcclInt32;
    case at::kLong: return hxcclInt64;
    case at::kHalf: return hxcclFloat16;
    case at::kFloat: return hxcclFloat32;
    case at::kDouble: return hxcclFloat64;
    case at::kBFloat16: return hxcclBfloat16;
    case at::kFloat8_e4m3fn: return hxcclFloat8e4m3;
    case at::kFloat8_e5m2: return hxcclFloat8e5m2;
    default: TORCH_CHECK_TYPE(false, "hxccl: unsupported dtype ", t);
  }
}
hxcclRedOp_t to_op(const ReduceOp& op) {
  switch (op.op_) {
    case ReduceOp::SUM: return hxcclSum;
    case ReduceOp::PRODUCT: return hxcclProd;
    case ReduceOp::MIN: return hxcclMin;
    case ReduceOp::MAX: return hxcclMax;
    case ReduceOp::AVG: return hxcclAvg;
    default: TORCH_CHECK_VALUE(false, "hxccl: unsupported reduce op");
  }
}
void check_tensor(const at::Tensor& t) {
  TORCH_CHECK_VALUE(t.is_contiguous(), "hxccl: tensors must be contiguous");
  TORCH_CHECK_VALUE(t.is_cuda(), "hxccl: tensors must be on a CUDA device");
}
void ok(hxcclResult_t r, hxcclComm_t c, const char* what) {
  TORCH_CHECK(r == hxcclSuccess, "hxccl ", what, " failed (", int(r), "): ",
              c ? hxcclGetLastError(c) : hxcclGetErrorString(r));
}
hxcclStream_t cur() { return at::cuda::getCurrentCUDAStream().stream(); }

class DoneWork : public Work {
 public:
  DoneWork(OpType t, std::vector<at::Tensor> out) : Work(-1, t) {
    fut_ = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
    fut_->markCompleted(c10::IValue(out));
  }
  bool isCompleted() override { return true; }
  bool isSuccess() const override { return true; }
  bool wait(std::chrono::milliseconds) override { return true; }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return fut_; }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> fut_;
};

class HxcclBackend : public Backend {
 public:
  HxcclBackend(const c10::intrusive_ptr<Store>& store, int rank, int size)
      : Backend(rank, size) {
    // A CUDA context on every rank before the communicator exists lets hxccl start its
    // device path now (I4), so a first send/recv between two ranks only cannot hang.
    cudaFree(nullptr);
    hxcclUniqueId id;
    if (rank == 0) {
      ok(hxcclGetUniqueId(&id), nullptr, "GetUniqueId");
      store->set("hxccl_uid",
                 std::vector<uint8_t>((uint8_t*)&id, (uint8_t*)&id + sizeof(id)));
    }
    auto v = store->get("hxccl_uid");
    TORCH_CHECK(v.size() == sizeof(id), "hxccl: bad unique id in the store");
    std::memcpy(&id, v.data(), sizeof(id));
    ok(hxcclCommInitRank(&comm_, size, id, rank), nullptr, "CommInitRank");
  }
  ~HxcclBackend() override {
    if (comm_) hxcclCommDestroy(comm_);  // a socket communicator: no MPI involved
  }
  const std::string getBackendName() const override { return "hxccl"; }

  c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor>& ts,
                                     const AllreduceOptions& o) override {
    for (auto& t : ts) {
      check_tensor(t);
      ok(hxcclAllReduce(t.data_ptr(), t.data_ptr(), t.numel(), to_dt(t.scalar_type()),
                        to_op(o.reduceOp), comm_, cur()),
         comm_, "AllReduce");
    }
    return c10::make_intrusive<DoneWork>(OpType::ALLREDUCE, ts);
  }
  c10::intrusive_ptr<Work> _allgather_base(at::Tensor& out, at::Tensor& in,
                                           const AllgatherOptions&) override {
    check_tensor(in);
    check_tensor(out);
    TORCH_CHECK_VALUE(out.numel() == in.numel() * getSize(),
                      "hxccl: output must hold world_size * input elements");
    ok(hxcclAllGather(in.data_ptr(), out.data_ptr(), in.numel(), to_dt(in.scalar_type()), comm_,
                      cur()),
       comm_, "AllGather");
    return c10::make_intrusive<DoneWork>(OpType::_ALLGATHER_BASE, std::vector<at::Tensor>{out});
  }
  c10::intrusive_ptr<Work> _reduce_scatter_base(at::Tensor& out, at::Tensor& in,
                                                const ReduceScatterOptions& o) override {
    check_tensor(in);
    check_tensor(out);
    TORCH_CHECK_VALUE(in.numel() == out.numel() * getSize(),
                      "hxccl: input must hold world_size * output elements");
    ok(hxcclReduceScatter(in.data_ptr(), out.data_ptr(), out.numel(), to_dt(in.scalar_type()),
                          to_op(o.reduceOp), comm_, cur()),
       comm_, "ReduceScatter");
    return c10::make_intrusive<DoneWork>(OpType::_REDUCE_SCATTER_BASE,
                                         std::vector<at::Tensor>{out});
  }
  c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor>& ts,
                                     const BroadcastOptions& o) override {
    for (auto& t : ts) {
      check_tensor(t);
      ok(hxcclBroadcast(t.data_ptr(), t.data_ptr(), t.numel(), to_dt(t.scalar_type()),
                        int(o.rootRank), comm_, cur()),
         comm_, "Broadcast");
    }
    return c10::make_intrusive<DoneWork>(OpType::BROADCAST, ts);
  }
  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& ts, int dst, int) override {
    check_tensor(ts[0]);
    ok(hxcclSend(ts[0].data_ptr(), ts[0].numel(), to_dt(ts[0].scalar_type()), dst, comm_, cur()),
       comm_, "Send");
    return c10::make_intrusive<DoneWork>(OpType::SEND, ts);
  }
  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& ts, int src, int) override {
    check_tensor(ts[0]);
    ok(hxcclRecv(ts[0].data_ptr(), ts[0].numel(), to_dt(ts[0].scalar_type()), src, comm_, cur()),
       comm_, "Recv");
    return c10::make_intrusive<DoneWork>(OpType::RECV, ts);
  }
  c10::intrusive_ptr<Work> barrier(const BarrierOptions&) override {
    auto t = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    std::vector<at::Tensor> v{t};
    return allreduce(v, AllreduceOptions());
  }

 private:
  hxcclComm_t comm_ = nullptr;
};

c10::intrusive_ptr<Backend> create_backend(const c10::intrusive_ptr<Store>& store, int rank,
                                           int size, const std::chrono::duration<float>&) {
  return c10::make_intrusive<HxcclBackend>(store, rank, size);
}

// If the program itself initialized MPI (for example mpi4py) and has not finalized it, finalize
// it before exit; with no MPI in use this does nothing.
void finalize() {
  int init = 0, fin = 0;
  MPI_Initialized(&init);
  MPI_Finalized(&fin);
  if (init && !fin) MPI_Finalize();
}
}  // namespace

// The backend builds its communicator with hxcclGetUniqueId/InitRank over sockets (spec E),
// so it needs no MPI: it runs under torchrun as well as mpirun.
// Under mpirun (OMPI_COMM_WORLD_SIZE set) it still initializes MPI as before: without that
// the CUDA IPC handles could not be opened and the device path fell back to the slow staged
// path (LEDGER E6, measured 2026-11-11). torchrun processes never touch MPI.
PYBIND11_MODULE(_C, m) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited && std::getenv("OMPI_COMM_WORLD_SIZE")) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  }
  m.def("create_backend", &create_backend);
  m.def("finalize", &finalize);
}
