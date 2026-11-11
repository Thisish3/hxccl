// X experiment (spec 7): a device-driven ring AllReduce on one GPU. Each rank maps its right
// neighbor's inbox and flag over CUDA IPC. One kernel per rank runs reduce-scatter then
// all-gather: at step s it writes a chunk into the right rank's inbox slot s, fences, raises
// the right rank's flag to s+1, and spins until its own flag reaches s+1.
// Every step has its own inbox slot, so a fast sender never overwrites unread data.
//
// Build: /usr/local/cuda/bin/nvcc -O2 -arch=sm_86 -std=c++17 experimental/device_allreduce.cu \
//   -o build/device_allreduce -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
// Run:   timeout 60 mpirun --oversubscribe -np 2 build/device_allreduce [count] [blocks]
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                               \
  do {                                                                                      \
    cudaError_t e_ = (x);                                                                   \
    if (e_ != cudaSuccess) {                                                                \
      std::printf("rank %d: %s failed: %s\n", rank, #x, cudaGetErrorString(e_));            \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
    }                                                                                       \
  } while (0)

// %globaltimer: one nanosecond clock for the whole GPU, so times from different processes
// on the same GPU can be compared.
__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ void wait_flag(volatile int* flag, int want, unsigned long long* seen) {
  if (threadIdx.x == 0) {
    while (*flag < want) {
    }
    *seen = now_ns();
  }
  __syncthreads();
}

__device__ void signal(int* peer_flag, int value, unsigned long long* sent) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicExch(peer_flag, value);
    *sent = now_ns();
  }
}

// One block (gridDim.x must be 1: the flags order whole steps, not blocks).
// times[0] = kernel start, then per step: times[1 + 2*step] = signal sent,
// times[2 + 2*step] = left neighbor's signal seen.
__global__ void ring_allreduce(float* data, float* inbox, float* right_inbox, int* my_flag,
                               int* right_flag, int rank, int n, size_t chunk,
                               unsigned long long* times) {
  auto mod = [n](int a) { return ((a % n) + n) % n; };
  if (threadIdx.x == 0) times[0] = now_ns();
  int step = 0;
  for (int s = 0; s < n - 1; ++s, ++step) {  // reduce-scatter
    const float* src = data + mod(rank - s) * chunk;
    float* dst = right_inbox + step * chunk;
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) dst[i] = src[i];
    signal(right_flag, step + 1, &times[1 + 2 * step]);
    wait_flag(my_flag, step + 1, &times[2 + 2 * step]);
    float* acc = data + mod(rank - s - 1) * chunk;
    const float* in = inbox + step * chunk;
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) acc[i] += in[i];
    __syncthreads();
  }
  for (int s = 0; s < n - 1; ++s, ++step) {  // all-gather
    const float* src = data + mod(rank + 1 - s) * chunk;
    float* dst = right_inbox + step * chunk;
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) dst[i] = src[i];
    signal(right_flag, step + 1, &times[1 + 2 * step]);
    wait_flag(my_flag, step + 1, &times[2 + 2 * step]);
    float* out = data + mod(rank - s) * chunk;
    const float* in = inbox + step * chunk;
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) out[i] = in[i];
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  const size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1u << 20);
  const int threads = 256;
  const size_t chunk = count / n;  // count must be a multiple of n
  CK(cudaSetDevice(0));
  float *data, *inbox;
  int* flag;
  const size_t slots = n > 1 ? 2 * size_t(n - 1) : 1;  // a zero-byte buffer has no IPC handle
  const int steps = 2 * (n - 1);
  unsigned long long* times;
  CK(cudaMalloc(&times, (1 + 2 * size_t(steps)) * sizeof(unsigned long long)));
  CK(cudaMalloc(&data, count * sizeof(float)));
  CK(cudaMalloc(&inbox, slots * chunk * sizeof(float)));
  CK(cudaMalloc(&flag, sizeof(int)));
  CK(cudaMemset(flag, 0, sizeof(int)));
  std::vector<float> h(count);
  for (size_t i = 0; i < count; ++i) h[i] = float(rank + 1) + float(i % 7);
  CK(cudaMemcpy(data, h.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  cudaIpcMemHandle_t mine[2], all[2 * 64];
  CK(cudaIpcGetMemHandle(&mine[0], inbox));
  CK(cudaIpcGetMemHandle(&mine[1], flag));
  MPI_Allgather(mine, sizeof mine, MPI_BYTE, all, sizeof mine, MPI_BYTE, MPI_COMM_WORLD);
  const int right = (rank + 1) % n;
  void *right_inbox = inbox, *right_flag = flag;
  if (n > 1) {
    CK(cudaIpcOpenMemHandle(&right_inbox, all[2 * right], cudaIpcMemLazyEnablePeerAccess));
    CK(cudaIpcOpenMemHandle(&right_flag, all[2 * right + 1], cudaIpcMemLazyEnablePeerAccess));
  }
  CK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const double t0 = MPI_Wtime();
  CK(cudaEventRecord(e0));
  ring_allreduce<<<1, threads>>>(data, inbox, static_cast<float*>(right_inbox), flag,
                                 static_cast<int*>(right_flag), rank, n, chunk, times);
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  const double wall = MPI_Wtime() - t0;
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));

  CK(cudaMemcpy(h.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  const float rank_sum = float(n) * float(n + 1) / 2;
  for (size_t i = 0; i < count; ++i) bad += h[i] != rank_sum + float(n) * float(i % 7);
  std::printf("rank %d/%d count=%zu: %s (%zu bad), kernel %.3f ms, wall %.3f ms\n", rank, n, count,
              bad ? "FAIL" : "PASS", bad, ms, wall * 1e3);

  // Step timeline on rank 0: every rank's globaltimer stamps, relative to the earliest start.
  std::vector<unsigned long long> t(1 + 2 * size_t(steps)), all_t(t.size() * n);
  CK(cudaMemcpy(t.data(), times, t.size() * sizeof(t[0]), cudaMemcpyDeviceToHost));
  MPI_Gather(t.data(), int(t.size()), MPI_UNSIGNED_LONG_LONG, all_t.data(), int(t.size()),
             MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  if (rank == 0 && n > 1) {
    unsigned long long base = all_t[0];
    for (int r = 0; r < n; ++r) base = std::min(base, all_t[r * t.size()]);
    for (int r = 0; r < n; ++r) {
      const unsigned long long* tr = &all_t[r * t.size()];
      std::printf("  timeline rank %d: start %+.3f ms", r, (tr[0] - base) / 1e6);
      for (int s = 0; s < steps; ++s)
        std::printf(" | s%d sent %+.3f seen %+.3f", s, (tr[1 + 2 * s] - base) / 1e6,
                    (tr[2 + 2 * s] - base) / 1e6);
      std::printf("\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (n > 1) {
    cudaIpcCloseMemHandle(right_inbox);
    cudaIpcCloseMemHandle(right_flag);
  }
  cudaFree(data);
  cudaFree(inbox);
  cudaFree(flag);
  MPI_Finalize();
}
