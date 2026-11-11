// rank 0 allocates a device buffer, fills it with 42, shares the IPC handle over MPI;
// rank 1 opens it, copies it into its own buffer and checks the values.
#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::printf("rank %d: %s failed: %s\n", rank, #x, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CK(cudaSetDevice(0));
  const size_t n = 1 << 20;
  float* buf; CK(cudaMalloc(&buf, n * sizeof(float)));
  cudaIpcMemHandle_t h{};
  if (rank == 0) {
    std::vector<float> host(n, 42.0f);
    CK(cudaMemcpy(buf, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaIpcGetMemHandle(&h, buf));
  }
  MPI_Bcast(&h, sizeof(h), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank == 1) {
    void* peer; CK(cudaIpcOpenMemHandle(&peer, h, cudaIpcMemLazyEnablePeerAccess));
    CK(cudaMemcpy(buf, peer, n * sizeof(float), cudaMemcpyDeviceToDevice));
    std::vector<float> host(n);
    CK(cudaMemcpy(host.data(), buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t bad = 0; for (float v : host) bad += v != 42.0f;
    std::printf("ipc_probe: %s (%zu bad)\n", bad ? "FAIL" : "PASS", bad);
    CK(cudaIpcCloseMemHandle(peer));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  CK(cudaFree(buf));
  MPI_Finalize();
}
