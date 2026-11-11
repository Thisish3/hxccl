#pragma once
// Thin CUDA/HIP portability layer. HX_HD marks code shared by host and device.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HX_HD __host__ __device__
#else
#define HX_HD
#endif
#ifdef HXCCL_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
#define gpuSuccess cudaSuccess
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuGetErrorString cudaGetErrorString
#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuIpcCloseMemHandle cudaIpcCloseMemHandle
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#elif defined(HXCCL_HIP)  // not built or run in B; kept so the names line up
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
#define gpuSuccess hipSuccess
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuGetErrorString hipGetErrorString
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcCloseMemHandle hipIpcCloseMemHandle
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#endif
