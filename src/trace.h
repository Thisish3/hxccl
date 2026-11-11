#pragma once
// NVTX ranges for Nsight Systems (spec 6). Empty in CPU builds.
#ifdef HXCCL_CUDA
#include <nvtx3/nvToolsExt.h>
#define HX_RANGE_PUSH(name) nvtxRangePushA(name)
#define HX_RANGE_POP() nvtxRangePop()
#else
#define HX_RANGE_PUSH(name) ((void)0)
#define HX_RANGE_POP() ((void)0)
#endif
