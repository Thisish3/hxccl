# hxccl B + C: GPU 버퍼(CUDA IPC)와 PyTorch 백엔드

- 작성일: 2026-11-11
- 상태: 사용자 리뷰 대기
- 선행: A(코어)·A2(CPU 쪽 NCCL/RCCL API 전체), main `9dbdb4b`.
- 순서 변경: A2 스펙은 C → B 순서였다. 이번에는 사용자 합의(2026-11-11)로 **B와 C를 한 번에** 하고, GPU가 주도하는 방식은 실험(X)으로 둔다.

## 1. 목표

1. **B**: 사용자가 GPU 메모리 포인터를 넘기면 hxccl이 CUDA IPC로 GPU 버퍼끼리 데이터를 주고받고, 리덕션도 GPU에서 한다. A·A2의 모든 집합 연산과 Send/Recv·Group이 GPU 버퍼로 동작한다.
2. **C**: `torch.distributed`에 `"hxccl"` 백엔드로 등록해 `dist.init_process_group("hxccl")`로 쓴다.
3. **X(실험)**: GPU가 주도하는 AllReduce 커널 하나를 `experimental/`에 만들어, GPU 한 장에서 동작하는지와 그 이유를 기록한다. 본 라이브러리에는 넣지 않는다.

## 2. 환경과 전제

- 개발·검증: 데스크탑 RTX 3060 Ti 8 GB(sm_86), WSL2 Ubuntu 22.04, CUDA 12.2(`/usr/local/cuda`), PyTorch 2.9.1+cu128, Nsight Systems·Compute. 맥은 CPU 빌드만 확인한다.
- **GPU가 한 장이다.** rank 여러 개(프로세스 여러 개)가 같은 GPU를 나눠 쓴다. 따라서 IPC 복사는 같은 GPU 안의 복사이고, GPU 사이 NVLink/PCIe 대역폭은 측정하지 않는다. 성능 숫자는 "구조가 어디서 시간을 쓰는가"를 보는 용도로만 쓴다.
- 확인해야 할 전제(B0에서 먼저 확인, 실패하면 진행을 멈추고 대안을 논의):
  - WSL2에서 `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`이 같은 GPU의 다른 프로세스 사이에서 동작한다.
- 확인만 하고 결과를 기록할 것:
  - PyTorch NCCL 백엔드가 한 GPU에 rank 여러 개를 허용하는지(측정 비교 대상이 될 수 있는지).
  - WSL2에서 MPS 사용 가능 여부(X 실험에 영향).

## 3. 구조

### 3.1 메모리 위치별 경로

- 호출마다 버퍼의 메모리 위치를 `cudaPointerGetAttributes`로 판정한다.
  - 전부 CPU 메모리: 기존 CPU 경로(A·A2 그대로).
  - 전부 GPU 메모리: GPU 경로(3.2~3.5).
  - 섞여 있음: `hxcclInvalidArgument`, 메시지 "send and recv buffers must both be host or both be device memory".
- 판정에 쓰는 버퍼: 그 호출의 send·recv 포인터(크기 0이면 판정하지 않고 성공).
- CUDA는 CMake 옵션 `HXCCL_CUDA`(기본 OFF)로만 켠다. OFF 빌드는 지금과 동작이 같고, GPU 포인터 판정 코드도 들어가지 않는다.
- `src/gpu.h`: CUDA API를 매크로로 감싼 얇은 이식 헤더(`gpuMalloc` → `cudaMalloc` 등). HIP 쪽 정의도 넣지만 AMD GPU에서 실행하지는 않는다.

### 3.2 `MemOps`: 알고리즘에서 메모리 작업 분리 (B1)

알고리즘(`algo_*.cc`)에 직접 들어 있는 메모리 작업을 인터페이스로 뽑는다.

```cpp
class MemOps {
 public:
  virtual hxcclResult_t copy(void* dst, const void* src, size_t bytes) = 0;
  virtual hxcclResult_t reduce(void* inout, const void* in, size_t count,
                               hxcclDataType_t dt, hxcclRedOp_t op) = 0;
  virtual hxcclResult_t alloc_tmp(size_t bytes, void** p) = 0;   // 호출 끝에 해제
  virtual Transport& transport() = 0;
};
```

- `HostOps`: `memcpy`, 기존 `reduce_into`, `std::vector`, `MpiTransport`.
- `DeviceOps`: `cudaMemcpyAsync`(사용자 stream), 리덕션 커널, `cudaMallocAsync`/`cudaFreeAsync`, `IpcTransport`.
- B1은 **CPU 경로 리팩터링만** 한다. 기존 테스트 78개가 그대로 통과해야 하고, 결과와 통신량(`bytes_sent`, `msgs_sent`)도 바뀌면 안 된다.

### 3.3 리덕션 커널 (B2)

- dtype 12종 × op(sum, prod, max, min, avg, PreMulSum) 전부.
- 계산 규칙은 CPU 경로(`src/reduce.cc`, `src/fp.h`)와 같다: fp16·bf16은 float로 바꿔 계산 후 round-to-nearest-even, fp8은 half로 바꿔 계산 후 변환, 정수 avg는 합 후 0 방향 버림, 실수 avg는 1/n을 곱한 뒤 합.
- **정답 기준: 같은 입력에서 CPU 경로와 비트 단위로 같다.**
- fp8 overflow(KNOWN_ISSUES #4): 같은 입력을 CUDA `__nv_fp8_e4m3`/`__nv_fp8_e5m2` 변환에도 넣어 hxccl 결과와 같은지 기록한다. 다르면 고치지 않고 KNOWN_ISSUES에 결과를 적는다.

### 3.4 `IpcTransport` (B3)

- 통신기 생성 시(GPU 경로가 처음 쓰일 때 지연 초기화):
  - 각 rank가 다른 rank마다 수신용 중간 버퍼(기본 4 MiB, 환경 변수 `HXCCL_IPC_CHUNK`로 변경)를 `cudaMalloc`한다.
  - 핸들(`cudaIpcGetMemHandle`)을 `MPI_Allgather`로 교환하고, 상대 rank의 버퍼를 `cudaIpcOpenMemHandle`로 연다.
  - 통신기 해제 시 `cudaIpcCloseMemHandle`, `cudaFree`.
- 덩어리(chunk) 하나 전송:
  1. 받는 쪽이 MPI로 "중간 버퍼 비었음"(ready)을 보낸다.
  2. 보내는 쪽이 ready를 받으면 상대의 중간 버퍼로 `cudaMemcpyAsync`(device→device)를 하고, stream을 동기화한 뒤 MPI로 "다 씀"(done)을 보낸다.
  3. 받는 쪽은 done을 받으면 중간 버퍼에서 목적지로 복사하거나 리덕션한다.
- 4 MiB보다 크면 덩어리로 나눠 반복한다.
- 동기화를 IPC 이벤트가 아니라 **stream 동기화 + MPI 메시지**로 하는 이유: 단순하고 결과가 확실하다. 느린 것은 B5에서 측정으로 보여 주고, X 실험의 근거로 쓴다.
- `Transport` 인터페이스(`sendrecv`, `isend`/`irecv`, `waitall`)를 그대로 구현해, Group과 비차단 송수신 알고리즘이 바뀌지 않게 한다. ready/done 메시지는 기존 태그와 겹치지 않는 별도 태그를 쓴다.
- 통신량 카운터(`bytes_sent`, `msgs_sent`)는 사용자 데이터 바이트만 센다(ready/done 제외). 그래야 CPU 경로와 같은 통신량 테스트를 그대로 쓸 수 있다.

### 3.5 stream

- GPU 작업은 사용자가 넘긴 `hxcclStream_t`(= `cudaStream_t`)에 넣는다. `NULL`이면 기본 stream.
- 호출은 **작업이 끝날 때까지 CPU가 기다렸다가 반환한다.** NCCL처럼 바로 반환하지 않는다는 점을 README와 헤더 주석에 적는다.

### 3.6 범위

| 항목 | 이번 B·C | 비고 |
|---|---|---|
| AllReduce, AllGather, ReduceScatter, Broadcast/Bcast, Reduce, Gather, Scatter, AlltoAll/AllToAll, AllToAllv, Send, Recv, Group | **B** | GPU 버퍼로 |
| PyTorch `all_reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast`, `send`, `recv`, `barrier` | **C** | |
| A2 스펙 3절에서 "B"로 미룬 MemAlloc/Free, CommRegister, Window*, CommInitAll, PutSignal 등, 비차단 통신기(`blocking=0`) | 이후(B2) | 이번 목표(GPU 버퍼 집합 연산)에 필요하지 않다 |
| 비동기 반환(stream에 넣고 바로 반환) | 이후(X 결과에 따라) | |
| 여러 GPU 사이 대역폭, RCCL·MI300X 비교 | D | GPU 한 장 |
| HIP 실행 | 이후 | 헤더만 |

## 4. C: PyTorch 백엔드

- `python/hxccl_torch/`: C++ 확장(`torch.utils.cpp_extension`), `c10d::Backend`를 상속한 `HxcclBackend`, Python에서 `torch.distributed.Backend.register_backend("hxccl", ...)`로 등록.
- 실행 방식: hxccl의 `hxcclCommInitRank`는 `MPI_COMM_WORLD`를 나눠 통신기를 만든다(`src/comm.cc`). torchrun으로 띄운 프로세스는 하나의 MPI 작업이 아니어서 서로를 찾지 못한다. 그래서 **`mpirun -np N python script.py`로 띄운다.** 실행 도우미(`python/hxccl_torch/launch.py`)가 `OMPI_COMM_WORLD_RANK`/`SIZE`/`LOCAL_RANK`로 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT`를 채운다. torchrun 지원(소켓 부트스트랩)은 E 단계로 남긴다.
- 부트스트랩: 확장 로드 시 MPI가 초기화되지 않았으면 `MPI_Init_thread`를 부른다. 백엔드 생성 시 rank 0이 `hxcclGetUniqueId` 결과를 PyTorch store에 쓰고, 모든 rank가 읽어 `hxcclCommInitRank`를 부른다.
- dtype 매핑: torch dtype → hxccl dtype(지원하지 않는 dtype은 `TypeError`). ReduceOp 매핑: SUM, PRODUCT, MIN, MAX, AVG.
- Work 객체: 호출이 끝난 뒤 반환하므로 이미 완료된 `Work`를 돌려준다(`async_op=True`도 같음).
- 텐서 조건: contiguous, 같은 device. 조건을 어기면 `ValueError`.

## 5. 테스트

| 대상 | 방법 | 정답 기준 |
|---|---|---|
| B1 | 기존 ctest 78개 | 결과·통신량 변화 없음 |
| B2 리덕션 | dtype 12 × op 6, 크기 1, 7, 4097, 1M | CPU 경로와 비트 일치 |
| B3·B4 집합 연산 | 기존 테스트를 GPU 버퍼 버전으로(`mpirun -np 2`, `-np 4`) | 같은 입력의 CPU 경로 결과와 비트 일치, 통신량 같음 |
| 경계 | 크기 0, 4 MiB±1 원소, 64 MiB, rank 수로 나눠떨어지지 않는 개수, in-place, CPU·GPU 섞인 버퍼(거절) | 위와 같음 / 오류 코드 |
| C | pytest를 `mpirun -np 2·4`로 실행 | 같은 입력을 `gloo` 백엔드로 돌린 결과와 같음(실수형은 리덕션 순서가 달라 허용 오차, 정수형은 정확히 일치) |
| CPU 빌드 | 맥에서 `HXCCL_CUDA=OFF` 빌드·테스트 | 지금과 같음 |

## 6. 측정 (B5, C2)

- AllReduce, 크기 4 KiB ~ 256 MiB(2배씩), rank 2·4, warm-up 5 + 측정 20회 중앙값. SM 클럭 1500 MHz 고정과 클럭 로그는 hxgemm과 같은 방식.
- 비교: hxccl CPU 경로(MPI), hxccl GPU 경로(IPC), PyTorch `gloo`, PyTorch NCCL(한 GPU에 여러 rank가 허용될 때만, 아니면 오류 메시지를 기록).
- Nsight Systems: NVTX 구간(`hxccl:<op>`, `ipc:copy`, `reduce`)을 넣고(대기 시간은 `hxccl:<op>`에서 나머지를 뺀 값으로 계산) rank별로 `nsys profile`. 크기 3개(64 KiB, 4 MiB, 64 MiB)에서 시간 구성(복사·대기·리덕션)을 표로 남긴다.
- Nsight Compute: 리덕션 커널(float sum, 64 MiB)의 DRAM 처리량만.
- 결과에 "같은 GPU 안의 복사"라는 한계를 같이 적는다.

## 7. X: GPU 주도 AllReduce 실험

- `experimental/device_allreduce.cu`: 상대 rank 버퍼를 IPC로 매핑하고, 커널 하나가 GPU 메모리 플래그로 서로를 기다리며 ring AllReduce를 한다.
- 목표는 동작 여부와 이유를 기록하는 것이다. 한 GPU에서 여러 프로세스의 커널이 시간을 나눠 쓰면, 서로를 기다리는 커널이 진행하지 못할 수 있다. MPS 유무에 따른 차이도 기록한다.
- 시간 상한: 측정까지 하루. 넘으면 그때까지의 관찰을 기록하고 끝낸다.

## 8. 작업 방식과 순서

- 맥 Claude가 이 스펙과 구현 계획을 쓰고, 데스크탑 Claude가 브랜치 `b-gpu`에서 계획대로 TDD로 구현한다. 계획과 달라진 결정은 `docs/LEDGER.md`, 맥에서 보내는 요청은 `docs/INBOX.md`.
- 순서와 완료 기준:

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| B0 | 환경: OpenMPI 설치, WSL IPC 시험 프로그램, PyTorch NCCL 한 GPU 여러 rank 확인 | IPC 시험 통과(실패 시 중단) |
| B1 | `MemOps` 분리 | 테스트 78개 그대로 통과 |
| B2 | 리덕션 커널 | CPU와 비트 일치 |
| B3 | `IpcTransport` + AllReduce | np 2·4, 4 MiB 초과 포함 CPU와 비트 일치 |
| B4 | 나머지 집합 연산, Send/Recv, Group | 기존 테스트 전부 GPU 버퍼로 통과 |
| B5 | 측정, NVTX, Nsight | 표, 시간 구성, 클럭 로그 |
| C1 | PyTorch 백엔드 (mpirun 실행 도우미 포함) | np 2·4에서 gloo와 일치 |
| C2 | PyTorch 측정 | 표(또는 NCCL이 안 되는 이유) |
| X | GPU 주도 실험 | 결과와 원인 기록 |
| R | 브랜치 전체 리뷰, main 병합 | 지적 반영 |

## 9. 완료 기준 (전체)

- GPU 버퍼로 3.6의 B 항목이 모두 동작하고 CPU 경로와 비트 단위로 같다.
- `dist.init_process_group("hxccl")`로 C 항목이 동작한다.
- 측정 표와 Nsight Systems 시간 구성이 있고, GPU 한 장이라는 한계를 문서에 적었다.
- README 상태가 "A, A2 done"에서 "B, C done"으로 바뀌고, 사용법(빌드 옵션, 호출이 끝날 때 반환한다는 점)이 들어간다.
- 끝난 뒤: 블로그 hxccl 개발 일지 #3(B)·#4(C), 이력서 hxccl 항목을 실제 결과로 수정.

## 환경 기록 (B0, 2026-11-11, 데스크탑)

| 항목 | 값 |
|---|---|
| GPU / 드라이버 | RTX 3060 Ti 8 GB (sm_86) / Windows 드라이버 581.80 |
| OS | WSL2 Ubuntu 22.04 |
| nvcc | 12.2 (V12.2.140, `/usr/local/cuda/bin/nvcc`) |
| 호스트 컴파일러 | g++ 11.4.0 |
| CMake | 3.22.1 |
| OpenMPI | 4.1.2 (apt `libopenmpi-dev openmpi-bin`) |
| PyTorch | 2.9.1+cu128 (`~/venvs/hxgemm`), NCCL 2.27.5 |

- **CUDA IPC (`experimental/ipc_probe.cu`, `mpirun -np 2`)**: `ipc_probe: PASS (0 bad)`. WSL2에서 같은 GPU의 두 프로세스 사이에 `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`이 동작한다. B 진행.
- **PyTorch NCCL, 한 GPU에 rank 2개** (`experimental/torch_nccl_same_gpu.py`, `python -m torch.distributed.run --nproc-per-node 2`): 두 rank 모두 `DistBackendError ... ncclInvalidUsage ... Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000`. NCCL은 한 GPU 여러 rank를 허용하지 않으므로 B5·C2 비교 대상에서 빼고 이 오류를 기록으로 남긴다.
- **MPS**: WSL에 `nvidia-cuda-mps-control`이 없다(`command not found`). WSL2에서 MPS를 쓸 수 있는지는 원인 미확인(바이너리가 없다는 것까지만 확인). X 실험은 MPS 없이 진행한다.
