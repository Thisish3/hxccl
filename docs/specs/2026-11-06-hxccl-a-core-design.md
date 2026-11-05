# hxccl A: 코어 (NCCL식 API, OpenMPI transport, CPU 알고리즘)

- 작성일: 2026-11-06
- 상태: 사용자 리뷰 대기
- 전체 프로젝트: NCCL/RCCL을 흉내 낸 collective 통신 라이브러리 **hxccl**를 직접 만들고(A~C), 같은 벤치마크로 OpenMPI·RCCL(가능하면 NCCL)과 비교한다(D).

## 0. 전체 분해 (사용자 합의)

| 순서 | 하위 프로젝트 | 개발 장비 | 결과물 |
|---|---|---|---|
| **A (이 문서)** | NCCL식 C API, MPI 기반 초기화, CPU 버퍼 ring / recursive doubling | Mac(conda OpenMPI) | `hxcclAllReduce` 등이 CPU에서 `MPI_*`와 같은 결과, 통신량 테스트 |
| B | GPU 경로: IPC 핸들 교환, GPU 리덕션 커널, 노드 내 ring over IPC, CUDA/HIP 이식 헤더 | RTX 3060 Ti(여러 프로세스가 GPU 1장 공유) | GPU 버퍼 정확성 |
| C | PyTorch `ProcessGroup` 백엔드(C++ 확장) | Mac(CPU) + 3060 Ti | `torch.distributed.all_reduce`가 hxccl 사용 |
| D | nccl-tests식 비교 벤치: hxccl / OpenMPI / RCCL (NCCL은 장비가 있을 때) | AMD Developer Cloud MI300X 8장, 짧은 세션 | 크기별 busbw 표와 분석 |

결정 근거: GPU 직접 통신(B)까지 하되, MI300X 8장은 제3자 자료 기준 시간당 약 $16이라 AMD 크레딧($100, AMD 공식 글 기준)으로 약 6시간이다. 그래서 개발은 3060 Ti에서 하고 8장 장비는 D의 측정에만 쓴다.

## 1. 목표 (A)

- NCCL과 같은 모양의 C API를 제공한다. 이후 B·C·D가 이 API 위에 올라간다.
- 알고리즘(ring, recursive doubling)과 통신(Transport)을 분리한다. A의 Transport는 OpenMPI.
- 결과가 `MPI_Allreduce` 등과 정확히 같고, 보낸 바이트가 이론식과 정확히 같음을 테스트로 고정한다.
- 같은 프로그램에서 hxccl과 OpenMPI를 크기별로 재는 벤치마크를 만든다.

## 2. 확인한 사실 (2026-11-06)

- NCCL API(`NVIDIA/nccl` `src/nccl.h.in`): `ncclGetUniqueId(ncclUniqueId*)`, `ncclCommInitRank(ncclComm_t*, int nranks, ncclUniqueId, int rank)`, `ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)`, `ncclAllGather(send, recv, sendcount, ...)`, `ncclReduceScatter(send, recv, recvcount, ...)`, `ncclBroadcast(send, recv, count, dtype, root, ...)`, `ncclCommDestroy`.
- nccl-tests `doc/PERFORMANCE.md`: `algbw = S/t`. busbw는 AllReduce `algbw × 2(n−1)/n`, ReduceScatter·AllGather `algbw × (n−1)/n`(S는 전체 배열 크기). Broadcast는 `busbw = algbw`.
- OpenMPI: conda-forge `openmpi` 5.0.11 설치 가능(이 Mac은 brew가 동작하지 않음).
- 기존 저장소 `~/Desktop/allreduce-practice`: 스레드 기반 naive / ring / recursive doubling, 96케이스 정확성, 통신량이 이론식과 일치하는 테스트. 알고리즘 로직을 옮겨 온다(저장소는 수정하지 않음).

## 3. API

```c
typedef struct hxcclComm* hxcclComm_t;
typedef struct { char internal[128]; } hxcclUniqueId;
typedef void* hxcclStream_t;   /* A에서는 무시, B에서 cudaStream_t / hipStream_t */

typedef enum { hxcclSuccess = 0, hxcclInvalidArgument, hxcclSystemError,
               hxcclInternalError, hxcclInvalidUsage } hxcclResult_t;
typedef enum { hxcclInt32, hxcclFloat32, hxcclFloat64 } hxcclDataType_t;   /* f16/bf16은 B */
typedef enum { hxcclSum, hxcclProd, hxcclMax, hxcclMin } hxcclRedOp_t;

hxcclResult_t hxcclGetUniqueId(hxcclUniqueId* id);
hxcclResult_t hxcclCommInitRank(hxcclComm_t* comm, int nranks, hxcclUniqueId id, int rank);
hxcclResult_t hxcclCommInitMPI(hxcclComm_t* comm, MPI_Comm mpi);
hxcclResult_t hxcclCommDestroy(hxcclComm_t comm);
hxcclResult_t hxcclCommCount(hxcclComm_t comm, int* count);
hxcclResult_t hxcclCommUserRank(hxcclComm_t comm, int* rank);

hxcclResult_t hxcclAllReduce(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             hxcclRedOp_t op, hxcclComm_t comm, hxcclStream_t s);
hxcclResult_t hxcclAllGather(const void* send, void* recv, size_t sendcount,
                             hxcclDataType_t dt, hxcclComm_t comm, hxcclStream_t s);
hxcclResult_t hxcclReduceScatter(const void* send, void* recv, size_t recvcount,
                                 hxcclDataType_t dt, hxcclRedOp_t op, hxcclComm_t comm,
                                 hxcclStream_t s);
hxcclResult_t hxcclBroadcast(const void* send, void* recv, size_t count, hxcclDataType_t dt,
                             int root, hxcclComm_t comm, hxcclStream_t s);

const char* hxcclGetErrorString(hxcclResult_t r);
const char* hxcclGetLastError(hxcclComm_t comm);
hxcclResult_t hxcclCommGetStats(hxcclComm_t comm, uint64_t* bytes_sent, uint64_t* msgs_sent);
hxcclResult_t hxcclCommResetStats(hxcclComm_t comm);
```

- 인자 순서와 의미는 NCCL과 같다(AllGather는 rank당 `sendcount`, ReduceScatter는 rank당 `recvcount`).
- in-place 허용: AllReduce `send == recv`, AllGather `send == recv + rank*sendcount*size`, ReduceScatter `recv == send + rank*recvcount*size`(NCCL 문서의 in-place 정의).
- `hxcclCommInitRank`: A에서는 MPI가 초기화돼 있어야 하고, `MPI_COMM_WORLD`에서 같은 `id`를 가진 rank들로 communicator를 만든다(`MPI_Comm_split`, color = id의 해시). `MPI_Comm_split`은 `MPI_COMM_WORLD`의 모든 rank가 같이 호출해야 하는 collective라, A에서는 "WORLD 전체가 같은 순서로 `hxcclCommInitRank`를 부른다"를 전제로 하고 문서에 적는다. 이 경로는 C의 PyTorch 백엔드가 쓴다. `hxcclGetUniqueId`는 무작위 128바이트.
- `hxcclCommInitMPI`: 주어진 `MPI_Comm`을 복제(`MPI_Comm_dup`)해 쓴다.

## 4. 구조

```
include/hxccl.h        공개 C API
src/comm.{h,cc}        hxcclComm: rank, nranks, Transport*, 통계, 마지막 오류
src/transport.h        Transport { send(peer, buf, n); recv(peer, buf, n); sendrecv(...) }
src/transport_mpi.cc   MPI_Sendrecv / MPI_Send / MPI_Recv (태그로 collective 순서 구분)
src/reduce.{h,cc}      dtype × op 원소별 리덕션 (CPU)
src/algo_ring.cc       allreduce = reduce-scatter + all-gather, reducescatter, allgather
src/algo_rd.cc         recursive doubling allreduce (2의 거듭제곱이 아닌 N은 남는 rank를 짝 rank에 접는 표준 방식)
src/algo_bcast.cc      broadcast: 이진 트리(작은 크기) / 파이프라인 체인(큰 크기)
src/select.{h,cc}      알고리즘 선택: 기본 임계값(바이트) + HXCCL_ALGO=ring|rd 강제
src/api.cc             C API 구현, 인자 검사, 오류 변환
tests/                 MPI 테스트 (ctest가 mpirun -np N으로 실행)
bench/hxccl_perf.cc    크기별 hxccl vs MPI 시간·algbw·busbw
CMakeLists.txt
```

- 알고리즘은 `Transport`만 안다. B에서 `transport_ipc.cc`를 추가해도 알고리즘 코드는 그대로다.
- 통계는 Transport가 `send`마다 센다(바이트, 메시지 수). 테스트가 이론식과 대조한다.
- 선택 임계값 기본값: allreduce는 64KB 미만 rd, 이상 ring(allreduce-practice의 M2 실측에서 64~256KB 경계였음). D에서 장비별로 다시 잰다.
- 언어: C++17 구현, C 헤더. 빌드: CMake ≥ 3.20, `find_package(MPI)`.

## 5. 오류 처리

- null comm, rank/nranks 범위 밖, 모르는 dtype/op, count 오버플로(`count * size`가 `size_t`를 넘음) → `hxcclInvalidArgument`, `hxcclGetLastError`에 원인 문자열.
- MPI 호출 실패 → `hxcclSystemError`(MPI 오류 문자열 포함). communicator는 `MPI_ERRORS_RETURN`으로 설정해 MPI가 프로세스를 죽이지 않게 한다.
- `count == 0`은 즉시 성공(통신 없음).

## 6. 테스트

| 대상 | 방법 |
|---|---|
| AllReduce 정확성 | N ∈ {1,2,3,4,5,8} × count ∈ {0,1,3,7,1000,4099} × algo ∈ {ring, rd, auto} × op ∈ {sum,prod,max,min} × dtype ∈ {i32,f32,f64}. 입력은 정수값(곱셈은 ±1, 2 등 작은 값)이라 순서와 무관하게 정확히 비교. 기준은 같은 입력의 `MPI_Allreduce` |
| in-place | AllReduce·AllGather·ReduceScatter의 in-place 호출 |
| AllGather / ReduceScatter / Broadcast | 같은 N·count 조합, 기준 `MPI_Allgather` / `MPI_Reduce_scatter_block` / `MPI_Bcast`(root 0, N−1) |
| 통신량 | ring allreduce: rank당 `2(N−1)/N·S`(나머지 처리 포함한 정확한 식), rd: N이 2의 거듭제곱일 때 log₂N 메시지 × S. `hxcclCommGetStats`로 확인 |
| 오류 | 잘못된 인자별 반환값과 메시지, MPI 실패 주입(잘못된 peer)으로 SystemError |
| 실행 | `ctest`가 각 테스트를 `mpirun --oversubscribe -np N`으로 실행 |

## 7. 벤치마크 (A 범위)

`bench/hxccl_perf -b 8 -e 64M -f 2 -n 20`: 크기를 8B~64MB에서 2배씩, 크기마다 워밍업 후 20회의 중앙값. 출력 열: size, count, hxccl 시간(µs)·algbw·busbw, MPI 시간·algbw·busbw, 선택된 알고리즘. busbw는 2절의 nccl-tests 정의. A에서는 Mac(공유 메모리)과 데스크탑 CPU에서 측정하고, README에 측정 사실과 확인되지 않은 해석을 나눠 적는다.

## 8. 범위 밖 (A)

GPU 버퍼(B), PyTorch(C), 노드 간 최적화, 비동기 실행과 스트림 의미론, 그룹 호출(`ncclGroupStart/End`), f16/bf16.
