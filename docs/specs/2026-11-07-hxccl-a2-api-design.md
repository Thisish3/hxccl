# hxccl A2: NCCL/RCCL API 전체 (CPU 버퍼)

- 작성일: 2026-11-07
- 상태: 사용자 리뷰 대기
- 선행: A(코어, main 7401794). 순서는 사용자 합의대로 A2 → C(PyTorch) → B(GPU) → E(Tree/PAT·소켓·토폴로지·튜너) → D(MI300X 비교).

## 1. 목표

NCCL v2.32.3과 RCCL rocm-7.2.4의 공개 API 중 **CPU 버퍼와 MPI만으로 의미가 성립하는 것**을 모두 구현한다. GPU 메모리·스트림·네트워크 부트스트랩이 있어야 의미가 생기는 API는 해당 단계로 보내고, 어느 단계인지 3절 표에 남긴다.

## 2. 확인한 사실 (2026-11-07, 원본 헤더·소스)

- `nccl.h.in`(v2.32.3-1): 결과 코드 `Success 0, UnhandledCudaError 1, SystemError 2, InternalError 3, InvalidArgument 4, InvalidUsage 5, RemoteError 6, InProgress 7, Timeout 8`.
- dtype: `Int8 0, Uint8 1, Int32 2, Uint32 3, Int64 4, Uint64 5, Float16 6, Float32 7, Float64 8, Bfloat16 9, Float8e4m3 10, Float8e5m2 11`(별칭 `Char`, `Int`, `Half`, `Float`, `Double`).
- op: `Sum 0, Prod 1, Max 2, Min 3, Avg 4, NumOps 5`. 사용자 op(`ncclRedOpCreatePreMulSum`)는 `NumOps + 인덱스`를 받는다.
- `ncclAvg` 구현(`src/enqueue/enqueue.cc`): 정수형은 **합한 뒤 nranks로 나누고**(0 방향 버림, `reduce_kernel.h` `FuncSumPostDiv`), 실수형은 **각 입력에 `1/nranks`(그 dtype으로 변환한 값)를 곱한 뒤 합한다**(PreMulSum).
- `ncclRedOpCreatePreMulSum(op*, scalar*, dtype, residence, comm)`: `ncclScalarHostImmediate`는 호출 시점에 값을 복사, `ncclScalarDevice`는 포인터를 저장해 실행 시 읽는다.
- fp16/bf16 리덕션: float로 바꿔 계산 후 다시 변환(`__float2half_rn`, `__float2bfloat16`, 둘 다 round-to-nearest-even). fp8 리덕션: half로 바꿔 계산 후 fp8로 변환.
- 시그니처: `ncclReduce(send, recv, count, dt, op, root, comm, stream)`, `ncclBcast(buff, count, dt, root, comm, stream)`, `ncclAlltoAll(send, recv, count, dt, comm, stream)`(rank j로 보낼 데이터는 `send + j*count`), `ncclGather(send, recv, count, dt, root, comm, stream)`(in-place: `send == recv + root*count`), `ncclScatter(send, recv, count, dt, root, comm, stream)`(in-place: `recv == send + root*count`), `ncclSend(send, count, dt, peer, comm, stream)`, `ncclRecv(recv, count, dt, peer, comm, stream)`, `ncclCommSplit(comm, color, key, newcomm*, config*)`(`NCCL_SPLIT_NOCOLOR = -1`), `ncclCommShrink(comm, excludeRanksList, excludeRanksCount, newcomm*, config*, shrinkFlags)`.
- Send/Recv 문서: 여러 Send/Recv가 동시에 진행돼야 하면 `ncclGroupStart/End`로 묶어야 한다. Group 안의 호출은 하나의 연산으로 합쳐지고 `GroupEnd`에서 시작된다.
- RCCL 추가: `ncclAllToAll`(대문자 T, NCCL `ncclAlltoAll`과 같은 의미), `ncclAllToAllv(send, sendcounts[], sdispls[], recv, recvcounts[], rdispls[], dt, comm, stream)`, `ncclAllReduceWithBias(send, recv, count, dt, op, comm, stream, acc)`.

## 3. 범위

| API | 단계 | 이유 |
|---|---|---|
| Reduce, Bcast, AlltoAll / AllToAll, AllToAllv, Gather, Scatter | **A2** | CPU에서 의미 그대로 |
| Send, Recv, GroupStart, GroupEnd | **A2** | MPI 비차단 송수신으로 구현 가능 |
| dtype 12종, Avg, RedOpCreatePreMulSum, RedOpDestroy | **A2** | 소프트웨어 변환으로 가능 |
| CommSplit, CommShrink, CommAbort, CommFinalize, CommGetAsyncError, CommInitRankConfig, GetVersion | **A2** | MPI communicator 연산으로 가능 |
| AllReduceWithBias | **A2** | 단, `acc` 의미를 구현 전에 RCCL `src/collectives.cc`와 커널에서 확인한다. 확인 못 하면 A2에서 빼고 이유를 남긴다 |
| MemAlloc/Free, CommRegister/Deregister, Window*, WinGetUserPtr, CommCuDevice, CommMemStats, CommInitAll(한 프로세스 여러 GPU), PutSignal/Signal/WaitSignal, 비차단 communicator(`blocking=0`, InProgress), Suspend/Resume | B | GPU 메모리·스트림이 있어야 의미가 있다 |
| CommGrow, CommInitRankScalable, CommGetUniqueId, CommRevoke, GroupSimulateEnd, Param*, 컬렉티브 `*Config` 변형, Timeout, SetEncryption, RCCL CommDump | E | 소켓 부트스트랩·튜너 모델이 있어야 의미가 있다 |

## 4. API 변경

- 결과 코드와 dtype 숫자를 **NCCL과 같게 바꾼다**(A의 `InvalidArgument 1`, `Float32 1` 등은 깨진다. 아직 외부 사용자가 없고, C단계 PyTorch 백엔드가 NCCL 숫자를 그대로 넘길 수 있게 된다). `hxcclUnhandledCudaError`, `hxcclRemoteError`, `hxcclInProgress`, `hxcclTimeout`은 값만 정의한다.
- 새 함수는 2절 시그니처에서 `nccl` → `hxccl`, `cudaStream_t` → `hxcclStream_t`.
- `hxcclConfig_t`: `size, magic, version, blocking, commName`만 둔다. `HXCCL_CONFIG_INITIALIZER`. A2는 `blocking == 1`(또는 미설정)만 받고 `0`이면 `hxcclInvalidArgument`("non-blocking communicators arrive in B").
- `hxcclGetVersion`: `0.2.0` → `200`(NCCL 방식 `major*10000 + minor*100 + patch`). NCCL 버전 공식은 구현 전 `NCCL_VERSION` 매크로로 확인한다.

## 5. 설계

### 5.1 Transport

`Transport`에 비차단 연산을 더한다.

```cpp
virtual hxcclResult_t isend(int to, const void* buf, size_t bytes, int tag) = 0;
virtual hxcclResult_t irecv(int from, void* buf, size_t bytes, int tag) = 0;
virtual hxcclResult_t waitall() = 0;   // 지금까지 걸어 둔 isend/irecv 전부
```

- 태그: collective는 0(A와 같음), Send/Recv는 1. MPI는 (source, tag, comm)마다 순서를 지키므로 같은 peer 사이의 Send/Recv는 호출 순서대로 짝이 맞는다(NCCL과 같은 규칙).
- 통계(`bytes_sent`, `msgs_sent`)는 isend에서도 센다.

### 5.2 Group

- 스레드 지역 깊이 카운터. `GroupStart`는 +1, `GroupEnd`는 −1이고 0이 될 때 실행한다(중첩 허용, NCCL과 같음).
- 그룹 안의 collective는 큐에 쌓았다가 `GroupEnd`에서 **호출 순서대로** 실행하고, 그다음 그룹 안의 Send/Recv를 전부 `isend/irecv`로 걸고 communicator마다 `waitall`한다. 서로 상대에게 Send→Recv 하는 두 rank도 멈추지 않는다.
- 깊이 0에서 `GroupEnd` → `hxcclInvalidUsage`. 그룹 안에서 인자 오류가 나면 그 호출이 즉시 오류를 돌려주고 큐에 넣지 않는다. `GroupEnd`에서 실행 중 오류가 나면 남은 작업을 버리고 첫 오류를 돌려준다.
- 그룹 밖 Send/Recv는 `MPI_Send/MPI_Recv`(차단)로 바로 실행한다. NCCL처럼 "동시에 진행돼야 하면 그룹으로 묶어야 한다"를 문서에 적는다.

### 5.3 알고리즘

| 연산 | 알고리즘 | 메시지 크기 상한 검사 |
|---|---|---|
| Reduce | 이진 트리(작은 크기), 512KB 조각 체인(큰 크기). Broadcast의 역방향 | 조각 크기 |
| Gather / Scatter | root와 각 rank의 직접 송수신(그룹 내부 isend/irecv) | count×es |
| AlltoAll / AllToAllv | pairwise: 단계 s=1..N−1에서 (r+s)에게 보내고 (r−s)에게서 받는다 | 가장 큰 블록. AllToAllv는 rank마다 블록이 달라 다른 rank의 크기를 모르므로, **자기 송신 블록 중 하나라도 INT_MAX를 넘으면 실행 전 `MPI_Allreduce`(MAX)로 모든 rank에 알리고 다 같이 거절**한다(A의 멈춤 버그 재발 방지) |
| Send / Recv | 그룹 밖: 차단, 그룹 안: 5.2 | count×es |
| Bcast | Broadcast(send = recv = buff) | A와 같음 |

### 5.4 dtype과 리덕션

- `reduce_into`를 dtype 12종으로 넓힌다. 정수 8종은 그대로, fp16/bf16은 float로 계산 후 RNE 변환, fp8 e4m3/e5m2는 half 값으로 계산 후 변환(2절의 NCCL 방식). 변환 함수는 `src/fp.h`에 순수 함수로 두고 단위 테스트로 고정한다.
- fp8 변환 규칙: OCP FP8(e4m3: bias 7, 최대 448, inf 없음 / e5m2: bias 15, inf 있음), RNE. **범위를 넘는 값의 처리**(포화인지 NaN인지)는 NCCL이 쓰는 `__nv_fp8_e4m3(__half)` 생성자의 모드를 구현 전에 CUDA `cuda_fp8.hpp`에서 확인해 그대로 따른다. 확인 결과를 스펙에 덧붙인다.
- Avg: 정수형은 합 후 0 방향 버림 나눗셈, 실수형은 각 입력에 `(dtype)(1/N)`을 곱하고 합한다(2절). 실수형 결과는 "합 / N"과 비트가 다를 수 있으며, 테스트 기준값도 같은 순서로 계산한다.
- PreMulSum: communicator별 사용자 op 표. `HostImmediate`는 값 복사, `Device`는 포인터 저장 후 collective 호출 시 읽는다(CPU에서는 둘 다 호스트 메모리). 다른 comm·다른 dtype으로 쓰면 `hxcclInvalidArgument`. 파괴된 op 사용도 `hxcclInvalidArgument`.

### 5.5 Communicator

- `CommSplit`: `MPI_Comm_split(color, key)`. `color == HXCCL_SPLIT_NOCOLOR`면 `MPI_UNDEFINED`로 참여하고 `*newcomm = NULL`.
- `CommShrink`: 모든 남는 rank가 같은 제외 목록을 넘긴다는 NCCL 전제를 따른다. `MPI_Group_excl` + `MPI_Comm_create_group`(남는 rank만 호출). 제외된 rank는 호출하지 않는다. 목록에 범위 밖 rank가 있으면 `hxcclInvalidArgument`. `shrinkFlags`는 0만 받는다(장애 rank 제외 모드는 E).
- `CommAbort`: 통신 없이 자원을 정리한다(`MPI_Comm_free` 대신 로컬 해제만, MPI 핸들은 누수 허용하고 문서화). 다른 rank가 멈춰 있어도 반환해야 한다.
- `CommFinalize`: 동기 실행이라 남은 작업이 없다. 상태를 finalized로 바꾸고, 이후 collective는 `hxcclInvalidUsage`. `CommDestroy`는 그대로 가능.
- `CommGetAsyncError`: 마지막 실패 결과(없으면 Success)를 돌려준다.
- `CommInitRankConfig`: `CommInitRank` + 4절 config 검사.

## 6. 오류 처리

A의 규칙을 그대로 따른다(인자 오류는 아무것도 보내기 전에, MPI 오류는 `SystemError`, 메시지 상한은 모든 rank가 같은 판단). 추가:

- root·peer 범위 밖 → `InvalidArgument`.
- AllToAllv의 counts/displs가 null → `InvalidArgument`. 블록이 버퍼 범위를 넘는지는 버퍼 크기를 모르므로 검사하지 않는다(NCCL과 같음).
- Abort/Finalize된 comm 사용 → `InvalidUsage`.

## 7. 테스트

| 대상 | 기준값 |
|---|---|
| Reduce, Gather, Scatter, AlltoAll, AllToAllv | `MPI_Reduce`, `MPI_Gather`, `MPI_Scatter`, `MPI_Alltoall`, `MPI_Alltoallv`. N ∈ {1,2,3,4,5,8}, root ∈ {0, N−1}, count ∈ {0,1,7,4099}, in-place 포함 |
| Send/Recv + Group | 서로에게 Send→Recv 하는 두 rank가 그룹 안에서 끝남(시간 제한으로 멈춤 검출), ring 시프트, 같은 peer 여러 메시지 순서, 그룹 안 collective + p2p 혼합, 중첩 그룹, 깊이 0 GroupEnd |
| dtype 12종 | 정수 8종은 MPI 기준. fp16/bf16/fp8은 MPI가 dtype을 모르므로 **각 rank가 모든 rank의 입력을 재계산해 로컬 기준값**을 만든다(입력이 rank 번호의 결정적 함수). 입력은 그 dtype에서 정확히 표현되는 작은 정수라 순서와 무관 |
| 변환 | fp16/bf16/e4m3/e5m2 ↔ float: 0, ±최소 정규·비정규, 최대값, RNE 경계(반올림 짝수), NaN, inf(e5m2·fp16·bf16) |
| Avg | 정수 음수 버림(−7/2 → −3), 실수 PreMul 순서 |
| PreMulSum | HostImmediate 값 복사 후 원본 변경해도 결과 불변, Device 포인터는 호출 시점 값, 다른 dtype 사용 거절, Destroy 후 사용 거절 |
| Split/Shrink | 짝/홀 color, key 역순, NOCOLOR, 0번과 마지막 rank 제외 |
| Abort | 상대 rank가 다른 collective에서 멈춰 있어도 Abort가 반환 |
| 메시지 상한 | AllToAllv에서 한 rank의 블록만 INT_MAX 초과 → 모든 rank가 `InvalidArgument`로 반환(멈춤 없음) |
| 통신량 | Reduce 트리: rank당 보낸 바이트, AlltoAll: rank당 (N−1)×count×es |

## 8. 벤치

`hxccl_perf -o reduce|alltoall|gather|scatter|sendrecv` 추가. busbw는 nccl-tests 정의를 쓰고, 연산별 식을 구현 전에 `nccl-tests/doc/PERFORMANCE.md`에서 확인해 README에 인용한다.

## 9. 블로그

개발 일지 #2(A2): Group이 왜 필요한지(Send→Recv 교착을 테스트로 재현), fp8/bf16을 소프트웨어로 만들며 본 반올림 규칙, NCCL의 avg가 정수와 실수에서 다르게 구현된 이유를 코드로 보여 준다. 비공개로 올린다.
