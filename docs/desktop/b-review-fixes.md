# hxccl B+C 최종 리뷰 반영 지시 (데스크탑)

맥에서 한 일:
- 맥 CPU 빌드 확인: clang/libc++, `HXCCL_CUDA=OFF`, 경고 0, 테스트 78/78 통과. R28 (5)는 해결됐다.
- `main..b-gpu`(`496f76c`) 독립 리뷰. CUDA 코드는 읽기로만 확인했다.

리뷰 결과: Critical 0건, Important 5건, Minor 4건. Review Focus 1~5는 읽기로 확인했고 모두 문제없다. 단 Focus 5는 I4 예외가 있다.

## 규칙

- 고치기 전에 GPU 테스트로 먼저 재현하고, 실패를 확인한 뒤 고친다.
- 멈춤(hang)을 재현하는 테스트에는 반드시 시간 제한을 둔다. ctest `TIMEOUT` 또는 `timeout 60 mpirun ...`을 쓴다. R26처럼 GPU가 멈추면 사용자에게 `wsl --shutdown`을 요청한다.
- 재현이 안 되면 "재현 안 됨"으로 기록하고, 원인을 짐작해서 쓰지 않는다.
- 결정은 `docs/LEDGER.md`에 R29부터 적는다. 고치지 않고 미루는 것은 `docs/KNOWN_ISSUES.md`에 적는다.
- 커밋해서 `b-gpu`에 push한다. main merge는 하지 않는다.
- 끝나면 전체 테스트를 다시 돌린다: CPU 78, CUDA 102 이상, PyTorch np2·np4.

## Important (순서대로)

### I4. device P2P 첫 사용 시 일부 rank만 호출하면 멈춤 (가장 실제적)

- 위치: `src/api_p2p.cc` (`ops_for`가 호출 시점에 판정)
- 증상: 첫 device Send/Recv가 통신기 전체의 `MPI_Allgather`(지연 초기화)를 돈다. 그래서 np4에서 rank0 `dist.send(t,1)`, rank1 `dist.recv(t,0)`만 처음 부르면 영원히 멈춘다. 파이프라인 병렬에서 흔한 패턴이다.
- 현재 `gpu_p2p` 테스트는 `ring_shift`가 먼저 모든 rank를 초기화해서 통과하고 있다.
- 재현 테스트: np4에서 rank0·1만 device Send/Recv를 처음 호출. 시간 제한을 둔다.
- 수정 후보:
  - CUDA 빌드에서 GPU가 있으면 통신기 생성 시점에 device transport를 바로 초기화한다(eager). **권장.** 통신기 생성은 모든 rank가 부르는 집합 동작이다.
  - 또는 쌍마다 필요할 때 IPC를 설정한다(복잡하다).
- eager로 하면 host만 쓰는 사용자도 CUDA 컨텍스트를 만든다. 방금 고친 #26과 충돌하므로 둘 다 지키는 방법을 정해 R로 기록한다. 예: `HXCCL_DEVICE=0`이면 eager 초기화를 끈다.

### I2. 지연 초기화 실패 시 다른 rank가 멈추고 서로 어긋남

- 위치: `src/memspace_cuda.cc:36-43`, `src/transport_ipc.cc:57-73`
- 증상 1: 한 rank의 `cudaMalloc`이 실패하면(OOM, 또는 `HXCCL_IPC_CHUNK`에 아주 큰 값·음수를 넣으면 `strtoull`이 받아들임) 그 rank는 Allgather 전에 반환한다. 나머지 rank는 Allgather에서 멈춘다.
- 증상 2: Allgather 뒤 `cudaIpcOpenMemHandle`이 실패하면 그 rank만 다음 호출에서 `init()`을 다시 돈다. 그래서 Allgather 횟수가 rank마다 어긋난다.
- 재현 테스트: 한 rank만 `HXCCL_IPC_CHUNK`를 비정상 값으로 주거나 테스트용 실패 주입을 쓴다. 시간 제한을 둔다.
- 수정:
  - 단계마다 `MPI_Allreduce(MIN)`으로 성공 여부를 모든 rank가 합의한다.
  - 실패하면 통신기에 device 오류 상태를 고정해서(sticky), 이후 device 호출이 재시도 없이 오류를 반환하게 한다.
  - `HXCCL_IPC_CHUNK` 값 범위도 검사한다.

### I1. 두 통신기에 걸친 Group의 device P2P가 서로 기다리며 멈춤(교착)

- 위치: `src/group.cc:55-56`, `src/transport_ipc.cc:195`
- 원인: `run_p2p`가 transport의 `waitall`을 하나씩 돈다.
- 시나리오:
  - rank0: {Send c1→1, Recv c2←1}
  - rank1: {Send c2→0, Recv c1←0}
  - 결과: 서로 상대 통신기의 READY를 기다리며 멈춘다.
- 수정:
  - 사용된 IPC transport 전부를 한 루프에서 진행시킨다.
  - 또는 여러 통신기가 섞인 device Group을 `hxcclInvalidUsage`로 거절하고 KNOWN_ISSUES에 기록한다. 간단한 쪽을 먼저 해도 된다.

### I3. `hxcclScalarDevice`에 실제 device 포인터를 주면 segfault

- 위치: `src/checks.cc:50`, `src/api.cc:98`
- 원인: `plan_op`가 host에서 `std::memcpy(p->scalar, u.ptr, ...)`를 한다. NCCL에서는 device scalar의 PreMulSum이 정상 사용법이다.
- 재현 테스트: `cudaMalloc`한 scalar로 PreMulSum AllReduce를 실행한다.
- 수정: device 포인터면 그 호출의 stream에서 `cudaMemcpyAsync`로 읽거나, 커널이 포인터로 읽게 한다.

### I5. 전송 도중 한 rank에서 CUDA/MPI 오류가 나면 상대가 무한 대기

- 위치: `src/transport_ipc.cc:218-220`, `Scratch`/`cudaMallocAsync` 오류 경로
- 증상: 송신자의 `cudaMemcpyAsync`가 실패하면 `cancel_all()` 후 반환한다. 수신자는 오지 않을 DONE을 CPU 100%로 기다린다(KNOWN #27과 겹침).
- 추가 문제: 다른 스레드에서 `hxcclCommAbort`를 부르면, `waitall`이 deque를 순회하는 중에 그 deque가 지워질 수 있다(use-after-free).
- 수정 최소안:
  - 대기 중인 상대에게 오류 표시가 붙은 DONE/READY를 보내고 반환한다.
  - poll 루프에 abort 플래그나 시간 제한을 둔다.
  - 이번에 다 못 하면 KNOWN_ISSUES에 시나리오와 함께 기록한다.

## Minor (시간 되면, 아니면 KNOWN_ISSUES)

- **M1** `python/hxccl_torch/csrc/backend.cpp:51`: `cur()`가 텐서의 device가 아니라 현재 device의 stream을 쓰고, `CUDAGuard`도 없다. 수정: `c10::cuda::CUDAGuard g(t.device()); getCurrentCUDAStream(t.device().index())`. 단일 GPU에서는 재현되지 않을 수 있으니 수정과 테스트만 한다.
- **M2** `backend.cpp:134-145`:
  - send/recv가 `tag`를 무시하고, 리스트 크기 확인 없이 `ts[0]`을 쓴다.
  - `backend.cpp:175`: `MPI_Init_thread`의 `provided` 값을 확인하지 않는다.
  - 수정: 크기 검사 추가, tag가 0이 아니면 오류, `provided` 확인.
- **M3** `transport_ipc.cc:187-189,203-205`: Group 안에서 자기 자신에게 보내는 Send가 잘리면(truncation) 바로 반환하면서 다른 상대의 작업까지 취소해, 그 상대들이 멈춘다. self가 아닌 경로처럼 끝까지 진행하게 맞춘다.
- **M4** `checks.cc:22`: device 경로는 chunk로 나눠 보내는데도 MPI의 메시지당 `INT_MAX` 제한을 그대로 적용해서, 2 GiB를 넘는 ring chunk를 거절한다. KNOWN_ISSUES에 기록만 해도 된다.

## 리뷰어가 GPU 없이 판단하지 못한 것 (가능하면 확인)

- `CommDestroy`에 barrier가 없다. 상대가 아직 핸들을 열고 있는 동안 `~IpcTransport`가 staging 메모리를 `cudaFree`해도 안전한지 확인한다.
- 8~16 B 제어 메시지의 `MPI_Send`가 OpenMPI에서 항상 eager로 나가는지. 교착이 없다는 근거가 이것에 기대고 있다.

## 끝나면

`b-gpu`에 push하고 맥에 보고한다:
- 항목별: 재현 여부 / 고침 / 미룸
- 새 R번호
- 테스트 결과
- 원인 미확인 항목

그 다음 ttalkkak Stage 5로 넘어간다. 시작 방법:

```bash
git clone https://github.com/Thisish3/ttalkkak.git ~/ttalkkak
```

그리고 `docs/desktop/stage5-desktop.md`를 진행한다.
