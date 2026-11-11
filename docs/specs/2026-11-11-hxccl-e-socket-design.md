# hxccl E(1차): 소켓 부트스트랩과 torchrun 지원

- 작성일: 2026-11-11
- 상태: 데스크탑 Claude가 사용자 요청("torchrun 지원 먼저")으로 작성, 구현 진행
- 선행: B+C(브랜치 `b-gpu`, `496f76c`). 브랜치 `e-socket`은 `b-gpu` 위에서 시작한다.

## 1. 목표

`torchrun --nproc-per-node N train.py`로 띄운 프로세스가 `dist.init_process_group("hxccl")`를 쓸 수 있게 한다. torchrun 프로세스는 하나의 MPI 작업이 아니다. 그래서 hxccl이 MPI 없이 통신기를 만들고 통신할 수 있어야 한다.

## 2. 범위

| 항목 | 이번 | 비고 |
|---|---|---|
| `hxcclGetUniqueId`/`hxcclCommInitRank[Config]`를 TCP 소켓 부트스트랩으로 | O | NCCL과 같은 쓰임새: rank 0이 id를 만들어 다른 rank에 전달 |
| 소켓 통신기의 host 전송(TCP), Split, Shrink, AllToAllv 합의 | O | |
| 소켓 통신기의 GPU 경로(IPC 제어 메시지를 소켓으로) | O | |
| PyTorch 백엔드가 MPI 없이 동작(torchrun), mpirun도 계속 동작 | O | |
| `hxcclCommInitMPI`(MPI 통신기) | 그대로 | MPI 전송 유지 |
| 여러 노드 | 이후 | 기본 주소는 127.0.0.1, `HXCCL_SOCKET_ADDR`로 바꿀 수 있지만 검증은 한 노드만 |
| 부트스트랩 timeout, 실패한 rank 감지 | 이후 | 오지 않는 rank를 기다리면 멈춘다(NCCL도 timeout 전까지 기다림) |

## 3. 구조

### 3.1 `Net`: 통신기의 제어 통신

```cpp
class Net {  // one per communicator
 public:
  virtual int rank() const = 0; virtual int size() const = 0;
  virtual hxcclResult_t isend(int peer, int tag, const void* buf, size_t bytes, int* req) = 0;
  virtual hxcclResult_t irecv(int peer, int tag, void* buf, size_t bytes, int* req) = 0;
  virtual hxcclResult_t test(int req, bool* done) = 0;  // frees req once done
  hxcclResult_t wait(int req);
  virtual void cancel(int req) = 0;
  virtual hxcclResult_t allgather(const void* mine, void* all, size_t bytes) = 0;
  virtual hxcclResult_t allreduce_max(int* v) = 0;
  std::string error;
};
```

- `MpiNet`: `MPI_Comm`을 감싼다(`MPI_Isend`/`Irecv`/`Test`/`Allgather`/`Allreduce`).
- `SockNet`: 아래 3.2의 TCP 메시와 통신기 id를 가진다.
- `IpcTransport`의 READY/DONE과 핸들 교환, AllToAllv의 인자 합의, 소켓 통신기의 Split이 `Net`을 쓴다.

### 3.2 TCP 메시(`SockMesh`)

- **연결:** 통신기를 만들 때 rank 쌍마다 TCP 연결 하나(rank r은 자기보다 작은 rank에 connect, 큰 rank는 accept)를 맺는다. `TCP_NODELAY`, non-blocking이다.
- **메시지:** 헤더 {comm id(64), tag(32), bytes(64)} + 본문. 받는 쪽은 (보낸 rank, comm id, tag)마다 게시 순서대로 짝짓는다. 짝이 없는 메시지는 "예상 밖 메시지" 목록에 복사해 둔다. 순서 보장은 MPI와 같다(같은 보낸 rank, comm, tag 안에서 순서 유지).
- **진행:** 모든 진행은 `poll()` 한 루프(한 스레드, 백그라운드 스레드 없음)로 한다. 보내기 큐와 받기를 함께 진행하므로, 두 rank가 동시에 큰 메시지를 보내도 멈추지 않는다.
- **크기 불일치:** 받는 버퍼보다 큰 메시지는 버퍼만큼 받고 나머지는 버린 뒤 `hxcclSystemError`("message truncated")를 낸다. MPI 경로와 같다.
- **자기 자신:** 자기 자신에게 보내는 메시지는 소켓 없이 복사한다.
- **공유:** Split/Shrink로 만든 하위 통신기는 부모의 메시를 공유하고(`shared_ptr`), comm id만 다르다. 상대가 연결을 닫으면 그 상대가 걸린 요청만 실패한다.

### 3.3 부트스트랩

- **unique id:** `hxcclGetUniqueId`는 리슨 소켓을 열고, id(128바이트)에 magic, IPv4 주소, 포트, 무작위 nonce를 담는다. 주소는 `HXCCL_SOCKET_ADDR`, 없으면 127.0.0.1이다.
- **루트:** id의 nonce가 이 프로세스의 리슨 소켓이면 이 프로세스가 루트다. id를 만든 프로세스는 반드시 참여해야 한다(NCCL과 같음).
- **등록:** 각 rank는 자기 메시 리슨 소켓을 연 뒤 루트에 {rank, nranks, 메시 포트}를 보낸다. 루트는 nranks개를 모아 검사한다. rank가 겹치거나 빠졌거나 nranks가 다르면 모두에게 실패를 보내고, 모든 rank가 `hxcclInvalidUsage`를 반환한다. 이상이 없으면 주소표를 돌려준다.
- **메시:** 주소표로 3.2의 메시를 만든다. 통신기 id는 nonce에서 나온다.

### 3.4 통신기

- `hxcclComm`에 `std::shared_ptr<Net> net`을 더한다. MPI 통신기는 `mpi`, `MpiTransport`, `MpiNet`을 쓰고, 소켓 통신기는 `mpi = MPI_COMM_NULL`, `NetTransport`(Transport를 Net 위에 구현), `SockNet`을 쓴다.
- **Split(소켓):** 부모 Net으로 (color, key)를 allgather해 구성원을 정한다. 자식 id는 (부모 id, 부모의 Split·Shrink 횟수, color)에서 만든다.
- **Shrink(소켓):** 남는 rank만 부르므로 통신 없이, (부모 id, 횟수, 제외 목록)에서 id를 만든다.
- **InitMPI:** `HXCCL_BOOTSTRAP=socket`이면 주어진 MPI 통신기로 id만 나눠 갖고 소켓 통신기를 만든다. 기존 테스트 전부를 소켓 경로로 돌리기 위한 것이고, MPI 사용자가 소켓 전송을 고를 수도 있다.
- **Abort:** 그 프로세스의 메시 참조를 놓는다(마지막 참조면 소켓을 닫는다). 상대와 통신하지 않는다.

### 3.5 PyTorch

- 확장 모듈이 import 때 `MPI_Init`을 부르지 않는다. `atexit`의 finalize는 MPI가 초기화돼 있을 때만 부른다.
- 백엔드는 이미 store로 unique id를 나누고 `hxcclCommInitRank`를 부르므로, torchrun과 mpirun 둘 다 그대로 동작한다.

## 4. 테스트

| 대상 | 방법 | 기준 |
|---|---|---|
| 기존 CPU 테스트 78개 | 같은 테스트를 `HXCCL_BOOTSTRAP=socket`으로 한 번 더(156개) | 전부 통과(MPI 경로와 같은 비트, 같은 통신량) |
| GPU 테스트 | 같은 방식으로 소켓 변형 추가 | 전부 통과 |
| InitRank 오류 | `test_errors`의 "모두가 rank 0이라고 주장" | 모든 rank가 `hxcclInvalidUsage`. MPI 시절에는 rank 0만 성공했는데, 소켓 부트스트랩에서는 겹친 rank로 통신기를 만들 수 없다. 테스트 기대값을 바꾸고 LEDGER에 적는다. |
| PyTorch | `python -m torch.distributed.run --nproc-per-node 2·4 -m pytest python/tests`와 기존 mpirun | 둘 다 통과 |

## 5. 측정

- 클럭 고정 상태에서 `hxccl_perf`(host 버퍼) np2·4를 MPI 통신기와 소켓 통신기로 한 번씩 재서, TCP loopback이 MPI 공유 메모리보다 얼마나 느린지 기록한다.
- PyTorch `bench_allreduce.py`를 torchrun으로 다시 재서 mpirun 결과와 비교한다.
