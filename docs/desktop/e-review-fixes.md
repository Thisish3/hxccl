# hxccl E 브랜치 리뷰 반영 지시 (데스크탑)

**맥에서 한 일:**
- 맥 빌드 수정: `964440b`. `SOCK_CLOEXEC`/`accept4`는 리눅스 전용이라 분기했고, `std::min` 타입 불일치를 고쳤다. 리눅스 동작은 그대로다. 결과: CPU 156/156, 소켓 변형 반복 2회 통과.
- `496f76c..964440b` 독립 리뷰. GPU 경로는 코드를 읽어서만 확인했다.

**먼저 할 일:** `git pull origin e-socket`

**리뷰 결과:** Critical 0, Important 2. 그리고 Minor 중 하나를 Important로 올렸다(I4 부분 수정).

**리뷰어가 확인하고 문제없다고 본 것:**
- 소켓 framing: 부분 읽기·쓰기, EINTR/EAGAIN, 0바이트 메시지
- 매칭 순서, truncation 처리
- 연결 끊김 시 이미 다 받은 unexpected 메시지 보존
- 부트스트랩: rank 중복과 nranks 불일치 거절, mesh 연결 순서(교착 없음)
- I2 합의 단계, I3, I5의 "연결 닫힘은 ABORT가 아님" 수정
- progress 스레드는 없다

## 규칙 (지난번과 같음)

- 고치기 전에 재현 테스트를 먼저 만들고 실패를 확인한다. 멈춤 테스트에는 반드시 시간 제한(TIMEOUT 60 s)을 둔다.
- 재현이 안 되면 "재현 안 됨"으로 적는다. 원인을 짐작해서 쓰지 않는다.
- `docs/LEDGER.md`에 R번호를 이어서 적는다. 미루는 것은 `docs/KNOWN_ISSUES.md`에 적는다.
- `e-socket`에 커밋하고 push한다. main merge는 하지 않는다.
- 끝나면 전체 테스트를 다시 돌린다: CPU 156, CUDA 258 이상, PyTorch(torchrun·mpirun, np2·np4).

## Important

### E-I1. 취소된 send가 이미 해제된 메모리를 계속 읽는다 (조용한 데이터 손상)

- **위치:** `src/net_sock.cc:142-143`(`SockMesh::cancel`), `src/transport_ipc.cc:513-518`(`cancel_all`), `src/net_mpi.cc`의 `cancel`
- **원인:** `SockMesh::cancel`은 `s.req=-1`만 바꾼다. `SendOp`은 `sendq`에 그대로 남고, `buf`는 여전히 호출자 메모리를 가리킨다.
- **시나리오:**
  1. 다른 rank의 ABORT(`check_abort`)나 group 취소로 `IpcTransport::cancel_all`이 돈다.
  2. `staged_ops_.clear()`가 `o.host`와 리스트 노드 안의 `size_msg`를 해제한다.
  3. 세 번째 rank로 가는 staged send가 대기 중이거나 반쯤 써진 상태였다면, 해제된 heap 메모리를 그대로 보낸다.
  4. 받는 쪽은 그것을 staged 수신에 매칭해서 쓰레기 값을 device로 복사하고 성공으로 끝낸다.
- **같은 문제가 있는 곳:**
  - host `NetTransport::cancel_all`: group이 오류를 반환한 뒤 사용자가 버퍼를 해제할 수 있다.
  - `MpiNet::cancel`: send에 대해 `MPI_Request_free`를 부른다.
- 코드 주석 "nothing may still read a staged host buffer"는 stream만 다루고 Net은 다루지 않는다.
- **수정 후보:**
  - 취소된 send가 `off==0`(아직 한 바이트도 안 나감)이면 큐에서 뺀다. 일부가 이미 나갔으면 남은 바이트를 `SendOp`이 소유한 버퍼로 복사해 끝까지 보낸다. 프레임을 중간에 끊으면 스트림이 어긋나기 때문이다.
  - 또는 `StagedOp`을 그 Net 요청들이 끝날 때까지 살려 둔다.
- **재현 테스트:** 3 rank 이상에서 rank A가 B·C로 staged send를 걸고, 그 사이에 ABORT나 취소가 일어나 버퍼가 해제되게 만든다. C가 받은 값이 보낸 값과 다르거나 ASan에서 heap-use-after-free가 나야 한다. CPU 빌드의 host `NetTransport` 경로로 먼저 재현할 수 있으면 그쪽이 쉽다(소켓 변형, ASan 빌드).

### E-I2. I1 수정: `run_p2p`가 단순 truncation에도 다른 transport를 모두 취소한다

- **위치:** `src/group.cc:66-71`
- **원인:** `IpcTransport::cancel_all`은 `broken_`도 설정하지 않고 ABORT도 보내지 않는다.
- **시나리오:** 하나의 Group 안에 두 수신이 있다.
  - 통신기 A의 device Recv: 버퍼가 작다(치명적이지 않은 truncation).
  - 통신기 B의 device Recv: 정상이다.
  - B의 READY가 이미 나갔다면:
    1. B의 DONE 수신이 취소된다.
    2. 상대는 staging에 복사하고 DONE을 보낸 뒤 성공으로 끝난다.
    3. 그 DONE은 아무도 받지 않은 채 남는다.
    4. 다음 B device Recv가 그 오래된 DONE에 매칭된다. 상대가 새 데이터를 쓰는 중인 staging을 읽게 되고, 그 뒤로 그 쌍은 메시지 하나씩 어긋난다(조용한 손상).
  - READY가 아직 안 나갔다면: 상대가 영원히 기다린다.
- 이 변경 전에는 루프가 전부 끝날 때까지 기다렸기 때문에 두 경우 모두 없었다.
- **수정:**
  - 치명적이지 않은 오류(truncation): 나머지 transport는 끝까지 진행시키고 결과만 모은다.
  - 치명적 오류: `IpcTransport::fail` 경로(ABORT와 sticky `broken_`)로 취소한다.
- **재현 테스트:** np2에서 통신기 두 개를 만들고 위 Group을 실행한다. 이어서 B에서 정상 Send/Recv를 한 번 더 했을 때 값이 맞는지 확인한다.

### E-I3 (Minor에서 올림). I4가 부분만 고쳐졌다

- **위치:** `src/memspace_cuda.cc:83-99`
- **문제:** 통신기를 만들 때 한 rank라도 CUDA 컨텍스트가 없으면 device 경로를 여전히 지연 초기화한다. 그래서 원래 시나리오가 그대로 남는다: 순수 C 프로그램에서 일부 rank만 device P2P를 처음 부르면 멈춘다. KNOWN_ISSUES에도 없다.
- PyTorch 백엔드는 `cudaFree(nullptr)`로 미리 컨텍스트를 만들어서 이 경우를 피한다.
- **선택지:**
  - 이번에 고친다. 예: 지연 초기화가 필요한 첫 device P2P에서 일부 rank만 부르는 경우를 감지하고 오류를 낸다.
  - 또는 KNOWN_ISSUES에 "C API 사용자는 통신기 생성 전에 CUDA 컨텍스트를 만들거나 `HXCCL_DEVICE=1`을 쓸 것"을 시나리오와 함께 적고, README에도 적는다.
- `repro_p2p_first_partial`에 "한 rank가 컨텍스트 없이 생성" 변형을 추가해서 현재 동작(멈춤 또는 오류)을 기록한다.

## Minor (시간 되면 고치고, 아니면 KNOWN_ISSUES)

1. **root의 mesh listen 실패 시 다른 rank가 멈춤**
   - 위치: `net_sock.cc:533` vs `:536-543`
   - 문제: root 자신의 `listen_on`이 실패하면, id용 listener를 `listeners()`에서 빼기 전에 반환한다. 다른 rank는 그 backlog에 들어간 채 `read_all(status)`에서 영원히 기다린다.
   - 수정: 먼저 `root_fd`를 꺼내거나 닫는다.
2. **`Info`에 nonce가 없다**
   - 위치: `net_sock.cc:379`
   - 문제: 포트가 재사용되면 옛 id를 가진 rank를 root가 받아들인다. nonce는 mesh handshake에서만 확인한다.
3. **취소된 수신의 처리가 일관되지 않다**
   - 위치: `net_sock.cc:140-141`
   - 문제: 진행 중인 unexpected 메시지를 claim한 수신을 취소하면, 그 메시지가 큐로 돌아가 다음 수신에 매칭된다. 그런데 posted 수신을 body 중간에 취소하면 메시지를 버린다(`dest_cancelled`).
4. **`hdr.bytes`를 검사 없이 쓴다**
   - 위치: `net_sock.cc:226`
   - 문제: 상대가 보낸 값을 그대로 `resize`에 쓰므로, 깨진 헤더면 `bad_alloc` 예외가 C API 밖으로 나간다.
   - 수정: 상한을 검사하고 `hxcclSystemError`를 반환한다.
5. **`SockMesh`에 lock이 없다**
   - 위치: `net_sock.cc:359-364`
   - 문제: 부모 통신기와 Split/Shrink 자식이 `SockMesh` 하나를 공유한다. NCCL은 통신기마다 다른 스레드를 허용하므로 여기서는 data race다.
   - 조치: README와 KNOWN_ISSUES에 "같은 메시의 통신기들은 한 스레드에서" 제약을 적는다.
6. **staged 수신 truncation 처리가 host 경로와 다르다**
   - 위치: `transport_ipc.cc:276`
   - 문제: 오류 문자열에서 "truncated"를 검색해서 판정하고, truncation이면 device 복사를 아예 건너뛴다. host 경로는 앞 R바이트를 채운다.
   - 수정: 오류 코드나 플래그로 판정하고 동작을 맞춘다.
7. **자식이 남은 부모의 Abort가 상대에게 알리지 않는다**
   - 문제: 자식 통신기가 살아 있는 부모에 `hxcclCommAbort`를 부르면 소켓이 열린 채로 남는다. 이 rank를 기다리는 상대들은 알 수 없다.

## 리뷰어가 판단하지 못한 것 (가능하면 확인)

- GPU에서만 확인 가능한 것:
  - device 결과의 정확성
  - staged 경로 처리량
  - 새로 들어간 `cuInit`/`cuDevicePrimaryCtxGetState` 호출 비용
- 읽지 않은 데이터가 남은 채 close하면 TCP RST로 상대 send 버퍼의 데이터가 버려질 수 있는지(테스트 안 함)

## 끝나면

`e-socket`에 push하고 맥에 보고한다:
- 항목별: 재현 여부 / 고침 / 미룸
- 새 R번호
- 테스트 결과
- 원인 미확인 항목

그 다음 ttalkkak Stage 5로 넘어간다:

```bash
git clone https://github.com/Thisish3/ttalkkak.git ~/ttalkkak
```

그리고 `docs/desktop/stage5-desktop.md`를 진행한다.
