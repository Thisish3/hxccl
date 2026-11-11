# hxccl E(1차) 소켓 부트스트랩 Implementation Plan

**Spec:** `docs/specs/2026-11-11-hxccl-e-socket-design.md`. 브랜치 `e-socket`. 계획과 다른 결정은 `docs/LEDGER.md`에 `E<n>`으로 적는다.

### Task E0: `Net` 인터페이스와 `MpiNet` (동작 변화 없음)
- Create `src/net.h`, `src/net_mpi.cc`. `hxcclComm`에 `std::shared_ptr<Net> net`.
- `IpcTransport`·`DeviceOps`가 `MPI_Comm` 대신 `Net&`를 쓴다. AllToAllv 합의는 `net->allreduce_max`.
- 완료: CPU 78, CUDA 108 그대로 통과. Commit "Route control traffic through a Net interface".

### Task E1: `SockMesh`/`SockNet`, `NetTransport`, 부트스트랩
- Create `src/net_sock.h`, `src/net_sock.cc`, `src/transport_net.cc`.
- `hxcclGetUniqueId`/`InitRank`를 소켓으로. Split/Shrink를 소켓 통신기에서. `HXCCL_BOOTSTRAP=socket`이면 InitMPI도 소켓.
- CMake: 모든 MPI 테스트와 GPU 테스트에 `_sock` 변형(`-x HXCCL_BOOTSTRAP=socket`).
- `test_errors`의 중복 rank 기대값 변경.
- 완료: CPU 156, CUDA 전부 통과. Commit "Bootstrap communicators over TCP sockets".

### Task E2: PyTorch를 MPI 없이
- `backend.cpp`의 import 시 `MPI_Init` 제거, finalize는 MPI가 초기화됐을 때만.
- 완료: torchrun np2·4, mpirun np2·4에서 pytest 통과. Commit "Run the torch backend under torchrun".

### Task E3: 측정과 문서
- `hxccl_perf` host np2·4: MPI 통신기와 `HXCCL_BOOTSTRAP=socket`. torch bench를 torchrun으로.
- README(상태, 사용법, 결과), KNOWN_ISSUES(#19 해결, 새 제약). Commit "Measure the socket transport and document torchrun".
