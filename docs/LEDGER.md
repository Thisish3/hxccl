# hxccl B+C 실행 기록 (ledger)

계획(`docs/plans/2026-11-11-hxccl-b-c.md`)과 다르게 한 결정을 `R<n>`으로 적는다.

- **R1** (Task 0, 2026-11-11) 실행 방식: 계획 머리말의 `superpowers:executing-plans` 스킬이 데스크탑 세션에 설치되어 있지 않아, 계획의 Task·Step을 순서대로 직접 따라 실행했다.
- **R2** (Task 0 Step 4) `~/venvs/hxgemm/bin/torchrun`이 없다(venv bin에 torchrun 진입점이 없음). 같은 동작인 `~/venvs/hxgemm/bin/python -m torch.distributed.run --nproc-per-node 2 ...`로 실행했다.
- **R3** (Task 0 Step 5) WSL에 `nvidia-cuda-mps-control`이 없다(`/usr/local/cuda/bin`, `/usr/lib/wsl/lib`에도 없음). 계획의 명령은 "command not found"로 끝났고, 결과를 그대로 환경 기록에 적었다.
- **R4** (Task 0 Step 6) `origin`이 HTTPS URL이라 `git push`가 자격 증명 입력을 기다리며 멈췄다. 이 데스크탑은 GitHub에 등록된 SSH 키를 쓰므로 `origin`을 `git@github.com:Thisish3/hxccl.git`로 바꿨다(로컬 설정만, 저장소 내용 변화 없음).
- **R5** (Task 1 Step 1) 데스크탑 첫 빌드 기준선: OpenMPI 4.1.2, g++ 11.4, `100% tests passed, 0 tests failed out of 78`(27 s).
- **R6** (Task 1 Step 2) `HostOps`가 계획의 `Transport&` 대신 통신기의 소유 포인터 `std::unique_ptr<Transport>&`를 들고 `tr()`에서 역참조한다. 이유: `tests/test_comm.cc`·`tests/test_errors.cc`가 `f->tr = std::make_unique<FailingTransport>()`로 transport를 바꿔 끼우는데, `Transport&`를 들면 해제된 객체를 가리켜 `test_comm` np2~8이 segfault로 실패했다(측정으로 확인). 기존 테스트를 고치지 않고 통과시키려고 이렇게 했다.
- **R7** (Task 1 Step 4) `allreduce_with(c, algo, ...)`는 시그니처를 그대로 두고 안에서 `*c->host_ops`를 쓴다(테스트 `test_allreduce`·`test_dtypes`가 직접 부른다). `MemOps&`를 받는 형태로 바꾸는 것은 계획대로 Task 4 Step 6에서 한다.
- **R8** (Task 1 Step 4) 실패 메시지 선택(`m.error`가 비었으면 `m.tr().error`)을 `checks.h`의 `ops_fail(c, m, r)` 하나로 모았다. `m.error`는 읽은 뒤 지워 다음 호출로 넘어가지 않게 했다. 원래 `api_rooted.cc`·`api_alltoall.cc`에 있던 `fail_if`는 이것으로 대체했다.
- **R9** (Task 2 Step 1) 데스크탑 CMake가 3.22라 `CUDA::nvtx3` 타깃이 없다(3.25에서 추가). CMake가 "links to target CUDA::nvtx3 but the target was not found"로 실패했다. NVTX3는 헤더만 있고 cudart include 경로에 들어 있으므로 `if(TARGET CUDA::nvtx3)`일 때만 링크하게 했다. `.gitignore`에 `build-cuda/`를 추가했다.
- **R10** (Task 2 Step 7) GPU 리덕션이 float32 sum/prod에서 CPU와 달랐다(4097개 중 36개). 원소를 찍어 확인한 원인: 입력에 NaN이 있으면 x86 SSE(CPU 경로)는 첫 피연산자가 NaN이면 그것을, 아니면 둘째를 quiet 비트만 세워 돌려준다(예: a=0x496056dd, b=0x7f9d0578 → 0x7fdd0578). 반면 sm_86 fp32 덧셈·곱셈은 표준 NaN 0x7fffffff를 낸다. float64는 1M개에서 이미 같았다. `elem_apply`에 device 전용(`__CUDA_ARCH__`)으로 CPU 규칙을 적었다(float·double, sum·prod). host 코드는 바뀌지 않는다.
- **R11** (Task 2 Step 7) fp8 max/min이 1M개 중 17·21개에서 달랐다. 원인: +0과 −0 비교. glibc `fmaxf`/`fminf`(x86 `maxss`/`minss`)는 같은 값이면 둘째 피연산자를 돌려주고(예: max(+0,−0) → −0), CUDA `fmaxf`는 −0 < +0으로 본다. `elem_apply_small`의 `fmaxf`/`fminf`를 glibc 동작을 그대로 적은 `hx_fmax`/`hx_fmin`으로 바꿨다(host·device 공용). host 결과가 그대로인지 `experimental/fmax_rule_check.cc`로 확인했다: fp8 두 종류의 모든 쌍 + fp16·bf16 무작위 2500만 쌍씩 + ±0·±NaN·±inf에서 `std::fmax`/`std::fmin`과 비트가 다른 경우 0/50,131,121. 계획에 없던 host 쪽 변경이다.
- **R12** (Task 4 Step 4) `IpcTransport`를 계획 코드에서 세 가지 바꿨다. 기존 테스트 의미(통신량, 오류)를 GPU 경로에서도 지키기 위해서다. (a) 0바이트 메시지를 건너뛰지 않는다. `MpiTransport`는 0바이트 송신도 `msgs_sent`에 세므로(ring에서 count < N일 때), READY/DONE 한 번을 주고받고 센다. (b) READY에 수신 크기 R, DONE에 송신 크기 S와 덩어리 길이를 실어 min(S, R)만 옮긴다. S > R이면 받는 쪽이 `hxcclSystemError`("message truncated")를 낸다. MPI의 `MPI_ERR_TRUNCATE`(`test_p2p` `recv_too_small`)와 같은 결과를 내려는 것이다. (c) 자기 자신에게 보내는 Send/Recv(np=1의 `test_p2p`)는 staging 없이 게시 순서대로 짝지어 device 복사를 한다. 제어 메시지는 `MPI_UINT64_T`다.
- **R13** (Task 4 Step 5·6) `allreduce_with(MemOps&, c, ...)`를 추가하고, 테스트가 부르는 `allreduce_with(c, ...)`는 host 경로 래퍼로 남겼다(R7). `hxcclCommGetStats`/`ResetStats`는 host·device 두 transport의 카운터를 더한다(계획에 없음, `gpu_allreduce`의 통신량 비교에 필요). `memspace.cc`(CPU용 `ops_for`)는 CUDA 빌드에서 FILTER로 뺐다.
- **R14** (Task 4 Step 2) TDD 순서: 실패 테스트를 따로 먼저 돌려 보지 않고 구현과 같이 작성했다. 결과: CUDA 빌드 90개(GPU 12개 포함, np 1·2·4와 `HXCCL_IPC_CHUNK=4096`), CPU 빌드 78개 통과. `cudaMallocAsync`/`cudaFreeAsync`는 WSL2에서 오류 없이 동작했다.
- **R15** (Task 5 Step 2) 아홉 API에 같은 틀(`ops_for` → NVTX 구간 → 본문 → `sync_ops` → `ops_fail`)을 반복하지 않고 `src/run.h`의 `run_on(c, stream, a, b, range, f)` 하나로 묶었다. AllReduce도 이것으로 바꿨다. 경로 판정에 넘기는 포인터: Broadcast non-root는 `send`를 빼고, Reduce·Gather non-root는 `recv`를, Scatter non-root는 `send`를 뺀다(그 rank가 실제로 쓰는 버퍼만).
- **R16** (Task 5 Step 3) Send/Recv의 경로 판정(`ops_for`)은 GroupEnd가 아니라 **호출 시점**에 한다. 그래서 device 경로의 지연 초기화(`MPI_Allgather`)는 각 rank가 처음 device Send/Recv를 부를 때 일어난다. device P2P를 처음 쓸 때는 모든 rank가 하나씩은 불러야 한다(README에 적을 것). 섞인 그룹 거절은 GroupEnd의 `run_p2p`에서 하고, 메시지는 "a group cannot mix host and device Send/Recv buffers"다.
- **R17** (Task 5 Step 4) 계획의 테스트에 몇 가지를 더했다. `gpu_collectives`에는 AllGather·ReduceScatter·AlltoAll in-place, Bcast in-place, ReduceScatter·Reduce Avg를 넣었다. 1<<20 크기에서는 블록 하나가 1<<20/n이 되게 per를 나눴다. `gpu_p2p`에는 그룹 밖 blocking 송수신, 작은 수신 버퍼(truncate), host와 같은 통신량을 넣었다. 결과: CUDA 빌드에서 `gpu_collectives`·`gpu_p2p`가 np 1·2·4와 chunk4k 변형 모두 통과했다(가장 오래 걸린 것은 np4 chunk4k, 30 s).
- **R18** (Task 6 Step 1·2) 측정 도구만 먼저 커밋한다. 측정은 SM 클럭 고정(Windows 관리자 `nvidia-smi -lgc 1500,1500`)을 사용자에게 받은 뒤 한다. `hxccl_perf` 변경 내용: (a) `-d`이면 hxccl 쪽 버퍼를 `cudaMalloc` 복사본으로 쓰고, MPI 열은 "-"로 둔다. apt OpenMPI 4.1.2가 CUDA-aware인지는 확인하지 않았고, 비교 대상(spec 6)도 MPI가 아니라 hxccl host 경로라서 뺐다. (b) 인자 파싱을 값이 없는 플래그도 받게 바꿨다. (c) warm-up을 3회에서 spec 6의 5회로 바꿨다(A 결과와 조건이 다르다). (d) `mem` 열을 추가했다. `measure_b.sh`의 클럭 로그는 `noheader,nounits` CSV로 받게 해 awk 검사가 MHz 숫자를 그대로 비교한다.
- **R19** (Task 6, 관찰) 클럭을 고정하지 않은 동작 확인(np 2, 5회, 측정 아님)에서 device AllReduce가 4 KiB~4 MiB 전 구간에서 약 2.9~3.3 ms로 거의 같았다. device ReduceScatter·Reduce도 약 2.7~3.1 ms였고, Gather·Scatter·Broadcast·AllGather·AlltoAll·SendRecv는 0.24~0.41 ms였다. 리덕션이 들어가는 연산만 느리다. **원인 미확인.** Task 6 Step 3(Nsight Systems)에서 구간을 나눠 확인한다.
  - 갱신(Task 6 Step 3): np2 64 KiB 호출 하나(평균 4.09 ms)를 나눠 보니 `cudaMallocAsync`가 0.80 ms(중앙값 0.31 ms, 호출마다 1회), `ipc:copy` 4회가 1.74 ms(한 번 중앙값 0.14 ms, 그중 `cudaStreamSynchronize` 중앙값 0.13 ms), `reduce` launch가 0.05 ms였다. 나머지 1.5~2.1 ms는 MPI 대기 등이며, MPI 호출은 트레이스되지 않아 더 나누지 못했다. 바닥값의 구성 요소는 측정했지만, 각 요소가 왜 이만큼 걸리는지(풀 재사용이 안 되는 이유, WSL2의 동기화 비용)는 **원인 미확인**이다. 리덕션 커널 자체는 원인이 아니다(1M fp32 기준 29~31 µs).
- **R20** (Task 6 Step 2) 측정 조건: SM 클럭은 1500 MHz로 고정됐고 96개 샘플 모두 1500이다. 메모리 클럭은 고정하지 않았다(`-lgc`는 SM만 고정). host 측정 구간의 샘플 66개가 810 MHz였다. host 경로 결과는 CPU 메모리만 쓰므로 영향이 없을 것으로 보지만 확인하지 않았다. device 구간 샘플은 6800~7000 MHz였다.
- **R21** (Task 6 Step 3) Nsight Systems 2023.2.3(WSL)은 GPU 커널·memcpy 트레이스를 남기지 않았다("does not contain CUDA kernel data"). 그래서 시간 구성은 CPU 쪽 NVTX 구간과 CUDA API 시간으로 만들었다. 계획의 "복사·대기·리덕션" 중 리덕션은 launch 시간만 잡힌다. `-t mpi`는 MPI_Init/Finalize만 기록했다. 계획에 없던 `cuda_api_sum` CSV를 같이 커밋한다. `.nsys-rep`·`.sqlite`는 커밋하지 않는다.
- **R22** (Task 6 Step 4) `/usr/bin/ncu`(apt, 2021.3)는 section 경로 오류로 실패해서 `/usr/local/cuda/bin/ncu`(2026.3.1)를 썼다. 계획 명령(SpeedOfLight)으로 잰 fp32 sum 1M 커널은 299 µs, DRAM 78.7%였다. 같은 실행에서 잰 `dram__cycles_elapsed.avg.per_second`가 0.81 GHz여서, 이 값은 낮아진 메모리 클럭 기준이다(측정으로 확인). 지표만 수집한 1 pass 실행에서는 DRAM 6.79 GHz, 31.5 µs, 386 GB/s(정격 448 GB/s의 86%)였다. 둘 다 결과 파일에 남겼다. 여러 pass replay에서 메모리 클럭이 내려간 이유는 **원인 미확인**이다.
- **R23** (Task 7 Step 4) 계획의 `setup.py`(libraries `mpi`, `cudart`)로 빌드한 `_C.so`가 import 시 `undefined symbol: _ZN3MPI8Datatype4FreeEv`로 실패했다. C++에서 `mpi.h`를 include하면 OpenMPI C++ 바인딩이 같이 들어오기 때문이다. `libmpi_cxx`를 링크 목록에 추가했다.
- **R24** (Task 7 Step 5) 테스트가 모두 통과했는데도 mpirun이 "exited without calling finalize"로 종료 코드 1을 냈다. `_C.finalize()`(MPI가 초기화됐고 아직 finalize하지 않았으면 `MPI_Finalize`)를 추가하고, `hxccl_torch` import 때 `atexit`에 등록했다. `HxcclBackend` 소멸자는 MPI가 이미 finalize됐으면 `hxcclCommDestroy`를 건너뛴다. 백엔드 생성 시 store에서 읽은 unique id 크기도 검사한다. `_allgather_base`/`_reduce_scatter_base`는 크기 관계(world_size배)를 검사해 `ValueError`를 낸다.
- **R25** (Task 7) torch는 venv가 아니라 `~/.local/lib/python3.10/site-packages`(2.9.1+cu128)에 있다. `~/venvs/hxgemm/bin/python`이 이것을 import한다. 결과: `mpirun -np 2·4 ... pytest -q python/tests` 모두 37 passed, 3 skipped(gloo에 정수 AVG가 없어 계획대로 skip). 실패 테스트를 먼저 돌려 보는 단계는 따로 하지 않았다.
- **R26** (Task 9) 실험 프로그램에 계획에 없던 `%globaltimer` 단계별 시각 기록을 넣었다. Nsight Systems가 WSL에서 GPU 트레이스를 남기지 않아서(R21), 두 프로세스의 커널이 동시에 도는지를 이것으로 대신 봤다. n=1에서는 inbox가 0바이트라 `cudaIpcGetMemHandle`이 invalid argument로 실패해서, 버퍼를 최소 1 덩어리로 잡게 고쳤다. 결과와 해석은 `experimental/README.md`에 있다. np4 두 번째 실행이 멈춘 뒤 GPU가 WSL·Windows 모두에서 응답하지 않았고(프로세스가 D 상태), 멈춘 **원인은 미확인**이다. `wsl --shutdown`은 자동 실행 권한이 거부되어 사용자에게 요청했다. 시간 상한(측정까지 하루) 안에서 X를 여기까지로 마친다.
- **R27** (Task 10) 사용자가 `wsl --shutdown`을 한 뒤 GPU가 복구됐다(`nvidia-smi` 응답, `gpu_reduce` 통과). 이때 SM 클럭이 1680 MHz여서 `-lgc` 고정은 풀린 것으로 보이지만, Task 10에는 측정이 없다. 전체 테스트: CPU 빌드 78/78, CUDA 빌드 102/102, PyTorch np2·np4 각각 37 passed, 3 skipped.
- **R28** (Task 10 Step 4) 브랜치 리뷰: 계획의 "superpowers 최종 리뷰 절차"가 없어서 `main..b-gpu`(42개 파일, +1647/−261)를 직접 다시 읽었다. 지적과 조치:
  (1) CUDA 빌드에서 host 전용 호출도 CUDA 컨텍스트를 만든다. 측정 결과 rank당 약 115 MiB였다. 판정 방식을 바꿔야 고칠 수 있어 KNOWN_ISSUES #26에 기록만 했다.
  (2) `IpcTransport::waitall`이 sleep 없이 polling한다. KNOWN_ISSUES #27에 기록했고 측정은 안 했다.
  (3) 모든 rank가 같은 종류의 버퍼를 써야 하는데 이를 검사하지 않는다. KNOWN_ISSUES #18에 적었다.
  (4) 비동기 반환, Group 혼합 거절, torchrun 미지원, 항상 완료된 Work를 KNOWN_ISSUES #16·#17·#19·#20에 추가하고, #4(fp8)는 Task 3 결과로 갱신돼 있다.
  (5) 맥 CPU 빌드(`HXCCL_CUDA=OFF`, clang/libc++)는 이 데스크탑에서 확인할 수 없다. `fp.h`·`elem.h`가 `<math.h>`의 `isnan`/`ldexpf` 등을 한정자 없이 쓰므로 맥에서 빌드와 78개 테스트 확인이 필요하다(스펙 5 "CPU 빌드" 행).
  그 밖의 코드 결함은 찾지 못했다.
- **R29** (#26 수정, R28 (1) 정정) R28 (1)의 원인 판단이 틀렸다. 확인한 것:
  - 시험 프로그램에서 host 포인터로 `cudaPointerGetAttributes`를 불러도 primary 컨텍스트는 비활성(active=0) 그대로였다. `cudaSetDevice(0)` 뒤에는 active=1이 됐다.
  - 115 MiB/rank는 `hxccl_perf`가 CUDA 빌드에서 `cudaSetDevice(0)`를 무조건 부른 탓이었다. `-d`일 때만 부르게 고치자, 수정 전 라이브러리와 driver API로 먼저 확인하게 바꾼 라이브러리 모두 host np4 실행 중 GPU 메모리가 441 MiB(idle과 같음)였다.
  - 그래서 라이브러리 변경(driver API 확인, libcuda 링크)은 되돌렸다. 회귀 테스트 `tests/gpu_host_no_context.cc`를 추가했고, 이 테스트만 `CUDA::cuda_driver`를 링크한다. host 구간에 `cudaFree(0)`를 넣으면 이 테스트가 실패하는 것도 확인했다.
  - 결과: CUDA 빌드 108/108, CPU 빌드 78/78.
  - 측정 영향: B·Task 6의 host 경로 측정은 rank마다 CUDA 컨텍스트가 있는 상태에서 잰 것이다. 시간에 영향이 있었는지는 확인하지 않았다.

## E (소켓 부트스트랩, 브랜치 `e-socket`)

- **E1** (Task E0) 스펙·계획은 데스크탑에서 사용자 요청("torchrun 지원 먼저")으로 썼다. 맥 Claude가 리뷰하지 않았다. `Net`/`MpiNet`을 추가하고 IPC 제어 메시지와 AllToAllv 합의를 그쪽으로 옮겼다. 동작은 바뀌지 않았다(CPU 78/78, CUDA 108/108 통과).
- **E2** (Task E1) `hxcclGetUniqueId`/`hxcclCommInitRank`가 이제 항상 소켓 부트스트랩을 쓴다. MPI는 필요 없다. 그래서 `nranks > MPI_COMM_WORLD 크기` 검사를 뺐다. 기존 테스트 두 곳을 바꿨다.
  - `test_errors`의 "모두가 rank 0이라고 주장" 기대값: rank 0만 성공 → 모든 rank가 `hxcclInvalidUsage`(겹친 rank로는 메시를 만들 수 없다).
  - `MPI_Comm_get_errhandler(c->mpi)` 검사: MPI 통신기일 때만 한다(소켓 통신기에는 MPI 핸들이 없다).
- **E3** (Task E1) 결과: CPU 빌드 156/156(MPI 78 + `HXCCL_BOOTSTRAP=socket` 78). 소켓 변형만 따로 3번 더 돌려 매번 78/78. CUDA 빌드 201/201(GPU 테스트의 소켓 변형 포함). 상대가 연결을 닫으면 그 상대가 걸린 요청만 실패시키고, 부트스트랩 connect는 30초 동안 재시도한다. 부트스트랩 timeout(오지 않는 rank 감지)은 넣지 않았다(스펙 2).
- **E4** (Task E2) torchrun에서 GPU 텐서가 `cudaIpcOpenMemHandle: invalid resource handle`로 실패했다.
  - 원인(측정): 이 WSL은 `ptrace_scope=1`이다. 핸들을 내보낸 프로세스가 ptrace를 허용해야(`prctl(PR_SET_PTRACER, ANY)`) 상대가 핸들을 연다. OpenMPI의 `MPI_Init`이 이 설정을 해 주고 있었다. ctypes로 `prctl`만 불러도 torchrun에서 성공했고, `MPI_Init` 뒤 설정을 0으로 되돌리면 C++에서도 같은 오류가 났다(`tests/gpu_no_ptracer.cc`). CUDA가 왜 ptrace 권한을 요구하는지는 **원인 미확인**.
  - 라이브러리에서 `prctl`을 부르는 수정은 자동 모드 권한 분류기가 "보안 약화"로 거부했다. 대신 IPC를 열 수 없으면 모든 rank가 합의해 **staged 경로**(device → host → Net → host → device)로 자동 전환하게 했다. `HXCCL_DEVICE_TRANSPORT=staged`로 강제할 수도 있다.
  - CUDA VMM의 POSIX fd 공유도 시험했다: API는 성공했지만 읽은 값이 틀렸다(1M 중 524288개). 그래서 쓰지 않았다. **원인 미확인.**
  - 중간 버전(검증 전 staged 코드)의 `gpu_no_ptracer` 실행이 끝나지 않았고, 그 뒤 GPU가 멈춰 `wsl --shutdown`이 필요했다. **원인 미확인.** 지금 코드에서는 np2·np4 모두 staged로 합의하고 정상 종료한다.
- **E5** (Task E2) PyTorch 확장이 import 때 `MPI_Init`을 부르지 않는다. 결과: torchrun np2·np4, mpirun np2·np4 모두 37 passed, 3 skipped.

### 맥 최종 리뷰(`docs/desktop/b-review-fixes.md`) 반영, `e-socket`에서

- **R30** 재현(커밋된 E1 코드 `53fb9fe`, 테스트를 하나씩 따로, 각 TIMEOUT 60 s):
  - I3: segfault
  - I1, I4, I2: Timeout
  - I5: Timeout. 실패 주입한 rank 0은 반환하고 rank 1은 대기
  - 처음 동시 실행은 앞선 중단 실행 때문에 GPU가 이미 멈춘 상태여서 무효로 보고 다시 쟀다.
  - 테스트: `tests/repro_*.cc`, MPI·소켓 변형, I2는 실패 주입 변형(`_inject`)까지 12개.
- **R31** I4: 통신기 생성 때(`device_eager_init`, 모든 rank가 부르는 집합 동작) 모든 rank에 CUDA 컨텍스트가 이미 있으면 device 경로를 바로 시작한다. 컨텍스트 확인은 driver API로 해서 새로 만들지 않는다(그래서 `CUDA::cuda_driver`를 링크). 컨텍스트가 없는 host 전용 프로그램은 #26 보장 그대로 지연 시작한다(`gpu_host_no_context` 통과). `HXCCL_DEVICE=1`은 강제 시작, `HXCCL_DEVICE=0`은 끈다. PyTorch 백엔드는 통신기를 만들기 전에 `cudaFree(nullptr)`로 컨텍스트를 만든다.
- **R32** I2:
  - `HXCCL_IPC_CHUNK`는 64 B~1 GiB인 숫자만 받고, 그 밖의 값은 기본 4 MiB로 둔다. 그 뒤 모든 rank의 최솟값으로 합의한다.
  - `IpcTransport::init`의 모든 단계(할당, 핸들 교환, 열기)를 실패와 상관없이 모든 rank가 거치고, 매 단계 합의한다. 어디서든 실패하면 모두 staged로 간다. 상대 핸들을 모두 닫은 뒤 합의를 한 번 더 하고 나서 자기 버퍼를 해제한다.
  - Net 자체가 실패하면 통신기에 오류를 고정한다(`dev_error`). 이후 device 호출은 다시 시도하지 않고 그 오류를 반환한다.
- **R33** I1: `Transport::step()`(한 번만 진행, 블록하지 않음)을 추가했다. `run_p2p`는 Group이 쓰는 모든 transport를 한 루프에서 번갈아 진행한다. 하나가 실패하면 나머지는 취소한다. MPI transport는 기본 구현(waitall)을 쓴다: MPI가 프로세스의 모든 요청을 함께 진행시키기 때문이다.
- **R34** I3: `plan_op`가 stream을 받는다. `hxcclScalarDevice` 포인터가 device 메모리면 그 stream을 동기화한 뒤 `cudaMemcpy`로 읽는다(`read_scalar`). 호출 시점의 값을 쓴다.
- **R35** I5:
  - 각 rank가 device 경로 시작 때 모든 상대로부터 오는 ABORT 수신을 걸어 둔다. 전송 중 치명적 오류가 나면 모든 상대에게 ABORT를 보낸다. 기다리던 rank는 `hxcclRemoteError`("rank N failed during a device transfer")로 반환하고, 그 통신기의 device 경로는 계속 실패 상태로 남는다.
  - 고치는 중에 오탐을 하나 찾았다. 상대가 통신기를 Destroy해 소켓을 닫으면 ABORT 수신이 "연결 닫힘"으로 완료돼 abort로 잘못 처리됐다(`repro_p2p_first_partial` 소켓 변형). 실제 메시지가 도착했을 때만 abort로 보게 고쳤다.
  - 미룸: 다른 스레드의 `hxcclCommAbort`와 진행 루프의 동시 실행(use-after-free 가능), poll 루프의 시간 제한 → KNOWN_ISSUES #28, #29.
- **R36** Minor:
  - M3는 고쳤다. 자기 자신에게 보낸 메시지가 잘리면 나머지를 취소하지 않고 끝까지 진행한 뒤 truncation을 보고한다.
  - M1, M2, M4는 KNOWN_ISSUES #30~#32에 기록만 했다.
  - 리뷰어 질문 두 가지는 확인하지 않았다: Destroy에 barrier 없이 staging 메모리를 `cudaFree`해도 안전한지, 8~16 B 제어 메시지가 항상 eager인지.
- **R37** 결과:
  - 재현 테스트 12/12(하나씩), `gpu_no_ptracer` np2·np4 통과(staged로 합의)
  - CPU 빌드 156/156, CUDA 빌드 258/258(staged·소켓 변형 포함)
  - PyTorch torchrun·mpirun np2·np4 각각 37 passed, 3 skipped
- **R38** (Nsight 캡처, `docs/desktop/nsight-captures.md`)
  - **Nsight Systems 그림:** 데스크탑 세션에서는 Windows GUI를 띄워 캡처할 수 없었다. 지시의 대체안대로, 보고서의 SQLite 데이터(NVTX와 CUDA API 구간)로 matplotlib 그림 3장을 그렸다(`scripts/plot_b_timeline.py`, `docs/results/figs/`). **GUI 화면이 아니라 보고서 데이터로 그린 그림이다.** GPU 트레이스가 없어서 모든 막대는 CPU 쪽 시간이다(R21).
  - **그림에서 새로 보인 것:**
    - device AllReduce 한 번의 바닥값(64K에서 3.58 ms) 중 가장 큰 부분은 **첫 `ipc:copy`의 `cudaStreamSynchronize` 한 번**이다: 64K에서 2080 µs, 64M에서 2532 µs. 나머지 copy는 0.1~0.3 ms다.
    - 4M에서 두 rank를 같은 시간축에 놓으면, rank 1의 이 긴 첫 동기화가 끝난 뒤에야 rank 0의 첫 copy가 시작한다.
    - 첫 동기화가 왜 2 ms인지는 **원인 미확인**이다. KNOWN_ISSUES #21의 "나머지"는 이것과 그 대기로 보이지만, 측정은 이 그림까지다.
  - **Nsight Compute 재측정:**
    - 조건: SM 클럭을 `nvidia-smi -lgc 1500,1500`으로 고정한 상태(사용자). 측정 전후 1500/7000 MHz.
    - 명령: `/usr/local/cuda/bin/ncu --clock-control none --set full`로 fp32 sum 1<<20 원소 `k_reduce<float>` 한 번(13번째 launch)만 쟀다.
    - 결과: 32.16 µs, DRAM read 8.39 MB + write 4.12 MB, **389 GB/s(정격 448 GB/s의 86.8%)**. ncu DRAM Throughput 89.6%. replay 중 실측 클럭은 SM 1.47 GHz, DRAM 6.78 GHz. 기존 1 pass 값(31.5 µs, 386 GB/s)과 2% 안에서 같다.
    - R22에서 본 replay 중 DRAM 0.81 GHz 하락은 이번에 재현되지 않았다(이번에는 `--set full`인데도 6.78 GHz). R22 때 무엇이 달랐는지는 **원인 미확인**이다.
    - Nsight Compute GUI 캡처도 같은 이유로 못 했다. `.ncu-repz`(142 KB)와 details 텍스트를 커밋한다.
  - **커밋한 보고서:** `.nsys-rep` 6개(각 0.31~0.40 MB, `.gitignore`에 걸려 `-f`로 추가). SQLite(각 0.8~1.1 MB)는 `nsys export --type sqlite`로 다시 만들 수 있어 커밋하지 않았다.
- **R39** (Nsight GUI 캡처, 사용자 요청 "GUI를 캡처해 보자")
  - **방식:** R38에서는 GUI를 캡처하지 못했다. hxgemm과 같은 방식으로 WSL 안의 GUI를 WSLg에 띄우고 X11 XTEST로 조작해 창 단위로 캡처했다. 대상은 Nsight Systems 2026.3.2와 Nsight Compute 2026.3.1이다. 조작 도구는 `tools/nsight_gui.py`(창 id 지정, drag·우클릭·문자 입력·스크롤 추가)다.
  - **그림:**
    - Nsight Systems 3장: `fig_b_nsys_64k.png`, `fig_b_nsys_64m.png`, `fig_b_nsys_r0r1.png`. r0r1은 File → New multi-report view로 두 보고서를 한 시간축에 놓았다.
    - Nsight Compute 2장: `fig_b_ncu_sol.png`, `fig_b_ncu_memory.png`.
    - R38의 matplotlib 그림 3장은 교차 확인용 보조 자료로 남겼다(같은 호출, 같은 숫자).
  - **겪은 문제:**
    - 처음에는 X11 창 목록에 아무 창도 잡히지 않았다. "사용 데이터 수집 동의" 대화 상자가 메인 창을 막고 있었다. **No**(수집 안 함)를 눌렀다.
    - Nsight Systems 2026.3.2는 이어서 EGL 초기화 실패로 죽었다(충돌 보고는 보내지 않음). `QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu`로 띄우니 정상이었다.
    - XWayland에서는 루트 창 캡처가 BadMatch로 막혀서, 우클릭 메뉴 같은 팝업 창을 따로 찾아 찍었다.
  - **GUI에서 확인한 숫자:** 앞선 데이터 그림과 같다.
    - 64K: 호출 3.581 ms, 첫 `ipc:copy` 2.080 ms
    - 64M: 호출 11.576 ms, 첫 `ipc:copy` 2.532 ms
    - 4M: rank 1의 첫 copy 2.103 ms 동안 rank 0이 2.188 ms 기다림
    - ncu: 32.16 µs, DRAM 89.63 %, 389.06 GB/s, 읽기 8.39 MB, 쓰기 4.12 MB
- **E6** (Task E3) E2에서 import 때 `MPI_Init`을 없앤 뒤로는 **mpirun에서도** PyTorch device 경로가 staged로 갔다. `HXCCL_DEBUG`로 확인했는데, 큰 텐서에서 약 12배 느린 후퇴였다. mpirun으로 띄운 경우(`OMPI_COMM_WORLD_SIZE`가 있을 때)에만 import 때 `MPI_Init_thread`를 다시 부르게 고쳤다. 이제 mpirun은 IPC, torchrun은 staged다. torchrun·mpirun np2·np4 pytest 모두 37 passed, 3 skipped. 이것은 OpenMPI의 원래 동작을 되살린 것이다(라이브러리가 `prctl`을 부르지는 않는다).
- **E7** (Task E3) 측정
  - 조건: SM 1500 MHz 고정, 샘플 271 + 16개 모두 1500. 데이터와 표: `scripts/measure_e.sh`, README "Measurements (E)".
  - 소켓 host 전송: MPI보다 4 KiB에서 2.5~2.8배, 16 MiB 이상에서 1.3~1.4배 느리다.
  - IPC 제어를 소켓으로 보내도 MPI와 거의 같다.
  - staged 경로(= torchrun의 device 경로): 256 MiB에서 IPC보다 10.8~12.6배, gloo보다 4.1배 느리다(KNOWN_ISSUES #33).
  - PyTorch mpirun: np2 256 MiB 41.8 ms로 C 측정(41.5 ms)과 같다. torchrun은 559 ms.
  - 바닥값(np2 약 3.2 ms)이 IPC copy가 없는 staged 경로에도 똑같이 있다. 그러니 IPC copy 자체가 원인은 아니다. **원인 미확인**(#35).
