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
