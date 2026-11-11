# hxccl B+C 실행 기록 (ledger)

계획(`docs/plans/2026-11-11-hxccl-b-c.md`)과 다르게 한 결정을 `R<n>`으로 적는다.

- **R1** (Task 0, 2026-11-11) 실행 방식: 계획 머리말의 `superpowers:executing-plans` 스킬이 데스크탑 세션에 설치되어 있지 않아, 계획의 Task·Step을 순서대로 직접 따라 실행했다.
- **R2** (Task 0 Step 4) `~/venvs/hxgemm/bin/torchrun`이 없다(venv bin에 torchrun 진입점이 없음). 같은 동작인 `~/venvs/hxgemm/bin/python -m torch.distributed.run --nproc-per-node 2 ...`로 실행했다.
- **R3** (Task 0 Step 5) WSL에 `nvidia-cuda-mps-control`이 없다(`/usr/local/cuda/bin`, `/usr/lib/wsl/lib`에도 없음). 계획의 명령은 "command not found"로 끝났고, 결과를 그대로 환경 기록에 적었다.
