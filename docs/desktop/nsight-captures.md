# hxccl 블로그 #3(B편)용 Nsight 캡처 지시 (데스크탑)

**전제: 리뷰 수정(I1~I5, `docs/desktop/b-review-fixes.md`)을 먼저 끝내고 커밋·push한 뒤에 시작한다.**

- 목표: 블로그에 넣을 Nsight 화면 캡처 3장 이상과, 각 그림이 무엇을 보여 주는지 적은 설명.
- 저장 위치: `docs/results/figs/` (PNG). 작업 중인 브랜치(`e-socket`)에 커밋해서 push한다.

## 1. Nsight Systems 타임라인 (기존 보고서 사용)

**파일**: `docs/results/2026-11-11_b_nsys_{64K,4M,64M}_r{0,1}.nsys-rep`

**여는 방법**: Windows의 Nsight Systems 2025.5.1 GUI로 연다.
- 실행 파일: `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.1` 아래 `nsys-ui.exe`
- WSL 쪽 파일 경로: `\\wsl$\Ubuntu\home\a\hxccl\...`

**캡처할 화면**:
- **캡처 1** (`fig_b_nsys_64k.png`): 64K r0에서 AllReduce 호출 하나를 확대한다.
  - NVTX 행(`cudaMallocAsync`, `ipc:copy` ×4, `reduce`)과 CUDA API 행이 함께 보여야 한다.
  - 구간 길이가 표시되도록 마우스로 범위를 선택한 상태에서 캡처한다.
- **캡처 2** (`fig_b_nsys_64m.png`): 64M r0에서 같은 구성으로 캡처한다. 크기가 커지면 `ipc:copy`가 길어지는 것을 비교하기 위해서다.
- **캡처 3** (`fig_b_nsys_r0r1.png`): 4M에서 r0과 r1을 같은 시간축으로 볼 수 있으면 캡처한다(READY/DONE 대기가 보이게). 안 되면 건너뛰고 이유를 적는다.

**GUI를 띄우고 캡처할 수 없는 환경이면**: 억지로 하지 말고 대체 방법을 쓴다.
- `nsys export --type sqlite`로 NVTX와 CUDA API 구간을 뽑아 matplotlib으로 타임라인 그림을 만든다.
- 파일 이름은 `fig_b_timeline_*.png`로 한다.
- "Nsight 화면이 아니라 보고서 데이터로 그린 그림"이라는 사실을 README와 LEDGER에 명시한다.

**한계**: WSL이라 GPU 커널 트레이스가 없다(R21). 그림 설명에 그대로 쓴다.

## 2. Nsight Compute 재측정 (리덕션 커널)

**준비**:
- SM 클럭을 고정해야 한다. R27에서 고정이 풀린 상태다.
- Windows 관리자 PowerShell에서 `nvidia-smi -lgc 1500,1500`을 실행해야 하므로, 사용자에게 요청하고 끝날 때까지 기다린다.

**측정**:
- 반드시 `/usr/local/cuda/bin/ncu`(2026.3.1)를 쓴다. `/usr/bin/ncu`는 R22에서 실패했다.
  ```bash
  /usr/local/cuda/bin/ncu --set full -k regex:reduce \
    --export docs/results/2026-11-11_b_ncu_reduce_full <1M fp32 sum 실행 명령>
  ```
- 측정 전후로 `nvidia-smi`로 SM·메모리 클럭을 기록한다. R22처럼 replay 중에 메모리 클럭이 내려가는지도 함께 기록한다.

**캡처**: Windows의 Nsight Compute GUI로 `.ncu-rep`를 열어 캡처한다.
- `fig_b_ncu_sol.png`: Speed Of Light (DRAM 사용률)
- `fig_b_ncu_memory.png`: Memory Workload Analysis 차트
- GUI 캡처가 안 되면 `ncu --import ... --page details` 출력을 텍스트로 저장하고 이유를 적는다.

**결과 비교**: 결과 숫자(시간, DRAM GB/s, 정격 448 GB/s 대비 %)를 기존 결과(31.5 µs, 386 GB/s)와 비교해 적는다. 다르면 원인을 측정한 만큼만 쓰고, 나머지는 "원인 미확인"으로 둔다.

## 3. 끝나면

- `docs/results/figs/README.md`에 그림마다 한 줄씩 적는다: 파일, 원본 보고서, 무엇을 보여 주는지, 한계.
- `.nsys-rep`와 `.ncu-rep`를 `docs/results/`에 함께 커밋한다(블로그에서 링크할 수 있도록).
  - 각각 수 MB 이내인지 확인한다. 크면 커밋하지 않고 크기만 적는다.
- `docs/LEDGER.md`에 R번호로 측정 조건과 결정을 적는다.
- push하고 맥에 보고한다:
  - 만든 그림 목록
  - 측정 숫자
  - 원인 미확인 항목
