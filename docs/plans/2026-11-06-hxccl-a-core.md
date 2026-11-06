# hxccl A (코어) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** NCCL 모양의 C API(`hxccl.h`)로 AllReduce / AllGather / ReduceScatter / Broadcast를 OpenMPI 위에서 CPU 버퍼로 구현하고, `MPI_*` 결과와의 일치·통신량·오류를 테스트하고, hxccl과 MPI를 나란히 재는 벤치마크를 만든다.

**Architecture:** `api.cc`(인자 검사·오류 변환) → `select`(알고리즘 선택) → `algo_*`(Transport만 사용) → `transport_mpi`(MPI_Sendrecv, 통계). 리덕션은 `reduce`.

**Tech Stack:** C++17(Apple clang 14), CMake 4.4 + Ninja, OpenMPI 5.0.11 (conda env `hxccl`: `/opt/anaconda3/envs/hxccl`).

**Spec:** `docs/specs/2026-11-06-hxccl-a-core-design.md`

## Global Constraints

- 빌드: `export PATH=/opt/anaconda3/envs/hxccl/bin:$PATH; cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ && cmake --build build && ctest --test-dir build --output-on-failure`.
- 모든 MPI 테스트는 `mpirun --oversubscribe -np N`(ctest가 실행). N ∈ {1,2,3,4,5,8}.
- 공개 API 이름·인자 순서는 스펙 3절 그대로.
- 알고리즘 코드는 `Transport` 외의 MPI 함수를 부르지 않는다.
- 커밋 author `Thisish3 <104387634+Thisish3@users.noreply.github.com>`, 메시지 끝 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. push는 사용자 승인 후.

## Review Focus

1. **count < N, count % N ≠ 0** → ring 조각 크기가 0인 rank가 있어도 결과·통신량 정확.
2. **N이 2의 거듭제곱이 아님**(3, 5) → rd가 남는 rank를 접어 정확한 결과.
3. **in-place 호출** 3종 → 결과가 out-of-place와 같음.
4. **`count * sizeof(dt)` 오버플로, null 포인터(count>0), 범위 밖 root** → InvalidArgument, 통신 없음.
5. **MPI 오류** → 프로세스가 죽지 않고 SystemError.

---

## File Structure

```
CMakeLists.txt
include/hxccl.h
src/comm.h  src/comm.cc        hxcclComm 구조체, 오류 문자열 저장
src/transport.h                Transport 인터페이스
src/transport_mpi.cc           MpiTransport
src/reduce.h  src/reduce.cc    reduce_into(dst, src, n, dt, op), dtype_size
src/chunks.h                   ring 조각 경계 계산 (테스트도 사용)
src/algo.h                     알고리즘 함수 선언
src/algo_ring.cc               ring allreduce / reducescatter / allgather
src/algo_rd.cc                 recursive doubling allreduce
src/algo_bcast.cc              binomial tree / chain broadcast
src/select.h  src/select.cc    알고리즘 선택
src/api.cc                     C API
tests/check.h                  MPI 테스트용 CHECK 매크로와 종료 처리
tests/test_*.cc
bench/hxccl_perf.cc
```

---

### Task 1: 빌드 골격, 공개 헤더, communicator 수명

**Files:** Create `CMakeLists.txt`, `include/hxccl.h`, `src/comm.h`, `src/comm.cc`, `src/transport.h`, `src/transport_mpi.cc`, `src/api.cc`, `tests/check.h`, `tests/test_init.cc`, `.gitignore`

**Produces:** 스펙 3절의 모든 선언(이 Task에서는 collective 함수 본문이 `hxcclInternalError`를 반환), `hxcclCommInitMPI/InitRank/Destroy/Count/UserRank/GetErrorString/GetLastError/GetStats/ResetStats` 구현, `struct hxcclComm { int rank, nranks; MPI_Comm mpi; std::unique_ptr<Transport> tr; std::string last_error; }`, `class Transport { virtual hxcclResult_t sendrecv(int to, const void* sbuf, size_t sbytes, int from, void* rbuf, size_t rbytes) = 0; virtual hxcclResult_t send(int to, const void*, size_t); virtual hxcclResult_t recv(int from, void*, size_t); uint64_t bytes_sent, msgs_sent; std::string error; }`.

- [ ] **Step 1: `tests/check.h`**

```cpp
#pragma once
#include <mpi.h>
#include <cstdio>
#include <cstdlib>

inline int g_fail = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_);                             \
      std::fprintf(stderr, "[rank %d] %s:%d CHECK(%s) failed\n", r_, __FILE__, \
                   __LINE__, #cond);                                           \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)

// every rank returns the same exit code: nonzero if any rank failed
inline int finish() {
  int any = 0;
  MPI_Allreduce(&g_fail, &any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Finalize();
  return any ? 1 : 0;
}
```

- [ ] **Step 2: 실패하는 테스트 `tests/test_init.cc`**

```cpp
#include "check.h"
#include "hxccl.h"
#include <cstring>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int wr, wn;
  MPI_Comm_rank(MPI_COMM_WORLD, &wr);
  MPI_Comm_size(MPI_COMM_WORLD, &wn);

  hxcclComm_t c = nullptr;
  CHECK(hxcclCommInitMPI(&c, MPI_COMM_WORLD) == hxcclSuccess);
  int n = -1, r = -1;
  CHECK(hxcclCommCount(c, &n) == hxcclSuccess && n == wn);
  CHECK(hxcclCommUserRank(c, &r) == hxcclSuccess && r == wr);
  uint64_t b = 1, m = 1;
  CHECK(hxcclCommGetStats(c, &b, &m) == hxcclSuccess && b == 0 && m == 0);
  CHECK(hxcclCommDestroy(c) == hxcclSuccess);

  hxcclUniqueId id;
  if (wr == 0) CHECK(hxcclGetUniqueId(&id) == hxcclSuccess);
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
  hxcclComm_t c2 = nullptr;
  CHECK(hxcclCommInitRank(&c2, wn, id, wr) == hxcclSuccess);
  CHECK(hxcclCommUserRank(c2, &r) == hxcclSuccess && r == wr);
  CHECK(hxcclCommDestroy(c2) == hxcclSuccess);

  CHECK(hxcclCommInitMPI(nullptr, MPI_COMM_WORLD) == hxcclInvalidArgument);
  CHECK(hxcclCommInitRank(&c2, wn, id, wn) == hxcclInvalidArgument);
  CHECK(std::strlen(hxcclGetErrorString(hxcclSystemError)) > 0);
  return finish();
}
```

`CMakeLists.txt`(전체):
```cmake
cmake_minimum_required(VERSION 3.20)
project(hxccl CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(MPI REQUIRED COMPONENTS CXX)
add_compile_options(-Wall -Wextra -Wpedantic)

file(GLOB HXCCL_SRC CONFIGURE_DEPENDS src/*.cc)
add_library(hxccl STATIC ${HXCCL_SRC})
target_include_directories(hxccl PUBLIC include PRIVATE src)
target_link_libraries(hxccl PUBLIC MPI::MPI_CXX)

enable_testing()
function(hxccl_mpi_test name)
  add_executable(${name} tests/${name}.cc)
  target_include_directories(${name} PRIVATE src tests)
  target_link_libraries(${name} PRIVATE hxccl)
  foreach(np 1 2 3 4 5 8)
    add_test(NAME ${name}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} ${np}
                     $<TARGET_FILE:${name}>)
  endforeach()
endfunction()
file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/test_*.cc)
foreach(src ${TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  hxccl_mpi_test(${name})
endforeach()

add_executable(hxccl_perf bench/hxccl_perf.cc)
target_link_libraries(hxccl_perf PRIVATE hxccl)
```
(`bench/hxccl_perf.cc`는 Task 7 전까지 `int main(){return 0;}`.)

Run: 빌드 명령(Global Constraints) → Expected: 컴파일 또는 링크 실패(심볼 없음).

- [ ] **Step 3: 구현** — `hxccl.h`는 스펙 3절 + `#include <mpi.h>`, `<stddef.h>`, `<stdint.h>`와 `extern "C"`. `comm.cc`/`api.cc`:
  - `hxcclCommInitMPI`: `comm` null → InvalidArgument. `MPI_Comm_dup`, `MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN)`, rank/size 조회, `MpiTransport` 생성.
  - `hxcclGetUniqueId`: `std::random_device`로 128바이트.
  - `hxcclCommInitRank`: `rank` 범위 밖·`nranks`가 WORLD 크기 초과 → InvalidArgument. color = id 바이트의 FNV-1a 해시 `& 0x7fffffff`, `MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub)`; 결과 크기 ≠ nranks → InvalidUsage; 이후 InitMPI와 같은 경로(sub를 소유).
  - `hxcclCommDestroy`: `MPI_Comm_free` 후 delete.
  - `hxcclGetErrorString`: 각 코드별 고정 문자열. `GetLastError`: `comm->last_error.c_str()`(null comm이면 `""`).
  - `MpiTransport::sendrecv`는 `MPI_Sendrecv(sbuf, sbytes, MPI_BYTE, to, 0, rbuf, rbytes, MPI_BYTE, from, 0, mpi, MPI_STATUS_IGNORE)`. 실패 시 `MPI_Error_string`을 `error`에 담아 SystemError. 성공 시 `bytes_sent += sbytes; ++msgs_sent`. `send`/`recv`도 같은 방식. `sbytes`가 `INT_MAX` 초과면 SystemError(A 범위에서는 64MB 이하만 쓴다, 메시지에 명시).

- [ ] **Step 4: 통과 확인** — Run: 빌드 + `ctest --test-dir build -R test_init` → Expected: 6개(np1..8) 통과.
- [ ] **Step 5: 커밋** "Add hxccl build, public header and communicator lifecycle".

### Task 2: 리덕션과 조각 계산

**Files:** Create `src/reduce.h`, `src/reduce.cc`, `src/chunks.h`, `tests/test_reduce.cc`

**Produces:** `size_t dtype_size(hxcclDataType_t)`(0이면 모르는 dtype), `bool valid_op(hxcclRedOp_t)`, `void reduce_into(void* dst, const void* src, size_t count, hxcclDataType_t, hxcclRedOp_t)`(dst = dst op src), `struct Chunk { size_t off, len; }; Chunk chunk_of(size_t count, int nranks, int i)`(앞쪽 `count % n`개 조각이 1개씩 더 큼).

- [ ] 실패하는 테스트: i32/f32/f64 × sum/prod/max/min 값 검사, `chunk_of(7,3,i)` = (0,3),(3,2),(5,2), `chunk_of(2,5,i)` 뒤 3개 len 0, 모든 조각 합 = count. (ctest가 np1..8로 돌리지만 rank 0만 검사해도 됨)
- [ ] 구현 → 통과 → 커밋 "Add element-wise reduction and ring chunking".

### Task 3: ring AllReduce와 통신량

**Files:** Create `src/algo.h`, `src/algo_ring.cc`, `tests/test_allreduce.cc`; Modify `src/api.cc`

**Produces:** `hxcclResult_t ring_allreduce(Transport&, int rank, int n, const void* send, void* recv, size_t count, hxcclDataType_t, hxcclRedOp_t)`; `hxcclAllReduce`가 `HXCCL_ALGO` 미설정 시 당분간 ring 호출.

알고리즘(스펙 4절): `recv`에 `send` 복사(in-place면 생략). 오른쪽 = (r+1)%n, 왼쪽 = (r−1+n)%n. reduce-scatter: s=0..n−2에서 보낼 조각 `(r−s)%n`, 받을 조각 `(r−s−1)%n`을 임시 버퍼로 받아 `reduce_into(recv+받을조각, tmp)`. all-gather: s=0..n−2에서 보낼 조각 `(r+1−s)%n`, 받을 조각 `(r−s)%n`을 `recv`에 바로 받음. n==1이면 복사만.

- [ ] 실패하는 테스트: 스펙 6절 AllReduce 조합(algo는 이 Task에서 ring만). 입력 `x[i] = (rank+1)*((i%5)+1)`(sum/max/min), prod는 `x[i] = (i+rank)%3==0 ? 2 : 1`로 정확히 표현되는 값. 기준 `MPI_Allreduce`(MPI_INT/FLOAT/DOUBLE, MPI_SUM/PROD/MAX/MIN). in-place 1회. 통신량: `ResetStats` 후 ring 호출, `bytes == Σ_{s=0}^{n-2} (len((r−s)%n) + len((r+1−s)%n)) × size`, `msgs == 2(n−1)`.
- [ ] 구현 → 통과 → 커밋 "Implement ring allreduce over the MPI transport".

### Task 4: recursive doubling과 선택

**Files:** Create `src/algo_rd.cc`, `src/select.h`, `src/select.cc`; Modify `src/api.cc`, `tests/test_allreduce.cc`

**Produces:** `rd_allreduce(...)`(ring과 같은 시그니처), `enum class Algo { Ring, Rd }; Algo choose_allreduce(size_t bytes)`(`HXCCL_ALGO=ring|rd` 우선, 그 외 64KB 미만 Rd), 내부 테스트용 `hxcclResult_t allreduce_with(Algo, ...)`(`src/algo.h`에 선언, 테스트가 include).

rd(스펙 4절): p = n 이하 최대 2의 거듭제곱, rem = n−p. r < 2·rem: 짝수 r은 오른쪽(r+1)에 전체 버퍼 send, 홀수 r은 recv 후 reduce. 참여 rank의 새 번호 nr = (r < 2rem) ? r/2 : r−rem (짝수 r<2rem은 불참). 참여 rank는 mask=1..p/2에서 상대 nr^mask(원래 번호로 환산: q<rem ? 2q+1 : q+rem)와 sendrecv 후 reduce. 끝나면 홀수 r<2rem이 짝 r−1에 결과 send.

- [ ] 실패하는 테스트: 기존 조합에 algo ∈ {Rd, auto}, N=4·8에서 rd 통신량 `msgs == log2 N`, `bytes == log2N × count × size`. `HXCCL_ALGO=rd`를 `setenv` 후 `hxcclAllReduce` → rd 통계와 일치.
- [ ] 구현 → 통과 → 커밋 "Add recursive doubling allreduce and size-based selection".

### Task 5: AllGather, ReduceScatter, Broadcast

**Files:** Modify `src/algo_ring.cc`, `src/api.cc`; Create `src/algo_bcast.cc`, `tests/test_collectives.cc`

**Produces:** `ring_reducescatter`(recvcount 기준, 조각 = rank별 recvcount), `ring_allgather`(sendcount 기준), `bcast(Transport&, rank, n, buf, bytes, root)`(bytes < 512KB이면 binomial tree, 아니면 512KB 조각 파이프라인 체인 root→root+1→…).

- [ ] 실패하는 테스트: 스펙 6절 해당 행, 기준 `MPI_Allgather`, `MPI_Reduce_scatter_block`, `MPI_Bcast`(root 0과 n−1), in-place 정의(스펙 3절) 확인.
- [ ] 구현 → 통과 → 커밋 "Add allgather, reducescatter and broadcast".

### Task 6: 인자 검사와 오류 경로

**Files:** Modify `src/api.cc`; Create `tests/test_errors.cc`

**Produces:** 모든 collective 앞단 검사: comm null, dtype/op 무효, count>0인데 buffer null, `count > SIZE_MAX / size`, root 범위 밖 → InvalidArgument + `last_error`. Transport 오류 → SystemError + `last_error`에 MPI 문자열.

- [ ] 실패하는 테스트: 각 잘못된 인자별 반환값·`hxcclGetLastError` 비어 있지 않음·통계 0 유지. `MpiTransport`를 직접 만들어 `send(peer=n)` → SystemError(프로세스 생존).
- [ ] 구현 → 통과 → 커밋 "Validate arguments and surface MPI errors".

### Task 7: 벤치마크와 README

**Files:** Modify `bench/hxccl_perf.cc`; Create `README.md`

**Produces:** `hxccl_perf -b <min> -e <max> -f <factor> -n <iters> [-o allreduce|allgather|reducescatter|broadcast]`, rank 0이 표 출력: `size count hxccl_us hxccl_algbw hxccl_busbw mpi_us mpi_algbw mpi_busbw algo`. 각 크기에서 워밍업 3회, `MPI_Barrier` 후 n회 각각 시간 재서 rank 최대값의 중앙값. busbw 계수: allreduce 2(n−1)/n, allgather·reducescatter (n−1)/n, broadcast 1.

- [ ] `mpirun --oversubscribe -np 4 build/hxccl_perf -b 8 -e 64M -f 4 -n 20`과 `-np 8` 실행, 결과를 README에 표로(측정 장비·날짜·명령 포함), 해석은 "측정한 사실"과 "확인하지 않은 설명"을 나눠 적음.
- [ ] 커밋 "Add hxccl_perf and README with M2 measurements".
