cmake_minimum_required(VERSION 3.20)
project(hxccl CXX)
option(HXCCL_CUDA "Build the GPU path (CUDA IPC transport, device reductions)" OFF)
if(HXCCL_CUDA)
  if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  find_package(CUDAToolkit REQUIRED)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(MPI REQUIRED COMPONENTS CXX)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

file(GLOB HXCCL_SRC CONFIGURE_DEPENDS src/*.cc)
if(HXCCL_CUDA)
  file(GLOB HXCCL_CU CONFIGURE_DEPENDS src/*.cu)
  list(APPEND HXCCL_SRC ${HXCCL_CU})
  list(FILTER HXCCL_SRC EXCLUDE REGEX "/memspace\\.cc$")  # the CPU-only ops_for
else()
  list(FILTER HXCCL_SRC EXCLUDE REGEX "(memops_device|transport_ipc|memspace_cuda)\\.cc$")
endif()
add_library(hxccl STATIC ${HXCCL_SRC})
# the PyTorch extension (python/) links libhxccl.a into a shared object
set_target_properties(hxccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hxccl PUBLIC include PRIVATE src)
target_link_libraries(hxccl PUBLIC MPI::MPI_CXX)
if(HXCCL_CUDA)
  target_compile_definitions(hxccl PUBLIC HXCCL_CUDA=1)
  # cuda_driver: memspace_cuda.cc asks whether a CUDA context exists without creating one
  target_link_libraries(hxccl PUBLIC CUDA::cudart CUDA::cuda_driver)
  # NVTX3 is header-only; CMake < 3.25 has no CUDA::nvtx3 and the cudart include dir has it
  if(TARGET CUDA::nvtx3)
    target_link_libraries(hxccl PUBLIC CUDA::nvtx3)
  endif()
endif()

enable_testing()
function(hxccl_mpi_test name)
  add_executable(${name} tests/${name}.cc)
  target_include_directories(${name} PRIVATE src tests)
  target_link_libraries(${name} PRIVATE hxccl)
  foreach(np 1 2 3 4 5 8)
    add_test(NAME ${name}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} ${np}
                     $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np} PROPERTIES TIMEOUT 60)
    # the same test with hxcclCommInitMPI building socket communicators (spec E 4)
    add_test(NAME ${name}_np${np}_sock
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -x HXCCL_BOOTSTRAP=socket
                     ${MPIEXEC_NUMPROC_FLAG} ${np} $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np}_sock PROPERTIES TIMEOUT 60)
  endforeach()
endfunction()
file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/test_*.cc)
foreach(src ${TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  hxccl_mpi_test(${name})
endforeach()

function(hxccl_gpu_test name)
  add_executable(${name} tests/${name}.cc)
  target_include_directories(${name} PRIVATE src tests)
  target_link_libraries(${name} PRIVATE hxccl CUDA::cuda_driver)  # gpu_host_no_context probes contexts
  foreach(np 1 2 4)
    add_test(NAME ${name}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} ${np}
                     $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np} PROPERTIES TIMEOUT 120)
    add_test(NAME ${name}_np${np}_chunk4k
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -x HXCCL_IPC_CHUNK=4096
                     ${MPIEXEC_NUMPROC_FLAG} ${np} $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np}_chunk4k PROPERTIES TIMEOUT 120)
    add_test(NAME ${name}_np${np}_sock
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -x HXCCL_BOOTSTRAP=socket
                     ${MPIEXEC_NUMPROC_FLAG} ${np} $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np}_sock PROPERTIES TIMEOUT 120)
    # device data through host memory instead of CUDA IPC (the fallback path)
    add_test(NAME ${name}_np${np}_staged
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -x HXCCL_DEVICE_TRANSPORT=staged
                     ${MPIEXEC_NUMPROC_FLAG} ${np} $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np}_staged PROPERTIES TIMEOUT 120)
    add_test(NAME ${name}_np${np}_staged_sock
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -x HXCCL_DEVICE_TRANSPORT=staged
                     -x HXCCL_BOOTSTRAP=socket ${MPIEXEC_NUMPROC_FLAG} ${np}
                     $<TARGET_FILE:${name}>)
    set_tests_properties(${name}_np${np}_staged_sock PROPERTIES TIMEOUT 120)
  endforeach()
endfunction()
# Reproductions of the Mac review findings I1-I5 (docs/desktop/b-review-fixes.md). A hang shows
# up as a TIMEOUT failure, so every one has a short limit. FIRST_ENV is given to rank 0 only
# (MPMD), the other ranks run without it. Each also runs on socket communicators.
function(hxccl_repro_test name np first_env)
  set(variant "${ARGV3}")  # optional: a second registration of the same test
  if(NOT TARGET ${name})
    add_executable(${name} tests/${name}.cc)
    target_include_directories(${name} PRIVATE src tests)
    target_link_libraries(${name} PRIVATE hxccl)
  endif()
  foreach(boot mpi socket)
    set(b "")
    if(boot STREQUAL "socket")
      set(b -x HXCCL_BOOTSTRAP=socket)
    endif()
    if(first_env STREQUAL "")
      set(cmd ${MPIEXEC_EXECUTABLE} --oversubscribe ${b} ${MPIEXEC_NUMPROC_FLAG} ${np}
              $<TARGET_FILE:${name}>)
    else()
      math(EXPR rest "${np} - 1")
      set(cmd ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 1 ${b}
              -x ${first_env} $<TARGET_FILE:${name}> : ${MPIEXEC_NUMPROC_FLAG} ${rest} ${b}
              $<TARGET_FILE:${name}>)
    endif()
    add_test(NAME ${name}${variant}_np${np}_${boot} COMMAND ${cmd})
    set_tests_properties(${name}${variant}_np${np}_${boot} PROPERTIES TIMEOUT 60)
  endforeach()
endfunction()
if(HXCCL_CUDA)
  hxccl_repro_test(repro_p2p_first_partial 4 "")
  hxccl_repro_test(repro_init_failure 2 "HXCCL_IPC_CHUNK=99999999999999")
  hxccl_repro_test(repro_init_failure 2 "HXCCL_TEST_FAIL_DEVICE_INIT=1" _inject)
  hxccl_repro_test(repro_two_comm_group 2 "")
  hxccl_repro_test(repro_device_scalar 2 "")
  hxccl_repro_test(repro_peer_failure 2 "HXCCL_TEST_FAIL_DEVICE_COPY=1")
  hxccl_repro_test(repro_group_truncation 2 "")
endif()

if(HXCCL_CUDA)
  file(GLOB GPU_TEST_SRC CONFIGURE_DEPENDS tests/gpu_*.cc)
  foreach(src ${GPU_TEST_SRC})
    get_filename_component(name ${src} NAME_WE)
    hxccl_gpu_test(${name})
  endforeach()
endif()

add_executable(hxccl_perf bench/hxccl_perf.cc)
target_link_libraries(hxccl_perf PRIVATE hxccl)
