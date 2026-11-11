#!/usr/bin/env bash
# Nsight Systems per rank for device AllReduce at 64K, 4M, 64M (spec 6, plan Task 6).
set -euo pipefail
date=${1:?date}; out=docs/results; mkdir -p $out
for size in 64K 4M 64M; do
  mpirun --oversubscribe -np 2 nsys profile -t cuda,nvtx,mpi \
    -o $out/${date}_b_nsys_${size}_r%q{OMPI_COMM_WORLD_RANK} -f true \
    build-cuda/hxccl_perf -o allreduce -b $size -e $size -n 20 -d
  for r in 0 1; do
    nsys stats -r nvtx_sum -f csv -o $out/${date}_b_nsys_${size}_r$r $out/${date}_b_nsys_${size}_r$r.nsys-rep
  done
done
