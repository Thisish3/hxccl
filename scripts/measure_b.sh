#!/usr/bin/env bash
# Clock log + allreduce sweeps for host/device paths at np 2 and 4 (spec 6, plan Task 6).
# Lock the SM clock first, from a Windows admin prompt: nvidia-smi -lgc 1500,1500
set -euo pipefail
date=${1:?date}; out=docs/results; mkdir -p $out
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 1 > $out/${date}_b_clocks.csv &
log=$!
trap 'kill $log 2>/dev/null || true' EXIT
sleep 2
for np in 2 4; do
  for mem in host device; do
    flag=$([ $mem = device ] && echo -d || true)
    mpirun --oversubscribe -np $np build-cuda/hxccl_perf -o allreduce -b 4K -e 256M -f 2 -n 20 $flag \
      | tee $out/${date}_b_allreduce_np${np}_${mem}.txt
  done
done
kill $log
awk -F', ' '$2+0 != 1500 {bad++} END {print (bad ? bad " samples off 1500 MHz" : "all samples at 1500 MHz")}' \
  $out/${date}_b_clocks.csv
