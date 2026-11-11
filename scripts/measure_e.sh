#!/usr/bin/env bash
# E measurements (spec E 5): AllReduce over MPI vs socket communicators, host and device buffers,
# plus the torch backend under torchrun. Lock the SM clock first, from a Windows admin prompt:
#   nvidia-smi -lgc 1500,1500
# usage: bash scripts/measure_e.sh DATE   (repo root, build-cuda built)
set -euo pipefail
date=${1:?date}; out=docs/results; mkdir -p $out
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 1 > $out/${date}_e_clocks.csv &
log=$!
trap 'kill $log 2>/dev/null || true' EXIT
sleep 2

perf() {  # perf NP NAME [env...] -- [hxccl_perf flags]
  local np=$1 name=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=(-x "$1"); shift; done
  shift
  timeout 1800 mpirun --oversubscribe "${envs[@]}" -np $np build-cuda/hxccl_perf \
    -o allreduce -b 4K -e 256M -f 4 -n 20 "$@" | tee $out/${date}_e_allreduce_np${np}_${name}.txt
}
for np in 2 4; do
  perf $np host_mpi --
  perf $np host_socket HXCCL_BOOTSTRAP=socket --
  perf $np device_ipc_mpi -- -d
  perf $np device_ipc_socket HXCCL_BOOTSTRAP=socket -- -d
  perf $np device_staged_socket HXCCL_BOOTSTRAP=socket HXCCL_DEVICE_TRANSPORT=staged -- -d
done

# the torch backend under torchrun (no MPI_Init, so CUDA IPC falls back to the staged path)
csv=$out/${date}_e_torch_allreduce.csv
echo "backend,launcher,nranks,bytes,median_us,GBps" > $csv
for np in 2 4; do
  timeout 1800 ~/venvs/hxgemm/bin/python -m torch.distributed.run --nproc-per-node $np \
    python/bench_allreduce.py --backend hxccl --out /tmp/e_torch_$np.csv | grep "np="
  sed "s/^hxccl,/hxccl,torchrun,/" /tmp/e_torch_$np.csv >> $csv
  rm -f /tmp/e_torch_$np.csv
done

kill $log
awk -F', ' '$2+0 != 1500 {bad++} END {print (bad ? bad " samples off 1500 MHz" : "all samples at 1500 MHz"), NR " samples"}' \
  $out/${date}_e_clocks.csv
