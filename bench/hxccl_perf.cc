// nccl-tests style sweep: hxccl vs MPI, same buffers, rank-max time per iteration,
// median over iterations. busbw factors from nccl-tests doc/PERFORMANCE.md and the
// *GetBw functions in nccl-tests src/*.cu.
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "hxccl.h"

namespace {

size_t parse_size(const char* s) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (*end == 'K' || *end == 'k') v *= 1024;
  if (*end == 'M' || *end == 'm') v *= 1024 * 1024;
  if (*end == 'G' || *end == 'g') v *= 1024.0 * 1024 * 1024;
  return static_cast<size_t>(v);
}

double median_us(const std::function<void()>& fn, int iters) {
  for (int i = 0; i < 3; ++i) fn();
  std::vector<double> t(iters);
  for (int i = 0; i < iters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    fn();
    double mine = MPI_Wtime() - t0, worst = 0;
    MPI_Allreduce(&mine, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    t[i] = worst * 1e6;
  }
  std::nth_element(t.begin(), t.begin() + iters / 2, t.end());
  return t[iters / 2];
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  size_t minb = 8, maxb = 64 << 20;
  int factor = 2, iters = 20;
  std::string op = "allreduce";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "-b") minb = parse_size(argv[i + 1]);
    else if (k == "-e") maxb = parse_size(argv[i + 1]);
    else if (k == "-f") factor = std::atoi(argv[i + 1]);
    else if (k == "-n") iters = std::atoi(argv[i + 1]);
    else if (k == "-o") op = argv[i + 1];
  }
  hxcclComm_t c;
  hxcclCommInitMPI(&c, MPI_COMM_WORLD);
  const double bus = op == "allreduce" ? 2.0 * (n - 1) / n
                     : op == "broadcast" || op == "reduce" || op == "sendrecv" ? 1.0
                                                                              : 1.0 * (n - 1) / n;
  // ops whose size is count-per-rank x n (nccl-tests: S = count * nranks * typesize)
  const bool per_rank = op == "allgather" || op == "reducescatter" || op == "gather" ||
                        op == "scatter" || op == "alltoall";
  if (rank == 0) {
    std::printf("# hxccl_perf op=%s nranks=%d iters=%d (median of rank-max)\n", op.c_str(), n,
                iters);
    std::printf("%12s %12s %10s %9s %9s %10s %9s %9s %6s\n", "size(B)", "count", "hxccl_us",
                "algbw", "busbw", "mpi_us", "algbw", "busbw", "algo");
  }
  for (size_t bytes = minb; bytes <= maxb; bytes *= factor) {
    // S = total array size (nccl-tests); allgather/reducescatter use bytes/n per rank
    size_t count = bytes / sizeof(float);
    size_t per = std::max<size_t>(1, count / n);
    std::vector<float> a(op == "allgather" || op == "gather" ? per : count, 1.0f), b(count + per * n, 0.0f);
    std::function<void()> h, m;
    if (op == "allreduce") {
      h = [&] { hxcclAllReduce(a.data(), b.data(), count, hxcclFloat32, hxcclSum, c, nullptr); };
      m = [&] {
        MPI_Allreduce(a.data(), b.data(), (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      };
    } else if (op == "allgather") {
      h = [&] { hxcclAllGather(a.data(), b.data(), per, hxcclFloat32, c, nullptr); };
      m = [&] {
        MPI_Allgather(a.data(), (int)per, MPI_FLOAT, b.data(), (int)per, MPI_FLOAT,
                      MPI_COMM_WORLD);
      };
    } else if (op == "reducescatter") {
      a.assign(per * n, 1.0f);
      h = [&] {
        hxcclReduceScatter(a.data(), b.data(), per, hxcclFloat32, hxcclSum, c, nullptr);
      };
      m = [&] {
        MPI_Reduce_scatter_block(a.data(), b.data(), (int)per, MPI_FLOAT, MPI_SUM,
                                 MPI_COMM_WORLD);
      };
    } else if (op == "reduce") {
      h = [&] { hxcclReduce(a.data(), b.data(), count, hxcclFloat32, hxcclSum, 0, c, nullptr); };
      m = [&] {
        MPI_Reduce(a.data(), b.data(), (int)count, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      };
    } else if (op == "gather") {
      h = [&] { hxcclGather(a.data(), b.data(), per, hxcclFloat32, 0, c, nullptr); };
      m = [&] {
        MPI_Gather(a.data(), (int)per, MPI_FLOAT, b.data(), (int)per, MPI_FLOAT, 0,
                   MPI_COMM_WORLD);
      };
    } else if (op == "scatter") {
      a.assign(per * n, 1.0f);
      h = [&] { hxcclScatter(a.data(), b.data(), per, hxcclFloat32, 0, c, nullptr); };
      m = [&] {
        MPI_Scatter(a.data(), (int)per, MPI_FLOAT, b.data(), (int)per, MPI_FLOAT, 0,
                    MPI_COMM_WORLD);
      };
    } else if (op == "alltoall") {
      a.assign(per * n, 1.0f);
      h = [&] { hxcclAlltoAll(a.data(), b.data(), per, hxcclFloat32, c, nullptr); };
      m = [&] {
        MPI_Alltoall(a.data(), (int)per, MPI_FLOAT, b.data(), (int)per, MPI_FLOAT,
                     MPI_COMM_WORLD);
      };
    } else if (op == "sendrecv") {
      const int right = (rank + 1) % n, left = (rank + n - 1) % n;
      h = [&] {
        hxcclGroupStart();
        hxcclSend(a.data(), count, hxcclFloat32, right, c, nullptr);
        hxcclRecv(b.data(), count, hxcclFloat32, left, c, nullptr);
        hxcclGroupEnd();
      };
      m = [&] {
        MPI_Sendrecv(a.data(), (int)count, MPI_FLOAT, right, 0, b.data(), (int)count,
                     MPI_FLOAT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      };
    } else {
      h = [&] { hxcclBroadcast(a.data(), a.data(), count, hxcclFloat32, 0, c, nullptr); };
      m = [&] { MPI_Bcast(a.data(), (int)count, MPI_FLOAT, 0, MPI_COMM_WORLD); };
    }
    const double S =
        static_cast<double>(per_rank ? per * n * sizeof(float) : count * sizeof(float));
    const double hu = median_us(h, iters), mu = median_us(m, iters);
    const char* algo = "-";
    if (op == "allreduce") {
      const char* e = std::getenv("HXCCL_ALGO");
      algo = e ? e : (count * sizeof(float) < 64 * 1024 ? "rd" : "ring");
    }
    if (rank == 0)
      std::printf("%12.0f %12zu %10.1f %9.3f %9.3f %10.1f %9.3f %9.3f %6s\n", S,
                  per_rank ? per : count, hu, S / hu / 1e3,
                  S / hu / 1e3 * bus, mu, S / mu / 1e3, S / mu / 1e3 * bus, algo);
  }
  hxcclCommDestroy(c);
  MPI_Finalize();
  return 0;
}
