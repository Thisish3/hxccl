# Figures for blog post #3 (B)

## Nsight GUI captures

**What and how:** screenshots of the Nsight Systems 2026.3.2 and Nsight Compute 2026.3.1 GUIs, run in WSL through WSLg (2026-11-11). The windows were driven through X11 XTEST with `tools/nsight_gui.py`, the same way as for hxgemm (its LEDGER R6), and each capture was cropped to the relevant part.

**Launch settings:** `QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu`. Nsight Systems 2026.3.2 crashed at startup without them (EGL initialization failed).

**Limit (LEDGER R21):** WSL recorded no GPU kernel or memcpy trace. So every range in the Nsight Systems captures is CPU-side time:
- `reduce` is only the kernel launch.
- A copy's GPU time shows up inside `cudaStreamSynchronize`.

| File | Source | What it shows |
|---|---|---|
| `fig_b_nsys_64k.png` | `../2026-11-11_b_nsys_64K_r0.nsys-rep`, call 16 of 25 (median length), whole call selected | One 64 KiB device AllReduce on rank 0, np2.<br>• `hxccl:allreduce [3.581 ms]`; the selection reads `Duration: 3.576 ms`.<br>• The first `ipc:copy [2.080 ms]` is almost all one `cudaStreamSynchronize`.<br>• Before it comes `cudaMallocAsync`; after it, three short copies. |
| `fig_b_nsys_64m.png` | `../2026-11-11_b_nsys_64M_r0.nsys-rep`, call 11 of 25, whole call selected | 64 MiB: `hxccl:allreduce [11.576 ms]`.<br>• The first `ipc:copy [2.532 ms]` is again long.<br>• It is followed by 31 short chunk copies (4 MiB chunks). |
| `fig_b_nsys_r0r1.png` | Multi-report view of `../2026-11-11_b_nsys_4M_r0.nsys-rep` and `_r1.nsys-rep`, call 6 of 25 | Both ranks on one time axis (File → New multi-report view).<br>• Rank 1's first `ipc:copy [2.103 ms]` is one long `cudaStreamSynchronize`.<br>• Rank 0 makes no CUDA call during it and starts its first copy only after it ends.<br>• The selection marks that wait on rank 0: `Duration: 2.188 ms`. |
| `fig_b_ncu_sol.png` | `../2026-11-11_b_ncu_reduce_full.ncu-repz`, Details → GPU Speed Of Light | `k_reduce<float>`, fp32 sum, 1<<20 elements: 32.16 µs, DRAM Throughput 89.63 %.<br>• The measured clocks (DRAM 6.78 GHz, SM 1.47 GHz) are in the "Command line profiler metrics" rows.<br>• The "frequencies not fixed by the profiler" warning comes from `--clock-control none`: the SM clock was locked with `nvidia-smi -lgc 1500,1500` instead. |
| `fig_b_ncu_memory.png` | same report, Details → Memory Workload Analysis → Memory Chart | Device memory traffic: 8.39 MB read, 4.12 MB written (389.06 GB/s in the section's table).<br>• L1 hit rate 32.74 %, L2 hit rate 33.54 %. |

## Supplement: drawn from the report data

Before the GUI capture worked, `scripts/plot_b_timeline.py` drew the same three views with matplotlib, from the reports' SQLite export (NVTX and CUDA API ranges). **These are not GUI screenshots.** They show the same calls with the same numbers and are kept as a cross-check: `fig_b_timeline_64k.png`, `fig_b_timeline_64m.png`, `fig_b_timeline_4m_r0r1.png`.
