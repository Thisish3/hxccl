# Figures for blog post #3 (B)

**These are not Nsight GUI screenshots.** The desktop session could not drive the Windows GUI. So, as `docs/desktop/nsight-captures.md` allows, the figures are drawn with matplotlib (`scripts/plot_b_timeline.py`) from the Nsight Systems reports' data: the NVTX ranges and CUDA runtime API calls in each report's SQLite export (`nsys export --type sqlite`).

The limit applies to every figure: WSL recorded no GPU kernel or memcpy trace (LEDGER R21). So every bar is CPU-side time.
- `reduce` is only the kernel launch.
- A copy's GPU time shows up inside `cudaStreamSynchronize`.

| File | Source report | What it shows |
|---|---|---|
| `fig_b_timeline_64k.png` | `2026-11-11_b_nsys_64K_r0.nsys-rep`, call 16 of 25 (median length) | One 64 KiB device AllReduce on rank 0 (np2) takes 3.58 ms.<br>• `cudaMallocAsync`: 414 µs.<br>• The first of the 4 `ipc:copy` ranges: 2080 µs, almost all of it one `cudaStreamSynchronize`.<br>• The other copies: about 150 µs each.<br>Why that first synchronize takes 2 ms is not confirmed. |
| `fig_b_timeline_64m.png` | `2026-11-11_b_nsys_64M_r0.nsys-rep`, call 11 of 25 | 64 MiB takes 11.58 ms, with 32 `ipc:copy` ranges (4 MiB chunks).<br>• The first copy: again long, 2532 µs.<br>• The other 31: 0.1–0.3 ms each. |
| `fig_b_timeline_4m_r0r1.png` | `2026-11-11_b_nsys_4M_r{0,1}.nsys-rep`, call 6 of 25 on both | The same 4 MiB call on both ranks, aligned by each report's `TARGET_INFO_SESSION_START_TIME` (two nsys processes on one machine; the alignment was not checked against another clock).<br>• Rank 1's first copy runs 0.43–2.53 ms.<br>• Rank 0 starts its first copy only after that, at 2.55 ms.<br>So one rank's long first synchronize holds up its peer through the READY/DONE handshake. |

Nsight Compute: no figure. The `.ncu-repz` report (`../2026-11-11_b_ncu_reduce_full.ncu-repz`) can be opened in the GUI; its details page is saved as text in `../2026-11-11_b_ncu_reduce_full.details.txt`. Numbers are in LEDGER R38.
