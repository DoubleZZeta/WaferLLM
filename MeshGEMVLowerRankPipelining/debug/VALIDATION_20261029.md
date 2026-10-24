# Pipeline verification — 2026-10-29

Current local workspace, SDK 2.10.0 WSE-3 simulator. No kernel changes were needed during this verification.

Eight simulator cases passed, checking output on every PE. Six host unit tests passed. Python/shell syntax checks passed. The production bench command plans all twelve qkvo/gate_up/down cases at 0.2/0.4/0.6/0.8, P=64, G=8.

| Run | Grid | M | R | N | Local Mt/Rt/Nt | Profile | Repeats | Relative L2 |
|---|---|---:|---:|---:|---|---|---:|---:|
| verify_pipeline_20261029_down | 4×4 | 688 | 148 | 256 | 172/37/64 | True | 1 | 0.002031 |
| verify_pipeline_20261029_gate | 4×4 | 256 | 148 | 688 | 64/37/172 | True | 1 | 0.001516 |
| verify_pipeline_20261029_profile | 8×8 | 512 | 48 | 512 | 64/6/64 | True | 1 | 0.001396 |
| verify_pipeline_20261029_profile | 8×8 | 512 | 104 | 512 | 64/13/64 | True | 1 | 0.001417 |
| verify_pipeline_20261029_profile | 8×8 | 512 | 152 | 512 | 64/19/64 | True | 1 | 0.001367 |
| verify_pipeline_20261029_profile | 8×8 | 512 | 208 | 512 | 64/26/64 | True | 1 | 0.001644 |
| verify_pipeline_20261029_repeat | 8×8 | 512 | 152 | 512 | 64/19/64 | False | 5 | 0.001367 |
| verify_pipeline_20261029_repeat | 8×8 | 512 | 208 | 512 | 64/26/64 | False | 5 | 0.001644 |

Six profiled runs produced valid phase metrics and all-PE timeline CSV/JSON files. The decoder checks required event ordering on every PE. Outer timing ends after the final X collective completes. Y intervals include callback delay, and timestamp instrumentation changes timing.

These tests reproduce production local tile dimensions on smaller grids (4×4 and 8×8, G=2). They do not certify full 64×64 execution, full-grid timing, or speedup. The production server must use this updated local source, including aligned Z storage and the profiler/runner updates.

## Subsequent change: match the original diagnostic dispatch structure

The pipeline profiler now has a separate meshgemv_profile_entry/exit_profile
path, unmarked warmups, unconditional measured markers selected by comptime
specialisation, and the loop-end marker in the dispatcher after X completes.
The outer timer still surrounds the measured iteration. Async callbacks retain
one profile-path dispatch; no per-marker runtime warmup checks remain.

After this change, five simulator cases passed on 4x4/G=2:
- M=N=256, R=76 and 104, profile: 2064 and 2156 mean cycles.
- The same two shapes, normal mode, five repetitions: 1858 and 1950 cycles,
  matching the prior normal means. Relative L2 remains 0.001475/0.001415.
- M=256, N=688, R=148, serial profile: passed, relative L2 0.001516;
  both reported compute/communication interval overlaps were zero.

Six host unit tests passed. Timeline CSV/JSON export and event ordering checks
passed. Profiling still uses 16 markers versus seven in the original sequential
low-rank kernel: matching structure does not equalise marker cost. Y intervals
still measure launch-to-ready including callback delay, not pure fabric time.
