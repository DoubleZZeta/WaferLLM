# Profiled grid comparison

Source archive: profiled_sweep_all_logs_20261015_194347_UTC.zip
Selected runs: `profile_full_dense64_20261003` and `profile_full_rect_20261003`.

- 31 completed low-rank cases; 3 completed dense baseline cases.
- 8 complete grid/ratio block totals. Partial blocks are not plotted.
- Fixed dense 64x64 baseline: 17,973 profiled mean cycles for the seven projections.
- Speedup = dense mean cycles / low-rank mean cycles; block cycles = 4*qkvo + 2*gate_up + down.
- qkvo denotes a representative Q/K/V/O projection; gate_up represents gate and up. These are not measurements of individual transformer depth indices.
- All selected cases use phase profiling and one measured repetition. No unprofiled runs or duplicate smoke cases are mixed in.
- Mean cycles are used throughout; max-PE phase costs are not summed.
- Communication is measured all-reduce time, including broadcast and waiting. It excludes host transfers and is not a pure link-bandwidth measurement.
- Communication share uses the inner profiled iteration, whose compute + reductions + control sum is checked. Outer timed cycles are used for speedup and include extra timing/exit work.
- No output redistribution or other decoder operations are included in block totals.
- Target-ratio fields were blank in the uploaded low-rank CSV. Labels are recovered from the agreed exact rank schedule: qkvo 384/768/1280/1664; gate_up/down 640/1152/1792/2432 correspond to targets 0.2/0.4/0.6/0.8. Actual ratios remain available in completed_projections.csv.
- Eight PEs per local group: grids 64x64 / 32x128 / 128x32 use group counts (8,8) / (4,16) / (16,4).
- Missing values are not filled, extrapolated, or treated as zero.

Figures are provided as PNG and vector PDF. The CSVs preserve per-case source JSON paths.
Regenerate with `python3 results/plot_profiled_sweep.py` from the repository root.
