# Fused low-rank MeshGEMV

This directory evaluates a low-rank approximation without returning the
intermediate vector to the host:

```text
Z = X @ A      X: [1,M], A: [M,R]
Y = Z @ B      Z: [1,R], B: [R,N]
```

`Z` remains in PE memory between the two GEMVs. This is the key difference
from `MeshGEMVSparse`, which launches two separate artifacts and transfers
the intermediate through the host.

## Data layout

- PE row `py` stores one `M/P` block of `X`.
- PE `(py, px)` stores `A[py input block, px rank block]`.
- Stage 1 reduces along Y, leaving rank block `px` replicated down column `px`.
- PE `(py, px)` stores `B[px rank block, py output block]`.
- Stage 2 reduces along X, leaving output block `py` replicated across row `py`.

Both stages use separate, statically resident two-level reductions. PEs first
reduce concurrently inside groups, then the group roots reduce among
themselves. Stage 1 has dedicated Y colors/queues and stage 2 has dedicated X
colors/queues, so the timed loop selects a collective without changing routes.

Each level is an alternating-color chain rather than a balanced binary tree.
Its dependency depth is approximately `P/group_num + group_num`; choosing a
group count near `sqrt(P)` balances the two levels for larger grids.

## Run in the simulator

All dimensions must be divisible by `P`; `R` cannot exceed `min(M,N)`.
`group_num` must be positive and divide `P`.

```bash
cd MeshGEMVSparseFused
bash run_sim.sh P M R N group_num

# Small correctness example
bash run_sim.sh 4 64 16 64 2
```

To attribute one warmed iteration to local compute, the two collectives, and
loop/control overhead, add the profiling flag:

```bash
bash run_sim.sh 4 512 128 512 2 --profile-phases
```

Profiling inserts timestamp markers and therefore perturbs that run. Use the
ordinary command for comparable benchmark totals; use the profile only to see
where cycles are spent. `control_and_markers` includes loop dispatch and the
timestamp instructions themselves.

The launcher checks two distinct errors:

- `device_vs_host_svd_relative_l2_error` compares the WSE result with the host
  factorized operation `X @ A @ B`; this should be small;
- `device_vs_host_dense_relative_l2_error` compares the WSE result directly
  with the dense host operation `X @ W`; it includes low-rank approximation
  error and can be large for the deliberately random test matrix.

## Important files

- `src/meshgemv.csl`: both local GEMVs and the fused execution sequence.
- `src/comm_lib/comm_pe.csl`: resident Y and X two-level collectives.
- `src/layout.csl`: PE program, colors, and exported symbols.
- `host_common.py`: SVD factor generation and PE-major tensor tiling.
- `launch_sim.py`: transfers inputs once, launches one kernel, reconstructs Y,
  validates it, and reports cycles.
- `run_sim.sh` and `run_device.sh`: original single-case simulator/device entry
  points retained from WaferLLM.
- `run_benchmark.sh`: the only benchmark launcher added by this project.
- `benchmark_fused.py`: selects cases, runs comparisons, and writes CSVs.

## Run on WSE-3

With appliance credentials and the SDK environment configured:

```bash
bash run_device.sh P M R N group_num
```

The simulator should be used first. Device compilation uses the same source
and writes the artifact id under `compile_out/`.

## Benchmark sweep

`run_benchmark.sh` is the single entry point for benchmark sweeps. It starts a
detached tmux session and calls `benchmark_fused.py`. The Python program
compares fused cycles against dense and host-mediated low-rank cycles, reuses
matching rows from `MeshGEMVSparse/benchmark_results_simple.csv`, and appends
each result immediately so interrupted runs can resume.

```bash
# Test the same configurations as the existing unfused benchmark
bash run_benchmark.sh matched --match-baseline

# A smaller custom sweep
bash run_benchmark.sh small \
  --sizes 128,256,512 --retained-work-ratios .25,.5,.75 --groups 2

# Write the verbose rather than simple result schema
bash run_benchmark.sh detailed --match-baseline --verbose

# Preview without starting tmux or running kernels
python3 benchmark_fused.py --run-name preview --match-baseline --dry-run
```

Every invocation owns one subfolder under `benchmark/`:

```text
benchmark/<run_name>/
├── results.csv       # simple by default; verbose with --verbose
├── break_even.csv    # first tested winning square size per series
├── run.log           # complete tmux console output
└── logs/             # compiler/simulator output for individual cases
```

The default simple CSV contains only parameters, mean cycles, the two fused
comparisons, their speedups, and the two accuracy checks. `--verbose` adds the
failure message, PE count, rank/compression diagnostics, and maximum cycles.
Do not resume the same run once with simple mode and once with verbose mode;
the script detects the schema mismatch and asks for a new run name.

Monitor a run with:

```bash
tmux attach -t fused_matched
# Detach without stopping it: Ctrl-b, then d
```

For this square-only sweep, `square_matrix_size` means `M=N` and
`retained_work_ratio = 2R/M`. `--retained-work-ratios` selects candidate ranks
with `R = retained_work_ratio * M / 2`; combinations that do not produce an
integer rank are skipped rather than rounded. Use `--ranks` when an exact rank
is required. Speedups and faster/slower booleans are calculated from mean
cycles so they correspond directly to the three mean-cycle columns in the
simple CSV.

## Simulator upload fix

The simulator launcher uses bounded uploads with a device readback after each
batch to prevent SDK 2.10 ingress-ring overflow (HCF on large matrices).
Each kernel folder includes `sim_transfers.py` so SDK wrappers can mount the
folder independently. Uploads finish before kernel timing, and this fix does
not change the CSL computation. Llama2Projections uses this fix by default.
