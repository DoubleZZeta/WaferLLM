# Llama-2 7B projection benchmark

Simulator benchmark comparing dense `MeshGEMV` with fused low-rank
`MeshGEMVLowerRank` on the linear-projection shapes of one Llama-2 7B block:

| Group     | GEMVs per block | Shape                  |
|-----------|-----------------|------------------------|
| `qkvo`    | 4               | `[1,4096]  @ [4096,4096]`  |
| `gate_up` | 2               | `[1,4096]  @ [4096,11008]` |
| `down`    | 1               | `[1,11008] @ [11008,4096]` |

Dense and fused are separate commands with separate result files, so either
can be run, resumed, or rerun without the other. The kernels themselves stay
in their own folders; this folder only drives them.

## Running

```bash
cd Llama2Projections

# Dense baseline: one case per projection group
bash run.sh dense p64_fixed --P 64 --group-num 8 --suppress-simfab-trace

# Fused low-rank: one case per projection group and retained-work ratio.
# P, group_num, repetitions and hcf_fix are inherited from config.json.
bash run.sh fused p64_fixed --retained-work-ratios .125,.25,.5,.75 --suppress-simfab-trace

# Rebuild the joined summaries on demand
python3 bench.py summary --run-name p64_fixed
```

`run.sh` starts `bench.py` in a detached tmux session named
`llama_<dense|fused>_<run_name>`, adds the SDK to `PATH`, always passes
`--resume`, and appends console output to `runs/<run_name>/<dense|fused>.log`.
The two commands of one run can run at the same time. Use
`tmux attach -t llama_dense_p64_fixed` to watch, and `Ctrl-b d` to detach.

`bench.py` can also be called directly, e.g. inside your own tmux session or
for `--dry-run`. Useful options:

- `--repetitions N`: measured executions per simulator invocation, after one
  warm-up (default 3).
- `--suppress-simfab-trace`: disable bulk instruction/wavelet traces, which
  can otherwise produce multi-GB archives at P=64.
- Both dense and fused launchers use bounded, device-confirmed uploads by
  default. No flag is needed for the HCF fix.
- `--hcf-fix`: legacy selection of the separate `MeshGEMVPatched` snapshot.
  Saved configurations remain supported; `--no-hcf-fix` selects the normal
  implementations, which now also include the fix.
- `--build-root DIR`: put compiled artifacts and simulator logs, the large part
  of a run, under `DIR/runs/<run_name>/` instead of `runs/<run_name>/build/`.
  `DIR` must be inside `/tmp` (see Notes). It is saved in `config.json` and
  inherited by the other command and by resumes; passing a different directory
  later sends new builds there. CSVs and per-case logs stay in `runs/<run_name>/`.
- `--dry-run`: print the planned cases and ranks without compiling or writing.

The SDK location defaults to `../../cs_sdk`; set `CEREBRAS_SDK_DIR` otherwise.

## Output

```text
runs/<run_name>/
├── config.json              # P, group_num, repetitions, hcf_fix; fixed by the first command
├── dense.csv                # one row per projection group
├── fused.csv                # one row per projection group and target ratio
├── projection_summary.csv   # dense joined with fused, per group and ratio
├── block_summary.csv        # multiplicity-weighted totals per ratio
├── dense.log, fused.log     # tmux console output
├── logs/                    # compile and simulator output, one file per case
└── build/                   # compiled artifacts and simulator logs
```

Rerunning a case replaces its row, and `--resume` skips rows already marked
`ok`. Delete the build directory (`runs/<run_name>/build/`, or
`DIR/runs/<run_name>/` with `--build-root`) once a run is finished to reclaim
disk.

### Summaries

A dense result does not depend on the retained-work ratio, so every fused row
is paired with the dense row for its projection group. For each ratio:

```text
dense_block_cycles = 4·dense[qkvo]    + 2·dense[gate_up]    + dense[down]
fused_block_cycles = 4·fused[qkvo, ρ] + 2·fused[gate_up, ρ] + fused[down, ρ]
```

A summary row is `ok` only when all six inputs have succeeded; otherwise it is
`incomplete` and `missing` lists what is absent, e.g. `dense[down]
fused[gate_up]`. A block total is still filled in when its own side is
complete. Both summary files are rebuilt after every case by either command.
The block total estimates the linear-projection portion of a block, not
end-to-end decoder latency.

Rank is chosen as the multiple of `P` closest to `ρ·MN/(M+N)`, so the actual
`retained_work_ratio = R(M+N)/(MN)` is recorded next to the target.

## Parameter sweeps

`sweep.py` runs every combination of the values you give, instead of the fixed
Llama shapes, like the older `MeshGEMVLowerRank/benchmark_fused.py`. Dense and
fused stay separate commands and reuse this folder's launchers, compile paths,
`--hcf-fix` selection and resume behaviour.

```bash
# Dense: every (P, group_num, M, N) combination
bash run_sweep.sh dense square --P 4,8 --group-num 2 --M 256,512 --N 256,512

# Fused: every (P, group_num, M, N) combination for each rank
bash run_sweep.sh fused square --P 4,8 --group-num 2 --M 256,512 --N 256,512 \
  --retained-work-ratios .25,.5,.75 --ranks 64

# Rebuild the summary, or preview a plan without running anything
python3 sweep.py summary --run-name square
python3 sweep.py fused --run-name preview --M 256 --N 512 --ranks 32,64 --dry-run
```

- `--P` and `--group-num` default to 4 and 2; `--M` and `--N` are required.
  Every option except `--repetitions` takes a comma-separated list.
- Fused needs `--ranks`, `--retained-work-ratios`, or both. A ratio targets
  `R(M+N)/(MN)` and is rounded to the nearest multiple of P, as in `bench.py`;
  a rank already planned for a shape is not run twice.
- Combinations the kernels cannot run (group_num not dividing P, M or N not
  divisible by P, R not divisible by P or above min(M,N)) are listed as `SKIP`
  in the plan and not run.
- Only `repetitions` and `hcf_fix` are fixed per run in `config.json`. The
  sweep axes are stored on every row, so later commands in the same run may
  add new values; `--resume` skips cases already marked `ok`.
- `--build-root DIR` works as for `bench.py`; builds go to
  `DIR/sweeps/<run_name>/`.

Output goes to `sweeps/<run_name>/`: `dense.csv` (one row per P, group_num, M,
N), `fused.csv` (one row per P, group_num, M, N, R), and `summary.csv`, which
joins every fused row with its dense row and reports the speedup or what is
`missing`. `logs/`, `build/` and the tmux `dense.log`/`fused.log` are laid out
as for `bench.py`.

## Notes

- `config.json` exists because dense and fused rows are only comparable with
  the same `P`, `group_num`, `repetitions` and upload implementation. A command that asks for
  different values under an existing run name is rejected.
- Inputs are generated directly rather than from a host-side SVD: kernel cycle
  counts only need correctly shaped matrices. The fused launcher checks the
  device result against `X @ A @ B` computed on the host.
- The SDK's `cslc` and `cs_python` wrappers mount only the working directory
  into their container. `bench.py` therefore runs every compile and launch
  from the repository root with repository-relative paths, using each kernel's
  `compile.py --out-dir`.
- Patched entry points are `launch_dense_patched.py` and `launch_fused_patched.py`.
  They reuse the benchmark launchers' computation, validation and CSV reporting,
  and load the upload helper from `MeshGEMVPatched/sim_transfers.py`.
  The patched dense source is compiled from `MeshGEMVPatched/src`; fused still
  compiles `MeshGEMVLowerRank/src`. Normal entry points share
  `MeshGEMV/sim_transfers.py`, containing the same verified upload fix.
- Each case's artifact directory reaches the compiler and the launcher through
  `--out-dir`. The SDK containers mount only the working directory and the
  host's `${TMPDIR:-/tmp}` at `/tmp`. With `--build-root`, both commands run
  with `TMPDIR` unset, so a build root inside `/tmp` has the same path on the
  host and in both containers. Other locations would need extra container
  mounts, which some Singularity installations refuse, so they are rejected.
- `compile.py` removes an old artifact with `shutil.rmtree` rather than `rm`,
  because some servers wrap `rm` in an interactive prompt that fails in tmux.
- Large unfenced uploads can overflow the SDK simulator's 100 MiB ingress ring.
  Normal simulator launchers cap each batch at 16 MiB and read back from every
  destination PE before continuing. These transfers precede kernel timing.
  Debug tools remain under [MeshGEMVPatched](../MeshGEMVPatched/README.md).

## Phase profiling (both runners)

Pass `--profile-phases` to either `run.sh` or `run_sweep.sh`. It selects the
existing instrumented kernel entry point, measuring one iteration after one
warm-up. Omit `--repetitions` or set it to 1. This mode is saved in config.json
and inherited on resume; switching between normal and profiling measurements
requires a new run name. Older configs without this setting mean normal mode.

For qkvo only, from this directory:

```bash
bash run_sweep.sh dense qkvo_profile --P 64 --group-num 8 --M 4096 --N 4096 --profile-phases --suppress-simfab-trace
bash run_sweep.sh fused qkvo_profile --P 64 --group-num 8 --M 4096 --N 4096 --retained-work-ratios .2,.4,.6,.8 --profile-phases --suppress-simfab-trace
```

For all Llama projection shapes, the equivalent runner option is:

```bash
bash run.sh dense llama_profile --P 64 --group-num 8 --profile-phases --suppress-simfab-trace
bash run.sh fused llama_profile --P 64 --group-num 8 --retained-work-ratios .2,.4,.6,.8 --profile-phases --suppress-simfab-trace
```

Each case writes `logs/<case>.phases.csv` and `.phases.json` alongside its log,
under `sweeps/<run>/` or `runs/<run>/`. The JSON includes case dimensions/rank;
the phase CSV contains `phase,mean_cycles,max_pe_cycles`. Dense reports local
compute and Y all-reduce; fused reports XA, Y all-reduce, ZB, and X all-reduce.
Control intervals and aggregate control/iteration metrics are also included.
Do not sum aggregate rows with their component rows, or sum per-phase maxima.
Timestamp markers perturb timing: totals and summaries in a profiling run are
diagnostic, not normal benchmark measurements. Inputs still use the HCF fix.

## Rectangular fused kernel

Both shell runners now accept `fused-rectangular`, backed by
`MeshGEMVLowerRankRectangular` and `launch_fused_rectangular.py`. The launcher
creates the same deterministic direct factors as the square benchmark for the
same M/R/N, avoiding a host SVD. It retains bounded uploads and numerical checks.

From this folder, run all Llama projection shapes on a specified grid:

```bash
bash run.sh fused-rectangular llama_rect --PX 32 --PY 128 --group-num-x 4 --group-num-y 8 --retained-work-ratios .4 --suppress-simfab-trace
```

Select particular shapes with `--projections qkvo` or `--projections gate_up down`.
For arbitrary matrix shapes use the sweep runner:

```bash
bash run_sweep.sh fused-rectangular rect_small --PX 4 --PY 8 --group-num-x 2 --group-num-y 2 --M 64 --N 128 --ranks 16 --suppress-simfab-trace
```

PX/PY and both group counts accept comma-separated lists, as do M/N in sweep
mode. Lists form a Cartesian product, NOT paired layouts. Invalid divisibility
combinations are printed as SKIP. Use separate invocations for specific layout
pairs; `--resume` permits adding new configurations to the same rectangular run.

**For a fair comparison use the same exact rank, not merely the same target
ratio.** Rectangular ranks are rounded to multiples of PX; e.g. gate/up at 0.4
selects R=1184 with PX=32, versus the previous R=1216 with P=64. To reproduce
that earlier rank and profile down on the new layout:

```bash
bash run.sh fused-rectangular down_rect_profile --projections down --PX 32 --PY 128 --group-num-x 4 --group-num-y 8 --ranks 1216 --profile-phases --suppress-simfab-trace
```

qkvo at the previous 0.4 setting uses R=832; gate/up and down use R=1216.
Exact-rank rows leave target_retained_work_ratio blank and record the actual
R(M+N)/(MN). `--repetitions`, `--profile-phases`, `--build-root /tmp/...`,
`--resume`, and `--dry-run` work as in the square runners. Omit repetitions or
set it to 1 for profiling. Both normal and profiling modes include one warm-up.
Preview with `python3 bench.py fused-rectangular ... --dry-run` or
`python3 sweep.py fused-rectangular ... --dry-run`.

Results are `runs/<run>/fused-rectangular.csv` or
`sweeps/<run>/fused-rectangular.csv`. Rows explicitly record Px/Py, both group
counts, total PEs, dimensions, rank, repetitions, profiling mode, errors, and
cycle metrics. Per-case logs, phase CSV/JSON files, and artifacts have layout-
specific names. Resume keys include layout, group counts, dimensions, and rank.
Use a separate run name from square/dense runs; mixing kernel modes is rejected.

Rectangular mode does not generate the square runners' dense speedup or block
summaries. Compare its rows with a deliberately chosen dense baseline, matching
matrix shape, timing mode, repetitions, and accounting for PE counts. Existing
dense/fused commands and their summaries remain available.

## Rectangular dense and the 45-case profiled grid suite

Both `bench.py` / `run.sh` and `sweep.py` / `run_sweep.sh` accept
`dense-rectangular` as well as `fused-rectangular`. Dense uses the separate
`MeshGEMVRectangular/` kernel; it takes `--PX`, `--PY`, `--group-num-x` and
`--group-num-y` but no rank options. Dense partitions M over Y and N over X;
fused partitions M and N over Y and rank over X. Dense uses only the Y group
count for its collective. Each rectangular execution mode needs its own run name.

```bash
bash run.sh dense-rectangular dense_rect_profile \
  --PX 128 --PY 32 --group-num-x 16 --group-num-y 4 \
  --projections qkvo gate_up down --profile-phases --suppress-simfab-trace
bash run_sweep.sh dense-rectangular dense_custom_profile \
  --PX 32 --PY 128 --group-num-x 4 --group-num-y 16 \
  --M 4096 --N 11008 --profile-phases --suppress-simfab-trace
```

The complete experiment can be reviewed and launched from `Llama2Projections/`:

```bash
python3 profile_grid_suite.py --run-name grid45_v1 --dry-run
# Run in tmux to survive SSH disconnects. --jobs counts only this queue.
python3 profile_grid_suite.py --run-name grid45_v1 --jobs 2
# Resume after interruption (completed successful cases are skipped):
python3 profile_grid_suite.py --run-name grid45_v1 --jobs 2 --resume
```

Set `CEREBRAS_SDK_DIR` to the directory containing executable `cslc` and
`cs_python` launchers, or put them on PATH. The default SDK directory is
`../cs_sdk` relative to the repository. Optional `--build-root /tmp/wafer-builds`
is passed to each case. `--runner sweep` exercises the custom-shape runner
instead of the fixed-shape runner, with the same 45 physical cases.

The suite pairs configurations explicitly (it does not form a Cartesian product
of independent PX/PY/group lists):

| PX | PY | Gx | Gy | PEs per group, either axis |
|---:|---:|---:|---:|---:|
| 64 | 64 | 8 | 8 | 8 |
| 32 | 128 | 4 | 16 | 8 |
| 128 | 32 | 16 | 4 | 8 |

It runs qkvo (4096 x 4096), gate_up (4096 x 11008), and down (11008 x 4096).
Each shape/grid has one dense case and four low-rank cases: 9 dense + 36 fused.
Explicit ranks are identical across grids:

| Target retained-work ratio | qkvo rank | gate_up/down rank |
|---:|---:|---:|
| 0.2 | 384 | 640 |
| 0.4 | 768 | 1152 |
| 0.6 | 1280 | 1792 |
| 0.8 | 1664 | 2432 |

All cases use phase profiling, one warmup and one measured iteration, and
suppressed bulk traces. The target ratios are labels, not instructions to
round ranks; actual ratios use `R*(M+N)/(M*N)`. The individual rectangular
commands accept `--rank-target-ratios` paired with `--ranks` to preserve labels.
Dense uses an empty rank/target and actual retained-work ratio 1.

Every case gets its own run, build and log paths. The suite writes
`suites/<name>/manifest.json`, queue logs, `summary.csv`, and `phases.csv`;
merged CSVs refresh after each case completes. Missing and failed cases remain
visible. Completed case outputs are under `runs/` (or `sweeps/`). No files from
prior ZIP packages or other suites are merged.

Collective phase timings include the broadcast; there is no separate broadcast
marker. Profiling adds overhead: do not mix these total cycles with unprofiled
numbers. The dense and fused kernels retain their original different reduction
algorithms, and their final output replication axes differ. Model accuracy is
not measured here: dense is checked against host GEMV, and fused against host
factorized GEMV with directly generated factors.

### Keeping the original dense baseline

Rectangular dense is an optional control, not a replacement for the original
64x64 baseline. To vary only the low-rank configuration, use:

```bash
python3 profile_grid_suite.py --run-name lowrank36_v1 --low-rank-only --jobs 2
```

This runs the same 36 fused cases without rerunning or changing dense. Compare
against the existing original dense 64x64 measurements with matching profiling
mode; if necessary, collect them separately with `run.sh dense --P 64
--group-num 8 --profile-phases`. Keep distinct labels for speedup relative to
fixed dense 64x64, same-grid dense, and best measured dense. The 45-case suite
supports the latter two comparisons as additional controls.
