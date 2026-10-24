# P/G/payload profiles using the existing GEMV runners

Run from the SDK environment, in this directory:

```bash
bash run_pgl_profile.sh pgl_v2 --dry-run
bash run_pgl_profile.sh pgl_v2
```

The script executes sequentially and resumes completed rows. It runs 13
configurations in each of three variants, plus dense Y at P=64, G=8,
L=172 (40 cases total: 14 dense, 13 low-rank Y, 13 low-rank X). Grids are square.
G is the number of groups on each axis, not group size.

| Sweep | Values | Fixed |
|---|---|---|
| P | 16,32,64,96,128 | G=8, L=64 |
| G | 2,4,8,16,32 | P=64, L=64 |
| L | 16,33,48,64,80 | P=64, G=8 |

The reference P=64,G=8,L=64 is executed only once per variant.
The additional dense case uses M=4096, N=11008 and is appended to the same
dense results file. No low-rank L=172 case is added.

| Variant | M | N | R | Phase to compare |
|---|---|---|---|---|
| Dense | 64P | LP | — | dense_y_allreduce |
| Low-rank Y | (L+1)P | (L+1)P | LP | stage1_y_allreduce |
| Low-rank X | 64P | LP | 8P | stage2_x_allreduce |

These execute full GEMVs and use their existing phase profiler, including
broadcast and in-kernel waiting. They are not isolated collective tests.
The Y probes have rank close to the matrix dimensions: they are valid
factorized products, but do not save arithmetic. Do not interpret their
end-to-end cycle counts as compression speedups.

Largest FP16 matrix allocations per PE are 22,016 bytes (dense), 25,920
bytes (low-rank Y), and 2,304 bytes (low-rank X), plus vectors/code/runtime.
The previous low-rank L=172/256 Y probes are intentionally excluded because their
factor tiles cannot fit. Smaller M alone cannot solve this under R<=M,N.

All 40 cases pass runner planning. The maximum local Y tiles (Mt=Nt=81,
Rt=80) compiled and passed the SDK simulator on 4x4 with G=2, profiling on:
relative L2 error 0.002037. This checks local memory feasibility, not the full
production grid configurations, which remain to be run.

Results: sweeps/PREFIX_dense/dense.csv and
sweeps/PREFIX_y/fused-rectangular.csv, sweeps/PREFIX_x/fused-rectangular.csv.
Per-case phase CSV/JSON files are in each run's logs directory.
Existing profile mode measures one iteration after one warmup; do not
mistake this for five statistical repetitions. Inspect status/error_message
in result CSVs: a completed script is not proof every case passed.
