# Rectangular fused low-rank MeshGEMV

Computes `Z = X @ A`, then `Y = Z @ B` on a grid of PX columns and PY rows.
The intermediate stays on wafer. Each PE stores X[Mt], A[Mt,Rt], B[Rt,Nt],
where Mt=M/PY, Rt=R/PX and Nt=N/PY. A uses input-row/rank-column block
placement; B uses rank-column/output-row placement. Stage 1 all-reduces Rt
values vertically; stage 2 all-reduces Nt values horizontally.

M and N must be divisible by PY; R must be divisible by PX and no greater
than min(M,N). Each group count must divide its own axis. NumPy PE arrays
use [PY,PX,elements], while SDK copies take width=PX and height=PY.

## Small simulator checks

With cslc and cs_python available:

```bash
bash MeshGEMVLowerRankRectangular/run_sim.sh 4 8 64 16 128 2 2
bash MeshGEMVLowerRankRectangular/run_sim.sh 8 4 64 16 128 2 2 --profile-phases
```

Normal runs measure 10 iterations after one warm-up. Profiling measures one
instrumented iteration after one warm-up. Use normal timing for speedups.
The host uses SVD-generated factors; a large host SVD can be expensive.

The standalone launcher also accepts --out-dir and --suppress-simfab-trace.
For example, from the repository root:

```bash
python3 MeshGEMVLowerRankRectangular/compile.py --mode sim --PX 4 --PY 8 --M 64 --R 16 --N 128 --group_num_x 2 --group_num_y 2 --out-dir MeshGEMVLowerRankRectangular/out_test
cs_python MeshGEMVLowerRankRectangular/launch_sim.py --PX 4 --PY 8 --M 64 --R 16 --N 128 --group_num_x 2 --group_num_y 2 --out-dir MeshGEMVLowerRankRectangular/out_test --suppress-simfab-trace
```

Simulator uploads retain the bounded, device-confirmed HCF workaround.
The host checks the kernel against FP32 X@A@B separately from SVD approximation
error relative to X@W. Llama2Projections selects this variant with the fused-rectangular command;
see its README for direct-factor benchmarks, exact ranks, and profiling.

## Verification

```bash
cs_python -m unittest discover -s MeshGEMVLowerRankRectangular -p test_host_common.py
```

Inside an SDK environment where cslc and Python runtime bindings are directly
available, `python MeshGEMVLowerRankRectangular/debug/smoke.py` compiles and
runs two rectangle orientations, a square, and a one-PE-group/one-group case.
Artifacts are written below debug/runs. No accelerator access is needed.

## Hardware entry point

`bash MeshGEMVLowerRankRectangular/run_device.sh PX PY M R N Gx Gy` uses the
same rectangular mapping. Requires appliance credentials. Hardware execution
has not been validated locally.
