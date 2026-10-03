# MeshGEMV

## Overview

This folder contains the implementation of the **MeshGEMV** algorithm, which computes general matrix multiplication of the form $[1,N]=[1,M]@[M,N]$

## Platform

- **Cerebras SDK version**: 2.10.0
- **Hardware**: WSE-3 only

## Parameters

- `P`: Number of PEs in each dimension (creates P×P PE grid)
- `M`: Length of the input vector (shared dimension)
- `N`: Number of columns in the matrix (output vector length)
- `group_num`: Number of PE groups for parallel execution

## Run with Simulator

The simulator allows you to test and debug your MeshGEMV implementation before deploying to actual hardware.

```bash
# ./run_sim.sh P M N group_num   (small grids only — sim cost scales with P*P)
# Runs [1, M]@[M, N] on a P x P PE grid on the Cerebras simulator.
# Example
./run_sim.sh 4 64 64 2
```

**Note:** The simulator provides cycle-approximate performance estimates and allows debugging without consuming actual hardware resources.

## Run with Cerebras

Deploy and execute your MeshGEMV algorithm on the actual WSE-3 hardware.

```bash
# ./run_device.sh P M N group_num   (drives a real WSE-3 run via the cloud compiler)
# Runs [1, M]@[M, N] on a P x P PE grid on the Cerebras WSE-3.
# Example
./run_device.sh 256 4096 4096 16
```

**Prerequisites:**
- Ensure you have access to a WSE-3 system
- Verify your environment is properly configured with Cerebras SDK
- Check that you have the necessary permissions to run on hardware

**Performance Considerations:**
- The WSE-3 provides massive parallelism with thousands of cores
- Optimal performance is achieved when matrix dimensions are divisible by P
- Consider memory constraints when selecting matrix and vector sizes
- The `group_num` parameter allows for a trade-off between routing resources and allreduce latency

Benchmark CSVs use `retained_work_ratio = R(M+N)/(MN)` and
`compression_fraction = 1 - retained_work_ratio`. These fields compare the
factorized parameter/MAC count with the dense operation without assuming a
square matrix. The benchmark's `--rank-fractions` option only selects ranks as
`R/min(M,N)` for a sweep.

## Compare Dense and Low-Rank Simulator Runs

Run the comparison driver from inside the Cerebras SDK VM. `P` is the grid
dimension, so `P=4` uses 16 PEs. The following sweep tests four square matrix
sizes, four rank fractions, and one reduction-group configuration:

```bash
cd /Users/zhangao/cerebras/WaferLLM
export PATH="/Users/zhangao/cerebras/cs_sdk:$PATH"

python3 MeshGEMVSparese/benchmark_compare.py \
  --p-values 4 \
  --sizes 64,128,256,512 \
  --rank-fractions .125,.25,.375,.5 \
  --groups 2 \
  --output MeshGEMVSparese/benchmark_results_simple.csv \
  --log-dir MeshGEMVSparese/benchmark_logs \
  --resume
```

To sweep explicit ranks instead of ratios:

```bash
python3 MeshGEMVSparese/benchmark_compare.py \
  --p-values 4,8 \
  --sizes 256,512 \
  --ranks 32,64,128 \
  --groups 2,4 \
  --output MeshGEMVSparese/benchmark_results_simple.csv \
  --resume
```

Use `--dry-run` first to display the configurations and invalid combinations
without compiling. The compact CSV contains parameters, dense and low-rank
cycles, whether low rank is faster, speedup, cycle reduction, and two relative
L2 precision metrics. Full command output remains available in the log
directory. Each result is appended immediately, `--resume` skips successful
rows, and the script prints the first tested break-even size for every
`(P, group_num, compression_fraction)` series.

To upgrade an existing CSV from the historical `rank_ratio` schema without
running any kernels, use:

```bash
python3 MeshGEMVSparese/benchmark_compare.py \
  --output MeshGEMVSparese/benchmark_results_simple.csv --migrate-only
```
