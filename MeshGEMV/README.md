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

### Phase timing

Use the optional profile mode to split one warmed dense iteration into local
compute, the Y-axis all-reduce, and loop/control overhead:

```bash
bash run_sim.sh 4 512 512 2 --profile-phases
```

For a direct comparison with fused rank `R`, run the matching fused profile:

```bash
cd ../MeshGEMVSparseFused
bash run_sim.sh 4 512 128 512 2 --profile-phases
```

The phase means within each run are additive. Do not add `max_pe` values: each
phase can have a different slowest PE. Profiling markers add overhead, so use
ordinary runs without `--profile-phases` for the final dense-versus-fused total
cycle comparison.

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

## Simulator upload fix

The simulator launcher uses bounded uploads with a device readback after each
batch to prevent SDK 2.10 ingress-ring overflow (HCF on large matrices).
Each kernel folder includes `sim_transfers.py` so SDK wrappers can mount the
folder independently. Uploads finish before kernel timing, and this fix does
not change the CSL computation. Llama2Projections uses this fix by default.
