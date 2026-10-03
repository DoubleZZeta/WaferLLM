# DecodeLowerRank

## Overview

This folder incrementally adapts the WaferLLM **Decode** implementation to use
low-rank projection weights. Q/K/V, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj` now all use two-factor low-rank GEMVs.

Q, K, and V each use `[dim, rank_qkv] @ [rank_qkv, dim]`. Their configured
ranks must currently be equal so the three rank intermediates and three final
outputs can use two fused collectives rather than six independent collectives.
Likewise, `rank_up` and `rank_gate` must be equal so those two FFN projections
retain fused collectives.

`o_proj` is represented as two factors:

```text
O ~= O_A @ O_B
O_A: [dim, rank_o]
O_B: [rank_o, dim]
```

On each decoder iteration, the device performs:

```text
attention_output @ O_A -> X-axis all-reduce
rank intermediate @ O_B -> Y-axis all-reduce
attention residual add
```

The rank intermediate remains on wafer between the two GEMVs.

`down_proj` follows the same fused pattern with rectangular factors:

```text
DOWN ~= DOWN_A @ DOWN_B
DOWN_A: [ffn_dim, rank_down]
DOWN_B: [rank_down, dim]

z3 @ DOWN_A -> X-axis all-reduce
rank intermediate @ DOWN_B -> Y-axis all-reduce
FFN residual add
```

## Platform

- **Cerebras SDK version**: 2.10.0
- **Hardware**: WSE-3 only

## Configuration

The Decode implementation uses JSON configuration files to specify model parameters. Example configuration files can be found in `model_config/` (e.g. `test-sim.json` for a small P=8 sim run, `llama8B_4k_1_256.json` for the device-scale 8B-4K run).

**Configuration Parameters:**
- `P`: Number of PEs in each dimension (creates a P×P PE grid)
- `group_num`: Number of PE groups for the two-phase all-reduce
- `bsz`: Batch size
- `dim`: Model hidden dimension
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for grouped-query attention)
- `head_dim`: Dimension per attention head
- `seq_len`: Maximum sequence length
- `ffn_dim`: Feed-forward network hidden dimension
- `layer_num`: Number of transformer layers (used by the device throughput report)

The separate rank JSON currently consumes `rank_q`, `rank_k`, `rank_v`,
`rank_o`, `rank_up`, `rank_gate`, and `rank_down`. Q/K/V ranks must be equal;
Up/Gate ranks must also be equal. `dim`, `ffn_dim`, and all consumed ranks must
be divisible by `P`.

## Run with Simulator

The simulator runs the whole fabric on the host, so cost scales with P×P. Keep `P` small (the provided `test-sim.json` uses P=8). The config argument must be the full path to the JSON file.

```bash
# ./run_sim.sh <model-config.json> <rank-config.json>
./run_sim.sh model_config/test-sim.json rank_config/test-sim.json
```

To also build and run the separate projection profiler:

```bash
./run_sim.sh \
  model_config/test-sim.json \
  rank_config/test-sim.json \
  --profile-projections
```

`run_sim.sh` compiles with `cslc` into
`out_<model-config>__<rank-config>/` and then launches `launch_sim.py`. It
reports the mean per-step cycle count and validates `v_proj`, `o_proj`,
`up_proj`, and `down_proj` against host references. (`q_proj` and `k_proj`
are subsequently modified by RoPE; `gate_proj` is subsequently modified by
SiLU.) The diagnostic D2H copies occur after the timed decoder region.

The reported `cycles per decoder layer` is the measured time for one invocation
of this decoder-layer kernel. `estimated cycles for N layers` multiplies that
measurement by the model configuration's `layer_num`; Decode does not load and
timestamp N distinct layer weights in one run.

Projection profiling deliberately uses a second artifact named
`out_<model-config>__<rank-config>__projection-profile/`. It reports `qkv_proj`,
`o_proj`, `up_gate_proj`, and `down_proj` timing from one diagnostic decoder
pass. Q/K/V and Up/Gate are reported as groups because each group shares fused
collectives; splitting them would assign shared reduction time arbitrarily.
Timestamp markers perturb the diagnostic values slightly, but the normal
artifact is compiled with the profiler disabled and contains no projection
profile symbols, buffers, branches, or marker calls. Use the normal artifact's
`cycles per decoder layer` for performance comparisons.

## Run with Cerebras WSE-3

Compiles via the cloud SdkCompiler and runs on a real WSE-3 appliance. The config argument must be the full path to the JSON file.

```bash
# ./run_device.sh <model-config.json> <rank-config.json>
./run_device.sh \
  model_config/llama8B_4k_1_256.json \
  rank_config/llama8B_4k_25percent.json
```

`run_device.sh` compiles into
`compile_out/artifact_<model-config>__<rank-config>.json` and then launches
`launch_device.py`, which reports the mean per-step cycle count and throughput
per request.

**Prerequisites:**
- Access to a WSE-3 system
- Environment configured with Cerebras SDK 2.10.0
- Permissions to run on the hardware

**Performance Considerations:**
- Optimal performance is achieved when the model dimensions are divisible by `P`
- The `group_num` parameter trades off routing resources against all-reduce latency
