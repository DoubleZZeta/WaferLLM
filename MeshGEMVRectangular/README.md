# Rectangular dense MeshGEMV

Separate rectangular implementation of the original dense `MeshGEMV` algorithm,
including its two-level, bidirectional Y reduction and broadcast. The original
`MeshGEMV/` source is unchanged.

For `x[1,M] @ W[M,N]`, a `PX`-column by `PY`-row grid assigns PE `(py,px)`:

- `X`: input block `py`, `M/PY` elements, replicated across columns.
- `W`: input rows `py` and output columns `px`, tile `(M/PY) x (N/PX)`.
- `res`: `N/PX` output elements, summed and broadcast along Y.

`M` must divide evenly by `PY`, and `N` by `PX`. Both group counts must divide
their respective axes. Only `group_num_y` affects dense communication;
`group_num_x` is accepted and recorded for matching experiment configurations.
This original collective requires at least two Y groups and two PEs per Y group.
It is not the fused kernel's trailing-root collective, so comparisons retain
that algorithmic difference as well as the different matrix placements.

Use the repository-root commands (with the SDK launchers on PATH):

```bash
python3 MeshGEMVRectangular/compile.py --mode sim \
  --PX 32 --PY 128 --M 4096 --N 11008 \
  --group_num_x 4 --group_num_y 16 --out-dir MeshGEMVRectangular/out_gate
cs_python Llama2Projections/launch_dense_rectangular.py \
  --PX 32 --PY 128 --M 4096 --N 11008 \
  --group_num_x 4 --group_num_y 16 --repetitions 1 \
  --profile-phases --suppress-simfab-trace --out-dir MeshGEMVRectangular/out_gate
```

The launcher uses the bounded, readback-fenced upload helper from
`MeshGEMV/sim_transfers.py`. Transfers are outside device kernel timing.
Phase profiling captures local compute (including output clearing), Y all-reduce
(including broadcast), and control/marker intervals. Reduction and broadcast are
not separately instrumented. `profile_iteration` covers the inner iteration;
the headline total also includes surrounding timing/exit work, so it need not
match the phase sum. Phase maxima must not be added across PEs.

`host_common.py` handles rectangular tiling and 48-bit timestamps, including
high words. `test_host_common.py` checks tiling reconstruction and timestamp
arithmetic. `debug/runner_smoke.py` runs small SDK cases through both runners for
square, tall and wide grids, verifying numerical results and phase fields.
