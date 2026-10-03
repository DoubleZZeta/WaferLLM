# MeshGEMV with the simulator HCF workaround

This is a separate copy of the dense MeshGEMV implementation. Its CSL sources
match `MeshGEMV/src` at the time of the split. The host launcher uses bounded
uploads and device readbacks to avoid the SDK 2.10 local simulator's 100 MiB
ingress ring overwrite.

The server-tested upload fix is now also the default in `MeshGEMV` and
`MeshGEMVLowerRank`, including the normal Llama2Projections launchers.
This folder remains as the isolated snapshot and debug toolkit.

## Run through Llama2Projections

```bash
cd Llama2Projections
bash run.sh dense p64_fixed --P 64 --group-num 8 --repetitions 1 \
  --hcf-fix --suppress-simfab-trace
bash run.sh fused p64_fixed --hcf-fix --retained-work-ratios .2,.4,.6,.8 \
  --suppress-simfab-trace
```

`--hcf-fix` selects `launch_dense_patched.py`, compiles this folder's kernel,
and is saved in the run's `config.json`. The fused command inherits it and
selects `launch_fused_patched.py`, which applies this folder's upload helper
to the existing `MeshGEMVLowerRank` kernel. Both new entry points use the
benchmark's normal result reporting and `--out-dir` handling.

A new run uses the normal, now-fixed implementation when `--hcf-fix` is absent.
The flag remains for compatibility with saved runs and selects this snapshot;
`--no-hcf-fix` also uses safe uploads through the normal implementation.

## Direct dense invocation

From the repository root, using an output directory under that root:

```bash
mkdir -p MeshGEMVPatched/out
python3 MeshGEMVPatched/compile.py --mode sim --P 64 --M 4096 --N 8192 \
  --group_num 8 --out-dir MeshGEMVPatched/out/p64_4096x8192
cs_python MeshGEMVPatched/launch_sim.py --P 64 --M 4096 --N 8192 \
  --group_num 8 --repetitions 1 --out-dir MeshGEMVPatched/out/p64_4096x8192 \
  --suppress-simfab-trace
```

## Debug tools and verification

- `sim_transfers.py`: caps each payload at 16 MiB and reads one input value
  from every affected PE before submitting the next batch.
- `debug/`: compact HCF collector, isolated reproduction runner, and evidence.
- `test_sim_transfers.py`: batch sizing, layout, and completion regression tests.

All transfer checkpoints happen before kernel timing. They change upload
scheduling and wall-clock time; small simulator cycle variations remain possible.
The original and patched small dense tests measured the same 373 mean / 379
maximum cycles. Small fused tests passed with 377.5 / 384 cycles for original
and 377.25 / 383 for patched. Both produced matching validated numerical results.
The user subsequently confirmed that the fix resolved the failing server run.

```bash
cs_python -m unittest discover -s MeshGEMVPatched -p test_sim_transfers.py
python3 -m unittest discover -s Llama2Projections -p test_bench.py
cs_python MeshGEMVPatched/debug/run_diagnostic.py 64 4096 8192 8 \
  --suppress-simfab-trace --result-values-per-pe 128
```

For instruction capture, omit `--suppress-simfab-trace`. To reproduce the
unfenced SDK overflow deliberately, add `--no-fence-h2d` to the diagnostic.
Historical ZIPs are not needed to run the benchmark.
See [the diagnosis](debug/INVESTIGATION.md).
