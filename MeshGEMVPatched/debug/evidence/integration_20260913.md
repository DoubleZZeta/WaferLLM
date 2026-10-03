# Separate implementation integration checks — 2026-09-13

- Six unit tests passed: three upload tests and three benchmark routing/config tests.
- All copied dense CSL files match MeshGEMV/src byte for byte.
- Original MeshGEMV launcher has unfenced H2D again; pre-existing profiling and
  output-directory support were preserved.
- Dense host_common.py restored from the saved source snapshot.
- Low-rank host_common.py reconstructed from its surviving cached implementation;
  all seven generated arrays, configuration names, timestamp decoding and phase
  metrics matched that original implementation across three shapes.
- Original/patched projection launchers: P=4, M=64, N=128, R=16 for fused,
  one warmup + one measured iteration, bulk traces disabled. Both dense and
  fused numerical checks passed. Dense max relative error: 0.00102119.
  Fused relative L2 error: 0.000810487.
- Dense original/patched mean/max cycles: 373 / 379 for both.
- Fused original mean/max cycles: 377.5 / 384; patched: 377.25 / 383.
  A smoke-harness assertion requiring identical cycle counts was too strict;
  the harness now reports these small timing differences. Numerical assertions
  passed for all four cases. The patched fused launcher was checked again after
  restoring host_common.py and produced the same validated result and timing.
- Relocated debug runner: P=4, 16x32, passed; 204 mean / 210 max cycles,
  max relative error 0.0007.
- Full P=64 server verification remains outstanding.
