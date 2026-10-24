# Pipeline fix validation — 2026-10-25

SDK 2.10.0, WSE-3 simulator in Lima. These are correctness and diagnostic tests, not full-grid performance claims.

The original Rt=26 local-tile reproducer produced relative L2 error 0.683923 with SVD-generated inputs. Aligned chunk storage fixes the reproduced failure. The table below uses seeded direct factors from the projection runner, so its errors should not be compared numerically with that different input dataset.

| Run | Grid | M=N | Rank | Local Rt | Repetitions | Profile | Mean cycles | Relative L2 | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| pipeline_fixed_regression | 4×4 | 256 | 8 | 2 | 10 | False | 1230.000 | 0.000861 | ok |
| pipeline_fixed_regression | 4×4 | 256 | 12 | 3 | 10 | False | 1248.000 | 0.001596 | ok |
| pipeline_fixed_regression | 4×4 | 256 | 20 | 5 | 10 | False | 1286.000 | 0.001644 | ok |
| pipeline_fixed_regression | 4×4 | 256 | 52 | 13 | 10 | False | 1433.000 | 0.001446 | ok |
| pipeline_fixed_regression | 4×4 | 256 | 76 | 19 | 10 | False | 1858.000 | 0.001475 | ok |
| pipeline_fixed_regression | 4×4 | 256 | 104 | 26 | 10 | False | 1950.000 | 0.001415 | ok |
| pipeline_rect_fixed_check | 4×8 | 512 | 76 | 19 | 3 | False | 1761.292 | 0.001534 | ok |
| pipeline_rect_fixed_check | 4×8 | 512 | 104 | 26 | 3 | False | 1852.667 | 0.001508 | ok |
| pipeline_serial_fixed_regression | 4×4 | 256 | 76 | 19 | 10 | False | 1864.000 | 0.001475 | ok |
| pipeline_serial_fixed_regression | 4×4 | 256 | 104 | 26 | 10 | False | 1956.000 | 0.001415 | ok |
| pipeline_profile_fixed_probe | 4×4 | 256 | 76 | 19 | 1 | True | 2130.000 | 0.001475 | ok |
| pipeline_profile_fixed_probe | 4×4 | 256 | 104 | 26 | 1 | True | 2222.000 | 0.001415 | ok |
| pipeline_serial_profile_probe | 4×4 | 256 | 76 | 19 | 1 | True | 2134.000 | 0.001475 | ok |
| pipeline_serial_profile_probe | 4×4 | 256 | 104 | 26 | 1 | True | 2226.000 | 0.001415 | ok |

Additional standalone test: `bash run_sim.sh 4 4 64 20 128 2 2 --profile-phases` passed (relative L2 versus factorized reference 0.001056). Four host profiler unit tests passed. The timeline plot was generated and visually checked.

Serial and pipeline controls use identical chunks; serial is not the original unchunked baseline. The uninstrumented pipeline saves six cycles in each matched small test. Profiled intervals include marker and callback overhead; interval overlap is not proof of fabric progress. Full 64×64 tests, including the previously crashing case, remain to be rerun on the server.
