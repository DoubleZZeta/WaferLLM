# Two-chunk low-rank GEMV pipeline

## What changed and why

`Z = X @ A` is split into two local rank chunks. Their starts are now aligned
on 32-bit words. With FP16, two elements occupy one word: splitting after 9
or 13 elements put chunk 1 at an odd halfword address. Packed communication
of those chunks produced wrong output. We reproduced the
rank-1664 local shape on a 4x4 grid: M=N=256, R=104, giving Mt=Nt=64, Rt=26.
The old code failed; aligned storage passes. One padding element is inserted
when needed, with additional end padding so no packed tail shares another
object. A/B host layout and mathematical rank are unchanged. The original
multidimensional A descriptor and vector compute are retained.

`split` is the mathematical rank offset in A/B; `z1_offset` is the padded
storage offset in Z. Do not use one in place of the other.

## Execution order

Pipeline: A0 -> launch Y0 -> A1 -> wait for Y0 -> launch Y1 -> B0 ->
wait for Y1 -> B1 -> final X collective.

Serial control: A0 -> A1 -> Y0 -> B0 -> Y1 -> B1 -> final X collective.

Both modes use identical chunks, collectives, buffers and callbacks; only
launch order differs. Compare both against MeshGEMVLowerRankRectangular to
separate overlap benefit from chunking overhead. Serial does not mean the
original unchunked kernel.

## Small tests (SDK environment)

```bash
bash run_sim.sh 4 4 256 104 256 2 2
bash run_sim.sh 4 4 256 104 256 2 2 --schedule serial
bash run_sim.sh 4 4 256 104 256 2 2 --profile-phases
```

The projection runner avoids expensive host SVD input generation:

```bash
cd ../Llama2Projections
python3 sweep.py fused-pipelining --run-name pipeline_check \
  --PX 4 --PY 4 --group-num-x 2 --group-num-y 2 \
  --M 256 --N 256 --ranks 76,104 --repetitions 3 --suppress-simfab-trace
```

Use a new run name with `--schedule serial`, or with `--profile-phases`
(omit `--repetitions`, which becomes one measured iteration plus one warmup).
The compiler saves pipeline_config.json; the host checks schedule, shape and
profiling mode before launching to prevent using a mismatched artifact.

## Understand the profiler in three steps

1. `mark(profiled, event)` in `src/meshgemv.csl` records a timestamp at a named boundary.
   The 16 event names, in the same order, are in `host_common.PROFILE_EVENTS`.
2. Each PE's timestamps are converted to cycles since its own iteration start.
   A0/A1/B0/B1 have compute start/end markers; Y0/Y1 have launch/ready markers;
   X and the overall iteration have start/end markers.
3. Read the printed durations or plot a timeline:

```bash
python3 plot_timeline.py /path/to/pipeline_profile.json --px 0 --py 0
```

This writes pipeline_timeline.png. Orange bars are communication intervals;
blue bars are compute. Also inspect a group-root/global-root PE (e.g. 3,3 for
4x4) because roles differ. All PE event rows are saved in pipeline_timeline.csv.
Runner runs additionally preserve `<case>.timeline.json` and `.timeline.csv`
in their `logs/` directory, beside `.phases.json` and the terminal log.

Y launch-to-ready includes descriptor setup, queue waiting, the three
communication phases, and callback scheduling delay. It is **not pure wire
latency**. Its intersection with compute is called interval overlap, not
measured hardware utilisation. Do not sum overlapping durations. Compare
uninstrumented totals in serial and pipeline modes for actual speedup.
The diagnostic dispatcher now mirrors the original low-rank profiler:
unmarked warmups, one measured iteration with unconditional timestamp calls,
then a loop-end marker after the final X collective returns to the dispatcher.
`fused_iteration(false)` and `fused_iteration(true)` specialise the same schedule
at compile time; individual markers have no runtime warmup checks. Async ready
callbacks make one diagnostic-path dispatch decision. Normal builds eliminate
that decision and all internal timestamp calls. Outer start/end timestamps
remain outside the measured iteration, as in the original implementation.

Timestamp overhead can still change scheduling. The pipeline uses 16 markers
because intervals overlap; the original sequential low-rank profiler uses 7.
Matching dispatch structure does not equalise instrumentation cost or turn
launch-to-ready intervals into isolated reduction durations. Warmups do not
overwrite the measured event array. The profile summary reports means
and maxima; the timeline graph shows one actual PE, not averaged endpoints.

## Validation and limits

Run `python3 -m unittest discover -s MeshGEMVLowerRankPipelining -p 'test_pipeline*.py'`
from the repository root for host profiling checks. Simulator regressions must
cover odd/even chunk boundaries and repeated iterations. Matching local tile
sizes does not substitute for the full 64x64 production runs; rerun the failed
server cases before publishing a speedup claim. The device/cloud launcher has
not been validated for these new profiling options; use the simulator runner.
