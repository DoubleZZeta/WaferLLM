# Experimental rank-chunk pipeline

This is a separate simulator prototype based on `MeshGEMVLowerRankRectangular`.
The original kernels remain available. It computes `y = (x A) B`, retains both
broadcasts, and keeps both axis routes configured throughout execution.
**Correctness and a faster schedule relative to serial chunking do not imply a
speedup over the original kernel.** Compare all three implementations.

## What changes

On a `Px × Py` grid, each PE has `Mt=M/Py`, `Rt=R/Px`, `Nt=N/Py`.
A remains `[Mt,Rt]`, B remains `[Rt,Nt]`, and host tiling is unchanged.

Original schedule:

```
compute all local Z → Y all-reduce/broadcast all Z
                    → compute all local Z @ B → X all-reduce/broadcast output
```

Chunked schedule (two chunks):

```
compute all local Z → Y collective Z0 → completed Z0
                                       ├─ launch Y collective Z1
                                       └─ compute Z0 @ B0
                         completed Z1 → compute Z1 @ B1
                                      → one final X collective
```

This first experiment only chunks the intermediate communication and second
GEMV. Stage-1 computation is unchanged. B chunks accumulate into one output
buffer; the final output reduction is performed once, not once per chunk.

`--schedule serial` uses the same chunk sizes and completion machinery, but
starts the next Y collective only after computing the preceding B chunk.
`--schedule pipeline` starts it before computing the preceding B chunk.
`--chunks 1` is an asynchronous control without inter-chunk overlap.

## How to implement it yourself

Read the implementation in this order:

1. **`compile.py` and `src/layout.csl`: pass the new parameters.** `Chunks`
   selects the number of pieces per PE, `Pipeline` selects scheduling order,
   and `Profile` enables timestamp instructions at compile time. These are
   not changes to the global matrix rank or grid. Keep the original colours,
   queues, roots, and A/B layout.
2. **`src/meshgemv.csl: chunk_start`: partition the local rank dimension.**
   Let `q=Rt//Chunks`, `e=Rt%Chunks`. The first `e` chunks contain `q+1`
   elements; the rest contain `q`. Chunk k starts at `k*q+min(k,e)`.
   For `Rt=19, Chunks=2`, use `[0:10]` and `[10:19]`. Never introduce an
   empty chunk, and do not use a runtime division on the critical path.
3. **`src/comm_lib/comm_pe.csl: start_y`: make one collective asynchronous.**
   Point the memory descriptor at the selected Z slice and set the lengths
   of the six Y fabric descriptors to that slice's length. Launch the local
   group operation with `.async=true`, a dedicated microthread, and a
   completion task. Completion of group reduction launches root reduction;
   its completion launches broadcast; broadcast completion notifies the
   parent program. A non-participating PE skips the appropriate step.
   Pass the memory DSD directly to asynchronous instructions so their DSR
   allocation is managed by the compiler. The initial attempt to carry a
   manually loaded source DSR across callbacks failed numerical validation.
4. **`src/meshgemv.csl: fused_iteration` and `chunk_ready`: manage ownership.**
   Clear the output once. Start the first collective, then return to the
   scheduler. A broadcast completion makes exactly one more chunk ready.
   In pipeline mode it also starts the next collective before scheduling B
   computation. There is at most one active Y collective on each PE; its
   descriptors are changed only after its completion callback.
5. **`compute_chunk`: compute only the ready slice.** Offset the Z descriptor
   by `start` and shorten it to `end-start`. Offset B by `start*Nt`, keeping
   each B row length `Nt`. Reload the compute DSRs and apply the existing FMA
   loop. Do **not** clear `res` here: each chunk contributes to the same sum.
   Process chunks in increasing order to preserve accumulation order.
6. **Finish only after every chunk is consumed.** Then call the existing
   synchronous X all-reduce/broadcast and advance the warmup/repetition loop.
   Use launched/ready/consumed counters, not just task activations: multiple
   activations of a task can coalesce.
7. **Validate before measuring speed.** `launch_chunked_rectangular.py` checks
   all output replicas, compares the result with host `x @ A @ B`, and rejects
   compile/launch parameter mismatches. Test single and uneven chunks,
   repeated iterations, both rectangular orientations, and group edge cases.

The local B `@map` is still synchronous. A previously launched fabric operation
can progress during it, but completion tasks wait until the current task yields.
Consequently this prototype does not guarantee overlap of the *whole* three-step
collective with computation. Splitting/yielding B computation more finely is a
separate tradeoff: it permits earlier callbacks but adds more scheduling work.

## Running

Use the existing SDK environment. Both runners and shell wrappers now accept
`fused-chunked`. Choose a separate run name for each chunk count, schedule, and
profiling mode; these settings are saved and cannot change on resume.

Small local verification, inside an SDK container from the repository root:

```
python MeshGEMVLowerRankChunked/debug/smoke.py
```

A targeted server experiment, starting from `Llama2Projections`:

```
bash run.sh fused-chunked gate_r1216_chunks2 --projections gate_up \
  --PX 64 --PY 64 --group-num-x 8 --group-num-y 8 --ranks 1216 \
  --chunks 2 --schedule pipeline --repetitions 3 --suppress-simfab-trace
```

Compare the same shape/rank using `fused-rectangular` (omit chunks/schedule),
`fused-chunked --chunks 1`, and `fused-chunked --chunks 2 --schedule serial`.
Do not launch a full sweep until this targeted comparison is worthwhile.

Custom matrix sizes are supported through `run_sweep.sh`, for example:

```
bash run_sweep.sh fused-chunked small_chunk_profile \
  --M 64 --N 128 --ranks 20 --PX 4 --PY 8 \
  --group-num-x 2 --group-num-y 2 --chunks 2 --schedule pipeline \
  --profile-phases --suppress-simfab-trace
```

Results are written to `fused-chunked.csv`. Existing block summary commands
expect the original dense/fused CSV schemas; they do not aggregate this
experimental backend automatically.

## Reading the profiles

Use normal runs for final cycle comparisons and separate profiled runs for
schedule diagnosis. Profiling forces one measured iteration after a warmup;
normal runs have no timestamp instructions at the internal markers.

The additive outer intervals are `stage1_local_compute`, `pipeline_elapsed`,
and `stage2_x_allreduce`. Their sum is `profile_iteration`, excluding entry/exit
control. `pipeline_elapsed` includes waiting, B computation, and chunk control.
Each chunk reports B-compute duration and elapsed time until its readiness
callback. **Readiness time is not the duration of that chunk's collective.**
These nested metrics must not be added to the outer intervals.

Full per-PE event timestamps are in `chunk_timeline.json` in the build directory
and copied to `logs/<case>.timeline.json` by the runners. Times are relative to
each PE's own iteration start; they are not globally aligned. Markers show
callback execution, not the precise instant the hardware transfer completed.
Profiling perturbs the schedule, especially for very small chunks.

## Local verification (2026-10-18)

SDK 2.10.0, WSE-3 simulator: all 12 cases in `debug/smoke.py` passed.
These cover original-kernel controls, one to four chunks, uneven chunks,
repeated iterations, both rectangular orientations, one group, singleton
PE groups, and both profiling schedules. Relative L2 errors were below 0.0013.
The detailed results are recorded in `debug/validation_results.json`.

Unprofiled mean cycles, `Px=4, Py=8, Gx=Gy=2`, one warmup and three repeats:

| M, R, N | Original | Two chunks, serial | Two chunks, pipeline |
|---|---:|---:|---:|
| 64, 20, 128 | 342.708 | 685 | 676 |
| 256, 128, 1024 | 1862.667 | 2206 | 2194 |

The overlap schedule saves 9–12 cycles relative to serial chunking here, but
both are slower than the original kernel. These small-grid controls establish
correctness and expose overhead; they do not predict performance on 4096 PEs.
Do not describe this prototype as an achieved GEMV speedup. In particular,
asynchronous fabric instructions alone do not remove task-launch overhead or
allow communication callbacks to preempt the synchronous B loop.

`debug/runner_smoke.py` separately exercises both runner entry points and
checks profile/timeline output and resume behavior. Host-only regression tests:

```
python3 -m unittest discover -s Llama2Projections -p test_chunked.py -v
```
