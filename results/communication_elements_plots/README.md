# Communication cycles versus vector length

Source: `profiled_sweep_all_logs_20261015_194347_UTC.zip`. Read directly from successful phase JSONs, cross-checked
against the run CSVs by plot_profiled_sweep.read_data. Selected runs:
`profile_full_dense64_20261003`, `profile_full_rect_20261003`.

31 completed low-rank cases (62 collective observations), 3 dense cases.
One measured profiled iteration per case. Excludes unfinished cases, duplicate
smoke runs and older unprofiled runs. No new simulator tests were run.

Vector lengths are elements per PE, not summed over the fabric:
- Dense Y: L=N/Px, axis length=Py, group count=Gy.
- Low-rank stage 1 Y: L=R/Px, axis length=Py, group count=Gy.
- Low-rank stage 2 X: L=N/Py, axis length=Px, group count=Gx.
All selected groups contain 8 PEs. Both-axis PE count remains 4096.
Grid notation is Px x Py. Dense only has the 64x64 grid and two distinct lengths.
Panels match participating axis length, not full grid orientation. No dense
values are extrapolated into the rectangular panels or unmeasured lengths.

Mean phase cycles include reduction, broadcast, waiting and profiling effects;
exclude host transfers. Case counts are different workloads, not statistical
repetitions. At identical stage/axis/length, all observed cycle values agree;
overlapping observations are collapsed for display, with counts in the CSV.
max_pe_cycles is retained in the observation CSV, not added across phases.

In all observed low-rank points, C = b + 3*ceil(L/2) exactly matches reported
means. This is a descriptive relationship, not a validated extrapolation or
proof of a hardware mechanism. The odd-length stage-1 points at axis length 32
show why a smooth 1.5*L relationship alone is incomplete. For axis lengths
32/64/128, b is 231.875/387.875/699.875 (stage 1) and
230.125/386.125/698.125 (stage 2). Each stage-2 relationship has only two distinct
lengths. Changing axis length also changes group count; effects are not isolated.

Matched comparison: at L=64, dense=479 and low-rank stage2=482.125 (0.65% slower).
At L=172, dense=803 and low-rank stage2=644.125 (19.79% fewer cycles).
Dense's two points imply a secant slope of 3 cycles/element; this is not a
measured dense scaling curve. Low-rank stage1 and stage2 cover different length
ranges, so their offsets are descriptive, not directly matched phase overheads.
In-kernel timing and axes differ; this does not establish topology-only causality.

Separately checked older `results/phased_cycles.csv`: its 3 low-rank cases at
ranks 832/1216 satisfy the same 64-PE relationships. They are not pooled into
the latest-run plots; its dense values also reproduce the two baseline values.

PNG and vector PDF figures are provided. Regenerate:
`python3 results/plot_communication_elements.py`
