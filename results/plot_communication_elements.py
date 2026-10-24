"""Compare measured collective cycles against local vector length; no new runs."""
import argparse
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/waferllm-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/waferllm-cache')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_profiled_sweep import read_data, setup, save, ROOT

SERIES = {
    'stage1_y': ('Low-rank stage 1 (Y)', '#2563a6', 'o'),
    'stage2_x': ('Low-rank stage 2 (X)', '#d87513', 's'),
    'dense_y': ('Dense (Y)', '#222222', 'D'),
}


def collect(archive):
    data, dense, _ = read_data(archive, 'profile_full_dense64_20261003',
                              'profile_full_rect_20261003')
    rows = []
    with zipfile.ZipFile(archive) as z:
        for r in pd.concat([data, dense.reset_index()], ignore_index=True).to_dict('records'):
            phases = json.loads(z.read(r['source']))['phases']
            common = {k: r[k] for k in ('projection', 'grid', 'Px', 'Py', 'M', 'N', 'rank',
                      'target_retained_work_ratio', 'actual_retained_work_ratio', 'source')}
            specs = [('dense_y', 'dense_y_allreduce', r['N'], r['Px'], r['Py'], r['group_num_y'])] \
                if r['execution'] == 'dense' else [
                    ('stage1_y', 'stage1_y_allreduce', r['rank'], r['Px'], r['Py'], r['group_num_y']),
                    ('stage2_x', 'stage2_x_allreduce', r['N'], r['Py'], r['Px'], r['group_num_x'])]
            for stage, phase, numerator, divisor, axis, groups in specs:
                assert numerator % divisor == 0
                assert axis % groups == 0 and axis // groups == 8
                rows.append(dict(common, stage=stage, axis_pes=int(axis), group_count=int(groups),
                    group_size=8, elements=int(numerator // divisor),
                    mean_cycles=float(phases[phase]['mean_cycles']),
                    max_pe_cycles=float(phases[phase]['max_pe_cycles']), source_archive=archive.name))
    result = pd.DataFrame(rows)
    assert len(result) == 2 * len(data) + len(dense)
    return result


def aggregate(raw):
    return raw.groupby(['stage', 'axis_pes', 'group_size', 'elements'], as_index=False).agg(
        mean_cycles=('mean_cycles', 'mean'), min_cycles=('mean_cycles', 'min'),
        max_cycles=('mean_cycles', 'max'), case_count=('mean_cycles', 'size'),
        projections=('projection', lambda s: ','.join(sorted(set(s)))),
        grids=('grid', lambda s: ','.join(sorted(set(s)))))


def axis_plots(points, output, zoom=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    title = 'Stage 1 detail: local rank-vector length versus communication cycles' if zoom else \
        'Communication cycles versus elements reduced per PE'
    fig.suptitle(title, fontsize=17, y=.99)
    handles = [Line2D([], [], color=c, marker=m, linestyle='None', markersize=7, label=label)
               for k, (label, c, m) in SERIES.items() if not zoom or k == 'stage1_y']
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    for ax, axis in zip(axes, [32, 64, 128]):
        part = points[points.axis_pes == axis]
        for stage, (_, color, marker) in SERIES.items():
            if zoom and stage != 'stage1_y':
                continue
            p = part[part.stage == stage].sort_values('elements')
            # Each point is a measured (axis, stage, vector-length) combination.
            # Lines connect measured lengths only; no interpolation data is created.
            if len(p):
                ax.plot(p.elements, p.mean_cycles, color=color, marker=marker,
                        linestyle='-' if stage == 'stage1_y' else '--', linewidth=1.4,
                        markersize=6, zorder=3)
                if not zoom:
                    for r in p.itertuples():
                        if stage == 'stage1_y':
                            continue
                        dy = -24 if stage == 'stage2_x' and axis == 64 and r.elements == 64 else 10
                        value=f'{r.mean_cycles:.3f}'.rstrip('0').rstrip('.')
                        ax.annotate(value, (r.elements, r.mean_cycles),
                                    xytext=(0,dy), textcoords='offset points', ha='center',
                                    fontsize=9, color=color)
        s1grid = {32:'128×32', 64:'64×64', 128:'32×128'}[axis]
        s2grid = {32:'32×128', 64:'64×64', 128:'128×32'}[axis]
        ax.set_title(f'{axis} PEs along reduction axis · {axis//8} groups of 8\n'
                     + (f'Grid {s1grid}' if zoom else f'Stage 1 grid {s1grid}; stage 2 grid {s2grid}'), fontsize=11)
        ax.set_xlabel('Elements reduced per PE')
        ax.set_xlim(0, 82 if zoom else 370)
        ax.set_ylim(0, 880 if zoom else 1330)
        ax.grid(alpha=.2)
        if not zoom and axis != 64:
            ax.text(.97,.08,'No dense measurements\nfor this axis length',transform=ax.transAxes,
                    ha='right',va='bottom',fontsize=9,color='.4')
    axes[0].set_ylabel('All-reduce mean cycles per PE')
    footer = ('Completed cases only; markers with identical coordinates overlap. Grid labels are Px × Py.\n'
              'Includes reduction, broadcast and waiting. Lines join observations, not a predictive model.')
    if zoom:
        footer = ('All 31 completed stage-1 observations; identical points are collapsed.\n'
                  'At 32 PEs, odd/even lengths reveal a staircase: L=5 and 6 match, as do 9 and 10, and 13 and 14.')
    fig.text(.5,.025,footer,ha='center',fontsize=10,color='.35')
    fig.tight_layout(rect=(0,.12,1,.84))
    save(fig,output,'stage1_elements_detail' if zoom else 'communication_by_elements')


def matched_plot(raw, output):
    matched = []
    for length in [64,172]:
        d=raw[(raw.stage=='dense_y') & (raw.axis_pes==64) & (raw.elements==length)]
        l=raw[(raw.stage=='stage2_x') & (raw.axis_pes==64) & (raw.elements==length)]
        assert len(d) and len(l)
        assert d.mean_cycles.nunique()==1 and l.mean_cycles.nunique()==1
        dc,lc=float(d.mean_cycles.iloc[0]),float(l.mean_cycles.iloc[0])
        matched.append(dict(elements=length,axis_pes=64,group_size=8,dense_cycles=dc,
            low_rank_stage2_cycles=lc,dense_cases=len(d),low_rank_cases=len(l),
            cycles_saved=dc-lc,percent_cycles_saved=100*(dc-lc)/dc))
    table=pd.DataFrame(matched)
    fig,ax=plt.subplots(figsize=(8.4,5.6))
    x=np.arange(2); w=.32
    for values,offset,label,color,counts in [
        (table.dense_cycles,-w/2,'Dense Y all-reduce','#444444',table.dense_cases),
        (table.low_rank_stage2_cycles,w/2,'Low-rank stage-2 X all-reduce','#d87513',table.low_rank_cases)]:
        bars=ax.bar(x+offset,values,w,label=label,color=color)
        for bar,value,count in zip(bars,values,counts):
            ax.text(bar.get_x()+bar.get_width()/2,value+12,f'{value:g}\n({count} case'+('s)' if count!=1 else ')'),
                    ha='center',va='bottom',fontsize=10)
    ax.set_xticks(x,['64 elements\nQKVO / Down','172 elements\nGate / Up'])
    ax.set_ylabel('All-reduce mean cycles per PE')
    ax.set_ylim(0,1000); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.set_title('Matched vector lengths on the 64×64 grid\n64 participating PEs; 8 groups of 8',fontsize=14)
    ax.legend(frameon=False,loc='upper left')
    fig.text(.5,.025,'Low-rank covers all four tested ranks; stage-2 cycles are identical across ranks.\n'
             'These are in-kernel measurements: dense uses Y, low-rank stage 2 uses X.',ha='center',fontsize=10,color='.35')
    fig.tight_layout(rect=(0,.12,1,1))
    save(fig,output,'dense_vs_lowrank_matched_elements')
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=ROOT/'profiled_sweep_all_logs_20261015_194347_UTC.zip')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'communication_elements_plots')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=collect(args.zip); points=aggregate(raw)
    assert np.allclose(points.min_cycles,points.max_cycles), 'Matched lengths have differing times; retain error ranges'
    setup(); axis_plots(points,args.output_dir); axis_plots(points,args.output_dir,zoom=True)
    matched=matched_plot(raw,args.output_dir)
    fits=[]
    for (stage,axis),p in points[points.stage!='dense_y'].groupby(['stage','axis_pes']):
        pairs=np.ceil(p.elements.to_numpy()/2)
        intercept=float(np.mean(p.mean_cycles.to_numpy()-3*pairs))
        residual=float(np.max(np.abs(p.mean_cycles.to_numpy()-(intercept+3*pairs))))
        assert residual < 1e-8
        fits.append(dict(stage=stage,axis_pes=axis,observed_min_elements=int(p.elements.min()),
                         observed_max_elements=int(p.elements.max()),distinct_lengths=len(p),
                         intercept_cycles=intercept,cycles_per_element_pair=3,max_abs_residual=residual))
    raw.to_csv(args.output_dir/'communication_observations.csv',index=False)
    points.to_csv(args.output_dir/'communication_unique_points.csv',index=False)
    matched.to_csv(args.output_dir/'matched_comparison.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output_dir/'observed_length_relationship.csv',index=False)
    # Older 0.4 profile is a separate consistency check, never pooled into plots.
    old=pd.read_csv(ROOT/'phased_cycles.csv')
    checks=[]
    for r in old.itertuples():
        if r.type=='fused':
            predicted1=387.875+3*np.ceil(r.r/r.P/2)
            predicted2=386.125+3*np.ceil(r.n/r.P/2)
            checks.append(bool(np.isclose(predicted1,r.reduce_1_mean_cycles) and np.isclose(predicted2,r.reduce_2_mean_cycles)))
        else:
            baseline=raw[(raw.stage=='dense_y') & (raw.projection==r.name)]
            assert len(baseline)==1 and np.isclose(baseline.mean_cycles.iloc[0],r.reduce_1_mean_cycles)
    assert checks and all(checks)
    (args.output_dir/'README.md').write_text(f'''# Communication cycles versus vector length

Source: `{args.zip.name}`. Read directly from successful phase JSONs, cross-checked
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
''')
    print(f'{len(raw)} collective observations; {len(points)} distinct stage/axis/length points')
    print(matched.to_string(index=False))
    print(pd.DataFrame(fits).to_string(index=False))
    print(f'Older profile consistency check: {len(checks)} cases passed')
    print(args.output_dir.resolve())


if __name__=='__main__':
    main()
