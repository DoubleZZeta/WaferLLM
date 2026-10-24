"""Plot completed profiled sweep results directly from the supplied ZIP.

No extraction of binaries or simulator traces. Older unprofiled runs and smoke
runs are deliberately excluded. See the generated README for measurement scope.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RATIOS = [.2, .4, .6, .8]
RANKS = {'qkvo': [384, 768, 1280, 1664],
         'gate_up': [640, 1152, 1792, 2432], 'down': [640, 1152, 1792, 2432]}
MULT = {'qkvo': 4, 'gate_up': 2, 'down': 1}
TITLES = {'qkvo': 'Q / K / V / O', 'gate_up': 'Gate / Up', 'down': 'Down'}
GRIDS = ['64×64', '32×128', '128×32']
GROUPS = [(8, 8), (4, 16), (16, 4)]
COLORS = ['#2563a6', '#d87513', '#008b76']
MARKERS = ['o', 's', '^']


def read_data(archive, dense_run, fused_run):
    completed, baseline = [], []
    with zipfile.ZipFile(archive) as z:
        for run in [dense_run, fused_run]:
            config = json.loads(z.read(f'runs/{run}/config.json'))
            if config.get('profile_phases') is not True:
                raise ValueError(f'{run} is not a profiled run')
        for run, filename, execution in [(dense_run, 'dense.csv', 'dense'),
                                          (fused_run, 'fused-rectangular.csv', 'fused')]:
            frame = pd.read_csv(io.BytesIO(z.read(f'runs/{run}/{filename}')))
            frame = frame[frame.status == 'ok']
            profiles = {}
            for name in z.namelist():
                if name.startswith(f'runs/{run}/logs/') and name.endswith('.phases.json'):
                    profile = json.loads(z.read(name))
                    c = profile['case']
                    if c.get('status') != 'ok':
                        continue
                    key = ((c['projection_group'],) if execution == 'dense' else
                           (c['projection_group'], c['Px'], c['Py'], c['rank']))
                    if key in profiles:
                        raise ValueError(f'Duplicate phase record: {key}')
                    profiles[key] = (profile, name)
            for _, row in frame.iterrows():
                name = row.projection_group
                if name not in MULT:
                    raise ValueError(f'Unexpected projection {name}')
                if int(row.multiplicity) != MULT[name]:
                    raise ValueError('Unexpected projection multiplicity')
                if execution == 'dense':
                    key = (name,)
                    px = py = int(row.P)
                    gx = gy = int(row.group_num)
                    rank, ratio, actual = None, 1., 1.
                else:
                    px, py, rank = int(row.Px), int(row.Py), int(row['rank'])
                    gx, gy = int(row.group_num_x), int(row.group_num_y)
                    key = (name, px, py, rank)
                    # The uploaded CSV omitted target labels. Recover them only
                    # from the exact agreed rank schedule, never nearest rounding.
                    ratio = RATIOS[RANKS[name].index(rank)]
                    if pd.notna(row.target_retained_work_ratio):
                        assert np.isclose(ratio, row.target_retained_work_ratio)
                    actual = rank*(row.M+row.N)/(row.M*row.N)
                    assert np.isclose(actual, row.retained_work_ratio)
                    assert str(row.profile_phases).lower() == 'true'
                grid = f'{px}×{py}'
                assert (gx, gy) == GROUPS[GRIDS.index(grid)]
                profile, source = profiles[key]
                assert profile['case']['profile_phases'] is True
                phases = profile['phases']
                def mean(phase):
                    value = float(phases[phase]['mean_cycles'])
                    assert np.isfinite(value) and value >= 0
                    return value
                total = float(row[f'{execution}_mean_cycles'])
                assert total > 0 and np.isfinite(total)
                assert np.isclose(total, profile['case'][f'{execution}_mean_cycles'])
                c1 = mean('dense_local_compute' if execution == 'dense' else 'stage1_local_compute')
                c2 = 0. if execution == 'dense' else mean('stage2_local_compute')
                r1 = mean('dense_y_allreduce' if execution == 'dense' else 'stage1_y_allreduce')
                r2 = 0. if execution == 'dense' else mean('stage2_x_allreduce')
                control = mean('control_and_markers')
                inner = mean('profile_iteration')
                assert np.isclose(c1+c2+r1+r2+control, inner)
                record = dict(execution=execution, projection=name, grid=grid, Px=px, Py=py,
                    group_num_x=gx, group_num_y=gy, M=int(row.M), N=int(row.N),
                    rank=rank, target_retained_work_ratio=ratio, actual_retained_work_ratio=actual,
                    multiplicity=MULT[name], mean_cycles=total,
                    max_cycles=float(row[f'{execution}_max_cycles']), compute1_cycles=c1,
                    compute2_cycles=c2, reduce1_cycles=r1, reduce2_cycles=r2,
                    communication_cycles=r1+r2, compute_cycles=c1+c2,
                    control_cycles=control, profile_iteration_cycles=inner,
                    communication_share_percent=100*(r1+r2)/inner,
                    source=source)
                (baseline if execution == 'dense' else completed).append(record)
    dense = pd.DataFrame(baseline).set_index('projection')
    assert set(dense.index) == set(MULT) and dense.index.is_unique
    assert set(dense.grid) == {'64×64'}
    data = pd.DataFrame(completed)
    assert not data.duplicated(['projection', 'grid', 'target_retained_work_ratio']).any()
    data['dense_mean_cycles'] = data.projection.map(dense.mean_cycles)
    data['speedup'] = data.dense_mean_cycles/data.mean_cycles
    blocks = []
    dense_total = sum(dense.loc[p, 'mean_cycles']*m for p,m in MULT.items())
    for (grid, ratio), part in data.groupby(['grid', 'target_retained_work_ratio']):
        if set(part.projection) != set(MULT):
            continue  # Never extrapolate a partial block.
        total = float((part.mean_cycles*part.multiplicity).sum())
        blocks.append(dict(grid=grid, target_retained_work_ratio=ratio,
            mean_cycles=total, dense_mean_cycles=dense_total, speedup=dense_total/total,
            communication_cycles=float((part.communication_cycles*part.multiplicity).sum())))
    return data, dense, pd.DataFrame(blocks)


def setup():
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'axes.titlepad': 11})


def legend(fig, y=.91):
    handles = [Line2D([0], [0], color=c, marker=m, lw=2,
               label=f'{g}  (Gx={gx}, Gy={gy})')
               for g,(gx,gy),c,m in zip(GRIDS,GROUPS,COLORS,MARKERS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,y),
               ncol=3, frameon=False)


def lines(ax, data, metric):
    for grid, color, marker in zip(GRIDS, COLORS, MARKERS):
        part = data[data.grid == grid].set_index('target_retained_work_ratio')
        y = part[metric].reindex(RATIOS)
        # NaNs break lines; missing points are neither zero nor interpolated.
        ax.plot(RATIOS, y, color=color, marker=marker, lw=2, markersize=6)
    ax.set_xticks(RATIOS)
    ax.set_xlim(.17,.83)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)


def save(fig, output, stem):
    for ext in ['png','pdf']:
        fig.savefig(output/f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_speedups(data, dense, blocks, output):
    fig, axs = plt.subplots(2,2,figsize=(12,8.7))
    fig.suptitle('Low-rank speedup across PE grid configurations', y=.99, fontsize=17)
    legend(fig, .955)
    panels = [('Block linear projections', blocks)] + [
        (TITLES[p], data[data.projection == p]) for p in MULT]
    for ax,(title,part) in zip(axs.flat,panels):
        lines(ax, part, 'speedup')
        ax.axhline(1,color='.4',ls='--',lw=1)
        ax.set_title(title)
        ax.set_xlabel('Target retained-work ratio')
        ax.set_ylabel('Speedup over dense 64×64')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:.1f}×'))
        ax.set_ylim(.55,2.1)
        ax.set_yticks(np.arange(.6,2.01,.2))
    axs[0,0].text(.97,.95,'128×32: no complete block yet',transform=axs[0,0].transAxes,
                  ha='right',va='top',fontsize=10,color='.4')
    fig.text(.5,.025,'Profiled mean cycles; fixed profiled dense 64×64 baseline (1×). '
             'Block = 4×qkvo + 2×gate/up + down.\n'
             'Completed cases only; no cross-grid mixing. Other decoder operations and redistribution are excluded.',
             ha='center',fontsize=10,color='.35')
    fig.tight_layout(rect=(0,.085,1,.89),h_pad=2)
    save(fig,output,'speedup_by_grid')


def plot_communication(data, dense, output):
    fig,axs=plt.subplots(3,3,figsize=(14,11),sharex=True,sharey='row')
    fig.suptitle('Where communication cycles are spent',y=.99,fontsize=17)
    legend(fig,.96)
    metrics=['reduce1_cycles','reduce2_cycles','communication_cycles']
    labels=['Stage 1: Y all-reduce','Stage 2: X all-reduce','Total communication']
    for col,p in enumerate(MULT):
        part=data[data.projection==p]
        for row,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axs[row,col]
            lines(ax,part,metric)
            ax.set_ylim(bottom=0)
            if col==0: ax.set_ylabel(f'{label}\nMean cycles per PE')
            if row==0: ax.set_title(TITLES[p])
            if row==2:
                value=dense.loc[p,'communication_cycles']
                ax.axhline(value,color='.4',ls='--',lw=1)
                ax.text(.98,.04,f'Dense 64×64: {value:,.0f}',ha='right',va='bottom',
                        transform=ax.transAxes,fontsize=9,color='.35')
                ax.set_xlabel('Target retained-work ratio')
    for row, metric in enumerate(metrics):
        ceiling = float(data[metric].max())
        if row == 2:
            ceiling = max(ceiling, float(dense.communication_cycles.max()))
        axs[row,0].set_ylim(0, ceiling*1.12)
    fig.text(.5,.025,'Communication = timed all-reduce, including broadcast and waiting; excludes host transfers.\n'
             'Dashed lines in the bottom row show dense communication. Missing cases are left unplotted.',
             ha='center',fontsize=10,color='.35')
    fig.tight_layout(rect=(0,.075,1,.915),h_pad=1.8,w_pad=1.6)
    save(fig,output,'communication_stages')


def plot_share(data, output):
    fig,axs=plt.subplots(1,3,figsize=(14,4.9))
    fig.suptitle('Communication share of the profiled iteration',y=.99,fontsize=17)
    for ax,p in zip(axs,MULT):
        matrix=data[data.projection==p].pivot(index='grid',columns='target_retained_work_ratio',
                                              values='communication_share_percent').reindex(index=GRIDS,columns=RATIOS)
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=100,cmap=cmap,aspect='auto')
        for (r,c),v in np.ndenumerate(matrix.to_numpy()):
            if np.isfinite(v): ax.text(c,r,f'{v:.1f}%',ha='center',va='center',
                                       color='white' if v>55 else '#172b3a',fontsize=12)
        ax.set_title(TITLES[p])
        ax.set_xticks(range(4),[str(x) for x in RATIOS])
        ax.set_yticks(range(3),GRIDS)
        ax.set_xlabel('Target retained-work ratio')
    axs[0].set_ylabel('PE grid (X × Y)')
    fig.subplots_adjust(left=.08,right=.88,bottom=.24,top=.78,wspace=.35)
    cax=fig.add_axes([.91,.24,.014,.54]);fig.colorbar(im,cax=cax,label='Communication share (%)')
    fig.text(.5,.07,'Share = (stage-1 + stage-2 all-reduce) / profiled iteration. '
             'Includes broadcast; gray cells are unfinished.\n'
             'Every configuration uses eight PEs per local group on each axis.',ha='center',fontsize=10,color='.35')
    save(fig,output,'communication_share')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=ROOT/'profiled_sweep_all_logs_20261015_194347_UTC.zip')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'profiled_sweep_plots')
    parser.add_argument('--dense-run',default='profile_full_dense64_20261003')
    parser.add_argument('--fused-run',default='profile_full_rect_20261003')
    args=parser.parse_args()
    data,dense,blocks=read_data(args.zip,args.dense_run,args.fused_run)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    setup()
    plot_speedups(data,dense,blocks,output)
    plot_communication(data,dense,output)
    plot_share(data,output)
    data.to_csv(output/'completed_projections.csv',index=False)
    dense.to_csv(output/'dense_baseline.csv')
    blocks.to_csv(output/'complete_blocks.csv',index=False)
    dense_total=float(sum(dense.loc[p,'mean_cycles']*m for p,m in MULT.items()))
    (output/'README.md').write_text(f'''# Profiled grid comparison

Source archive: {args.zip.name}
Selected runs: `{args.dense_run}` and `{args.fused_run}`.

- {len(data)} completed low-rank cases; {len(dense)} completed dense baseline cases.
- {len(blocks)} complete grid/ratio block totals. Partial blocks are not plotted.
- Fixed dense 64x64 baseline: {dense_total:,.0f} profiled mean cycles for the seven projections.
- Speedup = dense mean cycles / low-rank mean cycles; block cycles = 4*qkvo + 2*gate_up + down.
- qkvo denotes a representative Q/K/V/O projection; gate_up represents gate and up. These are not measurements of individual transformer depth indices.
- All selected cases use phase profiling and one measured repetition. No unprofiled runs or duplicate smoke cases are mixed in.
- Mean cycles are used throughout; max-PE phase costs are not summed.
- Communication is measured all-reduce time, including broadcast and waiting. It excludes host transfers and is not a pure link-bandwidth measurement.
- Communication share uses the inner profiled iteration, whose compute + reductions + control sum is checked. Outer timed cycles are used for speedup and include extra timing/exit work.
- No output redistribution or other decoder operations are included in block totals.
- Target-ratio fields were blank in the uploaded low-rank CSV. Labels are recovered from the agreed exact rank schedule: qkvo 384/768/1280/1664; gate_up/down 640/1152/1792/2432 correspond to targets 0.2/0.4/0.6/0.8. Actual ratios remain available in completed_projections.csv.
- Eight PEs per local group: grids 64x64 / 32x128 / 128x32 use group counts (8,8) / (4,16) / (16,4).
- Missing values are not filled, extrapolated, or treated as zero.

Figures are provided as PNG and vector PDF. The CSVs preserve per-case source JSON paths.
Regenerate with `python3 results/plot_profiled_sweep.py` from the repository root.
''')
    print(f'{len(data)} fused cases, {len(dense)} dense cases, {len(blocks)} complete blocks')
    print(blocks.to_string(index=False))
    print(f'Figures saved to {output.resolve()}')


if __name__=='__main__':
    main()
