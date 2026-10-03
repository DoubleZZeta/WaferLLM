#!/usr/bin/env python3
"""Run the agreed 45-case grid comparison with isolated, profiled jobs.

Use --dry-run to review every command without compiling or creating artifacts.
Run inside tmux on the server. This queue's --jobs does not count other queues.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
from pathlib import Path
import re
import subprocess
import sys

import bench
import rectangular

GRIDS = ((64, 64, 8, 8), (32, 128, 4, 16), (128, 32, 16, 4))
RATIOS = (.2, .4, .6, .8)
RANKS = {'qkvo': (384, 768, 1280, 1664),
         'gate_up': (640, 1152, 1792, 2432),
         'down': (640, 1152, 1792, 2432)}


def cases(run_name, runner='bench', build_root=None, resume=False, low_rank_only=False):
    """Construct and validate exactly 9 dense + 36 fused cases, no grid product."""
    jobs = []
    for case in bench.CASES:
        for px, py, gx, gy in GRIDS:
            for rank, ratio in [(None, None), *zip(RANKS[case.name], RATIOS)]:
                if low_rank_only and rank is None:
                    continue
                execution = 'dense' if rank is None else 'fused'
                slug = f'{case.name}_px{px}_py{py}_gx{gx}_gy{gy}_' + (
                    'dense' if rank is None else f'r{rank}')
                name = f'{run_name}_{slug}'
                options = ['--run-name', name, '--PX', str(px), '--PY', str(py),
                           '--group-num-x', str(gx), '--group-num-y', str(gy),
                           '--profile-phases', '--repetitions', '1', '--suppress-simfab-trace']
                if runner == 'bench':
                    options += ['--projections', case.name]
                else:
                    options += ['--M', str(case.M), '--N', str(case.N)]
                if rank is not None:
                    options += ['--ranks', str(rank), '--rank-target-ratios', str(ratio)]
                if build_root:
                    options += ['--build-root', str(build_root)]
                if resume:
                    options += ['--resume']
                planned = rectangular.plan(rectangular.parse_args(options, runner, execution), runner)
                if len(planned) != 1 or planned[0][1]:
                    raise ValueError(f'invalid suite case {slug}: {planned}')
                row = planned[0][0]
                # Custom sweep names are dimension-based; preserve the logical projection here.
                row.update(projection_group=case.name, multiplicity=case.multiplicity,
                           execution=execution, profile_phases=True, repetitions=1,
                           pe_num_group_x=px//gx, pe_num_group_y=py//gy)
                root = bench.RUNS_DIR if runner == 'bench' else bench.BENCH_DIR/'sweeps'
                jobs.append(dict(slug=slug, row=row, run_dir=str(root/name),
                    command=[sys.executable, str(bench.BENCH_DIR/f'{runner}.py'),
                             f'{execution}-rectangular', *options]))
    return jobs


def collect(jobs, suite_dir):
    """Merge only this manifest's results, preserving missing/failed cases."""
    rows, phases = [], []
    for job in jobs:
        row = {**job['row'], 'status': 'not_run', 'error_message': ''}
        execution = row['execution']
        run_dir = Path(job['run_dir'])
        results = bench.read_rows(run_dir/f'{execution}-rectangular.csv')
        if len(results) == 1:
            saved = results[0]
            row.update(status=saved['status'], error_message=saved.get('error_message', ''))
            for metric in ['mean_cycles', 'max_cycles', 'relative_l2_error']:
                row[metric] = saved.get(f'{execution}_{metric}', '')
            if row['status'] == 'ok':
                profiles = list((run_dir/'logs').glob('*.phases.json'))
                if len(profiles) != 1:
                    row.update(status='failed', error_message='expected exactly one phase JSON')
                else:
                    profile = json.loads(profiles[0].read_text())
                    for phase, metrics in profile['phases'].items():
                        phases.append({**row, 'phase': phase,
                                       'phase_mean_cycles': metrics['mean_cycles'],
                                       'phase_max_pe_cycles': metrics['max_pe_cycles']})
        if job.get('return_code', 0):
            row.update(status='failed', error_message=row['error_message'] or
                       f"runner exited {job['return_code']}; see queue log")
        rows.append(row)
    fields = ('status', 'projection_group', 'execution', 'multiplicity', 'M', 'N', 'rank',
              'target_retained_work_ratio', 'retained_work_ratio', 'Px', 'Py',
              'group_num_x', 'group_num_y', 'pe_num_group_x', 'pe_num_group_y', 'pe_count',
              'profile_phases', 'repetitions', 'mean_cycles', 'max_cycles',
              'relative_l2_error', 'error_message')
    bench.write_rows(suite_dir/'summary.csv', fields, rows)
    bench.write_rows(suite_dir/'phases.csv', (*fields, 'phase', 'phase_mean_cycles',
                                           'phase_max_pe_cycles'), phases)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--runner', choices=['bench', 'sweep'], default='bench')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--low-rank-only', action='store_true',
                        help='run 36 fused cases; retain your separate original dense baseline')
    parser.add_argument('--build-root')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_name) or args.jobs < 1:
        parser.error('use a valid run name and a positive --jobs count')
    jobs = cases(args.run_name, args.runner, args.build_root, args.resume, args.low_rank_only)
    suite_dir = bench.BENCH_DIR/'suites'/args.run_name
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        print(f'{len(jobs)} cases: {0 if args.low_rank_only else 9} dense, 36 fused; all profiled; runner={args.runner}')
        return 0
    sdk_dir = Path(os.environ.get('CEREBRAS_SDK_DIR', bench.REPO_DIR.parent/'cs_sdk'))
    if (sdk_dir/'cslc').is_file() and (sdk_dir/'cs_python').is_file():
        os.environ['PATH'] = str(sdk_dir) + os.pathsep + os.environ['PATH']
    if not all(shutil.which(tool) for tool in ['cslc', 'cs_python']):
        parser.error('cslc/cs_python not found; set CEREBRAS_SDK_DIR or PATH to the SDK launchers')
    os.environ['PYTHONUNBUFFERED'] = '1'
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = suite_dir/'manifest.json'
    if manifest.exists() and not args.resume:
        parser.error('suite already exists; use --resume or a new run name')
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if [(j['slug'], j['run_dir'], j['row']) for j in previous] != [
                (j['slug'], j['run_dir'], j['row']) for j in jobs]:
            parser.error('suite definition changed; use a new run name')
    manifest.write_text(json.dumps(jobs, indent=2)+'\n')
    def run(job):
        print(f"START {job['slug']}", flush=True)
        with (suite_dir/f"{job['slug']}.log").open('a') as log:
            result = subprocess.run(job['command'], cwd=bench.REPO_DIR, stdout=log,
                                    stderr=subprocess.STDOUT)
        job['return_code'] = result.returncode
        print(f"FINISH {job['slug']}: exit {result.returncode}", flush=True)
        return job
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            collect(jobs, suite_dir)
    rows = collect(jobs, suite_dir)
    print(f"Results: {suite_dir}; {sum(r['status']=='ok' for r in rows)}/{len(rows)} passed")
    return int(any(r['status'] != 'ok' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
