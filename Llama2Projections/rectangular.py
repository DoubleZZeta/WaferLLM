"""Shared rectangular mode for the fixed-shape and custom-sweep runners."""
from __future__ import annotations

import argparse
import itertools
import re
from datetime import datetime

import bench

DEFAULT_CONFIG = {"kernel": "MeshGEMVLowerRankRectangular", "repetitions": 3,
                  "profile_phases": False}
FIELDS = ("status", "projection_group", "multiplicity", "Px", "Py", "group_num_x",
          "group_num_y", "pe_count", "M", "N", "rank", "target_retained_work_ratio",
          "retained_work_ratio", "repetitions", "profile_phases", "fused_mean_cycles",
          "fused_max_cycles", "fused_relative_l2_error", "error_message")
COLUMNS = {"mean_cycles": "fused_mean_cycles", "max_cycles": "fused_max_cycles",
           "relative_l2_error": "fused_relative_l2_error"}


def integers(value):
    try:
        values = [int(v.strip()) for v in value.split(',')]
        if not values:
            raise ValueError()
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers")


def parse_args(argv, mode, execution="fused"):
    dense = execution == "dense"
    p = argparse.ArgumentParser(description=f"Rectangular {execution} GEMV benchmark")
    p.add_argument('--run-name', default='rect_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--PX', '--px', dest='px', type=integers, required=True)
    p.add_argument('--PY', '--py', dest='py', type=integers, required=True)
    p.add_argument('--group-num-x', '--group_num_x', dest='gx', type=integers, required=True)
    p.add_argument('--group-num-y', '--group_num_y', dest='gy', type=integers, required=True)
    p.add_argument('--ranks', type=integers, default=[])
    p.add_argument('--retained-work-ratios', type=bench.comma_floats, default=[])
    p.add_argument('--rank-target-ratios', type=bench.comma_floats, default=[],
                   help='target labels paired with explicit --ranks (no rank rounding)')
    p.add_argument('--repetitions', type=int)
    p.add_argument('--profile-phases', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--suppress-simfab-trace', action='store_true')
    p.add_argument('--build-root')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    if mode == 'sweep':
        p.add_argument('--M', type=integers, required=True)
        p.add_argument('--N', type=integers, required=True)
    else:
        p.add_argument('--projections', nargs='+', choices=[c.name for c in bench.CASES],
                       default=[c.name for c in bench.CASES])
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_name):
        p.error('invalid run name')
    if min(args.px + args.py) < 3 or min(args.gx + args.gy) < 1:
        p.error('PX/PY must be at least 3 and group counts positive')
    if mode == 'sweep' and min(args.M + args.N) < 1:
        p.error('M/N must be positive')
    if args.ranks and min(args.ranks) < 1:
        p.error('ranks must be positive')
    if any(not 0 < r <= 1 for r in args.retained_work_ratios):
        p.error('retained-work ratios must be in (0,1]')
    if dense and (args.ranks or args.retained_work_ratios or args.rank_target_ratios):
        p.error("dense cases do not accept rank options")
    if args.rank_target_ratios and (len(args.rank_target_ratios) != len(args.ranks) or
            args.retained_work_ratios or any(not 0 < r <= 1 for r in args.rank_target_ratios)):
        p.error("--rank-target-ratios must pair with --ranks, with ratios in (0,1]")
    if not dense and not args.ranks and not args.retained_work_ratios:
        p.error('provide --ranks or --retained-work-ratios')
    if args.repetitions is not None and not 1 <= args.repetitions <= 32767:
        p.error('repetitions must be between 1 and 32767')
    args.execution = execution
    return args


def key(row):
    return (str(row['projection_group']),) + tuple(int(row[k]) if row[k] != '' else 0 for k in
            ('Px', 'Py', 'group_num_x', 'group_num_y', 'M', 'N', 'rank'))


def plan(args, mode):
    shapes = ([(f'm{m}_n{n}', m, n, 1) for m, n in itertools.product(args.M, args.N)]
              if mode == 'sweep' else
              [(c.name, c.M, c.N, c.multiplicity) for c in bench.CASES if c.name in args.projections])
    dense = args.execution == "dense"
    rows, seen = [], set()
    for px, py, gx, gy, (name, m, n, mult) in itertools.product(
            args.px, args.py, args.gx, args.gy, shapes):
        case = bench.ProjectionCase(name, m, n, mult)
        candidates = list(zip(args.ranks, args.rank_target_ratios or [None]*len(args.ranks))) + [
            (bench.choose_rank(case, ratio, px), ratio) for ratio in args.retained_work_ratios]
        if dense:
            candidates = [(0, None)]
        for rank, ratio in candidates:
            row = dict(status='failed', projection_group=name, multiplicity=mult,
                       Px=px, Py=py, group_num_x=gx, group_num_y=gy, pe_count=px*py,
                       M=m, N=n, rank='' if dense else rank, target_retained_work_ratio=ratio if ratio is not None else '',
                       retained_work_ratio=1.0 if dense else rank*(m+n)/(m*n), error_message='')
            if key(row) in seen:
                continue
            seen.add(key(row))
            reason = None
            if px % gx or py % gy:
                reason = 'each group count must divide its own axis'
            elif dense and (m % py or n % px):
                reason = "dense requires M divisible by PY and N divisible by PX"
            elif dense and (gy < 2 or py // gy < 2):
                reason = "dense requires at least 2 Y groups and 2 PEs per Y group"
            elif not dense and (m % py or n % py):
                reason = 'M and N must be divisible by PY'
            elif not dense and (rank % px or not 0 < rank <= min(m, n)):
                reason = 'require R divisible by PX and 0 < R <= min(M,N)'
            rows.append((row, reason))
    return rows


def shape_args(row):
    return [item for flag, field in [('PX', 'Px'), ('PY', 'Py'), ('M', 'M'), ('R', 'rank'),
                                    ('N', 'N'), ('group_num_x', 'group_num_x'),
                                    ('group_num_y', 'group_num_y')]
            if field != 'rank' or row[field] != ''
            for item in ('--'+flag, str(row[field]))]


def main(argv, mode, execution="fused"):
    args = parse_args(argv, mode, execution)
    dense = execution == "dense"
    kernel = "MeshGEMVRectangular" if dense else DEFAULT_CONFIG['kernel']
    defaults = {**DEFAULT_CONFIG, "kernel": kernel}
    fields = tuple(f.replace('fused_', 'dense_') for f in FIELDS) if dense else FIELDS
    columns = {k: v.replace('fused_', 'dense_') for k, v in COLUMNS.items()} if dense else COLUMNS
    run_dir = (bench.RUNS_DIR if mode == 'bench' else bench.BENCH_DIR/'sweeps') / args.run_name
    config = bench.resolve_config(args, run_dir, write=not args.dry_run, defaults=defaults)
    build_root = bench.resolve_build_root(args, run_dir, write=not args.dry_run)
    cases = plan(args, mode)
    print(f'Rectangular run: {run_dir}; profiling={config["profile_phases"]}; repetitions={config["repetitions"]}')
    for row, reason in cases:
        print(f'{"SKIP" if reason else "RUN"}: {row["projection_group"]} '
              f'PX={row["Px"]}, PY={row["Py"]}, Gx={row["group_num_x"]}, Gy={row["group_num_y"]}, '
              f'M={row["M"]}, R={row["rank"]}, N={row["N"]}, '
              f'actual_rho={row["retained_work_ratio"]:.6f}' + (f' ({reason})' if reason else ''))
    if args.dry_run:
        return 0
    path = run_dir/f'{execution}-rectangular.csv'
    bench.prepare_results(path, fields, args.resume)
    done = bench.completed_keys(path, key) if args.resume else set()
    failures = 0
    for row, reason in cases:
        if key(row) in done:
            continue
        if reason:
            row['error_message'] = reason
            bench.upsert_result(path, fields, key, row)
            failures += 1
            continue
        row.update(repetitions=config['repetitions'], profile_phases=config['profile_phases'])
        slug = (f'{row["projection_group"]}_px{row["Px"]}_py{row["Py"]}'
                f'_gx{row["group_num_x"]}_gy{row["group_num_y"]}'
                f'_m{row["M"]}_r{row["rank"]}_n{row["N"]}')
        commands = bench.kernel_commands(kernel, f'launch_{execution}_rectangular.py',
                    shape_args(row), config, run_dir/'build'/slug, args.suppress_simfab_trace, build_root)
        row = bench.run_case(row, commands, run_dir/'logs'/f'{slug}.log', columns)
        bench.upsert_result(path, fields, key, row)
        failures += row['status'] != 'ok'
    print(f'Results: {path}')
    return int(failures > 0)
