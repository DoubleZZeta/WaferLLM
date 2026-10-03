#!/usr/bin/env python3
"""Parameter sweep for dense MeshGEMV and fused low-rank GEMV on the simulator.

Unlike bench.py, which runs the fixed Llama-2 projection shapes, this runs
every combination of the configured P, group_num, M, N and rank values:

  sweep.py dense   -> sweeps/<run>/dense.csv    one row per (P, group_num, M, N)
  sweep.py fused   -> sweeps/<run>/fused.csv    one row per (P, group_num, M, N, R)
  sweep.py summary -> sweeps/<run>/summary.csv  fused rows joined with dense

A dense result does not depend on rank, so each fused row pairs with the dense
row of the same (P, group_num, M, N). Combinations the kernels cannot run are
listed as skipped. Compiling, launching, resuming and the --hcf-fix selection
are shared with bench.py.
"""
from __future__ import annotations

import argparse
import itertools
import re
from datetime import datetime

import bench

SWEEPS_DIR = bench.BENCH_DIR / "sweeps"

# Only run-wide settings are locked in config.json. P, group_num, M, N and R
# are sweep axes recorded on every row, so later commands may add new values.
DEFAULT_CONFIG = {"repetitions": 3, "hcf_fix": False, "profile_phases": False}

SHAPE_FIELDS = ("status", "P", "group_num", "repetitions", "M", "N")
RANK_FIELDS = ("rank", "target_retained_work_ratio", "retained_work_ratio")

DENSE_FIELDS = SHAPE_FIELDS + (
    "dense_mean_cycles", "dense_max_cycles", "dense_max_relative_error",
    "error_message",
)

FUSED_FIELDS = SHAPE_FIELDS + RANK_FIELDS + (
    "fused_mean_cycles", "fused_max_cycles", "fused_relative_l2_error",
    "error_message",
)

SUMMARY_FIELDS = SHAPE_FIELDS + RANK_FIELDS + (
    "dense_mean_cycles", "fused_mean_cycles",
    "fused_faster_than_dense", "fused_speedup_vs_dense",
    "fused_relative_l2_error", "missing",
)

DENSE_COLUMNS = {
    "mean_cycles": "dense_mean_cycles",
    "max_cycles": "dense_max_cycles",
    "max_relative_error": "dense_max_relative_error",
}

FUSED_COLUMNS = {
    "mean_cycles": "fused_mean_cycles",
    "max_cycles": "fused_max_cycles",
    "relative_l2_error": "fused_relative_l2_error",
}


# ------------------------------- Arguments -------------------------------- #

def comma_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--profile-phases", action=argparse.BooleanOptionalAction, default=None,
        help="profile one warmed iteration; use a separate run name (saved in config)")
    options.add_argument(
        "--run-name",
        default=f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="subfolder under sweeps/ shared by dense and fused (default: timestamp)",
    )
    options.add_argument("--P", type=comma_ints, default=[4],
                         help="comma-separated PE grid dimensions (default: 4)")
    options.add_argument("--group-num", dest="group_num", type=comma_ints, default=[2],
                         help="comma-separated reduce group counts (default: 2)")
    options.add_argument("--M", type=comma_ints, required=True,
                         help="comma-separated input dimensions")
    options.add_argument("--N", type=comma_ints, required=True,
                         help="comma-separated output dimensions")
    options.add_argument(
        "--repetitions", type=int,
        help=(
            "measured executions per simulator invocation, after one warm-up "
            f"(default: the run's config, else {DEFAULT_CONFIG['repetitions']})"
        ),
    )
    options.add_argument("--resume", action="store_true",
                         help="skip cases already marked ok in this run's CSV")
    options.add_argument("--dry-run", action="store_true",
                         help="print the planned cases without compiling or writing files")
    options.add_argument("--suppress-simfab-trace", action="store_true",
                         help="disable bulk simulator instruction/wavelet traces")
    options.add_argument(
        "--hcf-fix", action=argparse.BooleanOptionalAction, default=None,
        help="legacy: select the separate MeshGEMVPatched snapshot; normal launchers also include the fix",
    )
    options.add_argument(
        "--build-root", metavar="DIR",
        help=(
            "put compiled artifacts and simulator logs under DIR instead of "
            "the run's build/ folder (default: the run's saved choice)"
        ),
    )

    parser = argparse.ArgumentParser(
        description="Sweep dense and fused low-rank GEMV over parameter combinations"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("dense", parents=[options],
                           help="run the dense MeshGEMV baseline")
    fused = subcommands.add_parser("fused", parents=[options],
                                   help="run the fused low-rank kernel")
    fused.add_argument("--ranks", type=comma_ints, default=[],
                       help="comma-separated exact ranks R")
    fused.add_argument(
        "--retained-work-ratios", type=bench.comma_floats, default=[],
        help="comma-separated target R(M+N)/(MN); R is rounded to a multiple of P",
    )
    subcommands.add_parser("fused-rectangular", help="run rectangular low-rank GEMV; use fused-rectangular --help")
    summary = subcommands.add_parser("summary",
                                     help="rebuild summary.csv from dense.csv and fused.csv")
    summary.add_argument("--run-name", required=True)
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if not args.P or min(args.P) < 3:
        raise SystemExit("P values must be at least 3")
    if not args.group_num or min(args.group_num) < 1:
        raise SystemExit("group_num values must be positive")
    if not args.M or not args.N or min(args.M + args.N) < 1:
        raise SystemExit("M and N values must be positive")
    if args.command != "fused":
        return
    if not args.ranks and not args.retained_work_ratios:
        raise SystemExit("fused needs --ranks, --retained-work-ratios, or both")
    if args.ranks and min(args.ranks) < 1:
        raise SystemExit("ranks must be positive")
    for ratio in args.retained_work_ratios:
        if not 0 < ratio <= 1:
            raise SystemExit(f"retained-work ratio must be in (0,1], got {ratio}")


# ---------------------------------- Plan ---------------------------------- #

def retained_ratio(M: int, N: int, rank: int) -> float:
    return rank * (M + N) / (M * N)


def shape_problem(P: int, group_num: int, M: int, N: int) -> str | None:
    if P % group_num:
        return "group_num must divide P"
    if M % P or N % P:
        return "M and N must be divisible by P"
    return None


def dense_plan(args: argparse.Namespace) -> list[tuple]:
    """Every (P, group_num, M, N) as (key, None, reason it is skipped or None)."""
    return [
        (shape, None, shape_problem(*shape))
        for shape in itertools.product(args.P, args.group_num, args.M, args.N)
    ]


def fused_plan(args: argparse.Namespace) -> list[tuple]:
    """Every (P, group_num, M, N, R) as (key, target ratio, skip reason).

    Explicit ranks come first, then ranks derived from retained-work ratios. A
    ratio that rounds to a rank already planned for the shape adds no case.
    """
    plan = []
    for P, group_num, M, N in itertools.product(args.P, args.group_num, args.M, args.N):
        problem = shape_problem(P, group_num, M, N)
        case = bench.ProjectionCase("", M, N, 1)
        candidates = [(rank, None) for rank in args.ranks] + [
            (bench.choose_rank(case, ratio, P), ratio)
            for ratio in args.retained_work_ratios
        ]
        planned_ranks: set[int] = set()
        for rank, ratio in candidates:
            if rank in planned_ranks:
                continue
            planned_ranks.add(rank)
            rank_problem = problem
            if rank_problem is None and rank % P:
                rank_problem = "rank must be divisible by P"
            elif rank_problem is None and not 0 < rank <= min(M, N):
                rank_problem = "rank must satisfy 0 < R <= min(M,N)"
            plan.append(((P, group_num, M, N, rank), ratio, rank_problem))
    return plan


def describe(key: tuple, ratio: float | None) -> str:
    text = "P={}, group_num={}, M={}, N={}".format(*key[:4])
    if len(key) == 5:
        P, _, M, N, rank = key
        text += f", R={rank}"
        if ratio is not None:
            text += f", target_rho={ratio:g}"
        text += f", actual_rho={retained_ratio(M, N, rank):.6f}"
    return text


# ------------------------------- Results ---------------------------------- #

def dense_key(row: dict) -> tuple[int, ...]:
    return tuple(int(row[field]) for field in ("P", "group_num", "M", "N"))


def fused_key(row: dict) -> tuple[int, ...]:
    return dense_key(row) + (int(row["rank"]),)


def write_summary(run_dir) -> list[dict]:
    """Join every fused row with the dense row of its (P, group_num, M, N)."""
    dense = {
        dense_key(row): row
        for row in bench.read_rows(run_dir / "dense.csv") if row["status"] == "ok"
    }
    rows = []
    for fused_row in sorted(bench.read_rows(run_dir / "fused.csv"), key=fused_key):
        dense_row = dense.get(dense_key(fused_row))
        fused_ok = fused_row["status"] == "ok"
        missing = [
            name for name, present in (("dense", dense_row is not None), ("fused", fused_ok))
            if not present
        ]
        row: dict[str, object] = {
            field: fused_row[field] for field in SHAPE_FIELDS[1:] + RANK_FIELDS
        }
        row.update(status="incomplete" if missing else "ok", missing=" ".join(missing))
        if dense_row:
            row["dense_mean_cycles"] = dense_row["dense_mean_cycles"]
        if fused_ok:
            row["fused_mean_cycles"] = fused_row["fused_mean_cycles"]
            row["fused_relative_l2_error"] = fused_row["fused_relative_l2_error"]
        if not missing:
            dense_cycles = float(dense_row["dense_mean_cycles"])
            fused_cycles = float(fused_row["fused_mean_cycles"])
            row["fused_faster_than_dense"] = fused_cycles < dense_cycles
            row["fused_speedup_vs_dense"] = dense_cycles / fused_cycles
        rows.append(row)
    bench.write_rows(run_dir / "summary.csv", SUMMARY_FIELDS, rows)
    return rows


def print_summary(run_dir, rows: list[dict]) -> None:
    complete = [row for row in rows if row["status"] == "ok"]
    print(
        f"\nSummary: {run_dir / 'summary.csv'} "
        f"({len(complete)} complete, {len(rows) - len(complete)} incomplete)"
    )
    for row in complete:
        print(
            f"  P={row['P']}, group_num={row['group_num']}, M={row['M']}, "
            f"N={row['N']}, R={row['rank']}: "
            f"fused speedup {row['fused_speedup_vs_dense']:.3f}x"
        )


# -------------------------------- Running --------------------------------- #

def run_command(args: argparse.Namespace, run_dir, config: dict) -> int:
    kind = args.command
    if kind == "fused":
        plan = fused_plan(args)
        kernel, launcher = "MeshGEMVLowerRank", bench.fused_launcher(config)
        fields, key, columns = FUSED_FIELDS, fused_key, FUSED_COLUMNS
    else:
        plan = dense_plan(args)
        kernel, launcher = bench.dense_implementation(config)
        fields, key, columns = DENSE_FIELDS, dense_key, DENSE_COLUMNS

    runnable = [(case_key, ratio) for case_key, ratio, problem in plan if problem is None]
    print(f"Implementation: {kernel}/compile.py + {launcher}")
    print(
        f"Planned {kind} configurations: {len(runnable)} to run, "
        f"{len(plan) - len(runnable)} skipped"
    )
    for case_key, ratio, problem in plan:
        print(f"  {describe(case_key, ratio)}" + (f"; SKIP: {problem}" if problem else ""))
    if args.dry_run:
        return 0

    results_path = run_dir / f"{kind}.csv"
    bench.prepare_results(results_path, fields, args.resume)
    done = bench.completed_keys(results_path, key) if args.resume else set()
    for index, (case_key, ratio) in enumerate(runnable, 1):
        label = describe(case_key, ratio)
        if case_key in done:
            print(f"Skipping completed {kind} case {label}")
            continue
        print(f"\n=== [{kind} {index}/{len(runnable)}] {label} ===", flush=True)

        P, group_num, M, N = case_key[:4]
        row: dict[str, object] = {
            "status": "failed", "P": P, "group_num": group_num,
            "repetitions": config["repetitions"], "M": M, "N": N,
            "error_message": "",
        }
        shape = ["--P", str(P), "--M", str(M)]
        slug = f"{kind}_p{P}_g{group_num}_m{M}"
        if kind == "fused":
            rank = case_key[4]
            shape += ["--R", str(rank)]
            slug += f"_r{rank}"
            row.update(
                rank=rank,
                target_retained_work_ratio="" if ratio is None else ratio,
                retained_work_ratio=retained_ratio(M, N, rank),
            )
        shape += ["--N", str(N), "--group_num", str(group_num)]
        slug += f"_n{N}"

        row = bench.run_case(
            row,
            bench.kernel_commands(kernel, launcher, shape, config,
                                  run_dir / "build" / slug, args.suppress_simfab_trace,
                                  args.build_root),
            run_dir / "logs" / f"{slug}.log",
            columns,
        )
        bench.upsert_result(results_path, fields, key, row)
        write_summary(run_dir)
        print(f"Saved: {results_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "fused-rectangular":
        import rectangular
        return rectangular.main(arguments[1:], mode="sweep")
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        raise SystemExit(
            "run name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or dashes"
        )
    run_dir = SWEEPS_DIR / args.run_name

    if args.command == "summary":
        if not (run_dir / "config.json").exists():
            raise SystemExit(f"no sweep found at {run_dir}")
        print_summary(run_dir, write_summary(run_dir))
        return 0

    validate(args)
    config = bench.resolve_config(
        args, run_dir, write=not args.dry_run, defaults=DEFAULT_CONFIG
    )
    if not 1 <= config["repetitions"] <= 32767:
        raise SystemExit("repetitions must be between 1 and 32767")
    args.build_root = bench.resolve_build_root(args, run_dir, write=not args.dry_run)
    print(f"Run: {args.run_name} ({args.command})")
    print("Timing mode: " + ("phase profiling (instrumented)" if config.get("profile_phases") else "normal"))
    print(
        f"repetitions={config['repetitions']} (+ 1 warm-up), "
        f"hcf_fix={config['hcf_fix']}"
    )
    print(f"Output directory: {run_dir}")
    print(f"Build directory: {bench.build_location(run_dir, args.build_root)}")

    status = run_command(args, run_dir, config)
    if not args.dry_run:
        print_summary(run_dir, write_summary(run_dir))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
