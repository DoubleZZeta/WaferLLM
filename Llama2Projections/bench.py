#!/usr/bin/env python3
"""Simulator benchmark for the linear projections in one Llama-2 7B block.

One representative GEMV is run for each distinct projection shape:

  qkvo    : 4 x [1,4096]  @ [4096,4096]
  gate_up : 2 x [1,4096]  @ [4096,11008]
  down    : 1 x [1,11008] @ [11008,4096]

Dense and fused low-rank kernels are separate subcommands with separate result
CSVs, so either can be run, resumed, or repeated on its own:

  bench.py dense   -> runs/<run>/dense.csv  (one row per projection)
  bench.py fused   -> runs/<run>/fused.csv  (one row per projection and ratio)
  bench.py summary -> runs/<run>/projection_summary.csv and block_summary.csv

The summaries join the two CSVs and are also rebuilt after every case.
block_summary.csv weights cycles by multiplicities 4, 2 and 1: it estimates the
linear-projection portion of a block, not end-to-end decoder latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from common import REPO_DIR, RESULT_PREFIX

BENCH_DIR = Path(__file__).resolve().parent
RUNS_DIR = BENCH_DIR / "runs"

# Dense and fused rows are comparable only when these settings match, so they
# are recorded in runs/<run>/config.json by the first subcommand of a run.
DEFAULT_CONFIG = {"P": 64, "group_num": 8, "repetitions": 3, "hcf_fix": False, "dense_timing": False, "profile_phases": False}
DEFAULT_RATIOS = [0.125, 0.25, 0.5, 0.75]


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    M: int
    N: int
    multiplicity: int


CASES = (
    ProjectionCase("qkvo", 4096, 4096, 4),
    ProjectionCase("gate_up", 4096, 11008, 2),
    ProjectionCase("down", 11008, 4096, 1),
)

CASE_FIELDS = (
    "status", "P", "group_num", "repetitions",
    "projection_group", "multiplicity", "M", "N",
)

DENSE_FIELDS = CASE_FIELDS + (
    "dense_mean_cycles", "dense_max_cycles", "dense_max_relative_error",
    "error_message",
)

FUSED_FIELDS = CASE_FIELDS + (
    "target_retained_work_ratio", "rank", "retained_work_ratio",
    "fused_mean_cycles", "fused_max_cycles", "fused_relative_l2_error",
    "error_message",
)

PROJECTION_SUMMARY_FIELDS = CASE_FIELDS + (
    "target_retained_work_ratio", "rank", "retained_work_ratio",
    "dense_mean_cycles", "fused_mean_cycles",
    "fused_faster_than_dense", "fused_speedup_vs_dense",
    "fused_relative_l2_error", "missing",
)

BLOCK_SUMMARY_FIELDS = (
    "status", "P", "group_num", "repetitions", "target_retained_work_ratio",
    "projection_gemv_count", "dense_block_cycles", "fused_block_cycles",
    "fused_speedup_vs_dense", "max_fused_relative_l2_error", "missing",
)


# ------------------------------- Arguments -------------------------------- #

def comma_floats(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument("--dense-timing", action=argparse.BooleanOptionalAction, default=None,
        help="use MeshGEMVTiming detailed dense profiling (requires a separate run name)")
    run_options.add_argument("--profile-phases", action=argparse.BooleanOptionalAction, default=None,
        help="profile one warmed iteration; use a separate run name (saved in config)")
    run_options.add_argument(
        "--run-name",
        default=f"llama2_7b_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="subfolder under runs/ shared by dense and fused (default: timestamp)",
    )
    run_options.add_argument(
        "--P", type=int,
        help=f"PE grid dimension (default: the run's config, else {DEFAULT_CONFIG['P']})",
    )
    run_options.add_argument(
        "--group-num", dest="group_num", type=int,
        help=f"reduce group count (default: the run's config, else {DEFAULT_CONFIG['group_num']})",
    )
    run_options.add_argument(
        "--repetitions", type=int,
        help=(
            "measured executions per simulator invocation, after one warm-up "
            f"(default: the run's config, else {DEFAULT_CONFIG['repetitions']})"
        ),
    )
    run_options.add_argument(
        "--resume", action="store_true",
        help="skip cases already marked ok in this run's CSV",
    )
    run_options.add_argument(
        "--dry-run", action="store_true",
        help="print the planned cases without compiling or writing files",
    )
    run_options.add_argument(
        "--suppress-simfab-trace", action="store_true",
        help="disable bulk simulator instruction/wavelet traces",
    )
    run_options.add_argument(
        "--hcf-fix", action=argparse.BooleanOptionalAction, default=None,
        help="legacy: select the separate MeshGEMVPatched snapshot; normal launchers also include the fix",
    )
    run_options.add_argument(
        "--build-root", metavar="DIR",
        help=(
            "put compiled artifacts and simulator logs under DIR instead of "
            "the run's build/ folder (default: the run's saved choice)"
        ),
    )

    parser = argparse.ArgumentParser(
        description="Benchmark Llama-2 7B projection shapes on the simulator"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "dense", parents=[run_options], help="run the dense MeshGEMV baseline"
    )
    fused = subcommands.add_parser(
        "fused", parents=[run_options], help="run the fused low-rank kernel"
    )
    fused.add_argument(
        "--retained-work-ratios", type=comma_floats, default=DEFAULT_RATIOS,
        help="target R(M+N)/(MN) values (default: .125,.25,.5,.75)",
    )
    subcommands.add_parser("fused-chunked", help="run experimental rank-chunk pipeline; use fused-chunked --help")
    subcommands.add_parser("fused-rectangular", help="run rectangular low-rank GEMV; use fused-rectangular --help")
    subcommands.add_parser("dense-rectangular", help="run rectangular dense GEMV; use dense-rectangular --help")
    subcommands.add_parser("fused-pipelining", help="run two-chunk low-rank pipeline; use fused-pipelining --help")
    summary = subcommands.add_parser(
        "summary", help="rebuild the summary CSVs from dense.csv and fused.csv"
    )
    summary.add_argument("--run-name", required=True)
    return parser.parse_args(argv)


def resolve_config(
    args: argparse.Namespace, run_dir: Path, write: bool,
    defaults: dict = DEFAULT_CONFIG,
) -> dict:
    """Fill unspecified settings from the run's config.json, then lock them in.

    ``defaults`` names the locked settings; sweep.py passes its own.
    """
    path = run_dir / "config.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if saved is not None and saved.get("kernel") != defaults.get("kernel"):
        raise SystemExit("run uses a different kernel mode; use a new run name")
    base = {**defaults, **(saved or {})}
    config = {
        name: base[name] if getattr(args, name, None) is None else getattr(args, name, None)
        for name in defaults
    }
    if config.get("dense_timing"):
        if args.command != "dense":
            raise SystemExit("--dense-timing is supported only for dense runs")
        if getattr(args, "profile_phases", None) is False:
            raise SystemExit("--dense-timing requires profiling")
        config["profile_phases"] = True
    if config.get("profile_phases"):
        if getattr(args, "repetitions", None) not in (None, 1):
            raise SystemExit("--profile-phases requires --repetitions 1 (or omit it)")
        config["repetitions"] = 1
    if saved is not None:
        if any(config[name] != base[name] for name in defaults):
            raise SystemExit(
                f"run '{run_dir.name}' uses {saved}, but this invocation asked "
                f"for {config}; use a new run name"
            )
        return config
    if not write:
        return config

    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / f".config.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        # link() fails if the other subcommand created config.json first.
        os.link(temporary, path)
    except FileExistsError:
        return resolve_config(args, run_dir, write, defaults)
    finally:
        temporary.unlink()
    return config


def resolve_build_root(
    args: argparse.Namespace, run_dir: Path, write: bool
) -> Path | None:
    """Directory for compiled artifacts: --build-root, else the run's saved one.

    Unlike the locked settings this does not affect results, so it may change
    when resuming, e.g. to move builds off a full disk. The latest explicit
    choice is saved so that later commands of the run inherit it.
    """
    path = run_dir / "config.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.build_root is None:
        if not saved.get("build_root"):
            return None
        root = Path(saved["build_root"])
    else:
        root = Path(args.build_root).expanduser().resolve()
    check_build_root(root)
    if write and path.exists() and saved.get("build_root") != str(root):
        temporary = run_dir / f".config.json.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps({**saved, "build_root": str(root)}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return root


def check_build_root(root: Path) -> None:
    """Reject build roots that the SDK containers cannot see at the same path.

    cslc and cs_python mount only the working directory and the host's
    ${TMPDIR:-/tmp} at /tmp. Cases run with TMPDIR unset, so a directory inside
    /tmp has the same path on the host and in both containers. Anywhere else
    needs extra container mounts, which some Singularity installations refuse.
    """
    tmp = Path("/tmp").resolve()
    if root != tmp and tmp not in root.parents:
        raise SystemExit(
            f"--build-root must be inside /tmp, got {root}: the SDK containers "
            "share only /tmp and the working directory with the host"
        )


def build_location(run_dir: Path, build_root: Path | None) -> Path:
    """Host directory holding one run's compiled artifacts."""
    if build_root is None:
        return run_dir / "build"
    return Path(build_root) / run_dir.parent.name / run_dir.name


def validate(config: dict, ratios: list[float]) -> None:
    P, group_num = config["P"], config["group_num"]
    if P < 3:
        raise SystemExit("P must be at least 3")
    if group_num <= 0 or P % group_num:
        raise SystemExit("group_num must be positive and divide P")
    if not 1 <= config["repetitions"] <= 32767:
        raise SystemExit("repetitions must be between 1 and 32767")
    for case in CASES:
        if case.M % P or case.N % P:
            raise SystemExit(
                f"{case.name} shape {case.M}x{case.N} is not divisible by P={P}"
            )
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise SystemExit(f"retained-work ratio must be in (0,1], got {ratio}")


# --------------------------------- Ranks ---------------------------------- #

def actual_ratio(case: ProjectionCase, rank: int) -> float:
    return rank * (case.M + case.N) / (case.M * case.N)


def choose_rank(case: ProjectionCase, target_ratio: float, P: int) -> int:
    """Round the ideal rank to the closest rank that can be sharded over P."""
    ideal = target_ratio * case.M * case.N / (case.M + case.N)
    rank = int(math.floor(ideal / P + 0.5)) * P
    largest_legal = (min(case.M, case.N) // P) * P
    return min(largest_legal, max(P, rank))


# ---------------------------------- CSVs ---------------------------------- #

def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    """Replace a CSV atomically; the other subcommand may read it concurrently."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def prepare_results(path: Path, fields: tuple[str, ...], resume: bool) -> None:
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != fields:
            raise SystemExit(f"{path} has an incompatible schema; use a new run name")
        if not resume:
            raise SystemExit(f"{path} already exists; use --resume or a new run name")
        return
    write_rows(path, fields, [])


def dense_key(row: dict) -> str:
    return str(row["projection_group"])


def fused_key(row: dict) -> tuple[str, float]:
    return str(row["projection_group"]), float(row["target_retained_work_ratio"])


def completed_keys(path: Path, key) -> set:
    return {key(row) for row in read_rows(path) if row["status"] == "ok"}


def upsert_result(path: Path, fields: tuple[str, ...], key, row: dict) -> None:
    """Replace the same case's row when a stopped or failed case is rerun."""
    rows = [old for old in read_rows(path) if key(old) != key(row)]
    rows.append(row)
    write_rows(path, fields, rows)


# ------------------------------- Summaries -------------------------------- #

def write_summaries(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Join dense.csv with fused.csv into per-projection and per-block CSVs.

    The dense result for a projection does not depend on the retained-work
    ratio, so each fused row pairs with the dense row of its projection.
    """
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dense = {
        row["projection_group"]: row
        for row in read_rows(run_dir / "dense.csv") if row["status"] == "ok"
    }
    fused_rows = read_rows(run_dir / "fused.csv")
    fused = {fused_key(row): row for row in fused_rows if row["status"] == "ok"}
    ratios = sorted({float(row["target_retained_work_ratio"]) for row in fused_rows})

    projection_rows: list[dict] = []
    block_rows: list[dict] = []
    for ratio in ratios:
        block_missing: list[str] = []
        errors: list[float] = []
        for case in CASES:
            dense_row = dense.get(case.name)
            fused_row = fused.get((case.name, ratio))
            missing = []
            if dense_row is None:
                missing.append(f"dense[{case.name}]")
            if fused_row is None:
                missing.append(f"fused[{case.name}]")
            block_missing += missing

            rank = int(fused_row["rank"]) if fused_row else choose_rank(case, ratio, config["P"])
            row: dict[str, object] = {
                "status": "incomplete" if missing else "ok",
                **config,
                "projection_group": case.name,
                "multiplicity": case.multiplicity,
                "M": case.M,
                "N": case.N,
                "target_retained_work_ratio": ratio,
                "rank": rank,
                "retained_work_ratio": actual_ratio(case, rank),
                "missing": " ".join(missing),
            }
            if dense_row:
                row["dense_mean_cycles"] = dense_row["dense_mean_cycles"]
            if fused_row:
                row["fused_mean_cycles"] = fused_row["fused_mean_cycles"]
                row["fused_relative_l2_error"] = fused_row["fused_relative_l2_error"]
                errors.append(float(fused_row["fused_relative_l2_error"]))
            if not missing:
                dense_cycles = float(dense_row["dense_mean_cycles"])
                fused_cycles = float(fused_row["fused_mean_cycles"])
                row["fused_faster_than_dense"] = fused_cycles < dense_cycles
                row["fused_speedup_vs_dense"] = dense_cycles / fused_cycles
            projection_rows.append(row)

        block: dict[str, object] = {
            "status": "incomplete" if block_missing else "ok",
            **config,
            "target_retained_work_ratio": ratio,
            "projection_gemv_count": sum(case.multiplicity for case in CASES),
            "missing": " ".join(block_missing),
        }
        if all(case.name in dense for case in CASES):
            block["dense_block_cycles"] = sum(
                case.multiplicity * float(dense[case.name]["dense_mean_cycles"])
                for case in CASES
            )
        if all((case.name, ratio) in fused for case in CASES):
            block["fused_block_cycles"] = sum(
                case.multiplicity * float(fused[(case.name, ratio)]["fused_mean_cycles"])
                for case in CASES
            )
        if not block_missing:
            block["fused_speedup_vs_dense"] = (
                block["dense_block_cycles"] / block["fused_block_cycles"]
            )
        if errors:
            block["max_fused_relative_l2_error"] = max(errors)
        block_rows.append(block)

    write_rows(run_dir / "projection_summary.csv", PROJECTION_SUMMARY_FIELDS, projection_rows)
    write_rows(run_dir / "block_summary.csv", BLOCK_SUMMARY_FIELDS, block_rows)
    return projection_rows, block_rows


# -------------------------------- Running --------------------------------- #

def repo_relative(path: Path) -> str:
    return os.path.relpath(path, REPO_DIR)


def kernel_commands(
    kernel: str, launcher: str, shape: list[str], config: dict,
    build_dir: Path, suppress_trace: bool, build_root: Path | None = None,
) -> list[list[str]]:
    """Compile and launch one case from the repository root.

    Without ``build_root`` the artifact stays in ``build_dir`` inside the
    repository and is passed as a repository-relative path, because the SDK
    containers mount the working directory. With it, both commands run with
    TMPDIR unset, so the containers mount the host's /tmp at /tmp and see the
    artifact at the same /tmp path as the host (see check_build_root).
    """
    run_dir = build_dir.parent.parent
    host_dir = build_location(run_dir, build_root) / build_dir.name
    # cslc creates the output directory itself but requires its parent.
    host_dir.parent.mkdir(parents=True, exist_ok=True)
    if build_root is None:
        compile_dir = launch_dir = repo_relative(host_dir)
        prefix: list[str] = []
    else:
        tmp = Path("/tmp")
        compile_dir = launch_dir = str(tmp / host_dir.resolve().relative_to(tmp.resolve()))
        prefix = ["env", "-u", "TMPDIR"]
    compile_command = prefix + [
        sys.executable, f"{kernel}/compile.py", "--mode", "sim", *shape,
        "--out-dir", compile_dir,
    ]
    launch_command = prefix + [
        "cs_python", repo_relative(BENCH_DIR / launcher), *shape,
        "--repetitions", str(config["repetitions"]), "--out-dir", launch_dir,
    ]
    if config.get("profile_phases"):
        launch_command.append("--profile-phases")
    if suppress_trace:
        launch_command.append("--suppress-simfab-trace")
    return [compile_command, launch_command]


def run_logged(commands: list[list[str]], log_path: Path) -> str:
    """Run commands from the repository root, streaming and saving output.

    The SDK's cslc and cs_python wrappers mount only the working directory into
    their container, so the repository root must be the working directory.
    Returns the output of the final command.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            header = f"$ {' '.join(command)}\n"
            print("\n" + header, end="", flush=True)
            log.write(header)
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                output_lines.append(line)
            return_code = process.wait()
            if return_code:
                script = next((part for part in command if part.endswith(".py")), command[0])
                raise RuntimeError(
                    f"{Path(script).name} exited with status {return_code}; "
                    f"see {log_path}"
                )
    return "".join(output_lines)


def parse_result(output: str) -> dict[str, float]:
    lines = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise ValueError("launcher did not report a result")
    return json.loads(lines[-1][len(RESULT_PREFIX):])


def run_case(
    row: dict, commands: list[list[str]], log_path: Path, columns: dict[str, str]
) -> dict:
    """Run one case and copy launcher metrics into CSV columns."""
    try:
        result = parse_result(run_logged(commands, log_path))
        if row.get("profile_phases") and not result.get("phases"):
            raise ValueError("profiled launcher did not report phase timings")
        row.update({column: result[metric] for metric, column in columns.items()})
        if "phases" in result:
            profile_path = log_path.with_suffix(".phases.json")
            profile_path.write_text(json.dumps({"case": {**row, "status": "ok"}, "phases": result["phases"]}, indent=2) + "\n")
            phase_rows = [{"phase": phase, **metrics} for phase, metrics in result["phases"].items()]
            write_rows(log_path.with_suffix(".phases.csv"),
                       ("phase", "mean_cycles", "max_pe_cycles"), phase_rows)
        if any(part.endswith("launch_dense_timing.py") for part in commands[-1]):
            launch = commands[-1]
            artifact = Path(launch[launch.index("--out-dir") + 1])
            if not artifact.is_absolute():
                artifact = REPO_DIR / artifact
            for suffix in ("operations", "operation_rows"):
                source = artifact / f"timing_{row['P']}_{row['M']}_{row['N']}_{row['group_num']}.{suffix}.csv"
                log_path.with_suffix(f".{suffix}.csv").write_bytes(source.read_bytes())
        if row.get("chunks") and result.get("phases"):
            launch = commands[-1]
            artifact = Path(launch[launch.index("--out-dir") + 1])
            if not artifact.is_absolute():
                artifact = REPO_DIR / artifact
            # Preserve detailed events with the logs even if builds are removed.
            timeline = json.loads((artifact / "chunk_timeline.json").read_text())
            log_path.with_suffix(".timeline.json").write_text(json.dumps(timeline) + "\n")
        row["status"] = "ok"
    except Exception as exc:  # preserve partial progress in long sweeps
        row["error_message"] = str(exc)
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
    return row


def case_row(config: dict, case: ProjectionCase) -> dict[str, object]:
    return {
        "status": "failed",
        **config,
        "projection_group": case.name,
        "multiplicity": case.multiplicity,
        "M": case.M,
        "N": case.N,
        "error_message": "",
    }


def dense_implementation(config: dict) -> tuple[str, str]:
    """Kernel folder and launcher for the dense baseline."""
    if config.get("dense_timing"):
        return "MeshGEMVTiming", "launch_dense_timing.py"
    if config["hcf_fix"]:
        return "MeshGEMVPatched", "launch_dense_patched.py"
    return "MeshGEMV", "launch_dense.py"


def fused_launcher(config: dict) -> str:
    """Launcher for the fused kernel, which always compiles MeshGEMVLowerRank."""
    return "launch_fused_patched.py" if config["hcf_fix"] else "launch_fused.py"


def run_dense(args: argparse.Namespace, run_dir: Path, config: dict) -> int:
    results_path = run_dir / "dense.csv"
    kernel, launcher = dense_implementation(config)
    print(f"Dense implementation: {kernel} ({launcher})")
    print(f"Planned dense configurations: {len(CASES)}")
    for case in CASES:
        print(f"  {case.name}: M={case.M}, N={case.N}, multiplicity={case.multiplicity}")
    if args.dry_run:
        return 0

    prepare_results(results_path, DENSE_FIELDS, args.resume)
    done = completed_keys(results_path, dense_key) if args.resume else set()
    for index, case in enumerate(CASES, 1):
        if case.name in done:
            print(f"Skipping completed dense case {case.name}")
            continue
        print(
            f"\n=== [dense {index}/{len(CASES)}] {case.name}: "
            f"M={case.M}, N={case.N} ===", flush=True,
        )
        slug = f"dense_{case.name}_m{case.M}_n{case.N}"
        shape = [
            "--P", str(config["P"]), "--M", str(case.M), "--N", str(case.N),
            "--group_num", str(config["group_num"]),
        ]
        row = run_case(
            case_row(config, case),
            kernel_commands(kernel, launcher, shape, config,
                            run_dir / "build" / slug, args.suppress_simfab_trace,
                            args.build_root),
            run_dir / "logs" / f"{slug}.log",
            {
                "mean_cycles": "dense_mean_cycles",
                "max_cycles": "dense_max_cycles",
                "max_relative_error": "dense_max_relative_error",
            },
        )
        upsert_result(results_path, DENSE_FIELDS, dense_key, row)
        write_summaries(run_dir)
        print(f"Saved: {results_path}", flush=True)
    return 0


def run_fused(args: argparse.Namespace, run_dir: Path, config: dict) -> int:
    results_path = run_dir / "fused.csv"
    launcher = fused_launcher(config)
    print(f"Fused launcher: {launcher}")
    P = config["P"]
    planned = [
        (case, ratio, choose_rank(case, ratio, P))
        for ratio in args.retained_work_ratios
        for case in CASES
    ]
    print(f"Planned fused configurations: {len(planned)}")
    for case, ratio, rank in planned:
        print(
            f"  {case.name}: M={case.M}, R={rank}, N={case.N}, "
            f"target_rho={ratio:g}, actual_rho={actual_ratio(case, rank):.6f}"
        )
    if args.dry_run:
        return 0

    prepare_results(results_path, FUSED_FIELDS, args.resume)
    done = completed_keys(results_path, fused_key) if args.resume else set()
    for index, (case, ratio, rank) in enumerate(planned, 1):
        if (case.name, ratio) in done:
            print(f"Skipping completed fused case {case.name}, rho={ratio:g}")
            continue
        print(
            f"\n=== [fused {index}/{len(planned)}] {case.name}: "
            f"M={case.M}, R={rank}, N={case.N}, target rho={ratio:g} ===",
            flush=True,
        )
        slug = f"fused_{case.name}_m{case.M}_r{rank}_n{case.N}"
        shape = [
            "--P", str(P), "--M", str(case.M), "--R", str(rank), "--N", str(case.N),
            "--group_num", str(config["group_num"]),
        ]
        row = case_row(config, case)
        row.update(
            target_retained_work_ratio=ratio,
            rank=rank,
            retained_work_ratio=actual_ratio(case, rank),
        )
        row = run_case(
            row,
            kernel_commands("MeshGEMVLowerRank", launcher, shape, config,
                            run_dir / "build" / slug, args.suppress_simfab_trace,
                            args.build_root),
            run_dir / "logs" / f"{slug}.log",
            {
                "mean_cycles": "fused_mean_cycles",
                "max_cycles": "fused_max_cycles",
                "relative_l2_error": "fused_relative_l2_error",
            },
        )
        upsert_result(results_path, FUSED_FIELDS, fused_key, row)
        write_summaries(run_dir)
        print(f"Saved: {results_path}", flush=True)
    return 0


def print_summary(run_dir: Path, block_rows: list[dict]) -> None:
    print(f"\nPer-projection summary: {run_dir / 'projection_summary.csv'}")
    print(f"Multiplicity-weighted block summary: {run_dir / 'block_summary.csv'}")
    if not block_rows:
        print("  no fused results yet")
    for row in block_rows:
        ratio = row["target_retained_work_ratio"]
        if row["status"] == "ok":
            print(f"  rho={ratio:g}: fused speedup {row['fused_speedup_vs_dense']:.3f}x")
        else:
            print(f"  rho={ratio:g}: incomplete, missing {row['missing']}")


def main(argv: list[str] | None = None) -> int:
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "fused-chunked":
        import rectangular
        return rectangular.main(arguments[1:], mode="bench", execution="chunked")
    if arguments and arguments[0] == "dense-rectangular":
        import rectangular
        return rectangular.main(arguments[1:], mode="bench", execution="dense")
    if arguments and arguments[0] == "fused-rectangular":
        import rectangular
        return rectangular.main(arguments[1:], mode="bench")
    if arguments and arguments[0] == "fused-pipelining":
        import rectangular
        return rectangular.main(
            arguments[1:], mode="bench", execution="pipelining"
        )
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        raise SystemExit(
            "run name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or dashes"
        )
    run_dir = RUNS_DIR / args.run_name

    if args.command == "summary":
        if not (run_dir / "config.json").exists():
            raise SystemExit(f"no run found at {run_dir}")
        _, block_rows = write_summaries(run_dir)
        print_summary(run_dir, block_rows)
        return 0

    config = resolve_config(args, run_dir, write=not args.dry_run)
    validate(config, getattr(args, "retained_work_ratios", []))
    args.build_root = resolve_build_root(args, run_dir, write=not args.dry_run)
    print(f"Run: {args.run_name} ({args.command})")
    print("Timing mode: " + ("phase profiling (instrumented)" if config.get("profile_phases") else "normal"))
    print(
        f"P={config['P']}, group_num={config['group_num']}, "
        f"repetitions={config['repetitions']} (+ 1 warm-up)"
    )
    print(f"Output directory: {run_dir}")
    print(f"Build directory: {build_location(run_dir, args.build_root)}")

    if args.command == "dense":
        status = run_dense(args, run_dir, config)
    else:
        status = run_fused(args, run_dir, config)
    if not args.dry_run:
        _, block_rows = write_summaries(run_dir)
        print_summary(run_dir, block_rows)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
