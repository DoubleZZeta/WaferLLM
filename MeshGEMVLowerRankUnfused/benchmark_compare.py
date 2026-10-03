#!/usr/bin/env python3
"""Sweep dense versus host-mediated low-rank MeshGEMV simulator runs.

Run this script inside the Cerebras SDK VM, where ``cslc`` and ``cs_python``
are available. Results are appended to CSV after every rank configuration so
an interrupted simulator sweep can be resumed with ``--resume``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path


FIELDS = [
    "status",
    "error",
    "P",
    "pe_count",
    "matrix_size",
    "rank",
    "retained_work_ratio",
    "compression_fraction",
    "group_num",
    "dense_mean_cycles",
    "dense_max_cycles",
    "sparse_mean_cycles",
    "sparse_max_cycles",
    "sparse_faster",
    "speedup",
    "cycle_reduction_pct",
    "kernel_relative_l2_error",
    "svd_approximation_relative_l2_error",
]


def comma_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def comma_floats(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense and two-stage low-rank MeshGEMV simulator cycles"
    )
    parser.add_argument(
        "--p-values",
        type=comma_ints,
        default=[4],
        help="comma-separated PE-grid dimensions; P means a P×P grid (default: 4)",
    )
    parser.add_argument(
        "--sizes",
        type=comma_ints,
        default=[64, 128, 256, 512],
        help="comma-separated square matrix sizes M=N",
    )
    parser.add_argument(
        "--ranks",
        type=comma_ints,
        default=[],
        help="optional comma-separated absolute ranks",
    )
    parser.add_argument(
        "--rank-fractions",
        "--rank-ratios",
        dest="rank_fractions",
        type=comma_floats,
        default=None,
        help=(
            "R/min(M,N) values used to choose ranks (legacy alias: "
            "--rank-ratios); this square sweep has M=N. When neither ranks "
            "nor fractions are supplied, defaults to .125,.25,.375,.5"
        ),
    )
    parser.add_argument(
        "--groups",
        type=comma_ints,
        default=[2],
        help="comma-separated group_num values (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results_simple.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("benchmark_logs"),
        help="directory for full command logs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip successful configurations already present in the CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print configurations without compiling or running them",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="upgrade an existing output CSV schema, then exit without running kernels",
    )
    return parser.parse_args()


def run_logged(command: list[str], cwd: Path, log_path: Path) -> tuple[str, float, int]:
    """Run a command, stream combined output, and retain it for parsing."""
    print(f"\n$ {' '.join(command)}", flush=True)
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    elapsed = time.perf_counter() - start
    output = "".join(lines)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n\n{output}", encoding="utf-8"
    )
    return output, elapsed, return_code


def numbers(pattern: str, text: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(pattern, text, flags=re.IGNORECASE)
    ]


def one(pattern: str, text: str) -> float:
    values = numbers(pattern, text)
    if not values:
        raise ValueError(f"output did not contain: {pattern}")
    return values[-1]


def parse_dense(output: str) -> dict[str, float]:
    if "SUCCESS!" not in output:
        raise ValueError("dense run did not print SUCCESS!")
    return {
        "dense_mean_cycles": one(r"Mean cycle count:\s*([0-9.eE+-]+)", output),
        "dense_max_cycles": one(r"Max Cycle count:\s*([0-9.eE+-]+)", output),
    }


def parse_low_rank(output: str) -> dict[str, float]:
    if "STAGE 1 SUCCESS!" not in output or "STAGE 2 SUCCESS!" not in output:
        raise ValueError("low-rank run did not report success for both stages")

    means = numbers(r"Mean cycle count:\s*([0-9.eE+-]+)", output)
    maxima = numbers(r"Max cycle count:\s*([0-9.eE+-]+)", output)
    max_abs = numbers(r"Max absolute error:\s*([0-9.eE+-]+)", output)
    rel_l2 = numbers(r"Relative L2 error:\s*([0-9.eE+-]+)", output)
    if len(means) < 2 or len(maxima) < 2 or len(max_abs) < 2 or len(rel_l2) < 2:
        raise ValueError("low-rank output is missing per-stage metrics")

    return {
        "stage1_mean_cycles": means[0],
        "stage1_max_cycles": maxima[0],
        "stage2_mean_cycles": means[1],
        "stage2_max_cycles": maxima[1],
        "low_rank_sum_mean_cycles": one(
            r"Sum of stage mean cycles:\s*([0-9.eE+-]+)", output
        ),
        "low_rank_sum_max_cycles": one(
            r"Sum of stage max cycles:\s*([0-9.eE+-]+)", output
        ),
        "stage1_max_abs_error": max_abs[0],
        "stage1_relative_l2_error": rel_l2[0],
        "stage2_max_abs_error": max_abs[1],
        "stage2_relative_l2_error": rel_l2[1],
        "end_to_end_relative_l2_error": one(
            r"End-to-end relative L2 error:\s*([0-9.eE+-]+)", output
        ),
        "approximation_relative_l2_error": one(
            r"approximation relative L2 error:\s*([0-9.eE+-]+)", output
        ),
    }


def ranks_for_size(size: int, ranks: list[int], ratios: list[float]) -> list[int]:
    selected = set(ranks)
    for ratio in ratios:
        rank = size * ratio
        if rank.is_integer():
            selected.add(int(rank))
        else:
            print(
                f"Skipping size={size}, ratio={ratio}: rank is not an integer",
                file=sys.stderr,
            )
    return sorted(selected)


def work_ratios(M: int, N: int, rank: int) -> tuple[float, float]:
    """Return factorized/dense work and the corresponding removed fraction.

    Dense GEMV performs M*N multiply-accumulates.  The factorized operation
    performs M*R + R*N, so this definition remains meaningful for rectangular
    matrices unlike the old R/M field.
    """
    retained = rank * (M + N) / (M * N)
    return retained, 1.0 - retained


def validate(P: int, size: int, rank: int, group: int) -> str | None:
    if P < 3:
        return "P must be at least 3 for the current CSL kernel"
    if size % P != 0:
        return "matrix size must be divisible by P"
    if rank <= 0 or rank > size:
        return "rank must satisfy 0 < R <= min(M,N)"
    if rank % P != 0:
        return "rank must be divisible by P"
    if group <= 0 or P % group != 0:
        return "group_num must be positive and divide P"
    if group % 2 != 0:
        return "the current two-tree setup expects an even group_num"
    return None


def completed_keys(csv_path: Path) -> set[tuple[int, int, int, int]]:
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                keys.add(
                    tuple(
                        int(row[name])
                        for name in ("P", "matrix_size", "rank", "group_num")
                    )
                )
    return keys


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not needs_header:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != FIELDS:
            raise ValueError(
                f"{csv_path} uses a different CSV schema; choose a new output "
                "path or move the existing file"
            )
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def migrate_result_schema(csv_path: Path) -> None:
    """Replace historical rank_ratio while preserving every other CSV field."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    if "rank_ratio" not in old_fields:
        if "retained_work_ratio" in old_fields and "compression_fraction" in old_fields:
            return
        raise ValueError(f"{csv_path} uses an unsupported CSV schema")

    new_fields = []
    for field in old_fields:
        if field == "rank_ratio":
            new_fields.extend(("retained_work_ratio", "compression_fraction"))
        else:
            new_fields.append(field)
    if new_fields == old_fields:
        return

    migrated = []
    for row in rows:
        if all(row.get(name) not in (None, "") for name in ("M", "N", "R")):
            retained, _ = work_ratios(int(row["M"]), int(row["N"]), int(row["R"]))
        elif all(row.get(name) not in (None, "") for name in ("matrix_size", "rank")):
            size = int(row["matrix_size"])
            retained, _ = work_ratios(size, size, int(row["rank"]))
        else:
            # Historical break-even summaries contain only the square R/S ratio.
            retained = 2.0 * float(row["rank_ratio"])
        new_row = {field: row.get(field, "") for field in new_fields}
        new_row["retained_work_ratio"] = retained
        new_row["compression_fraction"] = 1.0 - retained
        migrated.append(new_row)

    temporary = csv_path.with_suffix(csv_path.suffix + ".schema-tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(migrated)
    temporary.replace(csv_path)
    print(f"Migrated {csv_path} from rank_ratio to work/compression ratios")


def ratios_from_row(row: dict[str, str]) -> tuple[float, float]:
    """Read the new schema, with compatibility for an older square CSV."""
    if row.get("retained_work_ratio") not in (None, ""):
        retained = float(row["retained_work_ratio"])
        compression = float(row["compression_fraction"])
        return retained, compression
    # Historical files stored R/S as rank_ratio and only supported M=N=S.
    retained = 2.0 * float(row["rank_ratio"])
    return retained, 1.0 - retained


def print_break_even_summary(csv_path: Path) -> None:
    """Print the first tested size with speedup > 1 for each sweep series."""
    if not csv_path.exists():
        return
    series: dict[tuple[int, int, float], list[tuple[int, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            _, compression = ratios_from_row(row)
            key = (int(row["P"]), int(row["group_num"]), compression)
            series.setdefault(key, []).append(
                (int(row["matrix_size"]), float(row["speedup"]))
            )

    if not series:
        return
    print("\n=== Observed break-even summary ===")
    for (P, group, compression), measurements in sorted(series.items()):
        faster = sorted((size, speedup) for size, speedup in measurements if speedup > 1.0)
        if faster:
            size, speedup = faster[0]
            result = f"first tested faster size={size} (speedup={speedup:.3f}×)"
        else:
            result = "not reached in tested sizes"
        print(f"P={P}, G={group}, compression={compression:.3f}: {result}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    dense_dir = repo_dir / "MeshGEMV"
    low_rank_dir = repo_dir / "MeshGEMVSparese"
    csv_path = args.output.resolve()
    log_dir = args.log_dir.resolve()

    if args.migrate_only:
        migrate_result_schema(csv_path)
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle), []) == FIELDS:
                    print_break_even_summary(csv_path)
        return 0

    if args.rank_fractions is None:
        rank_fractions = [] if args.ranks else [0.125, 0.25, 0.375, 0.5]
    else:
        rank_fractions = args.rank_fractions

    configurations = []
    for P, size, group in itertools.product(args.p_values, args.sizes, args.groups):
        for rank in ranks_for_size(size, args.ranks, rank_fractions):
            configurations.append((P, size, rank, group))

    print(f"Planned rank configurations: {len(configurations)}")
    if args.dry_run:
        for P, size, rank, group in configurations:
            problem = validate(P, size, rank, group)
            suffix = f" SKIP ({problem})" if problem else ""
            print(f"P={P} ({P*P} PEs), M=N={size}, R={rank}, G={group}{suffix}")
        return 0

    migrate_result_schema(csv_path)
    done = completed_keys(csv_path) if args.resume else set()

    dense_cache: dict[tuple[int, int, int], tuple[dict[str, float], float, str]] = {}

    for index, (P, size, rank, group) in enumerate(configurations, start=1):
        key = (P, size, rank, group)
        print(
            f"\n=== [{index}/{len(configurations)}] "
            f"P={P}, M=N={size}, R={rank}, G={group} ===",
            flush=True,
        )
        if key in done:
            print("Skipping completed configuration from CSV")
            continue

        retained_work_ratio, compression_fraction = work_ratios(size, size, rank)
        row: dict[str, object] = {
            "status": "failed",
            "error": "",
            "P": P,
            "pe_count": P * P,
            "matrix_size": size,
            "rank": rank,
            "retained_work_ratio": retained_work_ratio,
            "compression_fraction": compression_fraction,
            "group_num": group,
        }

        problem = validate(P, size, rank, group)
        if problem:
            row["status"] = "skipped"
            row["error"] = problem
            append_row(csv_path, row)
            print(f"Skipped: {problem}")
            continue

        dense_key = (P, size, group)
        try:
            if dense_key not in dense_cache:
                dense_log = log_dir / f"dense_p{P}_m{size}_n{size}_g{group}.log"
                dense_output, dense_wall, return_code = run_logged(
                    ["bash", "run_sim.sh", str(P), str(size), str(size), str(group)],
                    dense_dir,
                    dense_log,
                )
                if return_code != 0:
                    raise RuntimeError(f"dense command exited with status {return_code}")
                dense_cache[dense_key] = (
                    parse_dense(dense_output),
                    dense_wall,
                    str(dense_log),
                )

            dense_metrics, dense_wall, dense_log_name = dense_cache[dense_key]
            row.update(dense_metrics)

            low_log = log_dir / (
                f"lowrank_p{P}_m{size}_r{rank}_n{size}_g{group}.log"
            )
            low_output, low_wall, return_code = run_logged(
                [
                    "bash",
                    "run_sim_on_host.sh",
                    str(P),
                    str(size),
                    str(rank),
                    str(size),
                    str(group),
                ],
                low_rank_dir,
                low_log,
            )
            if return_code != 0:
                raise RuntimeError(f"low-rank command exited with status {return_code}")
            low_metrics = parse_low_rank(low_output)
            row["sparse_mean_cycles"] = low_metrics["low_rank_sum_mean_cycles"]
            row["sparse_max_cycles"] = low_metrics["low_rank_sum_max_cycles"]
            row["kernel_relative_l2_error"] = low_metrics[
                "end_to_end_relative_l2_error"
            ]
            row["svd_approximation_relative_l2_error"] = low_metrics[
                "approximation_relative_l2_error"
            ]

            dense_mean = float(row["dense_mean_cycles"])
            dense_max = float(row["dense_max_cycles"])
            low_mean = float(row["sparse_mean_cycles"])
            low_max = float(row["sparse_max_cycles"])
            row["sparse_faster"] = low_max < dense_max
            row["speedup"] = dense_max / low_max
            row["cycle_reduction_pct"] = 100.0 * (dense_max - low_max) / dense_max
            row["status"] = "ok"
            print(
                f"Result: dense={dense_max:.1f}, low-rank={low_max:.1f}, "
                f"speedup={row['speedup']:.3f}×, "
                f"reduction={row['cycle_reduction_pct']:.2f}%"
            )
        except Exception as exc:  # preserve failures and continue a long sweep
            row["error"] = str(exc)
            print(f"FAILED: {exc}", file=sys.stderr)

        append_row(csv_path, row)
        print(f"Appended result to {csv_path}", flush=True)

    print_break_even_summary(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
