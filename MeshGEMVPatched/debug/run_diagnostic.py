#!/usr/bin/env python3
"""Compile an isolated probe, watch for HCF, and save a compact diagnostic ZIP."""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from collect_hcf import collect, find_halt


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    except ProcessLookupError:
        child.wait()
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("P", type=int)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("group_num", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--fence-h2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-traces", action="store_true")
    parser.add_argument("--suppress-simfab-trace", action="store_true")
    parser.add_argument("--result-values-per-pe", type=int, default=0)
    args = parser.parse_args()
    p, m, n, g = args.P, args.M, args.N, args.group_num
    if p < 4 or p % 2 or m <= 0 or n <= 0 or m % p or n % p:
        parser.error("use an even P >= 4 and positive M and N divisible by P")
    if g < 2 or p % g or p // g < 2:
        parser.error("this two-tree kernel requires at least two groups and two PEs per group")
    if not 1 <= args.repetitions <= 32766:
        parser.error("repetitions must be 1..32766 (plus one warmup)")

    probe_dir = Path(__file__).resolve().parent
    kernel_dir = probe_dir.parent
    runs = probe_dir / "runs"
    runs.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"p{p}_{m}x{n}_", dir=runs))
    source = run_dir / "source"
    shutil.copytree(kernel_dir / "src", source / "src")
    for path in (probe_dir / "launch_sim_chunked.py", probe_dir / "run_diagnostic.py",
                 probe_dir / "collect_hcf.py", kernel_dir / "host_common.py",
                 kernel_dir / "sim_transfers.py"):
        shutil.copy2(path, source / path.name)
    out_dir = run_dir / "out"
    group_size = p // g
    params = (
        f"P:{p},Mt:{m // p},Nt:{n // p},pe_num_group:{group_size},"
        f"root_1st_phase:{group_size // 2},"
        f"root_2nd_phase:{(g // 2) * group_size + group_size // 2}"
    )
    compile_cmd = [
        "cslc", "--arch=wse3", str(source / "src/layout.csl"),
        f"--fabric-dims={p + 7},{p + 2}", "--fabric-offsets=4,1",
        f"--params={params}", "-o", str(out_dir), "--memcpy", "--channels", "1",
    ]
    print(f"Isolated run: {run_dir}", flush=True)
    subprocess.run(compile_cmd, check=True)
    command = [
        sys.executable, str(probe_dir / "launch_sim_chunked.py"),
        "--P", str(p), "--M", str(m), "--N", str(n), "--group_num", str(g),
        "--repetitions", str(args.repetitions), "--rows-per-copy", "1",
        "--result-values-per-pe", str(args.result_values_per_pe), "--out-dir", str(out_dir),
    ]
    if not args.fence_h2d:
        command.append("--no-fence-h2d")
    if args.suppress_simfab_trace:
        command.append("--suppress-simfab-trace")
    log_path = run_dir / "host.log"
    halted = False
    with log_path.open("w") as log, log_path.open() as tail:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while True:
                print(tail.read(), end="", flush=True)
                sim_log = out_dir / "sim.log"
                if sim_log.exists() and find_halt(sim_log.read_text(errors="replace")):
                    halted = True
                    print("HCF detected; stopping the host wait and extracting diagnostics.", flush=True)
                    # The model stops on HCF; allow its final trace writes to
                    # finish before terminating the host that waits for D2H.
                    time.sleep(2)
                    stop_child(child)
                    break
                if child.poll() is not None:
                    break
                time.sleep(1)
        finally:
            stop_child(child)
            print(tail.read(), end="", flush=True)

    if halted:
        report = run_dir / "hcf_report.zip"
        ok = collect(out_dir, report, log_path, source)
        if ok and not args.keep_traces:
            shutil.rmtree(out_dir / "simfab_traces", ignore_errors=True)
            print("Removed this run's bulk traces after successful extraction.", flush=True)
        else:
            print(f"Raw traces retained in {out_dir}", flush=True)
        print(f"Share this file: {report}", flush=True)
        raise SystemExit(1)
    print(f"No HCF detected; host exit code {child.returncode}. Logs: {run_dir}", flush=True)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
