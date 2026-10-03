#!/usr/bin/env python3
"""Extract a small HCF report from SDK traces before discarding a run directory.

Run with cs_python. Full simfab traces are read locally, never put in the ZIP.
"""
import argparse
import json
import platform
import re
import sys
import zipfile
from pathlib import Path


HALT = re.compile(r"@(\d+) FATAL: P(\d+)\.(\d+):.*?hcf", re.IGNORECASE)


def find_halt(log):
    match = HALT.search(log)
    if match is None:
        return None
    cycle, x, y = map(int, match.groups())
    return {"cycle": cycle, "x": x, "y": y}


def collect(out_dir, destination, host_log=None, source_dir=None, cycle=None):
    out_dir = Path(out_dir).resolve()
    destination = Path(destination).resolve()
    log = (out_dir / "sim.log").read_text(errors="replace")
    halt = find_halt(log)
    summary = {
        "halt": halt, "python": sys.version, "platform": platform.platform(),
        "artifact": str(out_dir), "trace_records": {}, "source_files": {}, "errors": [],
    }
    if halt is None and cycle is None:
        raise ValueError("No HCF found in sim.log (use --cycle for a decoder smoke test)")
    end_cycle = halt["cycle"] if halt else cycle
    # For this project's --fabric-offsets=4,1, P1.1 is the SDK input
    # adaptor and P2.1 is its demultiplexer. Include both sides of the header.
    peers = [(1, 1), (2, 1)]
    if halt and (halt["x"], halt["y"]) not in peers:
        peers.append((halt["x"], halt["y"]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = set()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(out_dir / "sim.log", "sim.log")
        if host_log and Path(host_log).is_file():
            bundle.write(host_log, "host.log")
        for part in ("bin", "west/bin", "east/bin", "generated"):
            for path in sorted((out_dir / part).glob("*.elf")):
                bundle.write(path, "artifact/" + path.relative_to(out_dir).as_posix())
        for part in ("out.json", "west/out.json", "east/out.json", "simconfig.json"):
            path = out_dir / part
            if path.is_file():
                bundle.write(path, "artifact/" + part)
        if source_dir:
            for path in sorted(Path(source_dir).rglob("*")):
                if path.is_file() and path.suffix in (".csl", ".py", ".json"):
                    bundle.write(path, "source/" + path.relative_to(source_dir).as_posix())

        try:
            from cerebras.das_common.das_commonpybind import Coordinate
            from cerebras.sdk.debug.lib.symbol.csldebugpybind import SimfabInstructionTraceCtf
            import cerebras.sdk.runtime.sdkruntimepybind as runtime
            import numpy as np

            summary.update(numpy=np.__version__, runtime_module=runtime.__file__)
            readers = {}
            for x, y in peers:
                # The reported failure is in the west infrastructure. For a
                # different halt, also attempt the application ELF directory.
                elf_dir = out_dir / ("west/bin" if x < 4 else "bin")
                if elf_dir not in readers:
                    readers[elf_dir] = SimfabInstructionTraceCtf(
                        str(elf_dir), str(out_dir / "simfab_traces")
                    )
                reader = readers[elf_dir]
                records = []
                # The reader limits the number of returned records. Short
                # windows avoid silently truncating away the actual failure.
                for start in range(max(0, end_cycle - 4096), end_cycle + 1, 128):
                    stop = min(start + 128, end_cycle + 1)
                    response = reader.get_instruction_trace_at(Coordinate(x, y), start, stop)
                    for item in response.data:
                        if not start <= item.cycle < stop:
                            continue
                        record = {
                            "cycle": item.cycle, "pc": item.op_addr, "op": item.op_name,
                            "dest": item.dest, "src0": item.src0, "src1": item.src1,
                            "src2": item.src2, "task": item.task, "uthread": item.uthread,
                            "source": item.source, "line": item.source_line,
                        }
                        records.append(json.dumps(record))
                        if item.source != "?":
                            sources.add(item.source)
                name = f"P{x}.{y}"
                summary["trace_records"][name] = len(records)
                bundle.writestr(f"instructions/{name}.jsonl", "\n".join(records) + "\n")
                print(f"Extracted {len(records)} instructions for {name}", flush=True)
            for index, source in enumerate(sorted(sources)):
                path = Path(source)
                if path.is_file() and path.suffix == ".csl":
                    # Keep the exact SDK code corresponding to the trace's
                    # source locations, not a potentially different online SDK.
                    archive_name = f"referenced_sources/{index:03d}_{path.name}"
                    bundle.write(path, archive_name)
                    summary["source_files"][source] = archive_name
            if halt and summary["trace_records"].get(f"P{halt['x']}.{halt['y']}", 0) == 0:
                summary["errors"].append("No instructions decoded for the failing PE; retain raw traces")
        except Exception as error:
            summary["errors"].append(f"Trace extraction failed: {type(error).__name__}: {error}")
        bundle.writestr("summary.json", json.dumps(summary, indent=2) + "\n")
    print(f"Report: {destination} ({destination.stat().st_size:,} bytes)", flush=True)
    for error in summary["errors"]:
        print(error, file=sys.stderr)
    return not summary["errors"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-log", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--cycle", type=int, help="decode a successful run at a known cycle")
    args = parser.parse_args()
    ok = collect(args.out_dir, args.output, args.host_log, args.source_dir, args.cycle)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
