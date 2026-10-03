#!/usr/bin/env python3
"""Diagnose large dense MeshGEMV HCF failures with fenced, chunked H2D copies.

Uses the existing CSL kernel, bounded uploads with device completion readbacks,
and optional numerical validation of one replicated output row. The old queued
schedule is available with --no-fence-h2d solely to reproduce the SDK overflow.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view


PROBE_DIR = Path(__file__).resolve().parent
MESH_GEMV_DIR = PROBE_DIR.parent
REPO_DIR = MESH_GEMV_DIR.parent
sys.path.insert(0, str(MESH_GEMV_DIR))

import host_common as hc  # noqa: E402
from sim_transfers import upload_f16  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense MeshGEMV with row-chunked weight transfers"
    )
    parser.add_argument("--P", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num", required=True, type=int)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--out-dir", type=Path, help="use an isolated compiled artifact")
    parser.add_argument(
        "--fence-h2d", action=argparse.BooleanOptionalAction, default=True,
        help="drain each upload on device (default; disable only to reproduce overflow)",
    )
    parser.add_argument(
        "--rows-per-copy",
        type=int,
        default=1,
        help="number of PE rows included in each weight memcpy (default: 1)",
    )
    parser.add_argument(
        "--result-values-per-pe",
        type=int,
        default=None,
        help=(
            "sampled FP16 values per PE (default: min(64, N/P)); "
            "use 0 to skip result D2H and test timing D2H only"
        ),
    )
    parser.add_argument("--suppress-simfab-trace", action="store_true",
                        help="disable bulk simulator instruction/wavelet traces")
    return parser.parse_args()


def main():
    random.seed(2025)
    np.random.seed(2025)
    args = parse_args()
    P, M, N, G = args.P, args.M, args.N, args.group_num
    repetitions = args.repetitions
    rows_per_copy = args.rows_per_copy

    if M % P or N % P:
        raise SystemExit(f"M={M} and N={N} must be divisible by P={P}")
    if G <= 0 or P % G:
        raise SystemExit(f"group_num={G} must be positive and divide P={P}")
    if not 1 <= repetitions <= 32767:
        raise SystemExit("repetitions must be between 1 and 32767")
    if not 1 <= rows_per_copy <= P:
        raise SystemExit(f"rows-per-copy must be between 1 and P={P}")

    Mt, Nt = M // P, N // P
    result_values_per_pe = (
        min(64, Nt)
        if args.result_values_per_pe is None
        else args.result_values_per_pe
    )
    if not 0 <= result_values_per_pe <= Nt:
        raise SystemExit(
            f"result-values-per-pe must be between 0 and Nt={Nt}"
        )
    cfg = hc.cfg_name(P, M, N, G)

    print(f"Generating host inputs for {M}x{N}", flush=True)
    X, tensor_X, tensor_W = hc.make_inputs(P, M, N)
    print("Host inputs generated", flush=True)

    # Keep the PE axes explicit.  tiled_W[py, px] is the Mt x Nt block owned
    # by logical PE (py, px), flattened into Mt*Nt FP16 values.
    print("Tiling weight matrix", flush=True)
    tiled_W = hc.tile_W(tensor_W, P, Mt, Nt).reshape(P, P, Mt * Nt)
    print("Weight matrix tiled", flush=True)

    out_dir = (args.out_dir or (MESH_GEMV_DIR / f"out_{cfg}")).resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"{out_dir} not found; compile the configuration first")

    os.chdir(out_dir)
    print(f"Starting simulator from {out_dir}", flush=True)
    runner = SdkRuntime(str(out_dir), suppress_simfab_trace=args.suppress_simfab_trace)
    runner.load()
    runner.run()
    print("Simulator started", flush=True)

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")
    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    if args.fence_h2d:
        upload_f16(runner, sym_X, tensor_X.reshape(P, P, Mt), label="X")
        upload_f16(runner, sym_W, tiled_W, label="W", max_rows=rows_per_copy)
    else:
        # Deliberately retain the old submission schedule for diagnosis only.
        print("WARNING: unfenced uploads can overflow the SDK ingress ring", flush=True)
        print("Copying X", flush=True)
        runner.memcpy_h2d(
            sym_X,
            input_array_to_u32(tensor_X.ravel(), 1, 1),
            0,
            0,
            P,
            P,
            Mt,
            streaming=False,
            data_type=dtype,
            order=order,
            nonblock=False,
        )
        print("X upload submitted", flush=True)

        # Each call covers only rows_per_copy rows of the logical PE rectangle.
        # The total values and final on-PE layout are identical to the old single
        # call, but each HCF transaction is much smaller.
        print(
            f"Copying W in chunks of {rows_per_copy} PE row(s): "
            f"local tile={Mt}x{Nt} ({Mt * Nt} FP16 values per PE)",
            flush=True,
        )
        for py_start in range(0, P, rows_per_copy):
            row_count = min(rows_per_copy, P - py_start)
            print(
                f"  W rows {py_start}-{py_start + row_count - 1} of {P - 1}",
                flush=True,
            )
            chunk_u32 = input_array_to_u32(
                tiled_W[py_start : py_start + row_count].ravel(), 1, 1
            )
            runner.memcpy_h2d(
                sym_W,
                chunk_u32,
                0,
                py_start,
                P,
                row_count,
                Mt * Nt,
                streaming=False,
                data_type=dtype,
                order=order,
                nonblock=False,
            )
        print("W uploads submitted", flush=True)

    warmups = 1
    print("Launching init_task", flush=True)
    runner.launch("init_task", nonblock=False)
    print(
        f"Launching meshgemv_host: warmups={warmups}, repetitions={repetitions}",
        flush=True,
    )
    runner.launch(
        "meshgemv_host",
        np.int16(warmups),
        np.int16(repetitions),
        nonblock=False,
    )
    print("GEMV launch submitted; readback will wait for device completion", flush=True)

    # Read the tiny 32-bit timing buffers before touching the FP16 result.
    # This separates a failure in every post-kernel D2H transaction from one
    # specific to the result symbol or the 16-bit memcpy path.
    time_words = np.zeros((P, P, 3), dtype=np.float32)
    ref_words = np.zeros((P, P, 2), dtype=np.float32)
    print("Copying timestamps one PE row at a time", flush=True)
    for py in range(P):
        print(f"  timestamp row {py}/{P - 1}", flush=True)
        time_row = np.zeros(P * 3, dtype=np.float32)
        runner.memcpy_d2h(
            time_row,
            sym_time,
            0,
            py,
            P,
            1,
            3,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=order,
            nonblock=False,
        )
        time_words[py] = time_row.reshape(P, 3)

        ref_row = np.zeros(P * 2, dtype=np.float32)
        runner.memcpy_d2h(
            ref_row,
            sym_ref,
            0,
            py,
            P,
            1,
            2,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=order,
            nonblock=False,
        )
        ref_words[py] = ref_row.reshape(P, 2)
    print("All timestamps copied", flush=True)

    mean_cycles, max_cycles = hc.decode_timing(
        time_words,
        ref_words,
        P,
        repetitions,
    )
    print(f"Mean cycle count: {mean_cycles}", flush=True)
    print(f"Max Cycle count: {max_cycles}", flush=True)

    if result_values_per_pe == 0:
        runner.stop()
        print("Simulator stopped", flush=True)
        print(
            f"P: {P}, M: {M}, N: {N}, group_num: {G}, "
            f"repetitions: {repetitions}, rows_per_copy: {rows_per_copy}, "
            "result_values_per_pe: 0 (validation skipped)"
        )
        print("CHUNKED DENSE TIMING SUCCESS! (result validation skipped)")
        return

    print(
        "Copying one replicated result row: "
        f"{result_values_per_pe}/{Nt} values from each PE",
        flush=True,
    )
    # The Y-axis all-reduce broadcasts each output block to every PE row, so
    # py=0 contains the complete N-element result across all P PE columns.
    result_u32 = np.zeros(P * result_values_per_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32,
        sym_res,
        0,
        0,
        P,
        1,
        result_values_per_pe,
        streaming=False,
        data_type=dtype,
        order=order,
        nonblock=False,
    )
    actual_sample = memcpy_view(result_u32, np.dtype(np.float16)).reshape(
        P, result_values_per_pe
    )
    print("Sampled result row copied", flush=True)

    runner.stop()
    print("Simulator stopped", flush=True)

    expected = X.astype(np.float32) @ tensor_W.astype(np.float32)
    # Each PE column owns one Nt-element output block. Compare the same prefix
    # of every block that was selected by the shortened D2H operation.
    expected_sample = expected.reshape(P, Nt)[:, :result_values_per_pe]
    relative_error = np.abs(actual_sample.astype(np.float32) - expected_sample) / (
        np.abs(expected_sample) + 1e-3
    )
    print(
        f"P: {P}, M: {M}, N: {N}, group_num: {G}, "
        f"repetitions: {repetitions}, rows_per_copy: {rows_per_copy}, "
        f"result_values_per_pe: {result_values_per_pe}/{Nt}"
    )
    print(f"max rel err: {relative_error.max():.4f}")
    assert relative_error.max() < 0.1
    print("CHUNKED DENSE SUCCESS!")


if __name__ == "__main__":
    main()
