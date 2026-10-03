#!/usr/bin/env python3
"""Dense MeshGEMV simulator run for one Llama-2 projection shape.

bench.py runs this with cs_python from the repository root, after compiling
the artifact with ``MeshGEMV/compile.py --mode sim --out-dir``. The SDK
container mounts only the working directory, so --out-dir must lie beneath it.
"""
import argparse
import os
import random

import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

from common import emit_result, load_kernel_module


upload_f16_default = load_kernel_module("MeshGEMV", "sim_transfers").upload_f16

def parse_args():
    parser = argparse.ArgumentParser(description="Dense Llama-shaped MeshGEMV")
    parser.add_argument("--P", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num", required=True, type=int)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--out-dir", required=True,
                        help="compiled simulator artifact directory")
    parser.add_argument("--suppress-simfab-trace", action="store_true",
                        help="disable bulk simulator instruction/wavelet traces")
    parser.add_argument("--profile-phases", action="store_true")
    return parser.parse_args()


def main(upload_f16=upload_f16_default, kernel="MeshGEMV"):
    hc = load_kernel_module(kernel, "host_common")
    random.seed(2025)
    np.random.seed(2025)
    args = parse_args()
    P, M, N, G = args.P, args.M, args.N, args.group_num
    repetitions = 1 if args.profile_phases else args.repetitions
    if not 1 <= repetitions <= 32767:
        raise SystemExit("repetitions must be between 1 and 32767")

    Mt, Nt = M // P, N // P
    X, tensor_X, tensor_W = hc.make_inputs(P, M, N)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run MeshGEMV/compile.py first")

    # SdkRuntime writes simulator logs to CWD, so keep them inside the artifact.
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir, suppress_simfab_trace=args.suppress_simfab_trace)
    runner.load()
    runner.run()

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")
    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    upload_f16(runner, sym_X, tensor_X.reshape(P, P, Mt), label="X")
    upload_f16(runner, sym_W,
               hc.tile_W(tensor_W, P, Mt, Nt).reshape(P, P, Mt * Nt), label="W")

    warmups = 1
    runner.launch("init_task", nonblock=False)
    if args.profile_phases:
        runner.launch("meshgemv_profile_host", np.int16(warmups), nonblock=False)
    else:
        runner.launch(
            "meshgemv_host",
            np.int16(warmups),
            np.int16(repetitions),
            nonblock=False,
        )


    result_u32 = np.zeros(P * N, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, P, P, Nt, streaming=False,
        data_type=dtype, order=order, nonblock=False,
    )
    actual = memcpy_view(result_u32, np.dtype(np.float16)).reshape(P, N)

    time_words = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_words, sym_time, 0, 0, P, P, 3, streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False,
    )
    ref_words = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(
        ref_words, sym_ref, 0, 0, P, P, 2, streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False,
    )
    phases = None
    if args.profile_phases:
        profile_words = np.zeros(P * P * 10, dtype=np.float32)
        runner.memcpy_d2h(
            profile_words, runner.get_id("profile_time"), 0, 0, P, P, 10,
            streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
            order=order, nonblock=False,
        )
        phases = hc.decode_phase_timing(
            profile_words.reshape(P, P, 10), ref_words.reshape(P, P, 2), P)
        for phase, metrics in phases.items():
            print(f"{phase}: mean={metrics['mean_cycles']:.3f}, "
                  f"max PE={metrics['max_pe_cycles']:.3f} cycles")
    runner.stop()

    expected = X.astype(np.float32) @ tensor_W.astype(np.float32)
    expected_b = np.broadcast_to(expected, actual.shape)
    relative_error = np.abs(actual.astype(np.float32) - expected_b) / (
        np.abs(expected_b) + 1e-3
    )
    mean_cycles, max_cycles = hc.decode_timing(
        time_words.reshape(P, P, 3),
        ref_words.reshape(P, P, 2),
        P,
        repetitions,
    )

    print(
        f"P: {P}, M: {M}, N: {N}, group_num: {G}, "
        f"repetitions: {repetitions}"
    )
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max Cycle count: {max_cycles}")
    print(f"max rel err: {relative_error.max():.4f}")
    assert relative_error.max() < 0.1, (
        f"dense result mismatch: max rel err {relative_error.max()}"
    )
    print("SUCCESS!")
    emit_result(
        phases=phases,
        mean_cycles=mean_cycles,
        max_cycles=max_cycles,
        max_relative_error=relative_error.max(),
    )


if __name__ == "__main__":
    main()
