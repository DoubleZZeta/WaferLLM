#!/usr/bin/env python3
"""Fused low-rank GEMV simulator run for one Llama-2 projection shape.

Factors A and B are generated directly rather than by SVD: large projection
shapes do not need a costly host-side dense SVD to measure kernel cycles, only
correctly shaped factors.

bench.py runs this with cs_python from the repository root, after compiling
the artifact with ``MeshGEMVLowerRankPipelining/compile.py --mode sim --out-dir``. The
SDK container mounts only the working directory, so --out-dir must lie
beneath it.
"""
import argparse
import os

import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

from common import emit_result, load_kernel_module

hc = load_kernel_module("MeshGEMVLowerRankPipelining", "host_common")


upload_f16_default = load_kernel_module("MeshGEMV", "sim_transfers").upload_f16

def parse_args():
    parser = argparse.ArgumentParser(description="Fused Llama-shaped low-rank GEMV")
    parser.add_argument("--PX", required=True, type=int)
    parser.add_argument("--PY", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--R", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num_x", required=True, type=int)
    parser.add_argument("--group_num_y", required=True, type=int)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--out-dir", required=True,
                        help="compiled simulator artifact directory")
    parser.add_argument("--suppress-simfab-trace", action="store_true",
                        help="disable bulk simulator instruction/wavelet traces")
    parser.add_argument("--profile-phases", action="store_true")
    args = parser.parse_args()
    if args.profile_phases:
        parser.error("pipelining does not yet support --profile-phases")
    if args.PX < 3 or args.R // args.PX < 2:
        parser.error("pipelining requires PX >= 3 and R/PX >= 2")
    return args


def make_direct_inputs(Px, Py, M, R, N, seed=2025):
    """Generate factors directly and convert them to fused-kernel PE layouts."""
    if min(Px, Py, M, R, N) <= 0 or M % Py or N % Py or R % Px:
        raise ValueError("Require positive dimensions, M%PY=N%PY=R%PX=0")
    if R > min(M, N):
        raise ValueError("R cannot exceed min(M,N)")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, M), dtype=np.float32).astype(np.float16)
    A = (rng.standard_normal((M, R), dtype=np.float32) / np.sqrt(M)).astype(
        np.float16
    )
    B = (rng.standard_normal((R, N), dtype=np.float32) / np.sqrt(R)).astype(
        np.float16
    )
    Mt, Rt, Nt = M // Py, R // Px, N // Py
    return (
        X,
        A,
        B,
        hc.tile_vector(X, Px, Py),
        hc.tile_A_for_y_reduce(A, Px, Py, Mt, Rt),
        hc.tile_B_for_x_reduce(B, Px, Py, Rt, Nt),
    )


def main(upload_f16=upload_f16_default):
    args = parse_args()
    Px, Py, M, R, N = args.PX, args.PY, args.M, args.R, args.N
    Gx, Gy = args.group_num_x, args.group_num_y
    if min(Px, Py) < 3 or min(Gx, Gy) < 1 or Px % Gx or Py % Gy:
        raise SystemExit("Invalid rectangular grid/group counts")
    Mt, Rt, Nt = M // Py, R // Px, N // Py
    repeats = 1 if args.profile_phases else args.repetitions
    if not 1 <= repeats <= 32766:
        raise SystemExit("repetitions must be between 1 and 32766 (plus one warmup)")
    X, A, B, tensor_X, tensor_A, tensor_B = make_direct_inputs(Px, Py, M, R, N)

    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        raise SystemExit(
            f"Host: {out_dir} not found — run MeshGEMVLowerRankPipelining/compile.py first"
        )

    # SdkRuntime writes simulator logs to CWD, so keep them inside the artifact.
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir, suppress_simfab_trace=args.suppress_simfab_trace)
    runner.load()
    runner.run()

    sym_X = runner.get_id("X")
    sym_A = runner.get_id("A")
    sym_B = runner.get_id("B")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")
    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    upload_f16(runner, sym_X, tensor_X.reshape(Py, Px, Mt), label="X")
    upload_f16(runner, sym_A, tensor_A.reshape(Py, Px, Mt * Rt), label="A")
    upload_f16(runner, sym_B, tensor_B.reshape(Py, Px, Rt * Nt), label="B")

    print("Initializing fused kernel", flush=True)
    warmups = 1
    runner.launch("init_task", nonblock=False)
    print("Executing fused kernel", flush=True)
    if args.profile_phases:
        runner.launch("meshgemv_profile_host", np.int16(warmups), nonblock=False)
    else:
        runner.launch(
            "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
        )


    print("Collecting fused results", flush=True)
    result_u32 = np.zeros(Px * Py * Nt, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, Px, Py, Nt, streaming=False,
        data_type=dtype, order=order, nonblock=False,
    )
    result_pe = memcpy_view(result_u32, np.dtype(np.float16)).reshape(Py, Px, Nt)
    # X-axis broadcast replicates each output block across PE columns. Selecting
    # px=0 from every row reconstructs [block 0, block 1, ..., block Py-1].
    actual = result_pe[:, 0, :].reshape(1, N).astype(np.float32)

    time_words = np.zeros(Px * Py * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_words, sym_time, 0, 0, Px, Py, 3, streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False,
    )
    ref_words = np.zeros(Px * Py * 2, dtype=np.float32)
    runner.memcpy_d2h(
        ref_words, sym_ref, 0, 0, Px, Py, 2, streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False,
    )
    phases = None
    if args.profile_phases:
        profile_words = np.zeros(Px * Py * 14, dtype=np.float32)
        runner.memcpy_d2h(
            profile_words, runner.get_id("profile_time"), 0, 0, Px, Py, 14,
            streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
            order=order, nonblock=False,
        )
        phases = hc.decode_phase_timing(
            profile_words.reshape(Py, Px, 14), ref_words.reshape(Py, Px, 2), Px, Py)
        for phase, metrics in phases.items():
            print(f"{phase}: mean={metrics['mean_cycles']:.3f}, "
                  f"max PE={metrics['max_pe_cycles']:.3f} cycles")
    runner.stop()

    expected = X.astype(np.float32) @ A.astype(np.float32) @ B.astype(np.float32)
    max_abs_error = np.max(np.abs(actual - expected))
    relative_l2 = np.linalg.norm(actual - expected) / (
        np.linalg.norm(expected) + 1e-8
    )
    mean_cycles, max_cycles = hc.decode_timing(
        time_words.reshape(Py, Px, 3), ref_words.reshape(Py, Px, 2), Px, Py, repeats
    )

    print(
        f"PX={Px}, PY={Py}, M={M}, R={R}, N={N}, Gx={Gx}, Gy={Gy}, "
        f"repetitions={repeats}"
    )
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max Cycle count: {max_cycles}")
    print(f"Max absolute error: {max_abs_error:.6f}")
    print(f"Device vs host factorized relative L2 error: {relative_l2:.6f}")
    assert np.allclose(actual, expected, rtol=0.1, atol=0.02), (
        f"fused result mismatch: max abs={max_abs_error}, rel L2={relative_l2}"
    )
    print("FUSED SUCCESS!")
    emit_result(
        phases=phases,
        mean_cycles=mean_cycles,
        max_cycles=max_cycles,
        relative_l2_error=relative_l2,
    )


if __name__ == "__main__":
    main()
