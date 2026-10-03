#!/usr/bin/env python3
"""Run the two unfused GEMVs using directly generated low-rank factors.

This is the Llama-benchmark counterpart of ``launch_sim_on_host.py`` and is
kept separate so the existing SVD-based benchmark retains its old interface.
"""
import argparse
import os

import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
)

import host_common as hc
from launch_sim_on_host import read_timing, start_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Unfused Llama-shaped GEMVs")
    parser.add_argument("--stage", required=True, choices=("1", "2"))
    parser.add_argument("--P", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--R", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num", required=True, type=int)
    parser.add_argument("--state-file", required=True)
    return parser.parse_args()


def make_direct_inputs(P, M, R, N, seed=2025):
    for name, value in (("M", M), ("R", R), ("N", N)):
        if value % P:
            raise ValueError(f"{name}={value} must be divisible by P={P}")
    if R > min(M, N):
        raise ValueError(f"R={R} cannot exceed min(M,N)={min(M, N)}")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, M), dtype=np.float32).astype(np.float16)
    A = (rng.standard_normal((M, R), dtype=np.float32) / np.sqrt(M)).astype(
        np.float16
    )
    B = (rng.standard_normal((R, N), dtype=np.float32) / np.sqrt(R)).astype(
        np.float16
    )
    return X, hc.tile_vector(X, P), A, B


def run_stage1(args, module_dir, state_path):
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    Mt, Rt = M // P, R // P
    X, tensor_X, A, B = make_direct_inputs(P, M, R, N)

    runner = start_runtime(os.path.join(module_dir, f"out_{hc.cfg_name(P, M, R, G)}"))
    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")
    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    runner.memcpy_h2d(
        sym_X, input_array_to_u32(tensor_X.ravel(), 1, 1),
        0, 0, P, P, Mt, streaming=False, data_type=dtype,
        order=order, nonblock=False,
    )
    runner.memcpy_h2d(
        sym_W, input_array_to_u32(hc.tile_W(A, P, Mt, Rt), 1, 1),
        0, 0, P, P, Mt * Rt, streaming=False, data_type=dtype,
        order=order, nonblock=False,
    )

    warmups, repeats = 1, 10
    runner.launch("init_task", nonblock=False)
    runner.launch(
        "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
    )
    result_u32 = np.zeros(P * R, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, P, P, Rt, streaming=False,
        data_type=dtype, order=order, nonblock=False,
    )
    actual = memcpy_view(result_u32, np.dtype(np.float16)).reshape(P, R)
    mean_cycles, max_cycles = read_timing(
        runner, sym_time, sym_ref, P, repeats
    )
    runner.stop()

    expected = X.astype(np.float32) @ A.astype(np.float32)
    expected_b = np.broadcast_to(expected, actual.shape)
    actual_f32 = actual.astype(np.float32)
    max_abs_error = np.max(np.abs(actual_f32 - expected_b))
    relative_l2 = np.linalg.norm(actual_f32 - expected_b) / (
        np.linalg.norm(expected_b) + 1e-8
    )
    assert np.allclose(actual_f32, expected_b, rtol=0.1, atol=0.02)

    Z = actual[0:1, :].copy()
    np.savez(
        state_path,
        X=X,
        A=A,
        B=B,
        Z=Z,
        stage1_mean_cycle=np.float64(mean_cycles),
        stage1_max_cycle=np.float64(max_cycles),
    )
    print(f"Stage 1: [1, {M}] @ [{M}, {R}]")
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max cycle count: {max_cycles}")
    print(f"Max absolute error: {max_abs_error:.6f}")
    print(f"Relative L2 error: {relative_l2:.6f}")
    print(f"Saved intermediate Z {Z.shape} to {state_path}")
    print("STAGE 1 SUCCESS!")


def run_stage2(args, module_dir, state_path):
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    Rt, Nt = R // P, N // P
    if not os.path.isfile(state_path):
        raise SystemExit(f"Host: {state_path} not found — run stage 1 first")
    with np.load(state_path) as state:
        X = state["X"].copy()
        A = state["A"].copy()
        B = state["B"].copy()
        Z = state["Z"].copy()
        stage1_mean = float(state["stage1_mean_cycle"])
        stage1_max = float(state["stage1_max_cycle"])

    runner = start_runtime(os.path.join(module_dir, f"out_{hc.cfg_name(P, R, N, G)}"))
    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")
    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    runner.memcpy_h2d(
        sym_X, input_array_to_u32(hc.tile_vector(Z, P).ravel(), 1, 1),
        0, 0, P, P, Rt, streaming=False, data_type=dtype,
        order=order, nonblock=False,
    )
    runner.memcpy_h2d(
        sym_W, input_array_to_u32(hc.tile_W(B, P, Rt, Nt), 1, 1),
        0, 0, P, P, Rt * Nt, streaming=False, data_type=dtype,
        order=order, nonblock=False,
    )

    warmups, repeats = 1, 10
    runner.launch("init_task", nonblock=False)
    runner.launch(
        "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
    )
    result_u32 = np.zeros(P * N, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, P, P, Nt, streaming=False,
        data_type=dtype, order=order, nonblock=False,
    )
    actual = memcpy_view(result_u32, np.dtype(np.float16)).reshape(P, N)
    mean_cycles, max_cycles = read_timing(
        runner, sym_time, sym_ref, P, repeats
    )
    runner.stop()

    stage2_expected = Z.astype(np.float32) @ B.astype(np.float32)
    expected_b = np.broadcast_to(stage2_expected, actual.shape)
    actual_f32 = actual.astype(np.float32)
    max_abs_error = np.max(np.abs(actual_f32 - expected_b))
    stage2_relative_l2 = np.linalg.norm(actual_f32 - expected_b) / (
        np.linalg.norm(expected_b) + 1e-8
    )
    factorized_expected = (
        X.astype(np.float32) @ A.astype(np.float32) @ B.astype(np.float32)
    )
    end_to_end_relative_l2 = np.linalg.norm(
        actual_f32[0:1] - factorized_expected
    ) / (np.linalg.norm(factorized_expected) + 1e-8)
    assert np.allclose(actual_f32, expected_b, rtol=0.1, atol=0.02)

    print(f"Stage 2: [1, {R}] @ [{R}, {N}]")
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max cycle count: {max_cycles}")
    print(f"Max absolute error: {max_abs_error:.6f}")
    print(f"Relative L2 error: {stage2_relative_l2:.6f}")
    print(f"End-to-end relative L2 error: {end_to_end_relative_l2:.6f}")
    print(f"Sum of stage mean cycles: {stage1_mean + mean_cycles}")
    print(f"Sum of stage max cycles: {stage1_max + max_cycles}")
    print("STAGE 2 SUCCESS!")


def main():
    args = parse_args()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.abspath(args.state_file)
    if args.stage == "1":
        run_stage1(args, module_dir, state_path)
    else:
        run_stage2(args, module_dir, state_path)


if __name__ == "__main__":
    main()
