"""Run one stage of host-mediated low-rank MeshGEMV on the simulator.

The simulator supports one SdkRuntime lifecycle per Python process. Stage 1
therefore saves its intermediate vector and inputs to an NPZ file, and stage 2
loads that file in a fresh ``cs_python`` process.
"""
import argparse
import os

import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

import host_common as hc


def parse_args():
    parser = argparse.ArgumentParser(description="Host-mediated low-rank MeshGEMV")
    parser.add_argument("--stage", required=True, choices=("1", "2"))
    parser.add_argument("--P", required=True, type=int, help="PE grid dimension")
    parser.add_argument("--M", required=True, type=int, help="Original input dimension")
    parser.add_argument("--R", required=True, type=int, help="Low-rank dimension")
    parser.add_argument("--N", required=True, type=int, help="Original output dimension")
    parser.add_argument("--group_num", required=True, type=int, help="Reduce group count")
    parser.add_argument(
        "--state-file",
        default="low_rank_stage1.npz",
        help="NPZ file used to pass Z and reference inputs between processes",
    )
    return parser.parse_args()


def read_timing(runner, time_symbol, ref_symbol, P, repeat_times):
    time_memcpy_1d = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_memcpy_1d, time_symbol, 0, 0, P, P, 3,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )
    time_memcpy = np.reshape(time_memcpy_1d, (P, P, 3), order="C")

    time_ref_1d = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(
        time_ref_1d, ref_symbol, 0, 0, P, P, 2,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )
    time_ref = np.reshape(time_ref_1d, (P, P, 2), order="C")
    return hc.decode_timing(time_memcpy, time_ref, P, repeat_times)


def start_runtime(out_dir):
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py --mode sim first")
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir)
    runner.load()
    runner.run()
    return runner


def run_stage1(args, module_dir, state_path):
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    Mt, Rt = M // P, R // P

    X, tensor_X, W, A, B = hc.make_low_rank_inputs(
        P, M, R, N, seed=2025
    )

    cfg = hc.cfg_name(P, M, R, G)
    runner = start_runtime(os.path.join(module_dir, f"out_{cfg}"))

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_X, X_u32, 0, 0, P, P, Mt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )

    tensor_A = hc.tile_W(A, P, Mt, Rt)
    A_u32 = input_array_to_u32(tensor_A, 1, 1)
    runner.memcpy_h2d(
        sym_W, A_u32, 0, 0, P, P, Mt * Rt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )

    warmups, repeats = 1, 10
    runner.launch("init_task", nonblock=False)
    runner.launch(
        "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
    )

    result_u32 = np.zeros(P * R, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, P, P, Rt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )
    result_fp16 = memcpy_view(result_u32, np.dtype(np.float16))
    stage1_result = result_fp16.reshape(P, R)
    mean_cycle, max_cycle = read_timing(runner, sym_time, sym_ref, P, repeats)
    runner.stop()

    expected = X.astype(np.float32) @ A.astype(np.float32)
    expected_b = np.broadcast_to(expected, stage1_result.shape)
    actual = stage1_result.astype(np.float32)
    abs_err = np.abs(actual - expected_b)
    rel_l2 = np.linalg.norm(actual - expected_b) / (
        np.linalg.norm(expected_b) + 1e-8
    )
    assert np.allclose(actual, expected_b, rtol=0.1, atol=0.02), (
        f"stage 1 mismatch: max absolute error {abs_err.max()}, "
        f"relative L2 error {rel_l2}"
    )

    Z = stage1_result[0:1, :].copy()
    np.savez(
        state_path,
        X=X,
        W=W,
        A=A,
        B=B,
        Z=Z,
        stage1_mean_cycle=np.float64(mean_cycle),
        stage1_max_cycle=np.float64(max_cycle),
        stage1_max_abs_err=np.float64(abs_err.max()),
        stage1_rel_l2=np.float64(rel_l2),
    )

    print(f"Stage 1: [1, {M}] @ [{M}, {R}]")
    print(f"Mean cycle count: {mean_cycle}")
    print(f"Max cycle count: {max_cycle}")
    print(f"Max absolute error: {abs_err.max():.6f}")
    print(f"Relative L2 error: {rel_l2:.6f}")
    print(f"Saved intermediate Z {Z.shape} to {state_path}")
    print("STAGE 1 SUCCESS!")


def run_stage2(args, module_dir, state_path):
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    Rt, Nt = R // P, N // P

    if not os.path.isfile(state_path):
        raise SystemExit(f"Host: {state_path} not found — run stage 1 first")
    with np.load(state_path) as state:
        X = state["X"].copy()
        W = state["W"].copy()
        A = state["A"].copy()
        B = state["B"].copy()
        Z = state["Z"].copy()
        stage1_mean_cycle = float(state["stage1_mean_cycle"])
        stage1_max_cycle = float(state["stage1_max_cycle"])

    cfg = hc.cfg_name(P, R, N, G)
    runner = start_runtime(os.path.join(module_dir, f"out_{cfg}"))

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time = runner.get_id("time_memcpy")
    sym_ref = runner.get_id("time_ref")

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    # Z is a normal [1, R] MeshGEMV input: split it over PE rows and
    # replicate those row chunks over PE columns.
    tensor_Z = hc.tile_vector(Z, P)
    Z_u32 = input_array_to_u32(tensor_Z.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_X, Z_u32, 0, 0, P, P, Rt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )

    # B is [R, N], hence every PE stores an [R/P, N/P] block.
    tensor_B = hc.tile_W(B, P, Rt, Nt)
    B_u32 = input_array_to_u32(tensor_B, 1, 1)
    runner.memcpy_h2d(
        sym_W, B_u32, 0, 0, P, P, Rt * Nt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )

    warmups, repeats = 1, 10
    runner.launch("init_task", nonblock=False)
    runner.launch(
        "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
    )

    result_u32 = np.zeros(P * N, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, P, P, Nt,
        streaming=False, data_type=io_dtype, order=order, nonblock=False,
    )
    result_fp16 = memcpy_view(result_u32, np.dtype(np.float16))
    stage2_result = result_fp16.reshape(P, N)
    mean_cycle, max_cycle = read_timing(runner, sym_time, sym_ref, P, repeats)
    runner.stop()

    # This reference uses the exact FP16 Z sent to stage 2, isolating its
    # kernel error from the rounding introduced by stage 1.
    expected = Z.astype(np.float32) @ B.astype(np.float32)
    expected_b = np.broadcast_to(expected, stage2_result.shape)
    actual = stage2_result.astype(np.float32)
    abs_err = np.abs(actual - expected_b)
    rel_l2 = np.linalg.norm(actual - expected_b) / (
        np.linalg.norm(expected_b) + 1e-8
    )
    assert np.allclose(actual, expected_b, rtol=0.1, atol=0.02), (
        f"stage 2 mismatch: max absolute error {abs_err.max()}, "
        f"relative L2 error {rel_l2}"
    )

    factorized_expected = (
        X.astype(np.float32)
        @ A.astype(np.float32)
        @ B.astype(np.float32)
    )
    end_to_end_rel_l2 = np.linalg.norm(
        actual[0:1] - factorized_expected
    ) / (np.linalg.norm(factorized_expected) + 1e-8)
    dense_expected = X.astype(np.float32) @ W.astype(np.float32)
    approximation_rel_l2 = np.linalg.norm(
        factorized_expected - dense_expected
    ) / (np.linalg.norm(dense_expected) + 1e-8)

    print(f"Stage 2: [1, {R}] @ [{R}, {N}]")
    print(f"Mean cycle count: {mean_cycle}")
    print(f"Max cycle count: {max_cycle}")
    print(f"Max absolute error: {abs_err.max():.6f}")
    print(f"Relative L2 error: {rel_l2:.6f}")
    print(f"End-to-end relative L2 error: {end_to_end_rel_l2:.6f}")
    print(f"Rank-{R} approximation relative L2 error: {approximation_rel_l2:.6f}")
    print(f"Sum of stage mean cycles: {stage1_mean_cycle + mean_cycle}")
    print(f"Sum of stage max cycles: {stage1_max_cycle + max_cycle}")
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
