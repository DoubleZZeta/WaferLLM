"""Compile-artifact launcher for one fused low-rank GEMV simulator runtime."""
import argparse
import os

import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

import host_common as hc

# Local copy supports SDK wrappers that mount only this kernel folder.
from sim_transfers import upload_f16


def parse_args():
    parser = argparse.ArgumentParser(description="Fused low-rank MeshGEMV")
    parser.add_argument("--PX", required=True, type=int)
    parser.add_argument("--PY", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--R", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num_x", required=True, type=int)
    parser.add_argument("--group_num_y", required=True, type=int)
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="run one warmed, timestamp-instrumented iteration and report phase costs",
    )
    parser.add_argument("--out-dir", help="compiled simulator artifact directory")
    parser.add_argument("--suppress-simfab-trace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    Px, Py, M, R, N, group_num_x, group_num_y = args.PX, args.PY, args.M, args.R, args.N, args.group_num_x, args.group_num_y
    if min(Px, Py) < 3 or min(M, R, N) <= 0:
        raise SystemExit("PX/PY must be at least 3; M/R/N must be positive")
    if M % Py or N % Py or R % Px or R > min(M, N):
        raise SystemExit("Invalid shape: require M%PY=N%PY=R%PX=0 and R<=min(M,N)")
    if group_num_x <= 0 or group_num_y <= 0 or Px % group_num_x or Py % group_num_y:
        raise SystemExit("Each positive group count must divide its own axis")
    Mt, Rt, Nt = M // Py, R // Px, N // Py

    X, W, A, B, tensor_X, tensor_A, tensor_B = hc.make_low_rank_inputs(
        Px, Py, M, R, N, seed=2025
    )

    cfg = hc.cfg_name(Px, Py, M, R, N, group_num_x, group_num_y)
    out_dir = os.path.abspath(args.out_dir or f"out_{cfg}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py first")

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
    sym_profile = runner.get_id("profile_time") if args.profile_phases else None

    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR

    upload_f16(runner, sym_X, tensor_X.reshape(Py, Px, Mt), label="X")
    upload_f16(runner, sym_A, tensor_A.reshape(Py, Px, Mt * Rt), label="A")
    upload_f16(runner, sym_B, tensor_B.reshape(Py, Px, Rt * Nt), label="B")

    warmups = 1
    repeats = 1 if args.profile_phases else 10
    runner.launch("init_task", nonblock=False)
    if args.profile_phases:
        runner.launch("meshgemv_profile_host", np.int16(warmups), nonblock=False)
    else:
        runner.launch(
            "meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False
        )

    # There are Px*Py PEs and Nt output values per PE.
    result_u32 = np.zeros(Px * Py * Nt, dtype=np.uint32)
    runner.memcpy_d2h(
        result_u32, sym_res, 0, 0, Px, Py, Nt,
        streaming=False, data_type=dtype, order=order, nonblock=False,
    )
    result_pe = memcpy_view(result_u32, np.dtype(np.float16)).reshape(Py, Px, Nt)

    # X-axis broadcast replicates each output block across PE columns. Selecting
    # px=0 from every row reconstructs [block 0, block 1, ..., block Py-1].
    result = result_pe[:, 0, :].reshape(1, N)

    time_words = np.zeros(Px * Py * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_words, sym_time, 0, 0, Px, Py, 3,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=order, nonblock=False,
    )
    time_hwl = time_words.reshape(Py, Px, 3)

    ref_words = np.zeros(Px * Py * 2, dtype=np.float32)
    runner.memcpy_d2h(
        ref_words, sym_ref, 0, 0, Px, Py, 2,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=order, nonblock=False,
    )
    ref_hwl = ref_words.reshape(Py, Px, 2)

    profile_hwl = None
    if args.profile_phases:
        profile_words = np.zeros(Px * Py * 14, dtype=np.float32)
        runner.memcpy_d2h(
            profile_words, sym_profile, 0, 0, Px, Py, 14,
            streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
            order=order, nonblock=False,
        )
        profile_hwl = profile_words.reshape(Py, Px, 14)
    runner.stop()

    # Use FP32 host arithmetic to evaluate kernel error independently of SVD
    # approximation quality.
    factorized = X.astype(np.float32) @ A.astype(np.float32) @ B.astype(np.float32)
    actual = result.astype(np.float32)
    max_abs_error = np.max(np.abs(actual - factorized))
    kernel_rel_l2 = np.linalg.norm(actual - factorized) / (
        np.linalg.norm(factorized) + 1e-8
    )
    dense = X.astype(np.float32) @ W.astype(np.float32)
    dense_rel_l2 = np.linalg.norm(actual - dense) / (
        np.linalg.norm(dense) + 1e-8
    )

    mean_cycles, max_cycles = hc.decode_timing(time_hwl, ref_hwl, Px, Py, repeats)
    print(f"Px={Px}, Py={Py}, M={M}, R={R}, N={N}, group_num_x={group_num_x}, group_num_y={group_num_y}")
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max Cycle count: {max_cycles}")
    print(f"Max absolute error: {max_abs_error:.6f}")
    print(f"Device vs host SVD relative L2 error: {kernel_rel_l2:.6f}")
    print(f"Device vs host dense relative L2 error: {dense_rel_l2:.6f}")

    if args.profile_phases:
        phase_metrics = hc.decode_phase_timing(profile_hwl, ref_hwl, Px, Py)
        iteration_mean = phase_metrics["profile_iteration"]["mean_cycles"]
        print("Phase profile (one warmed iteration; includes timestamp overhead):")
        for name in (
            "stage1_local_compute",
            "stage1_y_allreduce",
            "stage2_local_compute",
            "stage2_x_allreduce",
            "control_and_markers",
        ):
            metric = phase_metrics[name]
            percentage = 100.0 * metric["mean_cycles"] / max(iteration_mean, 1.0)
            print(
                f"  {name}: mean={metric['mean_cycles']:.1f}, "
                f"max_pe={metric['max_pe_cycles']:.1f}, share={percentage:.1f}%"
            )
        profile_total = phase_metrics["profile_iteration"]
        print(
            f"  profile_iteration: mean={profile_total['mean_cycles']:.1f}, "
            f"max_pe={profile_total['max_pe_cycles']:.1f}"
        )
        outer_control = mean_cycles - profile_total["mean_cycles"]
        print(
            f"  outer_timing_and_exit: mean={outer_control:.1f} "
            "(total timer minus profiled loop)"
        )
        print(
            "  Note: control_and_markers includes loop dispatch plus profiling "
            "marker cost; use the normal run for unperturbed total cycles."
        )

    assert np.allclose(actual, factorized, rtol=0.1, atol=0.02), (
        f"fused result mismatch: max abs={max_abs_error}, rel L2={kernel_rel_l2}"
    )
    print("FUSED SUCCESS!")


if __name__ == "__main__":
    main()
