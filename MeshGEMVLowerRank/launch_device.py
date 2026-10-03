"""Run the fused low-rank GEMV artifact on a WSE-3 (or cloud simulator)."""
import argparse
import json
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

import host_common as hc

OUT_PATH = "compile_out"


def cast_tensor_u32(tensor):
    """Pack FP16 words into the uint32 carrier type expected by memcpy_h2d."""
    return np.uint32(np.ascontiguousarray(tensor).view(np.uint16))


def parse_args():
    parser = argparse.ArgumentParser(description="Fused low-rank GEMV on WSE-3")
    parser.add_argument("--P", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--R", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--group_num", required=True, type=int)
    parser.add_argument("--simulator", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    Mt, Rt, Nt = M // P, R // P, N // P
    cfg = hc.cfg_name(P, M, R, N, G)

    # The same deterministic factorization is used by launch_sim.py. A and B
    # are already flattened in PE-major order by host_common.
    X, W, A, B, tensor_X, tensor_A, tensor_B = hc.make_low_rank_inputs(
        P, M, R, N, seed=2025
    )

    with open(f"{OUT_PATH}/artifact_{cfg}.json", "r", encoding="utf-8") as f:
        artifact_id = json.load(f)["artifact_id"]

    dtype = MemcpyDataType.MEMCPY_16BIT
    order = MemcpyOrder.ROW_MAJOR
    print("Start running: " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    with SdkRuntime(
        artifact_id, simulator=args.simulator, disable_version_check=True
    ) as runner:
        sym_X = runner.get_id("X")
        sym_A = runner.get_id("A")
        sym_B = runner.get_id("B")
        sym_res = runner.get_id("res")
        sym_time = runner.get_id("time_memcpy")
        sym_ref = runner.get_id("time_ref")

        runner.memcpy_h2d(sym_X, cast_tensor_u32(tensor_X.ravel()), 0, 0, P, P, Mt,
                          streaming=False, data_type=dtype, order=order, nonblock=False)
        runner.memcpy_h2d(sym_A, cast_tensor_u32(tensor_A), 0, 0, P, P, Mt * Rt,
                          streaming=False, data_type=dtype, order=order, nonblock=False)
        runner.memcpy_h2d(sym_B, cast_tensor_u32(tensor_B), 0, 0, P, P, Rt * Nt,
                          streaming=False, data_type=dtype, order=order, nonblock=False)

        warmups, repeats = 10, 100
        runner.launch("init_task", nonblock=False)
        runner.launch("meshgemv_host", np.int16(warmups), np.int16(repeats), nonblock=False)

        result_u32 = np.zeros(P * P * Nt, dtype=np.uint32)
        runner.memcpy_d2h(result_u32, sym_res, 0, 0, P, P, Nt,
                          streaming=False, data_type=dtype, order=order, nonblock=False)
        result_pe = sdk_utils.memcpy_view(result_u32, np.dtype(np.float16)).reshape(P, P, Nt)

        time_words = np.zeros(P * P * 3, dtype=np.float32)
        runner.memcpy_d2h(time_words, sym_time, 0, 0, P, P, 3, streaming=False,
                          data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False)
        ref_words = np.zeros(P * P * 2, dtype=np.float32)
        runner.memcpy_d2h(ref_words, sym_ref, 0, 0, P, P, 2, streaming=False,
                          data_type=MemcpyDataType.MEMCPY_32BIT, order=order, nonblock=False)

    # Stage 2 broadcasts each output block across px, so any one column (px=0)
    # contains the complete vector when its PE rows are concatenated.
    actual = result_pe[:, 0, :].reshape(1, N).astype(np.float32)
    factorized = X.astype(np.float32) @ A.astype(np.float32) @ B.astype(np.float32)
    dense = X.astype(np.float32) @ W.astype(np.float32)
    kernel_rel_l2 = np.linalg.norm(actual - factorized) / (np.linalg.norm(factorized) + 1e-8)
    dense_rel_l2 = np.linalg.norm(actual - dense) / (np.linalg.norm(dense) + 1e-8)
    mean_cycles, max_cycles = hc.decode_timing(
        time_words.reshape(P, P, 3), ref_words.reshape(P, P, 2), P, repeats
    )

    print(f"P={P}, M={M}, R={R}, N={N}, group_num={G}")
    print(f"Mean cycle count: {mean_cycles}")
    print(f"Max Cycle count: {max_cycles}")
    print(f"Device vs host SVD relative L2 error: {kernel_rel_l2:.6f}")
    print(f"Device vs host dense relative L2 error: {dense_rel_l2:.6f}")
    assert np.allclose(actual, factorized, rtol=0.1, atol=0.02)
    print("FUSED DEVICE SUCCESS!")


if __name__ == "__main__":
    main()
