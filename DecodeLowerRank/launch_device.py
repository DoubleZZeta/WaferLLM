import json
import os
import argparse
import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime, sdk_utils

out_path = "compile_out"

def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))

class Config:
    def __init__(self):
        self.P = 8
        self.group_num = 2
        self.bsz = 1
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.seq_len = 64
        self.ffn_dim = 64
        self.layer_num = 32


class RankConfig:
    def __init__(self):
        self.rank_q = 512
        self.rank_k = 512
        self.rank_v = 512
        self.rank_o = 512
        self.rank_gate = 896
        self.rank_up = 896
        self.rank_down = 896

def parse_args():
    parser = argparse.ArgumentParser(description="Move to right unit test")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    parser.add_argument("--rank_config", default="rank_config.json", type=str, help="Rank config file")
    parser.add_argument("--simulator", action="store_true", help="Runs on simulator")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Config()
    rank_config = RankConfig()
    
    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    if not os.path.exists(args.rank_config):
        print("Host: Use default rank config values.")
    else:
        with open(args.rank_config) as f:
            rank_config.__dict__.update(json.load(f))
            
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    rank_cfg_name = os.path.splitext(os.path.basename(args.rank_config))[0]
    artifact_name = f"{cfg_name}__{rank_cfg_name}"

    P = config.P
    pe_num_p_group = P // config.group_num
    bsz = config.bsz
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    layer_num = config.layer_num
    rank_o = rank_config.rank_o
    rank_down = rank_config.rank_down
    rank_q = rank_config.rank_q
    rank_up = rank_config.rank_up
    if not rank_q == rank_config.rank_k == rank_config.rank_v:
        raise ValueError("rank_q, rank_k, and rank_v must be equal for fused QKV collectives")
    if rank_up != rank_config.rank_gate:
        raise ValueError("rank_up and rank_gate must be equal for fused FFN collectives")
    if any(value % P for value in (dim, ffn_dim, rank_q, rank_o, rank_up, rank_down)):
        raise ValueError("dimensions and all implemented ranks must be divisible by P")
    if not 0 < rank_q <= dim:
        raise ValueError(f"QKV rank must be in [1, {dim}], got {rank_q}")
    if not 0 < rank_o <= dim:
        raise ValueError(f"rank_o must be in [1, {dim}], got {rank_o}")
    if not 0 < rank_down <= min(ffn_dim, dim):
        raise ValueError(f"rank_down must be in [1, {min(ffn_dim, dim)}], got {rank_down}")
    dim_p_pe = dim // P
    pes_p_head = P // n_heads
    pes_p_kv_head = P // n_kv_heads
    head_dim_p_pe = head_dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P
    rank_o_p_pe = rank_o // P
    rank_down_p_pe = rank_down // P
    rank_qkv_p_pe = rank_q // P
    rank_ffn_p_pe = rank_up // P

    print(f"Host: P: {P}, Batch size: {bsz}, dim_p_pe: {dim_p_pe}, pes_p_head: {pes_p_head}, pes_p_kv_head: {pes_p_kv_head}, head_dim_p_pe: {head_dim_p_pe}, ffn_dim_p_pe: {ffn_dim_p_pe}, simulator: {args.simulator}")
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    X = np.random.rand(1, bsz*dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz*dim_p_pe), reps=(1, P))
    
    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))
    
    qkv_factors = {}
    for name, seed in (("Q", 2023), ("K", 2024), ("V", 2025)):
        factor_rng = np.random.default_rng(seed)
        a = (factor_rng.standard_normal((dim, rank_q), dtype=np.float32) / np.sqrt(dim)).astype(np.float16)
        b = (factor_rng.standard_normal((rank_q, dim), dtype=np.float32) / np.sqrt(rank_q)).astype(np.float16)
        qkv_factors[f"{name}_A"] = a.reshape(P, dim_p_pe, P, rank_qkv_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * rank_qkv_p_pe)
        qkv_factors[f"{name}_B"] = b.reshape(P, rank_qkv_p_pe, P, dim_p_pe).transpose(2, 0, 1, 3).reshape(P, P, rank_qkv_p_pe * dim_p_pe)
    
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1
        
    freqs_sin = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe//2), reps=(1, P))
    freqs_cos = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe//2), reps=(1, P))
    
    tensor_XKCache = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(seq_len, dim).astype(np.float16)
    
    # Placeholder compressed o_proj factors.  Replace these arrays with the
    # checkpoint's O_A [dim, rank_o] and O_B [rank_o, dim] to run real weights.
    rng = np.random.default_rng(2026)
    tensor_o_a = (
        rng.standard_normal((dim, rank_o), dtype=np.float32) / np.sqrt(dim)
    ).astype(np.float16)
    tensor_o_b = (
        rng.standard_normal((rank_o, dim), dtype=np.float32) / np.sqrt(rank_o)
    ).astype(np.float16)

    # Mirror the two reduction axes used by the CSL kernel:
    #   stage 1: O_A[input block px, rank block py], reduce along X
    #   stage 2: O_B[rank block py, output block px], reduce along Y
    O_A_blocks = tensor_o_a.reshape(P, dim_p_pe, P, rank_o_p_pe)
    O_A_reshape = O_A_blocks.transpose(2, 0, 1, 3).reshape(
        P, P, dim_p_pe * rank_o_p_pe
    )
    O_B_blocks = tensor_o_b.reshape(P, rank_o_p_pe, P, dim_p_pe)
    O_B_reshape = O_B_blocks.transpose(0, 2, 1, 3).reshape(
        P, P, rank_o_p_pe * dim_p_pe
    )
    ffn_factors = {}
    for name, seed in (("UP", 2028), ("GATE", 2029)):
        factor_rng = np.random.default_rng(seed)
        a = (factor_rng.standard_normal((dim, rank_up), dtype=np.float32) / np.sqrt(dim)).astype(np.float16)
        b = (factor_rng.standard_normal((rank_up, ffn_dim), dtype=np.float32) / np.sqrt(rank_up)).astype(np.float16)
        ffn_factors[f"{name}_A"] = a.reshape(P, dim_p_pe, P, rank_ffn_p_pe).transpose(2, 0, 1, 3).reshape(P, P, dim_p_pe * rank_ffn_p_pe)
        ffn_factors[f"{name}_B"] = b.reshape(P, rank_ffn_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, rank_ffn_p_pe * ffn_dim_p_pe)
    # Placeholder compressed down_proj factors. Replace these with checkpoint
    # DOWN_A [ffn_dim, rank_down] and DOWN_B [rank_down, dim] for real weights.
    down_rng = np.random.default_rng(2027)
    tensor_down_a = (
        down_rng.standard_normal((ffn_dim, rank_down), dtype=np.float32) / np.sqrt(ffn_dim)
    ).astype(np.float16)
    tensor_down_b = (
        down_rng.standard_normal((rank_down, dim), dtype=np.float32) / np.sqrt(rank_down)
    ).astype(np.float16)
    DOWN_A_blocks = tensor_down_a.reshape(P, ffn_dim_p_pe, P, rank_down_p_pe)
    DOWN_A_reshape = DOWN_A_blocks.transpose(2, 0, 1, 3).reshape(
        P, P, ffn_dim_p_pe * rank_down_p_pe
    )
    DOWN_B_blocks = tensor_down_b.reshape(P, rank_down_p_pe, P, dim_p_pe)
    DOWN_B_reshape = DOWN_B_blocks.transpose(0, 2, 1, 3).reshape(
        P, P, rank_down_p_pe * dim_p_pe
    )
    
    with open(f"{out_path}/artifact_{artifact_name}.json", "r", encoding="utf8") as f:
        data = json.load(f)
        artifact_path = data["artifact_id"]

    with SdkRuntime(artifact_path, simulator=args.simulator) as runner:
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Get symbols ------------------------------ #
        # -------------------------------------------------------------------------- #
        
        sym_X = runner.get_id("X")
        sym_W = runner.get_id("W")
        qkv_symbols = {
            name: runner.get_id(f"{name}_weight")
            for name in ("Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B")
        }
        sym_freqs_sin = runner.get_id("freqs_sin")
        sym_freqs_cos = runner.get_id("freqs_cos")
        sym_XKCache = runner.get_id("XKCache")
        sym_XVCache = runner.get_id("XVCache")
        sym_O_A_weight = runner.get_id("O_A_weight")
        sym_O_B_weight = runner.get_id("O_B_weight")
        ffn_symbols = {
            name: runner.get_id(f"{name}_weight")
            for name in ("UP_A", "UP_B", "GATE_A", "GATE_B")
        }
        sym_DOWN_A_weight = runner.get_id("DOWN_A_weight")
        sym_DOWN_B_weight = runner.get_id("DOWN_B_weight")
        
        # timer symbol list:
        symbol_timer_buf = runner.get_id("timer_buf")
        symbol_timer_ref = runner.get_id("time_ref")

        # -------------------------------------------------------------------------- #
        # ------------------------------ H2D memcpy ------------------------------ #
        # -------------------------------------------------------------------------- #
        
        X_u32 = cast_tensor_u32(tensor_X.ravel())
        runner.memcpy_h2d(
            sym_X, X_u32, 0, 0, P, P, bsz*dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        W_u32 = cast_tensor_u32(tensor_W.ravel())
        runner.memcpy_h2d(
            sym_W, W_u32, 0, 0, P, P, dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        for name, tensor in qkv_factors.items():
            data_u32 = cast_tensor_u32(tensor.ravel())
            runner.memcpy_h2d(
                qkv_symbols[name], data_u32, 0, 0, P, P,
                dim_p_pe * rank_qkv_p_pe,
                streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
            )
        
        freqs_sin_u32 = cast_tensor_u32(tensor_freqs_sin.ravel())
        runner.memcpy_h2d(
            sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy freqs_cos
        freqs_cos_u32 = cast_tensor_u32(tensor_freqs_cos.ravel())
        runner.memcpy_h2d(
            sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy XKCache
        XKCache_reshape = tensor_XKCache.reshape(P, dim_p_pe, P, seq_len_p_pe)
        XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
        XKCache_reshape = XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe)
        XKCache_u32 = cast_tensor_u32(XKCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XKCache, XKCache_u32, 0, 0, P, P, dim_p_pe * seq_len_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy XVCache
        XVCache_reshape = tensor_XVCache.reshape(P, seq_len_p_pe, P, dim_p_pe)
        XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
        XVCache_reshape = XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe)
        XVCache_u32 = cast_tensor_u32(XVCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XVCache, XVCache_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy both o_proj factors; the dense O matrix is never materialized.
        O_A_u32 = cast_tensor_u32(O_A_reshape.ravel())
        runner.memcpy_h2d(
            sym_O_A_weight, O_A_u32, 0, 0, P, P, dim_p_pe * rank_o_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        O_B_u32 = cast_tensor_u32(O_B_reshape.ravel())
        runner.memcpy_h2d(
            sym_O_B_weight, O_B_u32, 0, 0, P, P, rank_o_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        for name, tensor in ffn_factors.items():
            data_u32 = cast_tensor_u32(tensor.ravel())
            local_size = (
                dim_p_pe * rank_ffn_p_pe
                if name.endswith("_A")
                else rank_ffn_p_pe * ffn_dim_p_pe
            )
            runner.memcpy_h2d(
                ffn_symbols[name], data_u32, 0, 0, P, P, local_size,
                streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
            )
        # Copy both down_proj factors; the dense DOWN matrix is never built.
        DOWN_A_u32 = cast_tensor_u32(DOWN_A_reshape.ravel())
        runner.memcpy_h2d(
            sym_DOWN_A_weight, DOWN_A_u32, 0, 0, P, P, ffn_dim_p_pe * rank_down_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        DOWN_B_u32 = cast_tensor_u32(DOWN_B_reshape.ravel())
        runner.memcpy_h2d(
            sym_DOWN_B_weight, DOWN_B_u32, 0, 0, P, P, rank_down_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Run WSE-3 ---------------------------- #
        # -------------------------------------------------------------------------- #
        runner.launch("init_task", nonblock=False)

        total_warmup_times, total_repeat_times = 5, 50
        runner.launch("decode_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Timer Check ------------------------------ #
        # -------------------------------------------------------------------------- #
        # Copy back timer_buf from all width x height PEs
        timer_buf_1d_u32 = np.zeros((P*P*3), dtype=np.uint32)
        runner.memcpy_d2h(
            timer_buf_1d_u32, symbol_timer_buf, 0, 0, P, P, 3, streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False
        )
        timer_buf_time_hwl = timer_buf_1d_u32.view(np.float32).reshape((P, P, 3))
        
    # -------------------------------------------------------------------------- #
    # ------------------------------ Compute time ------------------------------ #
    # -------------------------------------------------------------------------- #
    cycles_count = np.zeros((P, P))
    for pe_x in range(P):
            for pe_y in range(P):
                cycles_count[pe_y, pe_x] = sdk_utils.calculate_cycles(timer_buf_time_hwl[pe_y, pe_x, :])
        
    freq_ghz = 1.1

    cycles_count_mean = cycles_count.mean()
    cycles_count_mean_per_step = cycles_count_mean / total_repeat_times
    print(f"\nRepeat count: {total_repeat_times}")
    print(f"Mean cycles per decoder layer: {cycles_count_mean_per_step}")
    print(f"Estimated cycles for {layer_num} layers: {cycles_count_mean_per_step * layer_num}")

    throughput_p_request = 1 / ((cycles_count_mean_per_step*layer_num) / (freq_ghz*1e9))
    print(f"Throughput_p_request: {throughput_p_request}")
    
if __name__ == "__main__":
    main()
