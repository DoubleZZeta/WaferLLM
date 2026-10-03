import json
import os
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, calculate_cycles, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


class Config:
    def __init__(self):
        self.P = 8
        self.bsz = 1
        self.group_num = 2
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.seq_len = 64
        self.ffn_dim = 64
        self.layer_num = 1

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
    parser.add_argument(
        "--profile-projections",
        action="store_true",
        help="run the separate projection-profile artifact instead of the benchmark artifact",
    )
    args = parser.parse_args()
    return args


def make_o_proj_factors(dim, rank_o, seed=2026):
    """Create placeholder factors for O ~= O_A @ O_B.

    These random factors exercise the exact device dataflow without paying for
    a large host SVD.  Real SVD-LLM factors can later be loaded into these same
    two arrays without changing the CSL kernel.
    """
    rng = np.random.default_rng(seed)
    o_a = rng.standard_normal((dim, rank_o), dtype=np.float32) / np.sqrt(dim)
    o_b = rng.standard_normal((rank_o, dim), dtype=np.float32) / np.sqrt(rank_o)
    return o_a.astype(np.float16), o_b.astype(np.float16)


def make_square_proj_factors(dim, rank, seed):
    """Create A [dim, rank], B [rank, dim] for a square projection."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, rank), dtype=np.float32) / np.sqrt(dim)
    b = rng.standard_normal((rank, dim), dtype=np.float32) / np.sqrt(rank)
    return a.astype(np.float16), b.astype(np.float16)


def tile_a_for_y_reduce(a, P, dim_p_pe, rank_p_pe):
    """Place A[input block py, rank block px] on PE(py, px)."""
    return a.reshape(P, dim_p_pe, P, rank_p_pe).transpose(0, 2, 1, 3).reshape(
        P, P, dim_p_pe * rank_p_pe
    )


def tile_b_for_x_reduce(b, P, rank_p_pe, dim_p_pe):
    """Place B[rank block px, output block py] on PE(py, px)."""
    return b.reshape(P, rank_p_pe, P, dim_p_pe).transpose(2, 0, 1, 3).reshape(
        P, P, rank_p_pe * dim_p_pe
    )


def tile_o_a_for_x_reduce(o_a, P, dim_p_pe, rank_o_p_pe):
    """Place O_A[input block px, rank block py] on PE(py, px)."""
    blocks = o_a.reshape(P, dim_p_pe, P, rank_o_p_pe)
    return blocks.transpose(2, 0, 1, 3).reshape(P, P, dim_p_pe * rank_o_p_pe)


def tile_o_b_for_y_reduce(o_b, P, rank_o_p_pe, dim_p_pe):
    """Place O_B[rank block py, output block px] on PE(py, px)."""
    blocks = o_b.reshape(P, rank_o_p_pe, P, dim_p_pe)
    return blocks.transpose(0, 2, 1, 3).reshape(P, P, rank_o_p_pe * dim_p_pe)


def make_down_proj_factors(ffn_dim, dim, rank_down, seed=2027):
    """Create placeholder factors for DOWN ~= DOWN_A @ DOWN_B."""
    rng = np.random.default_rng(seed)
    down_a = rng.standard_normal((ffn_dim, rank_down), dtype=np.float32) / np.sqrt(ffn_dim)
    down_b = rng.standard_normal((rank_down, dim), dtype=np.float32) / np.sqrt(rank_down)
    return down_a.astype(np.float16), down_b.astype(np.float16)


def make_rectangular_proj_factors(input_dim, output_dim, rank, seed):
    """Create A [input_dim, rank], B [rank, output_dim]."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((input_dim, rank), dtype=np.float32) / np.sqrt(input_dim)
    b = rng.standard_normal((rank, output_dim), dtype=np.float32) / np.sqrt(rank)
    return a.astype(np.float16), b.astype(np.float16)


def tile_down_a_for_x_reduce(down_a, P, ffn_dim_p_pe, rank_down_p_pe):
    """Place DOWN_A[input block px, rank block py] on PE(py, px)."""
    blocks = down_a.reshape(P, ffn_dim_p_pe, P, rank_down_p_pe)
    return blocks.transpose(2, 0, 1, 3).reshape(P, P, ffn_dim_p_pe * rank_down_p_pe)


def tile_down_b_for_y_reduce(down_b, P, rank_down_p_pe, dim_p_pe):
    """Place DOWN_B[rank block py, output block px] on PE(py, px)."""
    blocks = down_b.reshape(P, rank_down_p_pe, P, dim_p_pe)
    return blocks.transpose(0, 2, 1, 3).reshape(P, P, rank_down_p_pe * dim_p_pe)


def rebuild_x_sharded_vector(pe_tensor, bsz, dimension):
    """Rebuild a vector sharded over PE columns and replicated over rows."""
    P = pe_tensor.shape[1]
    local_dim = dimension // P
    return (
        pe_tensor[0, :, :]
        .reshape(P, bsz, local_dim)
        .transpose(1, 0, 2)
        .reshape(bsz, dimension)
    )


def rebuild_y_sharded_vector(pe_tensor, bsz, dimension):
    """Rebuild a vector sharded over PE rows and replicated over columns."""
    P = pe_tensor.shape[0]
    local_dim = dimension // P
    return pe_tensor[:, 0, :].reshape(P, bsz, local_dim).transpose(1, 0, 2).reshape(
        bsz, dimension
    )

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
    profile_suffix = "__projection-profile" if args.profile_projections else ""
    artifact_name = f"{cfg_name}__{rank_cfg_name}{profile_suffix}"

    P = config.P
    bsz = config.bsz
    group_num = config.group_num
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    layer_num = config.layer_num

    rank_q = rank_config.rank_q
    rank_k = rank_config.rank_k
    rank_v = rank_config.rank_v
    rank_o = rank_config.rank_o
    rank_gate = rank_config.rank_gate 
    rank_up = rank_config.rank_up
    rank_down = rank_config.rank_down

    if not rank_q == rank_k == rank_v:
        raise ValueError("rank_q, rank_k, and rank_v must be equal for fused QKV collectives")
    if rank_up != rank_gate:
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
    
    print(f"Host: P: {P}, Batch size: {bsz}, dim_p_pe: {dim_p_pe}, pes_p_head: {pes_p_head}, pes_p_kv_head: {pes_p_kv_head}, head_dim_p_pe: {head_dim_p_pe}, seq_len_p_pe: {seq_len_p_pe}, ffn_dim_p_pe: {ffn_dim_p_pe}")
    print(f"Rank: rank_q: {rank_q}, rank_k: {rank_k}, rank_v: {rank_v}, rank_o: {rank_o}, rank_gate: {rank_gate}, rank_up: {rank_up}, rank_down: {rank_down}")
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    X = np.random.rand(1, bsz*dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz*dim_p_pe), reps=(1, P))
    
    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))
    
    q_a, q_b = make_square_proj_factors(dim, rank_q, 2023)
    k_a, k_b = make_square_proj_factors(dim, rank_k, 2024)
    v_a, v_b = make_square_proj_factors(dim, rank_v, 2025)
    q_a_pe = tile_a_for_y_reduce(q_a, P, dim_p_pe, rank_qkv_p_pe)
    k_a_pe = tile_a_for_y_reduce(k_a, P, dim_p_pe, rank_qkv_p_pe)
    v_a_pe = tile_a_for_y_reduce(v_a, P, dim_p_pe, rank_qkv_p_pe)
    q_b_pe = tile_b_for_x_reduce(q_b, P, rank_qkv_p_pe, dim_p_pe)
    k_b_pe = tile_b_for_x_reduce(k_b, P, rank_qkv_p_pe, dim_p_pe)
    v_b_pe = tile_b_for_x_reduce(v_b, P, rank_qkv_p_pe, dim_p_pe)
    
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1
    
    freqs_sin = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe//2), reps=(1, P))
    freqs_cos = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe//2), reps=(1, P))
    
    tensor_XKCache = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(seq_len, dim).astype(np.float16)
    
    # Both factorized projections use an X reduction followed by a Y reduction.
    tensor_o_a, tensor_o_b = make_o_proj_factors(dim, rank_o)
    tensor_o_a_pe = tile_o_a_for_x_reduce(tensor_o_a, P, dim_p_pe, rank_o_p_pe)
    tensor_o_b_pe = tile_o_b_for_y_reduce(tensor_o_b, P, rank_o_p_pe, dim_p_pe)
    tensor_up_a, tensor_up_b = make_rectangular_proj_factors(dim, ffn_dim, rank_up, 2028)
    tensor_gate_a, tensor_gate_b = make_rectangular_proj_factors(dim, ffn_dim, rank_gate, 2029)
    tensor_up_a_pe = tile_o_a_for_x_reduce(tensor_up_a, P, dim_p_pe, rank_ffn_p_pe)
    tensor_gate_a_pe = tile_o_a_for_x_reduce(tensor_gate_a, P, dim_p_pe, rank_ffn_p_pe)
    tensor_up_b_pe = tile_o_b_for_y_reduce(tensor_up_b, P, rank_ffn_p_pe, ffn_dim_p_pe)
    tensor_gate_b_pe = tile_o_b_for_y_reduce(tensor_gate_b, P, rank_ffn_p_pe, ffn_dim_p_pe)
    tensor_down_a, tensor_down_b = make_down_proj_factors(ffn_dim, dim, rank_down)
    tensor_down_a_pe = tile_down_a_for_x_reduce(tensor_down_a, P, ffn_dim_p_pe, rank_down_p_pe)
    tensor_down_b_pe = tile_down_b_for_y_reduce(tensor_down_b, P, rank_down_p_pe, dim_p_pe)

    # Run from inside out_<cfg> so simfab run artifacts (sim.log, sim_stats.json,
    # simconfig.json, simfab_traces/, out.core, wio_flows_tmpdir.*) land there
    # instead of polluting the module root. SdkRuntime has no output-dir kwarg;
    # it writes these to CWD. compile.py already populated out_<cfg> via `cslc -o`.
    out_dir = os.path.abspath(f"out_{artifact_name}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py --mode sim first")
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir, simfab_numthreads=64, msg_level='INFO')

    runner.load()
    runner.run()
    
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
    sym_attention_output = runner.get_id("attention_output")
    sym_o_proj_output = runner.get_id("o_proj_output")
    sym_down_proj_input = runner.get_id("down_proj_input")
    sym_down_proj_output = runner.get_id("down_proj_output")
    sym_qkv_input = runner.get_id("qkv_input")
    sym_qkv_output = runner.get_id("qkv_output")
    sym_up_gate_input = runner.get_id("up_gate_input")
    sym_up_gate_output = runner.get_id("up_gate_output")
    sym_up_gate_rank = runner.get_id("up_gate_rank")
    
    # timer symbol list:
    symbol_timer_buf = runner.get_id("timer_buf")
    symbol_timer_ref = runner.get_id("time_ref")
    symbol_projection_profile = (
        runner.get_id("projection_profile_buf") if args.profile_projections else None
    )


    # -------------------------------------------------------------------------- #
    # ------------------------------ H2D memcpy ------------------------------ #
    # -------------------------------------------------------------------------- #
    
    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_X, X_u32, 0, 0, P, P, bsz*dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    W_u32 = input_array_to_u32(tensor_W.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_W, W_u32, 0, 0, P, P, dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    # Copy the six Q/K/V factors. A participates in the Y reduction; B in X.
    for name, tensor, local_size in (
        ("Q_A", q_a_pe, dim_p_pe * rank_qkv_p_pe),
        ("Q_B", q_b_pe, rank_qkv_p_pe * dim_p_pe),
        ("K_A", k_a_pe, dim_p_pe * rank_qkv_p_pe),
        ("K_B", k_b_pe, rank_qkv_p_pe * dim_p_pe),
        ("V_A", v_a_pe, dim_p_pe * rank_qkv_p_pe),
        ("V_B", v_b_pe, rank_qkv_p_pe * dim_p_pe),
    ):
        data_u32 = input_array_to_u32(tensor.ravel(), 1, 1)
        runner.memcpy_h2d(
            qkv_symbols[name], data_u32, 0, 0, P, P, local_size,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
    
    freqs_sin_u32 = input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy freqs_cos
    freqs_cos_u32 = input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy XKCache
    XKCache_reshape = tensor_XKCache.reshape(P, dim_p_pe, P, seq_len_p_pe)
    XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
    XKCache_reshape = XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe)
    XKCache_u32 = input_array_to_u32(XKCache_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_XKCache, XKCache_u32, 0, 0, P, P, dim_p_pe * seq_len_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy XVCache
    XVCache_reshape = tensor_XVCache.reshape(P, seq_len_p_pe, P, dim_p_pe)
    XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
    XVCache_reshape = XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe)
    XVCache_u32 = input_array_to_u32(XVCache_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_XVCache, XVCache_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy the two o_proj factors.  No dense O matrix is materialized.
    O_A_u32 = input_array_to_u32(tensor_o_a_pe.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_O_A_weight, O_A_u32, 0, 0, P, P, dim_p_pe * rank_o_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    O_B_u32 = input_array_to_u32(tensor_o_b_pe.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_O_B_weight, O_B_u32, 0, 0, P, P, rank_o_p_pe * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    for name, tensor, local_size in (
        ("UP_A", tensor_up_a_pe, dim_p_pe * rank_ffn_p_pe),
        ("UP_B", tensor_up_b_pe, rank_ffn_p_pe * ffn_dim_p_pe),
        ("GATE_A", tensor_gate_a_pe, dim_p_pe * rank_ffn_p_pe),
        ("GATE_B", tensor_gate_b_pe, rank_ffn_p_pe * ffn_dim_p_pe),
    ):
        data_u32 = input_array_to_u32(tensor.ravel(), 1, 1)
        runner.memcpy_h2d(
            ffn_symbols[name], data_u32, 0, 0, P, P, local_size,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
    # Copy down_proj factors. No dense DOWN matrix is materialized.
    DOWN_A_u32 = input_array_to_u32(tensor_down_a_pe.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_DOWN_A_weight, DOWN_A_u32, 0, 0, P, P, ffn_dim_p_pe * rank_down_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    DOWN_B_u32 = input_array_to_u32(tensor_down_b_pe.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_DOWN_B_weight, DOWN_B_u32, 0, 0, P, P, rank_down_p_pe * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    # -------------------------------------------------------------------------- #
    # ------------------------------ Run simulator ---------------------------- #
    # -------------------------------------------------------------------------- #
    runner.launch("init_task", nonblock=False)

    if args.profile_projections:
        runner.launch("projection_profile_host", nonblock=False)
        projection_profile_u32 = np.zeros(P * P * 12, dtype=np.uint32)
        runner.memcpy_d2h(
            projection_profile_u32, symbol_projection_profile,
            0, 0, P, P, 12, streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR, nonblock=False,
        )
        projection_profile_hwl = projection_profile_u32.view(np.float32).reshape(P, P, 12)
        runner.stop()

        print("Host: projection profile (separate diagnostic artifact and pass):")
        for projection_name, offset in (
            ("qkv_proj", 0),
            ("o_proj", 3),
            ("up_gate_proj", 6),
            ("down_proj", 9),
        ):
            projection_cycles = np.zeros((P, P))
            for pe_x in range(P):
                for pe_y in range(P):
                    projection_cycles[pe_y, pe_x] = calculate_cycles(
                        projection_profile_hwl[pe_y, pe_x, offset : offset + 3]
                    )
            print(
                f"  {projection_name}: mean={projection_cycles.mean():.1f}, "
                f"max_pe={projection_cycles.max():.1f} cycles"
            )
        print(
            "  Note: these diagnostic values include timestamp-marker overhead; "
            "the normal artifact contains no projection profiler."
        )
        return
    
    repeat_steps = 1
    warmup_steps = 0
    runner.launch("decode_host", np.int16(warmup_steps), np.int16(repeat_steps), nonblock=False)

    # Read the input and output of o_proj after the full decoder run.  These
    # arrays are diagnostic only and are copied outside the timed region.
    attention_output_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        attention_output_u32, sym_attention_output, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    o_proj_output_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        o_proj_output_u32, sym_o_proj_output, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    down_proj_input_u32 = np.zeros(P * P * bsz * ffn_dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        down_proj_input_u32, sym_down_proj_input, 0, 0, P, P, bsz * ffn_dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    down_proj_output_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        down_proj_output_u32, sym_down_proj_output, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    qkv_input_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        qkv_input_u32, sym_qkv_input, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    qkv_output_u32 = np.zeros(P * P * bsz * 3 * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        qkv_output_u32, sym_qkv_output, 0, 0, P, P, bsz * 3 * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    up_gate_input_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        up_gate_input_u32, sym_up_gate_input, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    up_gate_output_u32 = np.zeros(P * P * bsz * 2 * ffn_dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        up_gate_output_u32, sym_up_gate_output, 0, 0, P, P, bsz * 2 * ffn_dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    up_gate_rank_u32 = np.zeros(P * P * bsz * 2 * rank_ffn_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(
        up_gate_rank_u32, sym_up_gate_rank, 0, 0, P, P, bsz * 2 * rank_ffn_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False,
    )
    
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

    runner.stop()

    attention_output_pe = memcpy_view(
        attention_output_u32, np.dtype(np.float16)
    ).reshape(P, P, bsz * dim_p_pe)
    o_proj_output_pe = memcpy_view(
        o_proj_output_u32, np.dtype(np.float16)
    ).reshape(P, P, bsz * dim_p_pe)
    attention_output_host = rebuild_x_sharded_vector(attention_output_pe, bsz, dim)
    o_proj_output_device = rebuild_x_sharded_vector(o_proj_output_pe, bsz, dim)
    o_proj_output_host = (
        attention_output_host.astype(np.float32)
        @ tensor_o_a.astype(np.float32)
        @ tensor_o_b.astype(np.float32)
    )
    o_proj_rel_l2 = np.linalg.norm(
        o_proj_output_device.astype(np.float32) - o_proj_output_host
    ) / (np.linalg.norm(o_proj_output_host) + 1e-8)
    down_proj_input_pe = memcpy_view(
        down_proj_input_u32, np.dtype(np.float16)
    ).reshape(P, P, bsz * ffn_dim_p_pe)
    down_proj_output_pe = memcpy_view(
        down_proj_output_u32, np.dtype(np.float16)
    ).reshape(P, P, bsz * dim_p_pe)
    down_proj_input_host = rebuild_x_sharded_vector(down_proj_input_pe, bsz, ffn_dim)
    down_proj_output_device = rebuild_x_sharded_vector(down_proj_output_pe, bsz, dim)
    down_proj_output_host = (
        down_proj_input_host.astype(np.float32)
        @ tensor_down_a.astype(np.float32)
        @ tensor_down_b.astype(np.float32)
    )
    down_proj_rel_l2 = np.linalg.norm(
        down_proj_output_device.astype(np.float32) - down_proj_output_host
    ) / (np.linalg.norm(down_proj_output_host) + 1e-8)
    qkv_input_pe = memcpy_view(qkv_input_u32, np.dtype(np.float16)).reshape(
        P, P, bsz * dim_p_pe
    )
    qkv_output_pe = memcpy_view(qkv_output_u32, np.dtype(np.float16)).reshape(
        P, P, bsz * 3 * dim_p_pe
    )
    qkv_input_host = rebuild_y_sharded_vector(qkv_input_pe, bsz, dim)
    # Q and K are changed in place by RoPE. V occupies the final third and is
    # untouched, making it a clean check of both low-rank GEMVs.
    v_output_pe = qkv_output_pe[:, :, 2 * bsz * dim_p_pe :]
    v_output_device = rebuild_y_sharded_vector(v_output_pe, bsz, dim)
    v_output_host = qkv_input_host.astype(np.float32) @ v_a.astype(np.float32) @ v_b.astype(np.float32)
    v_proj_rel_l2 = np.linalg.norm(
        v_output_device.astype(np.float32) - v_output_host
    ) / (np.linalg.norm(v_output_host) + 1e-8)
    up_gate_input_pe = memcpy_view(up_gate_input_u32, np.dtype(np.float16)).reshape(
        P, P, bsz * dim_p_pe
    )
    up_gate_output_pe = memcpy_view(up_gate_output_u32, np.dtype(np.float16)).reshape(
        P, P, bsz * 2 * ffn_dim_p_pe
    )
    up_gate_rank_pe = memcpy_view(up_gate_rank_u32, np.dtype(np.float16)).reshape(
        P, P, bsz * 2 * rank_ffn_p_pe
    )
    # Validate the actual distributed operation. The synthetic Decode pipeline
    # does not guarantee identical z_norm replicas on the unused PE axis, so a
    # single reconstructed global input is too strong an assumption here.
    input_blocks = up_gate_input_pe.reshape(P, P, bsz, dim_p_pe).astype(np.float32)
    up_a_blocks = tensor_up_a_pe.reshape(P, P, dim_p_pe, rank_ffn_p_pe).astype(np.float32)
    expected_rank = np.zeros((P, bsz, rank_ffn_p_pe), dtype=np.float32)
    for py in range(P):
        for px in range(P):
            expected_rank[py] += input_blocks[py, px] @ up_a_blocks[py, px]
    actual_rank = up_gate_rank_pe[:, :, : bsz * rank_ffn_p_pe].reshape(
        P, P, bsz, rank_ffn_p_pe
    ).astype(np.float32)
    expected_rank_replicated = np.repeat(expected_rank[:, None, :, :], P, axis=1)
    up_rank_rel_l2 = np.linalg.norm(actual_rank - expected_rank_replicated) / (
        np.linalg.norm(expected_rank_replicated) + 1e-8
    )

    up_b_blocks = tensor_up_b_pe.reshape(P, P, rank_ffn_p_pe, ffn_dim_p_pe).astype(np.float32)
    expected_up = np.zeros((P, bsz, ffn_dim_p_pe), dtype=np.float32)
    for px in range(P):
        for py in range(P):
            expected_up[px] += expected_rank[py] @ up_b_blocks[py, px]
    actual_up = up_gate_output_pe[:, :, : bsz * ffn_dim_p_pe].reshape(
        P, P, bsz, ffn_dim_p_pe
    ).astype(np.float32)
    expected_up_replicated = np.repeat(expected_up[None, :, :, :], P, axis=0)
    up_proj_rel_l2 = np.linalg.norm(actual_up - expected_up_replicated) / (
        np.linalg.norm(expected_up_replicated) + 1e-8
    )

    # -------------------------------------------------------------------------- #
    # ------------------------------ Compute time ------------------------------ #
    # -------------------------------------------------------------------------- #
    cycles_count = np.zeros((P, P))
    for pe_x in range(P):
        for pe_y in range(P):
            cycles_count[pe_y, pe_x] = calculate_cycles(timer_buf_time_hwl[pe_y, pe_x, :])
    
    cycles_count_mean = cycles_count.mean()
    cycles_per_layer = cycles_count_mean / repeat_steps
    print(f"Host: mean cycles per decoder layer: {cycles_per_layer}")
    print(f"Host: estimated cycles for {layer_num} layers: {cycles_per_layer * layer_num}")
    print(f"Host: o_proj device-vs-host factorized relative L2 error: {o_proj_rel_l2:.6f}")
    print(f"Host: down_proj device-vs-host factorized relative L2 error: {down_proj_rel_l2:.6f}")
    print(f"Host: v_proj device-vs-host factorized relative L2 error: {v_proj_rel_l2:.6f}")
    print(f"Host: up_proj device-vs-host factorized relative L2 error: {up_proj_rel_l2:.6f}")
    print(f"Host: up_proj stage-1 relative L2 error: {up_rank_rel_l2:.6f}")
    if not np.allclose(o_proj_output_device, o_proj_output_host, rtol=0.2, atol=0.05):
        raise AssertionError(
            "low-rank o_proj mismatch: "
            f"relative L2 error={o_proj_rel_l2:.6f}"
        )
    if not np.allclose(down_proj_output_device, down_proj_output_host, rtol=0.2, atol=0.05):
        raise AssertionError(
            "low-rank down_proj mismatch: "
            f"relative L2 error={down_proj_rel_l2:.6f}"
        )
    if not np.allclose(v_output_device, v_output_host, rtol=0.2, atol=0.05):
        raise AssertionError(
            "low-rank v_proj mismatch: "
            f"relative L2 error={v_proj_rel_l2:.6f}"
        )
    if up_proj_rel_l2 > 0.02 or up_rank_rel_l2 > 0.02:
        raise AssertionError(
            "low-rank up_proj mismatch: "
            f"relative L2 error={up_proj_rel_l2:.6f}"
        )

if __name__ == "__main__":
    main()
