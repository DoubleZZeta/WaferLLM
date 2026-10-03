"""Unified compile entry for DecodeLowerRank (model/rank-config driven).

  --mode sim     : local build into out_<model-cfg>__<rank-cfg>/
  --mode device  : cloud build -> compile_out/artifact_<model-cfg>__<rank-cfg>.json

Including both basenames prevents builds with different ranks from colliding.
"""
import argparse
import json
import os
import subprocess
import time


def derive_params(cj, rj):
    P = cj["P"]
    G = cj["group_num"]
    rank_o = rj["rank_o"]
    rank_down = rj["rank_down"]
    rank_q = rj["rank_q"]
    if not rank_q == rj["rank_k"] == rj["rank_v"]:
        raise ValueError("rank_q, rank_k, and rank_v must be equal for fused QKV collectives")
    rank_up = rj["rank_up"]
    if rank_up != rj["rank_gate"]:
        raise ValueError("rank_up and rank_gate must be equal for fused FFN collectives")

    for name in ("dim", "ffn_dim"):
        if cj[name] % P != 0:
            raise ValueError(f"{name}={cj[name]} must be divisible by P={P}")
    for name, rank in (("rank_qkv", rank_q), ("rank_o", rank_o), ("rank_ffn", rank_up), ("rank_down", rank_down)):
        if rank % P != 0:
            raise ValueError(f"{name}={rank} must be divisible by P={P}")
    if not 0 < rank_o <= cj["dim"]:
        raise ValueError(f"rank_o must be in [1, {cj['dim']}], got {rank_o}")
    max_rank_down = min(cj["ffn_dim"], cj["dim"])
    if not 0 < rank_down <= max_rank_down:
        raise ValueError(f"rank_down must be in [1, {max_rank_down}], got {rank_down}")
    if not 0 < rank_q <= cj["dim"]:
        raise ValueError(f"QKV rank must be in [1, {cj['dim']}], got {rank_q}")
    if not 0 < rank_up <= min(cj["dim"], cj["ffn_dim"]):
        raise ValueError("Up/Gate rank exceeds the projection dimensions")

    d = {
        "P": P,
        "bsz": cj["bsz"],
        "dim_p_pe": cj["dim"] // P,
        "rank_o_p_pe": rank_o // P,
        "rank_qkv_p_pe": rank_q // P,
        "rank_ffn_p_pe": rank_up // P,
        "rank_down_p_pe": rank_down // P,
        "pes_p_head": P // cj["n_heads"],
        "pes_p_kv_head": P // cj["n_kv_heads"],
        "head_dim_p_pe": cj["head_dim"] // P,
        "seq_len_p_pe": cj["seq_len"] // P,
        "ffn_dim_p_pe": cj["ffn_dim"] // P,
        "pe_num_p_group": P // G,
    }
    d["root_1st_phase"] = d["pe_num_p_group"] // 2
    d["root_2nd_phase"] = (G // 2) * d["pe_num_p_group"] + d["root_1st_phase"]
    return d


def main():
    ap = argparse.ArgumentParser(description="Compile Decode (WSE-3, SDK 2.10)")
    ap.add_argument("--mode", choices=["sim", "device"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rank_config", required=True)
    ap.add_argument(
        "--profile-projections",
        action="store_true",
        help="build a separate diagnostic artifact with projection timers",
    )
    args = ap.parse_args()

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    rank_cfg_name = os.path.splitext(os.path.basename(args.rank_config))[0]
    profile_suffix = "__projection-profile" if args.profile_projections else ""
    artifact_name = f"{cfg_name}__{rank_cfg_name}{profile_suffix}"
    with open(args.config, encoding="utf-8") as f:
        cj = json.load(f)
    with open(args.rank_config, encoding="utf-8") as f:
        rj = json.load(f)
    d = derive_params(cj, rj)
    # cslc's --params parser accepts booleans as integer literals.
    d["enable_projection_profile"] = 1 if args.profile_projections else 0
    P = d["P"]
    params = ",".join(f"{k}:{v}" for k, v in d.items())

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = f"out_{artifact_name}"
        subprocess.run(["rm", "-rf", out_dir], check=True)
        cmd = [
            "cslc", "--arch=wse3", "./src/layout.csl",
            f"--fabric-dims={P + 7},{P + 2}", "--fabric-offsets=4,1",
            f"--params={params}", "-o", out_dir, "--memcpy", "--channels", "1",
        ]
        subprocess.run(cmd, check=True)
    else:
        from cerebras.sdk.client import SdkCompiler
        os.makedirs("compile_out", exist_ok=True)
        options = (
            f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            f"-o out --memcpy --channels=4 --params={params}"
        )
        with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
            artifact_id = compiler.compile(
                app_path="src", csl_main="layout.csl", options=options, out_path="compile_out",
            )
        with open(f"compile_out/artifact_{artifact_name}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)

    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
