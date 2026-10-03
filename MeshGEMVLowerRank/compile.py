"""Compile the one-artifact fused low-rank MeshGEMV implementation.

  --mode sim     : local `cslc` build into out_<cfg>/
  --mode device  : cloud SdkCompiler build -> compile_out/artifact_<cfg>.json

M, R and N must all be divisible by P. ``group_num`` selects the number of
groups in both statically configured two-level reductions.
"""
import argparse
import json
import os
import shutil
import subprocess
import time


def parse_args():
    ap = argparse.ArgumentParser(description="Compile MeshGEMV (WSE-3, SDK 2.10)")
    ap.add_argument("--mode", choices=["sim", "device"], required=True)
    ap.add_argument("--P", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--group_num", type=int, required=True)
    # cslc mounts only the working directory, so this must lie beneath it.
    ap.add_argument("--out-dir", help="sim build directory (default: out_<cfg>)")
    return ap.parse_args()


def main():
    args = parse_args()
    P, M, R, N, G = args.P, args.M, args.R, args.N, args.group_num
    if P < 3:
        raise SystemExit("P must be at least 3")
    for name, value in (("M", M), ("R", R), ("N", N)):
        if value % P != 0:
            raise SystemExit(f"{name}={value} must be divisible by P={P}")
    if R > min(M, N):
        raise SystemExit(f"R={R} cannot exceed min(M,N)={min(M, N)}")
    if G <= 0 or P % G != 0:
        raise SystemExit(f"group_num={G} must be positive and divide P={P}")
    Mt = M // P
    Rt = R // P
    Nt = N // P
    pe_num_group = P // G
    # Trailing roots make each static reduction phase one-directional. This is
    # what lets the X and Y collectives coexist within six user input queues.
    root_1st_phase = pe_num_group - 1
    root_2nd_phase = P - 1
    cfg = f"{P}_{Mt}_{Rt}_{Nt}_{G}"
    params = (
        f"P:{P},Mt:{Mt},Rt:{Rt},Nt:{Nt},pe_num_group:{pe_num_group},"
        f"root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase}"
    )

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = args.out_dir or f"out_{cfg}"
        if os.path.lexists(out_dir):
            # Not `rm -rf`: some servers wrap rm in an interactive prompt.
            shutil.rmtree(out_dir)
        cmd = [
            "cslc", "--arch=wse3", os.path.join(os.path.dirname(__file__), "src", "layout.csl"),
            f"--fabric-dims={P + 7},{P + 2}", "--fabric-offsets=4,1",
            f"--params={params}", "-o", out_dir, "--memcpy", "--channels", "1",
        ]
        subprocess.run(cmd, check=True)
    else:
        from cerebras.sdk.client import SdkCompiler
        os.makedirs("compile_out", exist_ok=True)
        options = (
            f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            f"-o out --memcpy --channels=1 --params={params}"
        )
        with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
            artifact_id = compiler.compile(
                app_path="src", csl_main="layout.csl", options=options, out_path="compile_out",
            )
        with open(f"compile_out/artifact_{cfg}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)

    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
