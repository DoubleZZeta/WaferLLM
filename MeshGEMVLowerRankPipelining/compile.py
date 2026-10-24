"""Compile the one-artifact fused low-rank MeshGEMV implementation.

  --mode sim     : local `cslc` build into out_<cfg>/
  --mode device  : cloud SdkCompiler build -> compile_out/artifact_<cfg>.json

M and N must be divisible by PY, and R by PX. The two group counts
select groups independently for the X and Y reductions.
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
    ap.add_argument("--PX", type=int, required=True)
    ap.add_argument("--PY", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--group_num_x", type=int, required=True)
    ap.add_argument("--group_num_y", type=int, required=True)
    # cslc mounts only the working directory, so this must lie beneath it.
    ap.add_argument("--out-dir", help="sim build directory (default: out_<cfg>)")
    return ap.parse_args()


def main():
    args = parse_args()
    Px, Py, M, R, N, Gx, Gy = args.PX, args.PY, args.M, args.R, args.N, args.group_num_x, args.group_num_y
    if Px < 3 or Py < 3:
        raise SystemExit("PX and PY must be at least 3")
    for name, value in (("M", M), ("N", N)):
        if value % Py != 0:
            raise SystemExit(f"{name}={value} must be divisible by Py={Py}")
    if R % Px != 0:
        raise SystemExit(f"R={R} must be divisible by Px={Px}")
    if R > min(M, N):
        raise SystemExit(f"R={R} cannot exceed min(M,N)={min(M, N)}")
    if Gx <= 0 or Px % Gx != 0:
        raise SystemExit(f"group_num_x={Gx} must be positive and divide PX={Px}")
    if Gy <= 0 or Py % Gy != 0:
        raise SystemExit(f"group_num_y={Gy} must be positive and divide PY={Py}")
    if min(M, R, N) <= 0:
        raise SystemExit("M, R and N must be positive")
    Mt = M // Py
    Rt = R // Px
    Nt = N // Py
    pe_num_group_x = Px // Gx
    pe_num_group_y = Py // Gy
    # Trailing roots make each static reduction phase one-directional. This is
    # what lets the X and Y collectives coexist within six user input queues.
    root_1st_phase_x = pe_num_group_x - 1
    root_2nd_phase_x = Px - 1
    root_1st_phase_y = pe_num_group_y - 1
    root_2nd_phase_y = Py - 1
    cfg = f"{Px}_{Py}_{Mt}_{Rt}_{Nt}_{Gx}_{Gy}"
    params = (
        f"Px:{Px},Py:{Py},Mt:{Mt},Rt:{Rt},Nt:{Nt},pe_num_group_x:{pe_num_group_x},pe_num_group_y:{pe_num_group_y},"
        f"root_1st_phase_x:{root_1st_phase_x},root_2nd_phase_x:{root_2nd_phase_x},root_1st_phase_y:{root_1st_phase_y},root_2nd_phase_y:{root_2nd_phase_y}"
    )

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = args.out_dir or f"out_{cfg}"
        if os.path.lexists(out_dir):
            # Not `rm -rf`: some servers wrap rm in an interactive prompt.
            shutil.rmtree(out_dir)
        cmd = [
            "cslc", "--arch=wse3", os.path.join(os.path.dirname(__file__), "src", "layout.csl"),
            f"--fabric-dims={Px + 7},{Py + 2}", "--fabric-offsets=4,1",
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
