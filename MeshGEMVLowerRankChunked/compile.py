"""Compile the one-artifact fused low-rank MeshGEMV implementation.

  --mode sim     : local `cslc` build into out_<cfg>/

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
    ap.add_argument("--mode", choices=["sim"], required=True)
    ap.add_argument("--PX", type=int, required=True)
    ap.add_argument("--PY", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--group_num_x", type=int, required=True)
    ap.add_argument("--group_num_y", type=int, required=True)
    # cslc mounts only the working directory, so this must lie beneath it.
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--schedule", choices=["pipeline", "serial"], default="pipeline")
    ap.add_argument("--profile-phases", action="store_true")
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
    if not 1 <= args.chunks <= Rt:
        raise SystemExit("require 1 <= chunks <= R/PX (no empty chunks)")
    if args.mode != "sim":
        raise SystemExit("Chunked prototype currently supports simulator mode only")
    pe_num_group_x = Px // Gx
    pe_num_group_y = Py // Gy
    # Trailing roots make each static reduction phase one-directional. This is
    # what lets the X and Y collectives coexist within six user input queues.
    root_1st_phase_x = pe_num_group_x - 1
    root_2nd_phase_x = Px - 1
    root_1st_phase_y = pe_num_group_y - 1
    root_2nd_phase_y = Py - 1
    cfg = f"{Px}_{Py}_{Mt}_{Rt}_{Nt}_{Gx}_{Gy}"
    cfg += f"_c{args.chunks}_{args.schedule}_p{int(args.profile_phases)}"
    params = (
        f"Px:{Px},Py:{Py},Mt:{Mt},Rt:{Rt},Nt:{Nt},pe_num_group_x:{pe_num_group_x},pe_num_group_y:{pe_num_group_y},"
        f"root_1st_phase_x:{root_1st_phase_x},root_2nd_phase_x:{root_2nd_phase_x},root_1st_phase_y:{root_1st_phase_y},root_2nd_phase_y:{root_2nd_phase_y}"
    )

    params += f",Chunks:{args.chunks},Pipeline:{int(args.schedule == 'pipeline')},Profile:{int(args.profile_phases)}"

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
        with open(os.path.join(out_dir, "chunk_config.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": args.chunks, "schedule": args.schedule,
                       "profile_phases": args.profile_phases,
                       "PX": Px, "PY": Py, "M": M, "R": R, "N": N,
                       "group_num_x": Gx, "group_num_y": Gy}, f, indent=2)
    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
