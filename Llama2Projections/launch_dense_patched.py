#!/usr/bin/env python3
"""Run the separate MeshGEMVPatched implementation with bounded uploads."""
from common import load_kernel_module
from launch_dense import main


if __name__ == "__main__":
    main(upload_f16=load_kernel_module("MeshGEMVPatched", "sim_transfers").upload_f16,
         kernel="MeshGEMVPatched")
