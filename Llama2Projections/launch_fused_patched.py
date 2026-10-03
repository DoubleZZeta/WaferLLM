#!/usr/bin/env python3
"""Apply the separate HCF upload helper to the existing fused kernel."""
from common import load_kernel_module
from launch_fused import main


if __name__ == "__main__":
    main(upload_f16=load_kernel_module("MeshGEMVPatched", "sim_transfers").upload_f16)
