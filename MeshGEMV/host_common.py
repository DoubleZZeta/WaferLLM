"""Shared host-side numpy helpers for MeshGEMV (no Cerebras SDK imports).

Imported by both launch_sim.py (cs_python / runtime SDK) and launch_device.py
(appliance python / client SDK).
"""
import struct

import numpy as np


def cfg_name(P, M, N, group_num):
    """Per-config suffix used for compile artifacts (out_<cfg>, artifact_<cfg>.json)."""
    return f"{P}_{M // P}_{N // P}_{group_num}"


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


def make_inputs(P, M, N):
    """Random row-vector X (replicated across PE columns) and matrix W."""
    Mt = M // P
    X = np.random.rand(1, M).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, Mt), reps=(1, P))
    tensor_W = np.random.rand(M, N).astype(np.float16)
    return X, tensor_X, tensor_W


def tile_W(tensor_W, P, Mt, Nt):
    W1 = tensor_W.reshape(P, Mt, P, Nt)
    W2 = W1.transpose(0, 2, 1, 3)
    return W2.reshape(P, P, Mt * Nt).ravel()


def decode_timing(time_memcpy_hwl, time_ref_hwl, P, total_repeat_times):
    """Decode the on-chip TSC timestamps into (mean_cycle, max_cycle)."""
    time_start = np.zeros((P, P)).astype(int)
    time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            time_end[(h, w)] = make_u48(word)

    time_ref = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)

    for py in range(P):
        for px in range(P):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

    time_start = time_start - time_ref
    time_end = time_end - time_ref

    min_time_start = time_start.min()
    max_time_end = time_end.max()
    mean_cycle = np.mean(time_end - time_start) / total_repeat_times
    max_cycle = (max_time_end - min_time_start) / total_repeat_times
    return mean_cycle, max_cycle


def decode_phase_timing(profile_hwl, time_ref_hwl, P):
    """Decode one dense iteration split into compute, all-reduce, and control.

    Phase means are additive because adjacent timestamp differences are taken
    per PE before averaging. ``max_pe_cycles`` values are diagnostic and must
    not be added because a different PE may be slowest in each phase.
    """
    marker_count = 5
    markers = np.zeros((P, P, marker_count), dtype=np.int64)
    word = np.zeros(3, dtype=np.uint16)

    for h in range(P):
        for w in range(P):
            for marker in range(marker_count):
                first = int(float_to_hex(profile_hwl[h, w, 2 * marker]), 16)
                second = int(float_to_hex(profile_hwl[h, w, 2 * marker + 1]), 16)
                word[:] = (
                    first & 0xffff,
                    (first >> 16) & 0xffff,
                    second & 0xffff,
                )
                markers[h, w, marker] = make_u48(word)

    # Apply the same per-PE clock alignment used by decode_timing().
    time_ref = np.zeros((P, P), dtype=np.int64)
    for h in range(P):
        for w in range(P):
            first = int(float_to_hex(time_ref_hwl[h, w, 0]), 16)
            second = int(float_to_hex(time_ref_hwl[h, w, 1]), 16)
            word[:] = (
                first & 0xffff,
                (first >> 16) & 0xffff,
                second & 0xffff,
            )
            time_ref[h, w] = make_u48(word) - (w + h)
    markers -= time_ref[:, :, None]

    intervals = np.diff(markers, axis=2)
    names = (
        "control_before",
        "dense_local_compute",
        "dense_y_allreduce",
        "control_after",
    )

    metrics = {}
    for index, name in enumerate(names):
        per_pe = intervals[:, :, index]
        metrics[name] = {
            "mean_cycles": float(np.mean(per_pe)),
            "max_pe_cycles": float(np.max(per_pe)),
        }

    control = intervals[:, :, 0] + intervals[:, :, 3]
    total = markers[:, :, -1] - markers[:, :, 0]
    metrics["control_and_markers"] = {
        "mean_cycles": float(np.mean(control)),
        "max_pe_cycles": float(np.max(control)),
    }
    metrics["profile_iteration"] = {
        "mean_cycles": float(np.mean(total)),
        "max_pe_cycles": float(np.max(total)),
    }
    return metrics
