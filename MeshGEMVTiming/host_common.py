"""Shared host-side numpy helpers for MeshGEMV (no Cerebras SDK imports).

Imported by both launch_sim.py (cs_python / runtime SDK) and launch_device.py
(appliance python / client SDK).
"""
import struct
import csv
from pathlib import Path

import numpy as np


def cfg_name(P, M, N, group_num):
    """Per-config suffix used for compile artifacts (out_<cfg>, artifact_<cfg>.json)."""
    return f"{P}_{M // P}_{N // P}_{group_num}"


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


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


PROFILE_WORDS = 28


def decode_phase_timing(profile_hwl, time_ref_hwl, P, group_num, return_details=False):
    """Decode one dense iteration split into compute, all-reduce, and control.

    Phase means are additive because adjacent timestamp differences are taken
    per PE before averaging. ``max_pe_cycles`` values are diagnostic and must
    not be added because a different PE may be slowest in each phase.
    """
    if group_num < 2 or P % group_num or P // group_num < 2:
        raise ValueError("Require G >= 2, P divisible by G, and at least two PEs/group")
    if np.shape(profile_hwl) != (P, P, PROFILE_WORDS):
        raise ValueError(f"Expected {(P, P, PROFILE_WORDS)} profile words; rebuild/download all 28 words")
    marker_count = 14
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

    group_size = P // group_num
    root_y = (group_num // 2) * group_size + (group_size // 2)

    intervals = np.diff(markers[:, :, :6], axis=2)
    names = (
        "control_before",       # interval 0
        "dense_local_compute",  # interval 1
        "root_reduction",       # interval 2
        "root_broadcast",       # interval 3
        "control_after",        # interval 4
    )

    def summarise(values):
        return {
            "mean_cycles": float(np.mean(values)),
            "max_pe_cycles": float(np.max(values)),
        }

    metrics = {}
    for index, name in enumerate(names):
        if index in (2, 3):
            values = intervals[root_y, :, index]
        else:
            values = intervals[:, :, index]

        metrics[name] = summarise(values)

    # Original allreduce metric: compute end → allreduce end, on all PEs.
    allreduce = markers[:, :, 4] - markers[:, :, 2]
    metrics["dense_y_allreduce"] = summarise(allreduce)

    # Same interval, restricted to roots, for checking the split.
    metrics["root_allreduce"] = summarise(allreduce[root_y, :])

    control = intervals[:, :, 0] + intervals[:, :, 4]
    metrics["control_and_markers"] = summarise(control)

    total = markers[:, :, 5] - markers[:, :, 0]
    metrics["profile_iteration"] = summarise(total)

    # The appended pairs have role-dependent validity, not chronological order.
    detail_rows = []
    row_summaries = []
    for py in range(P):
        for label, start, end in reduction_operations(py, P, group_num):
            durations = markers[py, :, end] - markers[py, :, start]
            if np.any(durations < 0):
                raise ValueError(f"Negative duration for {label}, row {py}; check marker packing")
            row_summaries.append({"py": py, "operation": label, **summarise(durations)})
            for px in range(P):
                detail_rows.append({
                    "px": px, "py": py, "operation": label,
                    "start_aligned": int(markers[py, px, start]),
                    "end_aligned": int(markers[py, px, end]),
                    "start_from_compute_end": int(markers[py, px, start] - markers[py, px, 2]),
                    "end_from_compute_end": int(markers[py, px, end] - markers[py, px, 2]),
                    "cycles": int(durations[px]),
                })
    # All markers within this diagnostic run must be ordered on the roots.
    if np.any(intervals[root_y] < 0):
        raise ValueError("Root phase timestamps out of order; check packing/kernel version")
    if return_details:
        return metrics, row_summaries, detail_rows
    return metrics


def reduction_operations(py, P, G):
    """Mirror the branch order in two_tree_allreduce_y_profile exactly.

    A/B slots are not always first/second chronological receives: endpoints
    send, intermediates receive-add-send, and only roots consume two streams.
    """
    size = P // G
    local_root = size // 2
    group, position = divmod(py, size)
    global_root = (G // 2) * size + local_root
    ops = []
    if position == local_root:
        if position != size - 1:
            ops.append(("level1_root_receive_bottom", 6, 7))
        ops.append(("level1_root_receive_top", 8, 9))
    elif position == 0:
        ops.append(("level1_send_down", 6, 7))
    elif position == size - 1:
        ops.append(("level1_send_up", 6, 7))
    elif position < local_root:
        ops.append(("level1_receive_add_send_down", 6, 7))
    else:
        ops.append(("level1_receive_add_send_up", 8, 9))
    if py == global_root:
        if group != G - 1:
            ops.append(("level2_root_receive_bottom", 10, 11))
        ops.append(("level2_root_receive_top", 12, 13))
    elif position == local_root:
        if group == 0:
            ops.append(("level2_send_down", 10, 11))
        elif group == G - 1:
            ops.append(("level2_send_up", 10, 11))
        elif py < global_root:
            ops.append(("level2_receive_add_send_down", 10, 11))
        else:
            ops.append(("level2_receive_add_send_up", 12, 13))
    return ops


def report_operation_timing(summaries, rows, P, M, N, G, directory="."):
    """Write every PE's intervals and per-row summaries beside run artifacts."""
    root_y = (G // 2) * (P // G) + (P // G) // 2
    print(f"Blocking operations on second-level root row {root_y}:")
    for row in summaries:
        if row["py"] == root_y:
            print(f"  {row['operation']}: mean={row['mean_cycles']:.1f}, "
                  f"max_pe={row['max_pe_cycles']:.1f}")
    context = {"P": P, "G": G, "M": M, "N": N, "L": N // P}
    for suffix, records in (("operations", rows), ("operation_rows", summaries)):
        path = Path(directory) / f"timing_{P}_{M}_{N}_{G}.{suffix}.csv"
        records = [{**context, **record} for record in records]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved {path.resolve()}")
    print("Operation intervals include blocking waits, streaming arithmetic/send work, "
          "and marker overhead; they are not pure transfer times. "
          "Aligned times use the existing reference-clock convention.")
