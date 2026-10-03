"""NumPy input, PE-layout and timestamp helpers for fused low-rank GEMV."""
import struct

import numpy as np


def cfg_name(P, M, R, N, group_num):
    """Name the artifact using the actual per-PE tile dimensions."""
    return f"{P}_{M // P}_{R // P}_{N // P}_{group_num}"


def tile_vector(vector, P):
    """Split a [1,D] vector over PE rows and replicate it over PE columns."""
    _, dimension = vector.shape
    if dimension % P != 0:
        raise ValueError(f"vector dimension {dimension} must be divisible by P={P}")
    return np.tile(vector.reshape(P, dimension // P), reps=(1, P))


def tile_A_for_y_reduce(A, P, Mt, Rt):
    """Place A[input block py, rank block px] on PE(py, px)."""
    blocks = A.reshape(P, Mt, P, Rt)
    return blocks.transpose(0, 2, 1, 3).reshape(P, P, Mt * Rt).ravel()


def tile_B_for_x_reduce(B, P, Rt, Nt):
    """Place B[rank block px, output block py] on PE(py, px)."""
    blocks = B.reshape(P, Rt, P, Nt)
    return blocks.transpose(2, 0, 1, 3).reshape(P, P, Rt * Nt).ravel()


def make_low_rank_inputs(P, M, R, N, seed=2025):
    """Generate SVD-derived rank-R factors and PE-ready tensors."""
    for name, value in (("M", M), ("R", R), ("N", N)):
        if value % P != 0:
            raise ValueError(f"{name}={value} must be divisible by P={P}")
    if R > min(M, N):
        raise ValueError(f"R={R} cannot exceed min(M,N)={min(M, N)}")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, M), dtype=np.float32)
    W = rng.standard_normal((M, N), dtype=np.float32) / np.sqrt(M)
    U, singular_values, Vh = np.linalg.svd(W, full_matrices=False)
    sqrt_s = np.sqrt(singular_values[:R])
    A = U[:, :R] * sqrt_s[None, :]
    B = sqrt_s[:, None] * Vh[:R, :]
    X = X.astype(np.float16)
    W = W.astype(np.float16)
    A = A.astype(np.float16)
    B = B.astype(np.float16)
    Mt, Rt, Nt = M // P, R // P, N // P
    return (X, W, A, B, tile_vector(X, P),
            tile_A_for_y_reduce(A, P, Mt, Rt), tile_B_for_x_reduce(B, P, Rt, Nt))


def float_to_hex(value):
    return hex(struct.unpack("<I", struct.pack("<f", value))[0])


def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


def decode_timing(time_memcpy_hwl, time_ref_hwl, P, repeat_times):
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
    mean_cycle = np.mean(time_end - time_start) / repeat_times
    max_cycle = (max_time_end - min_time_start) / repeat_times
    return mean_cycle, max_cycle


def decode_phase_timing(profile_hwl, time_ref_hwl, P):
    """Decode one fused iteration split into both compute and reduction stages.

    Phase means are additive because adjacent timestamp differences are taken
    per PE before averaging. ``max_pe_cycles`` values are diagnostic and must
    not be added because a different PE may be slowest in each phase.
    """
    marker_count = 7
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
        "stage1_local_compute",
        "stage1_y_allreduce",
        "stage2_local_compute",
        "stage2_x_allreduce",
        "control_after",
    )

    metrics = {}
    for index, name in enumerate(names):
        per_pe = intervals[:, :, index]
        metrics[name] = {
            "mean_cycles": float(np.mean(per_pe)),
            "max_pe_cycles": float(np.max(per_pe)),
        }

    control = intervals[:, :, 0] + intervals[:, :, 5]
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
