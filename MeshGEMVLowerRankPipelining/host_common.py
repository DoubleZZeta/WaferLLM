"""NumPy input, PE-layout and timestamp helpers for fused low-rank GEMV."""
import struct

import numpy as np


def cfg_name(Px, Py, M, R, N, group_num_x, group_num_y):
    """Name the artifact using the actual per-PE tile dimensions."""
    return f"{Px}_{Py}_{M // Py}_{R // Px}_{N // Py}_{group_num_x}_{group_num_y}"


def tile_vector(vector, Px, Py):
    """Split a [1,D] vector over PE rows and replicate it over PE columns."""
    _, M = vector.shape
    if M % Py:
        raise ValueError(f"vector dimension {M} must be divisible by Py={Py}")
    Mt = M // Py
    return np.repeat(vector.reshape(Py, 1, Mt), Px, axis=1).ravel()

def tile_A_for_y_reduce(A, Px, Py, Mt, Rt):
    """Place A[input block py, rank block px] on PE(py, px)."""
    blocks = A.reshape(Py, Mt, Px, Rt)
    return blocks.transpose(0, 2, 1, 3).reshape(
        Py, Px, Mt * Rt
    ).ravel()


def tile_B_for_x_reduce(B, Px, Py, Rt, Nt):
    """B[px rank block, py output block]."""
    blocks = B.reshape(Px, Rt, Py, Nt)
    return blocks.transpose(2, 0, 1, 3).reshape(
        Py, Px, Rt * Nt
    ).ravel()


def make_low_rank_inputs(Px, Py, M, R, N, seed=2025):
    """Generate SVD-derived rank-R factors and PE-ready tensors."""
    if min(Px, Py, M, R, N) <= 0:
        raise ValueError("Dimensions must be positive")
    if M % Py or N % Py or R % Px:
        raise ValueError("M and N must divide over Py; R over Px")
    if R > min(M, N):
        raise ValueError("R cannot exceed min(M, N)")
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
    Mt, Rt, Nt = M // Py, R // Px, N // Py
    return (
        X, W, A, B, 
        tile_vector(X, Px, Py),
        tile_A_for_y_reduce(A, Px, Py, Mt, Rt), 
        tile_B_for_x_reduce(B, Px, Py, Rt, Nt)
    )


def float_to_hex(value):
    return hex(struct.unpack("<I", struct.pack("<f", value))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def decode_timing(time_memcpy_hwl, time_ref_hwl, Px, Py, repeat_times):
    """Decode the on-chip TSC timestamps into (mean_cycle, max_cycle)."""
    time_start = np.zeros((Py, Px)).astype(int)
    time_end = np.zeros((Py, Px)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(Px):
        for h in range(Py):
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

    time_ref = np.zeros((Py, Px)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(Px):
        for h in range(Py):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)

    for py in range(Py):
        for px in range(Px):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

    time_start = time_start - time_ref
    time_end = time_end - time_ref

    min_time_start = time_start.min()
    max_time_end = time_end.max()
    mean_cycle = np.mean(time_end - time_start) / repeat_times
    max_cycle = (max_time_end - min_time_start) / repeat_times
    return mean_cycle, max_cycle


def decode_phase_timing(profile_hwl, time_ref_hwl, Px, Py):
    """Decode one fused iteration split into both compute and reduction stages.

    Phase means are additive because adjacent timestamp differences are taken
    per PE before averaging. ``max_pe_cycles`` values are diagnostic and must
    not be added because a different PE may be slowest in each phase.
    """
    marker_count = 7
    markers = np.zeros((Py, Px, marker_count), dtype=np.int64)
    word = np.zeros(3, dtype=np.uint16)

    for h in range(Py):
        for w in range(Px):
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
    time_ref = np.zeros((Py, Px), dtype=np.int64)
    for h in range(Py):
        for w in range(Px):
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

# These names and positions match mark(event) in src/meshgemv.csl.
PROFILE_EVENTS = (
    'iteration_start', 'A0_start', 'A0_end', 'Y0_launch', 'Y0_ready',
    'A1_start', 'A1_end', 'Y1_launch', 'Y1_ready', 'B0_start', 'B0_end',
    'B1_start', 'B1_end', 'X_start', 'X_end', 'iteration_end',
)
PROFILE_WORDS = 2 * len(PROFILE_EVENTS)
PROFILE_INTERVALS = {
    'A0_compute': (1, 2), 'A1_compute': (5, 6),
    'Y0_launch_to_ready': (3, 4), 'Y1_launch_to_ready': (7, 8),
    'B0_compute': (9, 10), 'B1_compute': (11, 12),
    'X_collective': (13, 14), 'profile_iteration': (0, 15),
}


def decode_pipeline_timing(words):
    """Per-PE relative times; no cross-PE clock-alignment assumption.

    Y intervals include callback delay, not just active fabric execution.
    Interval intersections show opportunity/latency overlap, not proof of
    simultaneous arithmetic and transfer on every cycle.
    """
    raw = np.ascontiguousarray(words, dtype=np.float32).view(np.uint32)
    if raw.ndim != 3 or raw.shape[-1] != PROFILE_WORDS:
        raise ValueError(f'expected (Py, Px, {PROFILE_WORDS}) containers')
    stamps = raw[..., 0::2].astype(np.uint64) | (
        (raw[..., 1::2].astype(np.uint64) & 0xffff) << np.uint64(32))
    times = ((stamps - stamps[..., :1]) & np.uint64((1 << 48) - 1)).astype(np.int64)
    if np.any(times > (1 << 40)):
        raise ValueError('missing/invalid pipeline timestamps')
    metrics = {}
    for name, (start, end) in PROFILE_INTERVALS.items():
        delta = times[..., end] - times[..., start]
        if np.any(delta < 0):
            raise ValueError(f'negative duration: {name}')
        metrics[name] = dict(mean_cycles=float(delta.mean()), max_pe_cycles=float(delta.max()))
    for name, a, b in [('Y0_A1_interval_overlap', (3, 4), (5, 6)),
                       ('Y1_B0_interval_overlap', (7, 8), (9, 10))]:
        delta = np.maximum(0, np.minimum(times[..., a[1]], times[..., b[1]]) -
                           np.maximum(times[..., a[0]], times[..., b[0]]))
        metrics[name] = dict(mean_cycles=float(delta.mean()), max_pe_cycles=float(delta.max()))
    # Validate scheduling dependencies on every PE.
    for before, after in [(2,3), (4,7), (6,7), (4,9), (8,11), (10,11), (12,13), (14,15)]:
        if np.any(times[..., before] > times[..., after]):
            raise ValueError(f'bad dependency: {PROFILE_EVENTS[before]} -> {PROFILE_EVENTS[after]}')
    return metrics, times


def write_pipeline_profile(words, directory='.'):
    """Write a readable phase summary and all PE timelines beside the artifact."""
    import csv
    import json
    from pathlib import Path
    metrics, times = decode_pipeline_timing(words)
    out = Path(directory)
    with (out / 'pipeline_timeline.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['py', 'px', *PROFILE_EVENTS])
        for py in range(times.shape[0]):
            for px in range(times.shape[1]):
                writer.writerow([py, px, *times[py, px].tolist()])
    (out / 'pipeline_profile.json').write_text(json.dumps({
        'description': 'cycles relative to each PE iteration start; Y includes callback delay; intervals overlap',
        'events': PROFILE_EVENTS, 'intervals': PROFILE_INTERVALS, 'metrics': metrics,
        'times': times.tolist(),
    }, indent=2))
    for name, m in metrics.items():
        print(f'{name}: mean={m["mean_cycles"]:.3f}, max PE={m["max_pe_cycles"]:.3f} cycles')
    return metrics


def validate_pipeline_artifact(out_dir, args):
    import json
    from pathlib import Path
    path = Path(out_dir) / 'pipeline_config.json'
    if not path.is_file():
        raise SystemExit('Missing pipeline_config.json: recompile with the updated compiler script')
    config = json.loads(path.read_text())
    for name in ('PX', 'PY', 'M', 'R', 'N', 'group_num_x', 'group_num_y', 'profile_phases', 'schedule'):
        if config.get(name) != getattr(args, name):
            raise SystemExit(f'Compiled pipeline mismatch for {name}: recompile for these options')
