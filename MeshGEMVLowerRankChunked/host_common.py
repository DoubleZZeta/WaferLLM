"""NumPy input, PE-layout and timestamp helpers for fused low-rank GEMV."""
import struct

import numpy as np


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


def float_to_hex(value):
    return hex(struct.unpack("<I", struct.pack("<f", value))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def decode_chunk_timing(words, chunks):
    """Decode overlapping events, retaining per-PE timelines for diagnosis.

    Only the three outer phases are additive. Chunk compute/event intervals
    lie within pipeline_elapsed and must never be added to it.
    """
    bits = np.ascontiguousarray(words, dtype=np.float32).view(np.uint32)
    assert bits.shape[-1] == 2 * (4 + 3 * chunks)
    marks = (bits[..., 0::2].astype(np.uint64) +
             ((bits[..., 1::2].astype(np.uint64) & 0xffff) << 32)).astype(np.int64)
    relative = marks - marks[..., :1]
    phases = {}
    def interval(name, start, end):
        delta = relative[..., end] - relative[..., start]
        if np.any(delta < 0):
            raise ValueError(f"Invalid timestamp order for {name}")
        phases[name] = {"mean_cycles": float(delta.mean()), "max_pe_cycles": float(delta.max())}
    interval('stage1_local_compute',0,1)
    interval('pipeline_elapsed',1,2)
    interval('stage2_x_allreduce',2,3)
    interval('profile_iteration',0,3)
    for k in range(chunks):
        interval(f'chunk{k}_stage2_compute',4+3*k+1,4+3*k+2)
        interval(f'chunk{k}_ready_since_pipeline_start',1,4+3*k)
        if np.any(relative[...,4+3*k+1] < relative[...,4+3*k]):
            raise ValueError('Chunk compute started before its broadcast completed')
        if k and np.any(relative[...,4+3*k+1] < relative[...,4+3*(k-1)+2]):
            raise ValueError('Stage-2 chunk accumulation is out of order')
    return phases, {"units":"cycles relative to this PE iteration start",
                    "marker_names":["iteration_start","pipeline_start","pipeline_end","output_reduce_end"]+
                       [f"chunk{k}_{event}" for k in range(chunks) for event in ("ready","compute_start","compute_end")],
                    "markers_py_px_event":relative.tolist(),
                    "note":"Per-PE timelines; no cross-PE clock alignment. Chunk intervals overlap outer phases."}


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


