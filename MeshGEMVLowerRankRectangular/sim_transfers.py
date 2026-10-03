"""Bound uploads to the SDK simulator's ingress queue and drain each batch."""
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


# SDK 2.10's local streamer can overwrite unread data in its 100 MiB ring.
# FP16 memcpy uses FOUR host/stream bytes per value. Leave ample room for
# command/configuration headers and the completion readback command.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


def upload_f16(runner, symbol, tiles, *, label="input", max_rows=None):
    """Upload [PE row, PE column, local element] tiles, fencing every batch.

    Blocking H2D only waits for host-side submission in the local simulator.
    A blocking D2H from every destination establishes device completion before
    the next batch is submitted. Read the first uploaded element back as a
    small integrity check; this does not replace full numerical validation.
    Call before launching the application kernel.
    """
    tiles = np.asarray(tiles)
    if tiles.ndim != 3 or tiles.dtype != np.float16 or min(tiles.shape) < 1:
        raise ValueError("tiles must be a nonempty [height, width, length] FP16 array")
    height, width, length = tiles.shape
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    pe_budget = MAX_UPLOAD_BYTES // (4 * length)
    if pe_budget < 1:
        raise ValueError("one PE's tile exceeds the simulator upload budget")
    columns = min(width, pe_budget)
    rows = min(height, pe_budget // columns, max_rows or height)
    options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                   order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    for y in range(0, height, rows):
        h = min(rows, height - y)
        for x in range(0, width, columns):
            w = min(columns, width - x)
            if rows < height or columns < width:
                print(f"Uploading {label}: PE rows {y}-{y+h-1}, "
                      f"columns {x}-{x+w-1}", flush=True)
            # Zero-extend the raw FP16 bits; each value occupies one u32.
            source = np.ascontiguousarray(tiles[y:y+h, x:x+w])
            words = source.view(np.uint16).astype(np.uint32).ravel()
            runner.memcpy_h2d(symbol, words, x, y, w, h, length, **options)
            checkpoint = np.zeros(w * h, dtype=np.uint32)
            runner.memcpy_d2h(checkpoint, symbol, x, y, w, h, 1, **options)
            expected = source.view(np.uint16)[:, :, 0].ravel()
            if not np.array_equal(checkpoint & 0xffff, expected):
                raise RuntimeError(f"{label} upload readback mismatch at PE rectangle "
                                   f"({x}, {y}, {w}, {h})")
    print(f"{label} upload completed on device", flush=True)
