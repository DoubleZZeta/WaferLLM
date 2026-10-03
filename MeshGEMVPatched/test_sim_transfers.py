"""Run with cs_python -m unittest discover -s MeshGEMVPatched -p test_sim_transfers.py."""
import unittest
from unittest.mock import patch

import numpy as np

import sim_transfers as transfers


class CheckedRuntime:
    """Model bounded ingress and per-PE memory, including partial rectangles."""
    def __init__(self, shape, capacity):
        self.memory = np.zeros(shape, dtype=np.uint16)
        self.capacity = capacity
        self.pending = None
        self.uploads = 0

    def memcpy_h2d(self, symbol, data, x, y, w, h, length, **options):
        assert self.pending is None, "another upload queued before device completion"
        assert data.dtype == np.uint32 and data.nbytes <= self.capacity
        assert options['data_type'] == transfers.MemcpyDataType.MEMCPY_16BIT
        assert options['order'] == transfers.MemcpyOrder.ROW_MAJOR
        assert not options['streaming'] and not options['nonblock']
        self.pending = (symbol, x, y, w, h)
        self.memory[y:y+h, x:x+w] = data.reshape(h, w, length)
        self.uploads += 1

    def memcpy_d2h(self, data, symbol, x, y, w, h, length, **options):
        assert self.pending == (symbol, x, y, w, h), "fence omitted destination PEs"
        assert length == 1 and not options['nonblock']
        data[:] = self.memory[y:y+h, x:x+w, 0].ravel()
        self.pending = None


class UploadTests(unittest.TestCase):
    def test_split_rows_and_columns_preserves_bits_and_drains_every_batch(self):
        # Include sign bits, subnormals, NaNs and infinities: copy raw bits.
        bits = (np.arange(5 * 7 * 11, dtype=np.uint16) * 173).reshape(5, 7, 11)
        for pes, max_rows in [(14, None), (3, None), (100, 1)]:
            with self.subTest(pes=pes, max_rows=max_rows):
                capacity = pes * 11 * 4
                runner = CheckedRuntime(bits.shape, capacity)
                with patch.object(transfers, 'MAX_UPLOAD_BYTES', capacity):
                    transfers.upload_f16(runner, 17, bits.view(np.float16), max_rows=max_rows)
                np.testing.assert_array_equal(runner.memory, bits)
                self.assertIsNone(runner.pending)
                self.assertGreater(runner.uploads, 1)

    def test_corrupt_checkpoint_fails_before_next_upload(self):
        runner = CheckedRuntime((2, 3, 4), 48)
        read = runner.memcpy_d2h
        def corrupt(*args, **kwargs):
            read(*args, **kwargs)
            args[0][0] ^= 1
        runner.memcpy_d2h = corrupt
        with patch.object(transfers, 'MAX_UPLOAD_BYTES', 48):
            with self.assertRaisesRegex(RuntimeError, 'readback mismatch'):
                transfers.upload_f16(runner, 1, np.ones((2, 3, 4), dtype=np.float16))
        self.assertEqual(runner.uploads, 1)

    def test_rejects_a_tile_larger_than_budget(self):
        runner = CheckedRuntime((1, 1, 4), 8)
        with patch.object(transfers, 'MAX_UPLOAD_BYTES', 8):
            with self.assertRaises(ValueError):
                transfers.upload_f16(runner, 1, np.ones((1, 1, 4), dtype=np.float16))
        self.assertEqual(runner.uploads, 0)


if __name__ == '__main__':
    unittest.main()
