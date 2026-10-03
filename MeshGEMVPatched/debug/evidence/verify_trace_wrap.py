"""Verify the exact 100 MiB displacement in the supplied September 12 trace."""
import json
from pathlib import Path
import zipfile

import numpy as np

archive = Path(__file__).resolve().parents[3] / 'DenseHCFChunkedProbe' / 'waferllm_csv_results_and_hcf_debug_logs_only_20260912.zip'
with zipfile.ZipFile(archive) as z:
    path = next(n for n in z.namelist() if n.endswith('instructions/P1.1.jsonl'))
    records = [json.loads(line) for line in z.read(path).decode().splitlines()]
# demux_adaptor.csl line 421 receives the next transfer's packed header.
header = np.array([int(r['dest']) for r in records
                   if r['op'] == 'IMOV32.NF' and r['line'] == '421'], dtype=np.uint32)
assert len(header) == 36
np.random.seed(2025)
np.random.rand(4096)  # X is generated first.
w = np.random.rand(4096, 8192).astype(np.float16).view(np.uint16)
# W upload for PE row 49: local element first, then receiving PE column.
wire = w[49 * 64:50 * 64].reshape(64, 64, 128).transpose(0, 2, 1).ravel()
offset = 522488
np.testing.assert_array_equal(header, wire[offset:offset + len(header)])
# Each of the 50 row uploads has a 36-u32 adaptor header preceding its data.
displacement_words = 49 * (64 * 8192) + offset + 50 * 36
assert displacement_words * 4 == 100 * 1024 * 1024
num_receivers = int(header[:32].sum()) % 65536
assert num_receivers == 57135
assert int(header[35]) == 13681
print('All 36 header words match weight data from PE row 49.')
print('Stream displacement:', displacement_words * 4, 'bytes (100 MiB).')
print('Malformed num_receivers:', num_receivers)
print('Malformed memcpy_optype:', int(header[35]))
