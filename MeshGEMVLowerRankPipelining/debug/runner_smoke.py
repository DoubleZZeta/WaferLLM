"""End-to-end runner check inside the SDK environment (repository root CWD)."""
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import json
import csv

import importlib.util
root = Path.cwd()
benchdir = root/'Llama2Projections'
sys.path.insert(0, str(benchdir))
import bench
import sweep

# Local SDK container has Python bindings directly, rather than cs_python.
with tempfile.TemporaryDirectory(prefix='rect_runner_bin_') as td:
    shim = Path(td)/'cs_python'
    shim.write_text('#!/bin/sh\nexec python "$@"\n')
    shim.chmod(0o755)
    os.environ['PATH'] = td + ':' + os.environ['PATH']
    name = 'rect_smoke_' + Path(td).name
    args = ['fused-rectangular', '--run-name', name, '--PX', '4', '--PY', '8',
            '--group-num-x', '2', '--group-num-y', '2', '--ranks', '16',
            '--suppress-simfab-trace']
    assert sweep.main(args + ['--M', '64', '--N', '128', '--repetitions', '2']) == 0
    bench.CASES = [bench.ProjectionCase('qkvo', 64, 128, 4)]
    assert bench.main(args + ['--profile-phases']) == 0
    for parent in ['runs', 'sweeps']:
        run = benchdir/parent/name
        rows = list(csv.DictReader((run/'fused-rectangular.csv').open()))
        assert len(rows) == 1 and rows[0]['status'] == 'ok', rows
        if parent == 'runs':
            profile = json.loads(next((run/'logs').glob('*.phases.json')).read_text())
            assert profile['case']['status'] == 'ok'
            assert profile['phases']['stage1_local_compute']['mean_cycles'] > 0
        print('Verified runner output:', run, flush=True)
