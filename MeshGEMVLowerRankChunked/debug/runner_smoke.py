#!/usr/bin/env python3
"""SDK-container integration checks for both runner entry points."""
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Llama2Projections'))
import bench
import sweep

os.chdir(ROOT)
# Inside the container the SDK Python is already active; emulate the outer
# cs_python launcher so the actual runner commands are exercised unchanged.
with tempfile.TemporaryDirectory(prefix='out_runner_', dir=ROOT / 'MeshGEMVLowerRankChunked') as tmp:
    temp = Path(tmp)
    shim = temp / 'cs_python'
    shim.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    shim.chmod(0o755)
    os.environ['PATH'] = str(temp) + os.pathsep + os.environ['PATH']
    bench.CASES = [bench.ProjectionCase('qkvo', 64, 128, 4)]
    # Place generated runner output beneath ignored out_* directories.
    bench.RUNS_DIR = temp / 'runs'
    args = ['--run-name', 'bench', '--PX', '4', '--PY', '8', '--ranks', '20',
            '--group-num-x', '2', '--group-num-y', '2', '--chunks', '2',
            '--profile-phases', '--suppress-simfab-trace', '--projections', 'qkvo']
    assert bench.main(['fused-chunked', *args]) == 0
    log_dir = bench.RUNS_DIR / 'bench/logs'
    assert len(list(log_dir.glob('*.phases.csv'))) == 1
    assert len(list(log_dir.glob('*.timeline.json'))) == 1
    saved = json.loads(next(log_dir.glob('*.timeline.json')).read_text())
    assert len(saved['markers_py_px_event']) == 8
    assert bench.main(['fused-chunked', *args, '--resume']) == 0
    # Exercise the real sweep path with a unique temporary run name.
    sweep_args = ['--run-name', 'sweep', '--PX', '8', '--PY', '4', '--ranks', '40',
                  '--group-num-x', '2', '--group-num-y', '2', '--chunks', '3',
                  '--schedule', 'serial', '--M', '64', '--N', '128',
                  '--repetitions', '3', '--suppress-simfab-trace']
    name = temp.name
    sweep_args[sweep_args.index('--run-name') + 1] = name
    target = bench.BENCH_DIR / 'sweeps' / name
    try:
        assert sweep.main(['fused-chunked', *sweep_args]) == 0
        with (target / 'fused-chunked.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]['status'] == 'ok'
        assert rows[0]['chunks'] == '3' and rows[0]['schedule'] == 'serial'
        print('PASS: bench profiling/timeline/resume and sweep execution', flush=True)
    finally:
        import shutil
        if target.exists():
            shutil.rmtree(target)
