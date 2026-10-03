"""SDK integration checks. Run from the repository root inside the SDK container."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'Llama2Projections'))
import bench
import sweep

with tempfile.TemporaryDirectory(prefix='dense_rect_runner_') as td:
    shim = Path(td)/'cs_python'
    shim.write_text('#!/bin/sh\nexec python "$@"\n')
    shim.chmod(0o755)
    os.environ['PATH'] = td + ':' + os.environ['PATH']
    name = Path(td).name
    report = []
    # Scaled versions of all three suite geometries, with two PEs per group.
    for px, py, gx, gy in [(8, 8, 4, 4), (4, 16, 2, 8), (16, 4, 8, 2)]:
        for execution in ['dense', 'fused']:
            for profile in ([True, False] if execution == 'dense' else [True]):
                run_name = f'{name}_{execution}_{px}x{py}_{profile}'
                args = [f'{execution}-rectangular', '--run-name', run_name,
                        '--PX', str(px), '--PY', str(py), '--group-num-x', str(gx),
                        '--group-num-y', str(gy), '--suppress-simfab-trace',
                        '--repetitions', '1' if profile else '2']
                if execution == 'fused':
                    args += ['--ranks', '32']
                if profile:
                    args += ['--profile-phases']
                # Exercise both entry points, all grid orientations and both timing paths.
                bench.CASES = [bench.ProjectionCase('qkvo', 64, 128, 4)]
                if profile:
                    assert bench.main(args) == 0
                    run = bench.RUNS_DIR/run_name
                else:
                    assert sweep.main(args + ['--M', '64', '--N', '128']) == 0
                    run = bench.BENCH_DIR/'sweeps'/run_name
                rows = bench.read_rows(run/f'{execution}-rectangular.csv')
                assert len(rows) == 1 and rows[0]['status'] == 'ok', rows
                if profile:
                    phases = json.loads(next((run/'logs').glob('*.phases.json')).read_text())['phases']
                    expected = (['dense_local_compute', 'dense_y_allreduce'] if execution == 'dense'
                                else ['stage1_local_compute', 'stage1_y_allreduce',
                                      'stage2_local_compute', 'stage2_x_allreduce'])
                    assert all(phases[k]['mean_cycles'] > 0 for k in expected), phases
                report.append(rows[0])
    output = root/'MeshGEMVRectangular/debug/runs'/name
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('ALL SDK RUNNER SMOKES PASSED:', output, flush=True)
