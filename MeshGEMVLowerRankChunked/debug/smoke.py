#!/usr/bin/env python3
"""Run inside an SDK container from the repository root; small simulator tests only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MeshGEMVLowerRankChunked/out_validation'

def run(name, px, py, rank, gx, gy, chunks, schedule='pipeline', profile=False, baseline=False, m=64, n=128):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    kernel = 'MeshGEMVLowerRankRectangular' if baseline else 'MeshGEMVLowerRankChunked'
    launcher = 'launch_fused_rectangular.py' if baseline else 'launch_chunked_rectangular.py'
    shape = ['--PX', str(px), '--PY', str(py), '--M', str(m), '--R', str(rank), '--N', str(n),
             '--group_num_x', str(gx), '--group_num_y', str(gy)]
    extra = [] if baseline else ['--chunks', str(chunks), '--schedule', schedule]
    flags = ['--profile-phases'] if profile else []
    build = str(folder / 'build')
    commands = [
        [sys.executable, f'{kernel}/compile.py', '--mode', 'sim', *shape, *extra, *flags, '--out-dir', build],
        [sys.executable, f'Llama2Projections/{launcher}', *shape, *extra, *flags,
         '--repetitions', '1' if profile else '3', '--out-dir', build, '--suppress-simfab-trace'],
    ]
    result = None
    for i, command in enumerate(commands):
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=180)
        (folder / f'{i}.log').write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f'{name}: {completed.stdout[-4000:]}')
        if i == 1:
            result = json.loads(next(line[len('BENCH_RESULT '):] for line in completed.stdout.splitlines()
                                     if line.startswith('BENCH_RESULT ')))
    result.update(case=name, PX=px, PY=py, M=m, N=n, R=rank, chunks=chunks, schedule=schedule, profile=profile)
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    # Odd per-PE rank, both orientations, nonuniform chunks, singleton groups,
    # one group, single chunk, and profiling. All runs include a warmup.
    cases = [
        ('baseline',4,8,20,2,2,1,'serial',False,True),
        ('one',4,8,20,2,2,1),
        ('pipeline_odd',4,8,20,2,2,2),
        ('serial_odd',4,8,20,2,2,2,'serial'),
        ('reverse_three',8,4,40,2,2,3),
        ('singleton_groups',4,4,16,4,4,4),
        ('single_group',4,4,24,1,1,2),
        ('profile_pipeline',4,8,20,2,2,2,'pipeline',True),
        ('profile_serial',4,8,20,2,2,2,'serial',True),
        ('larger_baseline',4,8,128,2,2,1,'serial',False,True,256,1024),
        ('larger_pipeline',4,8,128,2,2,2,'pipeline',False,False,256,1024),
        ('larger_serial',4,8,128,2,2,2,'serial',False,False,256,1024),
    ]
    results = []
    for case in cases:
        results.append(run(*case))
        (OUT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} simulator cases; {OUT / "summary.json"}', flush=True)
