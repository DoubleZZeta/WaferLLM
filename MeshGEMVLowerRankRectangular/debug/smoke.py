"""Small SDK smoke cases; run from the repository root inside the SDK container."""
from pathlib import Path
import subprocess
import sys
import tempfile

kernel = Path(__file__).resolve().parents[1]
runs = kernel / 'debug' / 'runs'
runs.mkdir(exist_ok=True)
output = Path(tempfile.mkdtemp(prefix='rect_', dir=runs))
print(f'Smoke artifacts: {output}', flush=True)
for px, py, gx, gy in [(4, 8, 2, 2), (8, 4, 2, 2), (4, 4, 2, 2), (4, 8, 4, 1)]:
    shape = ['--PX', str(px), '--PY', str(py), '--M', '64', '--R', '16', '--N', '128',
             '--group_num_x', str(gx), '--group_num_y', str(gy)]
    artifact = output / f'p{px}x{py}_g{gx}x{gy}'
    subprocess.run([sys.executable, str(kernel/'compile.py'), '--mode', 'sim', *shape,
                    '--out-dir', str(artifact)], check=True)
    launch = [sys.executable, str(kernel/'launch_sim.py'), *shape, '--out-dir', str(artifact),
              '--suppress-simfab-trace']
    subprocess.run(launch, check=True)
    if px != py and gx == gy:
        subprocess.run(launch + ['--profile-phases'], check=True)
