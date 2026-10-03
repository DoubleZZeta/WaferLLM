"""Compile dense suite shapes on the requested grids, without simulation."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--projections', nargs='+', choices=['qkvo','gate_up','down'],
                    default=['qkvo','gate_up','down'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
shapes = {'qkvo': (4096,4096), 'gate_up': (4096,11008), 'down': (11008,4096)}
for name in args.projections:
    m,n = shapes[name]
    for px, py, gx, gy in [(64,64,8,8),(32,128,4,16),(128,32,16,4)]:
        subprocess.run([sys.executable, str(root/'MeshGEMVRectangular/compile.py'),
                        '--mode','sim','--PX',str(px),'--PY',str(py),
                        '--group_num_x',str(gx),'--group_num_y',str(gy),
                        '--M',str(m),'--N',str(n),
                        '--out-dir',str(root/f'MeshGEMVRectangular/out_full_{name}_{px}x{py}')], check=True)
print('ALL REQUESTED FULL-GRID DENSE COMPILES PASSED', flush=True)
