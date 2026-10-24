#!/usr/bin/env python3
"""Plot one PE's instrumented pipeline timeline; requires matplotlib and numpy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', type=Path, help='pipeline_profile.json from the artifact directory')
    p.add_argument('--px', type=int, default=0)
    p.add_argument('--py', type=int, default=0)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    data = json.loads(args.profile.read_text())
    times = np.asarray(data['times'])
    if not (0 <= args.py < times.shape[0] and 0 <= args.px < times.shape[1]):
        p.error('PE coordinates outside captured grid')
    t = times[args.py, args.px]
    rows = ['A0_compute', 'A1_compute', 'Y0_launch_to_ready', 'B0_compute',
            'Y1_launch_to_ready', 'B1_compute', 'X_collective']
    fig, ax = plt.subplots(figsize=(11, 5))
    for y, name in enumerate(rows):
        start, end = data['intervals'][name]
        color = '#d87513' if name.startswith(('Y', 'X')) else '#2563a6'
        ax.barh(y, t[end]-t[start], left=t[start], height=.65, color=color)
        ax.text(t[end]+5, y, f'{t[end]-t[start]}', va='center', fontsize=9)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlabel('Cycles since this PE started the instrumented iteration')
    ax.set_title(f'Pipeline timeline — PE ({args.px}, {args.py}); total {t[-1]} cycles')
    ax.set_xlim(0, t[-1] * 1.12)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, 'Orange Y bars include callback delay. Interval overlap is not proof of continuous fabric progress.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    out = args.output or args.profile.with_name('pipeline_timeline.png')
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == '__main__':
    main()
