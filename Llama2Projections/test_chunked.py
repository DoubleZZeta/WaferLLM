"""Host checks for runner wiring, resume safety and overlapping timing records."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import bench
import sweep

spec = importlib.util.spec_from_file_location('chunk_host', bench.REPO_DIR / 'MeshGEMVLowerRankChunked/host_common.py')
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


class ChunkTests(unittest.TestCase):
    def test_runners_pass_compile_and_launch_options_and_lock_resume(self):
        for runner in (bench, sweep):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory(prefix='out_test_', dir=bench.REPO_DIR) as tmp:
                root = Path(tmp)
                args = ['fused-chunked', '--run-name', 'case', '--PX', '4', '--PY', '8',
                        '--group-num-x', '2', '--group-num-y', '2', '--ranks', '20',
                        '--chunks', '2', '--schedule', 'serial', '--profile-phases']
                args += ['--M', '64', '--N', '128'] if runner is sweep else ['--projections', 'qkvo']
                def success(row, *unused):
                    return {**row, 'status': 'ok'}
                with patch.object(bench, 'RUNS_DIR', root / 'runs'), \
                     patch.object(bench, 'BENCH_DIR', root), \
                     patch.object(bench, 'CASES', [bench.ProjectionCase('qkvo',64,128,4)]), \
                     patch.object(bench, 'run_case', side_effect=success) as call, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(args), 0)
                    commands = call.call_args.args[1]
                    self.assertIn('MeshGEMVLowerRankChunked/compile.py', commands[0])
                    self.assertTrue(any(s.endswith('launch_chunked_rectangular.py') for s in commands[1]))
                    for command in commands:
                        self.assertIn('--profile-phases', command)
                        self.assertEqual(command[command.index('--chunks')+1], '2')
                        self.assertEqual(command[command.index('--schedule')+1], 'serial')
                    call.reset_mock()
                    self.assertEqual(runner.main(args + ['--resume']), 0)
                    call.assert_not_called()
                    changed = args.copy()
                    changed[changed.index('--chunks')+1] = '3'
                    with self.assertRaisesRegex(SystemExit, 'new run name'):
                        runner.main(changed + ['--resume'])

    def test_runner_archives_chunk_timeline_with_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'build'
            artifact.mkdir()
            timeline = {"units": "cycles", "markers_py_px_event": [[[0, 1]]]}
            (artifact / 'chunk_timeline.json').write_text(json.dumps(timeline))
            result = {"mean_cycles": 10, "phases": {"pipeline_elapsed":
                      {"mean_cycles": 8, "max_pe_cycles": 9}}}
            row = {"status": "pending", "chunks": 2, "profile_phases": True}
            with patch.object(bench, 'run_logged', return_value='BENCH_RESULT ' + json.dumps(result)):
                actual = bench.run_case(row, [['cs_python', '--out-dir', str(artifact)]],
                                        root / 'case.log', {"mean_cycles": "fused_mean_cycles"})
            self.assertEqual(actual['status'], 'ok')
            self.assertEqual(json.loads((root / 'case.timeline.json').read_text()), timeline)

    def test_overlapping_markers_are_not_added_as_serial_phases(self):
        # Outer pipeline [10,100] contains both ready events and B computations.
        times = np.array([0,10,100,130,30,40,60,70,75,90], dtype=np.uint64) + (1 << 40)
        words = np.empty((1,1,20), dtype=np.uint32)
        words[...,0::2] = times & 0xffffffff
        words[...,1::2] = times >> 32
        phases, timeline = hc.decode_chunk_timing(words.view(np.float32), 2)
        self.assertEqual(phases['pipeline_elapsed']['mean_cycles'], 90)
        self.assertEqual(phases['profile_iteration']['mean_cycles'], 130)
        self.assertEqual(phases['chunk1_stage2_compute']['mean_cycles'], 15)
        self.assertEqual(timeline['markers_py_px_event'][0][0][0], 0)
        # Invalid: B consumes a chunk before the completion callback.
        words[...,12] = 29  # chunk0 compute-end: also invalid within its interval
        with self.assertRaises(ValueError):
            hc.decode_chunk_timing(words.view(np.float32), 2)


if __name__ == '__main__':
    unittest.main()
