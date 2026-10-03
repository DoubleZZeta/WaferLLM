import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench
import rectangular as rect
import sweep


class RectangularTests(unittest.TestCase):
    def args(self, *extra):
        return rect.parse_args(['--PX', '4', '--PY', '8', '--group-num-x', '2',
            '--group-num-y', '2', '--M', '64', '--N', '128', *extra], 'sweep')

    def test_plan_axes_rank_and_invalid_cases(self):
        args = self.args('--ranks', '16', '--retained-work-ratios', '.375')
        rows = rect.plan(args, 'sweep')
        self.assertEqual(len(rows), 1)
        row, reason = rows[0]
        self.assertIsNone(reason)
        self.assertEqual(row['pe_count'], 32)
        self.assertEqual(row['retained_work_ratio'], .375)
        args.py = [8, 7]
        self.assertIsNotNone(rect.plan(args, 'sweep')[1][1])
        args.py = [8]
        args.ranks = [14]
        args.retained_work_ratios = []
        self.assertIn('divisible by PX', rect.plan(args, 'sweep')[0][1])

    def test_mode_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            args = self.args('--ranks', '16')
            bench.resolve_config(args, run, True, rect.DEFAULT_CONFIG)
            with self.assertRaisesRegex(SystemExit, 'different kernel'):
                bench.resolve_config(args, run, True)

    def test_both_entry_points_dispatch(self):
        for module, mode in [(bench, 'bench'), (sweep, 'sweep')]:
            with patch.object(rect, 'main', return_value=0) as call:
                self.assertEqual(module.main(['fused-rectangular', '--help']), 0)
                call.assert_called_once_with(['--help'], mode=mode)

    def test_results_resume_profiles_and_layout_keys(self):
        with tempfile.TemporaryDirectory(dir=bench.BENCH_DIR) as td:
            with patch.object(bench, 'BENCH_DIR', Path(td)):
                argv = ['--run-name', 'test', '--PX', '4', '--PY', '8', '--group-num-x', '2',
                        '--group-num-y', '2', '--M', '64', '--N', '128', '--ranks', '16',
                        '--profile-phases', '--build-root', '/tmp/rect_test']
                phases = {'profile_iteration': {'mean_cycles': 10, 'max_pe_cycles': 11}}
                response = bench.RESULT_PREFIX + json.dumps(dict(mean_cycles=12, max_cycles=13,
                    relative_l2_error=.001, phases=phases))
                def fake_run(commands, log):
                    log.parent.mkdir(parents=True, exist_ok=True)
                    return response
                with patch.object(bench, 'run_logged', side_effect=fake_run) as run:
                    self.assertEqual(rect.main(argv, 'sweep'), 0)
                    commands, log = run.call_args.args
                    self.assertIn('MeshGEMVLowerRankRectangular/compile.py', commands[0])
                    self.assertIn('--PX', commands[0])
                    self.assertIn('--profile-phases', commands[1])
                    self.assertEqual(json.loads(log.with_suffix('.phases.json').read_text())['case']['status'], 'ok')
                    self.assertEqual(rect.main(argv + ['--resume'], 'sweep'), 0)
                    self.assertEqual(run.call_count, 1)
                    # Another layout must not be mistaken for the completed case.
                    argv[argv.index('--PY')+1] = '4'
                    self.assertEqual(rect.main(argv + ['--resume'], 'sweep'), 0)
                    self.assertEqual(run.call_count, 2)
                rows = bench.read_rows(Path(td)/'sweeps/test/fused-rectangular.csv')
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(r['repetitions'] == '1' for r in rows))
