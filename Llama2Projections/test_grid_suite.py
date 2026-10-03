import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench
import sweep
import rectangular
import profile_grid_suite as suite


class GridSuiteTests(unittest.TestCase):
    def test_all_45_cases_through_both_runners_and_merge(self):
        for runner, module in [('bench', bench), ('sweep', sweep)]:
            with self.subTest(runner=runner), tempfile.TemporaryDirectory(dir=bench.BENCH_DIR) as td:
                root = Path(td)
                with patch.object(bench, 'BENCH_DIR', root), patch.object(bench, 'RUNS_DIR', root/'runs'):
                    jobs = suite.cases('all45', runner)
                    self.assertEqual(len(jobs), 45)
                    self.assertEqual(len({j['run_dir'] for j in jobs}), 45)
                    self.assertEqual(sum(j['row']['execution']=='dense' for j in jobs), 9)
                    for job in jobs:
                        row = job['row']
                        self.assertEqual(row['Px'] * row['Py'], 4096)
                        self.assertEqual(row['pe_num_group_x'], 8)
                        self.assertEqual(row['pe_num_group_y'], 8)
                        if row['execution'] == 'fused':
                            self.assertEqual(row['rank'] % 128, 0)
                            index = suite.RATIOS.index(row['target_retained_work_ratio'])
                            self.assertEqual(row['rank'], suite.RANKS[row['projection_group']][index])
                        def fake_run(commands, log):
                            dense = row['execution'] == 'dense'
                            expected_kernel = 'MeshGEMVRectangular' if dense else 'MeshGEMVLowerRankRectangular'
                            self.assertIn(expected_kernel+'/compile.py', commands[0])
                            self.assertIn('--profile-phases', commands[1])
                            self.assertEqual(commands[1][commands[1].index('--repetitions')+1], '1')
                            self.assertEqual('--R' in commands[0], not dense)
                            for flag, field in [('PX','Px'), ('PY','Py'), ('M','M'), ('N','N'),
                                                ('group_num_x','group_num_x'), ('group_num_y','group_num_y')]:
                                for command in commands:
                                    self.assertEqual(command[command.index('--'+flag)+1], str(row[field]))
                            log.parent.mkdir(parents=True, exist_ok=True)
                            return bench.RESULT_PREFIX + json.dumps(dict(mean_cycles=100, max_cycles=110,
                                relative_l2_error=.001, phases={'profile_iteration':
                                {'mean_cycles': 90, 'max_pe_cycles': 95}}))
                        with patch.object(bench, 'run_logged', side_effect=fake_run) as call, contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(module.main(job['command'][2:]), 0)
                            self.assertEqual(module.main(job['command'][2:]+['--resume']), 0)
                            self.assertEqual(call.call_count, 1)
                    rows = suite.collect(jobs, root)
                    self.assertEqual(len(rows), 45)
                    self.assertTrue(all(r['status']=='ok' for r in rows))
                    phases = bench.read_rows(root/'phases.csv')
                    self.assertEqual(len(phases), 45)
                    self.assertEqual({r['projection_group'] for r in phases}, {'qkvo','gate_up','down'})

    def test_low_rank_only_preserves_all_36_fused_cases(self):
        for runner in ['bench', 'sweep']:
            full = suite.cases('test', runner)
            fused = suite.cases('test', runner, low_rank_only=True)
            self.assertEqual(fused, [j for j in full if j['row']['execution'] == 'fused'])
            self.assertEqual(len(fused), 36)

    def test_missing_profile_fails_instead_of_silent_success(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(bench, 'run_logged', return_value=bench.RESULT_PREFIX+'{"mean_cycles": 10}'):
                row = bench.run_case({'status': 'failed', 'profile_phases': True}, [],
                                     Path(td)/'case.log', {'mean_cycles': 'cycles'})
            self.assertEqual(row['status'], 'failed')
            self.assertIn('phase timings', row['error_message'])

    def test_dense_uses_output_x_divisibility_and_rejects_singleton_reduction(self):
        args = rectangular.parse_args(['--PX','8','--PY','16','--group-num-x','2',
            '--group-num-y','4','--M','64','--N','24'], 'sweep', 'dense')
        self.assertIsNone(rectangular.plan(args, 'sweep')[0][1])
        args.gy = [16]
        self.assertIn('2 PEs', rectangular.plan(args, 'sweep')[0][1])

    def test_invalid_case_is_saved_and_returns_failure(self):
        with tempfile.TemporaryDirectory(dir=bench.BENCH_DIR) as td, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(bench, 'BENCH_DIR', Path(td)):
                args=['--run-name','bad','--PX','8','--PY','16','--group-num-x','2',
                      '--group-num-y','4','--M','64','--N','25']
                self.assertEqual(rectangular.main(args,'sweep','dense'),1)
                row=bench.read_rows(Path(td)/'sweeps/bad/dense-rectangular.csv')[0]
                self.assertEqual(row['status'],'failed')
                self.assertIn('divisible',row['error_message'])


if __name__ == '__main__':
    unittest.main()
