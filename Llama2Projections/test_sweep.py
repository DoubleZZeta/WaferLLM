"""Host-only checks for sweep planning, resuming and the dense/fused join."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench
import sweep


def fake_run_case(row, commands, log_path, columns):
    """Stand in for the simulator: every metric becomes 100."""
    row.update({column: 100.0 for column in columns.values()}, status="ok")
    return row


class PlanTests(unittest.TestCase):
    def test_dense_plan_covers_every_combination_and_explains_skips(self):
        args = sweep.parse_args(['dense', '--P', '4,6', '--group-num', '2,4',
                                 '--M', '64,96', '--N', '64'])
        plan = sweep.dense_plan(args)
        self.assertEqual(len(plan), 2 * 2 * 2 * 1)
        problems = {key: problem for key, _, problem in plan}
        self.assertIsNone(problems[(4, 2, 64, 64)])
        self.assertEqual(problems[(6, 4, 64, 64)], 'group_num must divide P')
        self.assertEqual(problems[(6, 2, 64, 64)], 'M and N must be divisible by P')

    def test_fused_plan_rounds_ratios_and_does_not_repeat_ranks(self):
        args = sweep.parse_args(['fused', '--P', '4', '--M', '64', '--N', '128',
                                 '--ranks', '20,6,200', '--retained-work-ratios', '.5,.47'])
        plan = sweep.fused_plan(args)
        ranks = [key[4] for key, _, _ in plan]
        # .5 and .47 both round to R=20, which --ranks already planned.
        self.assertEqual(ranks, [20, 6, 200])
        problems = {key[4]: problem for key, _, problem in plan}
        self.assertIsNone(problems[20])
        self.assertEqual(problems[6], 'rank must be divisible by P')
        self.assertEqual(problems[200], 'rank must satisfy 0 < R <= min(M,N)')

    def test_fused_requires_ranks_or_ratios(self):
        with self.assertRaisesRegex(SystemExit, '--ranks'):
            sweep.validate(sweep.parse_args(['fused', '--M', '64', '--N', '64']))


class RunTests(unittest.TestCase):
    def run_quietly(self, argv, run_dir, config):
        args = sweep.parse_args(argv)
        with patch.object(bench, 'run_case', side_effect=fake_run_case) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            sweep.run_command(args, run_dir, config)
        return run

    def test_commands_follow_each_case_and_hcf_selection(self):
        for fixed in [False, True]:
            with self.subTest(fixed=fixed), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                config = {'repetitions': 1, 'hcf_fix': fixed}
                dense = self.run_quietly(['dense', '--P', '4,8', '--group-num', '2',
                                          '--M', '64', '--N', '128'], run_dir, config)
                fused = self.run_quietly(['fused', '--P', '8', '--group-num', '4',
                                          '--M', '64', '--N', '128', '--ranks', '16'],
                                         run_dir, config)
                self.assertEqual(dense.call_count, 2)
                dense_compile, dense_launch = dense.call_args_list[1].args[1]
                fused_compile, fused_launch = fused.call_args.args[1]
                suffix = '_patched' if fixed else ''
                kernel = 'MeshGEMVPatched' if fixed else 'MeshGEMV'
                self.assertEqual(dense_compile[1], f'{kernel}/compile.py')
                self.assertEqual(dense_launch[1], f'Llama2Projections/launch_dense{suffix}.py')
                self.assertEqual(fused_compile[1], 'MeshGEMVLowerRank/compile.py')
                self.assertEqual(fused_launch[1], f'Llama2Projections/launch_fused{suffix}.py')
                self.assertEqual(dense_launch[dense_launch.index('--P') + 1], '8')
                self.assertEqual(fused_launch[fused_launch.index('--R') + 1], '16')
                self.assertEqual(fused_launch[fused_launch.index('--group_num') + 1], '4')
                for compile_command, launch_command in [(dense_compile, dense_launch),
                                                        (fused_compile, fused_launch)]:
                    build = compile_command[compile_command.index('--out-dir') + 1]
                    self.assertEqual(build, launch_command[launch_command.index('--out-dir') + 1])

    def test_resume_skips_completed_cases_and_adds_new_values(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {'repetitions': 1, 'hcf_fix': False}
            self.run_quietly(['dense', '--M', '64', '--N', '64'], run_dir, config)
            again = self.run_quietly(['dense', '--resume', '--M', '64,128', '--N', '64'],
                                     run_dir, config)
            self.assertEqual(again.call_count, 1)
            self.assertEqual(len(bench.read_rows(run_dir / 'dense.csv')), 2)
            with self.assertRaisesRegex(SystemExit, '--resume'):
                self.run_quietly(['dense', '--M', '64', '--N', '64'], run_dir, config)

    def test_summary_joins_fused_rows_with_matching_dense_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {'repetitions': 1, 'hcf_fix': False}
            self.run_quietly(['dense', '--M', '64', '--N', '64'], run_dir, config)
            self.run_quietly(['fused', '--M', '64,128', '--N', '64', '--ranks', '8'],
                             run_dir, config)
            rows = {(int(row['M']), int(row['rank'])): row
                    for row in sweep.write_summary(run_dir)}
            self.assertEqual(rows[(64, 8)]['status'], 'ok')
            self.assertEqual(rows[(64, 8)]['fused_speedup_vs_dense'], 1.0)
            self.assertEqual(rows[(128, 8)]['status'], 'incomplete')
            self.assertEqual(rows[(128, 8)]['missing'], 'dense')

    def test_repetitions_and_hcf_fix_are_locked_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = sweep.parse_args(['dense', '--M', '64', '--N', '64', '--hcf-fix'])
            config = bench.resolve_config(first, run_dir, True, sweep.DEFAULT_CONFIG)
            self.assertEqual(config, {'repetitions': 3, 'hcf_fix': True, 'profile_phases': False})
            inherited = sweep.parse_args(['fused', '--P', '8', '--M', '128', '--N', '64',
                                          '--ranks', '8'])
            self.assertEqual(
                bench.resolve_config(inherited, run_dir, True, sweep.DEFAULT_CONFIG), config)
            changed = sweep.parse_args(['dense', '--M', '64', '--N', '64', '--repetitions', '1'])
            with self.assertRaisesRegex(SystemExit, 'use a new run name'):
                bench.resolve_config(changed, run_dir, True, sweep.DEFAULT_CONFIG)


if __name__ == '__main__':
    unittest.main()
