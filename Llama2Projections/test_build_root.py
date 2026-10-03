"""Host-only checks for --build-root, which moves compiled artifacts into /tmp."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench
import sweep
from test_sweep import fake_run_case


def out_dir(command):
    return command[command.index('--out-dir') + 1]


def tmp_directory():
    """A temporary directory inside /tmp, whatever the platform default is."""
    return tempfile.TemporaryDirectory(dir='/tmp')


def as_tmp_path(path):
    """The /tmp spelling of a path, which is what both containers see."""
    return str(Path('/tmp') / Path(path).resolve().relative_to(Path('/tmp').resolve()))


class BuildRootTests(unittest.TestCase):
    def test_commands_unset_tmpdir_and_share_one_tmp_path(self):
        with tempfile.TemporaryDirectory() as runs, tmp_directory() as root:
            run_dir = Path(runs) / 'sweeps' / 'square'
            compile_command, launch_command = bench.kernel_commands(
                'MeshGEMV', 'launch_dense.py', ['--P', '4'], {'repetitions': 1},
                run_dir / 'build' / 'dense_case', False, Path(root))
            for command in (compile_command, launch_command):
                self.assertEqual(command[:3], ['env', '-u', 'TMPDIR'])
            expected = as_tmp_path(root) + '/sweeps/square/dense_case'
            self.assertEqual(out_dir(compile_command), expected)
            self.assertEqual(out_dir(launch_command), expected)
            self.assertTrue((Path(root) / 'sweeps' / 'square').is_dir())
            self.assertFalse(run_dir.exists())

    def test_build_roots_outside_tmp_are_rejected(self):
        # Not a repository path: the repository itself may be checked out in /tmp.
        outside_tmp = '/nonexistent/waferllm_builds'
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            outside = sweep.parse_args(['dense', '--M', '64', '--N', '64',
                                        '--build-root', outside_tmp])
            with self.assertRaisesRegex(SystemExit, 'inside /tmp'):
                bench.resolve_build_root(outside, run_dir, write=False)
            # An old saved choice outside /tmp is rejected rather than used.
            run_dir.mkdir()
            (run_dir / 'config.json').write_text(json.dumps({'build_root': outside_tmp}))
            inherited = sweep.parse_args(['dense', '--M', '64', '--N', '64'])
            with self.assertRaisesRegex(SystemExit, 'inside /tmp'):
                bench.resolve_build_root(inherited, run_dir, write=False)

    def test_build_root_is_saved_inherited_and_may_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, tmp_directory() as root:
            run_dir = Path(directory) / 'run'
            first = sweep.parse_args(['dense', '--M', '64', '--N', '64', '--build-root', root])
            bench.resolve_config(first, run_dir, True, sweep.DEFAULT_CONFIG)
            saved_root = bench.resolve_build_root(first, run_dir, write=True)
            self.assertEqual(saved_root, Path(root).resolve())

            later = sweep.parse_args(['fused', '--M', '64', '--N', '64', '--ranks', '8'])
            # A saved build_root must not trip the locked-settings check.
            self.assertEqual(bench.resolve_config(later, run_dir, True, sweep.DEFAULT_CONFIG),
                             {'repetitions': 3, 'hcf_fix': False, 'profile_phases': False})
            self.assertEqual(bench.resolve_build_root(later, run_dir, write=True), saved_root)

            elsewhere = (Path(root) / 'elsewhere').resolve()
            moved = sweep.parse_args(['dense', '--M', '64', '--N', '64',
                                      '--build-root', str(elsewhere)])
            self.assertEqual(bench.resolve_build_root(moved, run_dir, write=True), elsewhere)
            saved = json.loads((run_dir / 'config.json').read_text())
            self.assertEqual(saved['build_root'], str(elsewhere))

    def test_dry_run_does_not_save_build_root(self):
        with tempfile.TemporaryDirectory() as directory, tmp_directory() as root:
            run_dir = Path(directory) / 'run'
            args = bench.parse_args(['dense', '--build-root', root])
            bench.resolve_config(args, run_dir, True)
            bench.resolve_build_root(args, run_dir, write=False)
            self.assertNotIn('build_root', json.loads((run_dir / 'config.json').read_text()))

    def test_both_runners_pass_build_root_to_every_case(self):
        with tempfile.TemporaryDirectory() as directory, tmp_directory() as root:
            # main() creates run folders and config.json before running cases.
            run_dir = Path(directory) / 'runs' / 'p4'
            sweep_dir = Path(directory) / 'sweeps' / 's'
            run_dir.mkdir(parents=True)
            sweep_dir.mkdir(parents=True)
            config = {'P': 4, 'group_num': 2, 'repetitions': 1, 'hcf_fix': False}
            (run_dir / 'config.json').write_text(json.dumps(config))
            dense_args = bench.parse_args(['dense'])
            dense_args.build_root = Path(root)
            with patch.object(bench, 'CASES', bench.CASES[:1]), \
                 patch.object(bench, 'run_case', side_effect=fake_run_case) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.run_dense(dense_args, run_dir, config)
                bench_compile, bench_launch = run.call_args.args[1]
                sweep_args = sweep.parse_args(['fused', '--M', '64', '--N', '64', '--ranks', '8',
                                               '--build-root', root])
                sweep.run_command(sweep_args, sweep_dir, config)
                sweep_launch = run.call_args.args[1][1]
            tmp_root = as_tmp_path(root)
            self.assertEqual(bench_launch[:3], ['env', '-u', 'TMPDIR'])
            self.assertEqual(out_dir(bench_compile), out_dir(bench_launch))
            self.assertTrue(out_dir(bench_launch).startswith(f'{tmp_root}/runs/p4/dense_qkvo'))
            self.assertEqual(sweep_launch[:3], ['env', '-u', 'TMPDIR'])
            self.assertEqual(out_dir(sweep_launch), f'{tmp_root}/sweeps/s/fused_p4_g2_m64_r8_n64')
            self.assertFalse((run_dir / 'build').exists())


if __name__ == '__main__':
    unittest.main()
