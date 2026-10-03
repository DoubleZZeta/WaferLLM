"""Host-only checks for selecting and resuming the separate HCF implementation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench


class BackendTests(unittest.TestCase):
    def test_patch_selection_is_inherited_and_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = bench.parse_args(['dense', '--hcf-fix'])
            config = bench.resolve_config(first, run_dir, write=True)
            self.assertTrue(config['hcf_fix'])
            inherited = bench.resolve_config(bench.parse_args(['fused']), run_dir, write=True)
            self.assertEqual(inherited, config)
            with self.assertRaisesRegex(SystemExit, 'use a new run name'):
                bench.resolve_config(bench.parse_args(['dense', '--no-hcf-fix']),
                                     run_dir, write=True)

    def test_legacy_config_defaults_to_original_and_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            legacy = {'P': 64, 'group_num': 8, 'repetitions': 3}
            path = run_dir / 'config.json'
            path.write_text(json.dumps(legacy))
            config = bench.resolve_config(bench.parse_args(['dense']), run_dir, write=False)
            self.assertFalse(config['hcf_fix'])
            self.assertEqual(json.loads(path.read_text()), legacy)
            child = run_dir / 'dry'
            bench.resolve_config(bench.parse_args(['dense', '--hcf-fix']), child, write=False)
            self.assertFalse(child.exists())

    def test_both_commands_select_matching_compile_and_launch_paths(self):
        for fixed in [False, True]:
            with self.subTest(fixed=fixed), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                flag = '--hcf-fix' if fixed else '--no-hcf-fix'
                dense_args = bench.parse_args(['dense', flag, '--suppress-simfab-trace'])
                config = bench.resolve_config(dense_args, run_dir, write=True)
                fused_args = bench.parse_args(['fused', '--retained-work-ratios', '.2'])
                with patch.object(bench, 'CASES', bench.CASES[:1]), \
                     patch.object(bench, 'run_case', side_effect=lambda row, *args: row) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    bench.run_dense(dense_args, run_dir, config)
                    dense_commands = run.call_args.args[1]
                    bench.run_fused(fused_args, run_dir, config)
                    fused_commands = run.call_args.args[1]
                kernel = 'MeshGEMVPatched' if fixed else 'MeshGEMV'
                suffix = '_patched' if fixed else ''
                self.assertEqual(dense_commands[0][1], f'{kernel}/compile.py')
                self.assertEqual(dense_commands[1][1], f'Llama2Projections/launch_dense{suffix}.py')
                self.assertEqual(fused_commands[0][1], 'MeshGEMVLowerRank/compile.py')
                self.assertEqual(fused_commands[1][1], f'Llama2Projections/launch_fused{suffix}.py')
                self.assertIn('--suppress-simfab-trace', dense_commands[1])
                for compile_command, launch_command in [dense_commands, fused_commands]:
                    build = compile_command[compile_command.index('--out-dir') + 1]
                    self.assertEqual(build, launch_command[launch_command.index('--out-dir') + 1])


if __name__ == '__main__':
    unittest.main()
