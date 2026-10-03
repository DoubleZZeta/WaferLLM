import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bench
import sweep


class ProfileTests(unittest.TestCase):
    def test_modes_saved_inherited_and_cannot_mix(self):
        for module in (bench, sweep):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as td:
                extra = ['--M', '64', '--N', '128'] if module is sweep else []
                args = module.parse_args(['dense', '--profile-phases'] + extra)
                config = bench.resolve_config(args, Path(td), True, module.DEFAULT_CONFIG)
                self.assertEqual(config['repetitions'], 1)
                inherited = module.parse_args(['dense'] + extra)
                self.assertEqual(bench.resolve_config(inherited, Path(td), True, module.DEFAULT_CONFIG), config)
                changed = module.parse_args(['dense', '--no-profile-phases'] + extra)
                with self.assertRaisesRegex(SystemExit, 'new run name'):
                    bench.resolve_config(changed, Path(td), True, module.DEFAULT_CONFIG)
                commands = bench.kernel_commands('MeshGEMV', 'launch_dense.py', [], config,
                                                bench.BENCH_DIR / 'runs/test/build', True)
                self.assertIn('--profile-phases', commands[-1])

    def test_phase_files_preserve_metrics_and_case(self):
        phases = {'local_compute': {'mean_cycles': 12.5, 'max_pe_cycles': 14}}
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / 'case.log'
            output = bench.RESULT_PREFIX + json.dumps({'mean_cycles': 20, 'phases': phases})
            with patch.object(bench, 'run_logged', return_value=output):
                row = bench.run_case({'status': 'error', 'rank': 16}, [], log, {'mean_cycles': 'cycles'})
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(json.loads(log.with_suffix('.phases.json').read_text())['phases'], phases)
            self.assertIn('local_compute,12.5,14', log.with_suffix('.phases.csv').read_text())
