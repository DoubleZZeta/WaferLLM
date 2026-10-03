"""Helpers shared by the Llama-2 projection launchers and bench.py.

The launchers run under cs_python with the repository root as the working
directory, because the SDK container mounts only that directory. Kernel-folder
modules are loaded by file path: MeshGEMV and MeshGEMVLowerRank each have their
own host_common.py, so putting both folders on sys.path would be ambiguous.
"""
import importlib.util
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RESULT_PREFIX = "BENCH_RESULT "


def load_kernel_module(kernel, name):
    """Import ``<repo>/<kernel>/<name>.py`` under a kernel-qualified name."""
    path = REPO_DIR / kernel / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{kernel}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit_result(phases=None, **metrics):
    """Print the single machine-readable line that bench.py parses."""
    values = {name: float(value) for name, value in metrics.items()}
    if phases is not None:
        values["phases"] = phases
    print(RESULT_PREFIX + json.dumps(values), flush=True)

