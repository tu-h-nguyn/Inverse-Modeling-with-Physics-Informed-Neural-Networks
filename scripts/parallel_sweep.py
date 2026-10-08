"""Process workers execute the original notebook's training functions, unchanged."""
import contextlib
import io
import json
from pathlib import Path
import sys
import types

_namespace = None


def train_configuration(job):
    global _namespace
    source, configuration = job
    if _namespace is None:
        cells = json.loads(Path(source).read_text())["cells"]
        module = types.ModuleType("uploaded_pinn")
        sys.modules[module.__name__] = module
        _namespace = module.__dict__
        with contextlib.redirect_stdout(io.StringIO()):
            for index in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 28):
                code = "".join(cells[index]["source"])
                if index == 10:
                    code = code.rsplit("\nplot_problem_setup(data, cfg)", 1)[0]
                if index == 28:
                    code = code.split("# Isolation check:")[0]
                exec(compile(code, f"source_cell_{index}", "exec"), _namespace)
    result = _namespace["run_experiment"](_namespace["Config"](**configuration), use_cache=False)
    # Local Config classes differ across processes; only plain data crosses the boundary.
    result = dict(result)
    result.pop("cfg")
    return configuration, result
