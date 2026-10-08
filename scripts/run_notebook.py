"""Execute the uploaded notebook unchanged and retain reproducible run artifacts."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--workers", type=int, default=1, choices=range(1, 6))
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
for variable, folder in (("XDG_CACHE_HOME", "cache"), ("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
                         ("MPLCONFIGDIR", "matplotlib")):
    os.environ.setdefault(variable, str(output / folder))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

import nbformat
from nbclient import NotebookClient

nb = nbformat.read(source, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

class LoggedClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)

def finished(cell, cell_index, **kwargs):
    print(f"\nCompleted source cell {cell_index}", flush=True)
    nbformat.write(nb, output / "partial.ipynb")

nb.metadata["cloud_execution"] = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source": str(source),
    "source_cells_unchanged": True,
    "threads": 1,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
client = LoggedClient(nb, timeout=7200, kernel_name="python3", allow_errors=False,
                      resources={"metadata": {"path": str(output)}}, on_cell_executed=finished)
capture = '''# Capture artifacts after all original notebook cells have completed.
import json, platform, importlib.metadata
from pathlib import Path
torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
            "dtype": str(DTYPE), "metrics": metrics}, "baseline_checkpoint.pt")
np.savez_compressed("baseline_fields.npz", **fields)
np.savetxt("observations.csv", np.column_stack([to_np(data["obs"][k]).ravel() for k in ("x", "t", "u")]),
           delimiter=",", header="x,t,u", comments="")
artifact = {"baseline": metrics, "baseline_config": asdict(cfg),
            "baseline_training_seconds": baseline_runtime,
            "baseline_history": history,
            "device": str(DEVICE), "dtype": str(DTYPE), "threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "matplotlib", "nbclient")},
            "sweep_rows": len(all_runs), "sweep_groups": len(summary_rows),
            "unique_sweep_configs": len(unique_cfgs), "seeds": list(SEEDS), "quick_mode": QUICK_MODE}
Path("run_manifest.json").write_text(json.dumps(artifact, indent=2, allow_nan=False))
print("Saved baseline checkpoint, fields, observations and run manifest.")
'''
try:
    with client.setup_kernel():
        if args.workers > 1:
            precompute = '''# Precompute independent sweep configurations with the original training functions.
import sys, json, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, HELPER_DIR)
from parallel_sweep import train_configuration
_cells = json.loads(Path(SOURCE_FILE).read_text())["cells"]
exec("".join(_cells[30]["source"]).split("all_runs = []")[0])
# Reuse the already completed baseline, which has exactly the same configuration.
_RUN_CACHE[cfg] = {**metrics, "runtime_s": baseline_runtime,
    "alpha_crlb_rel": alpha_crlb_rel(data, cfg), "lbfgs_status": "not used", "lbfgs_evals": 0,
    "final_losses": {k: v.item() for k, v in final_parts.items()}, "history": history, "cfg": cfg}
torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
            "dtype": str(DTYPE), "metrics": metrics}, "baseline_checkpoint.pt")
Path("baseline_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
_pending = sorted((c for c in unique_cfgs if c not in _RUN_CACHE), key=lambda c: json.dumps(asdict(c), sort_keys=True))
print(f"Parallel sweep: {len(_pending)} pending configurations, WORKERS worker processes", flush=True)
with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
    _futures = [pool.submit(train_configuration, (SOURCE_FILE, asdict(c))) for c in _pending]
    for _number, _future in enumerate(as_completed(_futures), 1):
        _cfg_dict, _result = _future.result()
        _local_cfg = Config(**_cfg_dict)
        _RUN_CACHE[_local_cfg] = {**_result, "cfg": _local_cfg}
        print(f"Completed {_number}/{len(_pending)} | seed {_local_cfg.seed} | alpha {_result['alpha']:.6f} | rel.error {_result['alpha_rel_err']:.3e}", flush=True)
        _serializable = [{"configuration": asdict(c), "result": {k: v for k, v in r.items() if k != "cfg"}}
                         for c, r in _RUN_CACHE.items() if c in unique_cfgs]
        Path("sweep_cache.json").write_text(json.dumps(_serializable, indent=2, allow_nan=False))
'''
            precompute = "from pathlib import Path\n" + precompute
            precompute = precompute.replace("HELPER_DIR", repr(str(Path(__file__).parent)))
            precompute = precompute.replace("SOURCE_FILE", repr(str(source)))
            precompute = precompute.replace("WORKERS", str(args.workers))
            nb.cells.insert(30, nbformat.v4.new_code_cell(precompute, metadata={"tags": ["cloud-parallel-execution"]}))
            nb.metadata["cloud_execution"]["sweep_workers"] = args.workers
        for i, cell in enumerate(nb.cells):
            print(f"Executing source cell {i} ({cell.cell_type})", flush=True)
            client.execute_cell(cell, i)
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Cloud execution artifacts\nThe following cell saves this run; the original source cells above are unchanged. "
            "The discussion in section 18 describes the historical reference run. See the new CSVs and manifest for this execution."))
        nb.cells.append(nbformat.v4.new_code_cell(capture))
        client.execute_cell(nb.cells[-1], len(nb.cells) - 1)
    nbformat.write(nb, output / "inverse_pinn_heat_1d.executed.ipynb")
    figures = output / "figures"
    figures.mkdir()
    for i, cell in enumerate(nb.cells):
        for j, item in enumerate(cell.get("outputs", [])):
            png = item.get("data", {}).get("image/png")
            if png:
                (figures / f"cell_{i:02d}_{j:02d}.png").write_bytes(base64.b64decode(png))
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nb)
    (output / "report.html").write_text(body)
    (output / "partial.ipynb").unlink(missing_ok=True)
    print(f"SUCCESS: notebook and artifacts saved to {output}", flush=True)
except BaseException:
    nbformat.write(nb, output / "partial.ipynb")
    raise
