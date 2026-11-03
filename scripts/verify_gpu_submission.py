import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
evidence = Path(
    tempfile.mkdtemp(prefix="gpu-submission-parent-", dir=root / "artifacts")
)
challenges = json.loads((root / "data/arc-agi_test_challenges.json").read_text())
for label in ("root", "detached"):
    cwd = root if label == "root" else evidence / "detached"
    cwd.mkdir(exist_ok=True)
    output = evidence / (label + "-output")
    nb = nbformat.read(root / "notebooks/gpu_submission.ipynb", as_version=4)
    nb.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            f"import sys\nassert sys.executable == {sys.executable!r}"
        ),
    )
    manager = KernelManager(kernel_name="python3")
    assert manager.kernel_spec is not None
    manager.kernel_spec.argv[0] = sys.executable
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        GPU_SUBMISSION_MODE="cpu-check",
        GPU_SUBMISSION_INPUT_ROOT=str(root / "data"),
        GPU_SUBMISSION_OUTPUT_DIR=str(output),
    )
    NotebookClient(
        nb, km=manager, timeout=600, resources={"metadata": {"path": str(cwd)}}
    ).execute(env=env)
    nbformat.write(nb, evidence / (label + "-executed.ipynb"))
    pred = json.loads((output / "submission.json").read_text())
    manifest = json.loads((output / "gpu-submission/manifest.json").read_text())
    assert set(pred) == set(challenges) and len(pred) == 240
    assert sum(map(len, pred.values())) == 259
    for key, task in challenges.items():
        assert len(pred[key]) == len(task["test"])
        for pair in pred[key]:
            assert set(pair) == {"attempt_1", "attempt_2"}
            for grid in pair.values():
                assert isinstance(grid, list) and 1 <= len(grid) <= 30
                assert isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
                assert all(
                    isinstance(row, list)
                    and len(row) == len(grid[0])
                    and all(type(v) is int and 0 <= v <= 9 for v in row)
                    for row in grid
                )
    assert manifest["mode"] == "cpu-check" and manifest["status"] == "passed"
    assert not any(manifest["claims"].values()) and manifest["model_outputs"] == 0
    print(
        label,
        "PASS all cells/freshprojectkernel/full240keys259outputs/readback/noGPUclaims",
        flush=True,
    )
print("EVIDENCE", evidence)
