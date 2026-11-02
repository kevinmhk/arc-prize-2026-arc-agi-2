# ARC-AGI-2 solver experiments

## Overview

This repository explores ARC-AGI-2 through a CPU symbolic solver, per-task
classical classifiers, and an offline GPU grid-generation solver. The objective
is to build valid submissions, compare methods on fixed samples, and preserve
enough evidence to explain their limitations.

The full GPU entrypoint is [gpu_submission.ipynb](notebooks/gpu_submission.ipynb).
It uses the inference-only policy selected after the bounded GPU comparisons.
A recorded full run processed 240 placeholder tasks and 259 test outputs in
6,367.8 seconds, with 258 model outputs and one symbolic fallback. This establishes
execution and coverage; it does not establish hidden-test accuracy.

The original [symbolic notebook](notebooks/submission.ipynb) remains independently
runnable. Classical methods improved small training samples but scored zero on
the 172 public evaluation outputs. GPU test-time training improved a 22-output
public sample, but was not selected for the full integration because of the
declared runtime gate.

Start with the methods and results below. [docs/plan.md](docs/plan.md) contains
dated implementation checkpoints, corrections, and evidence references. Its older
status statements describe their respective checkpoints, not the latest state.

## Task and submission contract

Each task supplies demonstration pairs under `train` and one or more query grids
under `test`. The solver must infer a transformation from the demonstrations and
predict each query's output. Grids are rectangular lists of integer colors 0-9,
with height and width from 1 through 30. Boolean values are rejected as colors.

`submission.json` must contain every input task ID. Each task maps to a list in
the same order as its test inputs. Each list entry has exactly `attempt_1` and
`attempt_2`, and each attempt is a complete grid. For example:

```json
{
  "00000001": [{ "attempt_1": [[1]], "attempt_2": [[2]] }]
}
```

A test output scores one if either attempt matches the entire expected grid,
including shape and every cell. Otherwise it scores zero. The reported official
score is output-weighted: correct test outputs divided by all test outputs.
Fully solved tasks and task-averaged pass@2 are separate diagnostic metrics.

The supplied local data has 1,000 training tasks with 1,076 test outputs, 120 public
evaluation tasks with 172 outputs, and 240 placeholder test tasks with 259 outputs.
Kaggle replaces the placeholder challenges during hidden scoring. Production
submission inference does not load solution files. Local training and evaluation
runners may load solutions for validation and reporting, but do not pass test
answers to candidate fitting or selection.

## Repository map

| Location | Purpose |
| --- | --- |
| [notebooks/submission.ipynb](notebooks/submission.ipynb) | Self-contained standard-library symbolic runtime, validators, scoring, and submission entrypoint. |
| [notebooks/gpu1_nvarc_pilot.ipynb](notebooks/gpu1_nvarc_pilot.ipynb) | Training-only pilot comparing inference and TTT; embeds pinned vendor sources and the symbolic runtime. |
| [notebooks/gpu2_public_evaluation.ipynb](notebooks/gpu2_public_evaluation.ipynb) | Frozen public-evaluation inference-only comparison. |
| [notebooks/gpu3_ttt_evaluation.ipynb](notebooks/gpu3_ttt_evaluation.ipynb) | TTT comparison on the same 16 public tasks, with cached symbolic predictions. |
| [notebooks/gpu_submission.ipynb](notebooks/gpu_submission.ipynb) | Full dynamic test-set inference, complete fallback-first output, and incremental GPU replacement. |
| [scripts/diagnose_search.py](scripts/diagnose_search.py) | Bounded training-only search diagnostic and trusted symbolic-runtime loader. |
| [scripts/experiment_a.py](scripts/experiment_a.py) | Independent per-puzzle decision-tree and random-forest cell classifiers. |
| [scripts/experiment_a2.py](scripts/experiment_a2.py) | Whole-demonstration leave-one-out selection among symbolic and classifier sources. |
| [scripts/download_data.py](scripts/download_data.py) | Optional KaggleHub download helper; performs a network download when executed. |
| [tests/](tests/) | Eight unittest modules covering symbolic inference, diagnostics, classifiers, and all four GPU notebooks. |
| [pyproject.toml](pyproject.toml) | Python requirement and Ruff configuration. |
| [requirements-dev.txt](requirements-dev.txt) | Pinned notebook execution and quality tools. |
| [requirements-experiments.txt](requirements-experiments.txt) | Pinned NumPy/scikit-learn experiment dependencies. |
| [docs/overview.md](docs/overview.md) | Captured competition description and submission constraints. |
| [docs/data.md](docs/data.md) | Captured dataset description; the dataset itself remains local-only. |
| [docs/proposal.md](docs/proposal.md) | Historical proposal, including unimplemented learned search priors and later phases. |
| [docs/qwen-perspective.md](docs/qwen-perspective.md) | Conceptual comparison of direct grid generation and proposed program generation. |
| [docs/plan.md](docs/plan.md) | Detailed execution history and measured checkpoints. |

`data/`, `.venv/`, caches, and bulk `artifacts/` outputs remain ignored by Git.
A curated set of frozen selections, configurations, source snapshots, and compact
verification reports under `artifacts/` is explicitly tracked. These are historical
evidence, not current runtime configuration or a complete experiment backup.
Original snapshots are preserved byte-for-byte, including old schema versions;
do not rewrite them to make a new run resume. Raw predictions, cached baselines,
executed notebooks, and logs still need a separate backup before local deletion.
In particular, `artifacts/phase1c/verification.json` records hashes but does not
replace the ignored `training.json` baseline it verifies.
The proposed `arc_solver/` package layout is not the implemented layout: the
symbolic runtime lives in the notebook and is reused through trusted definitions.

## Symbolic DSL and bounded search

The symbolic solver enumerates small programs rather than training a shared
model. Its primitives cover:

- All eight D4 orientations: identity, rotations, horizontal and vertical
  reflections, transpose, and anti-diagonal reflection.
- Consistent bijective recoloring, including maps learned after structural
  transformations of demonstration inputs.
- Crop to non-background content, uniform-border removal, and shape relations
  for unchanged, transposed, scaled, cropped, or fixed-size outputs. Resizing
  uses nearest-neighbor cell replication or sampling.
- Same-color connected components with 4- or 8-connectivity, followed by
  deterministic largest/smallest object extraction.
- Horizontal, vertical, or rotational symmetry completion, and horizontal or
  vertical periodic completion with periods 1-6.

Background is the most frequent input color, with the lowest color breaking ties.
Programs contain one or two operations, capped at 1,024 programs per task.
Singletons precede compositions in deterministic enumeration order. This is a
candidate-count bound, not a production wall-clock timeout.

Only programs that reproduce every demonstration output exactly can supply
semantic predictions. Surviving programs retain enumeration order. The solver
then appends fallback candidates: the unchanged query, a consistent inferred
shape transformation, and a learned recoloring when available. It deduplicates
whole grids and chooses the first two distinct outputs. If only one remains,
it duplicates that grid to satisfy the submission schema. Partial demo fits
contribute diagnostics, not a separate weak-candidate ranking policy.

The representation is deliberately limited. Modal background and size-based
object selection can misidentify structure. Bijective maps cannot merge colors.
Periodic completion requires an intact leading template. The cap can omit later
depth-two compositions, and a valid demo fit does not guarantee generalization.

### Training-only search diagnostic

`diagnose_search.py` compares production search (1,024 programs, depth two),
expanded search (4,096, depth two), and deeper search (8,192, depth three) on
32 fixed training task IDs. Expanded generators preserve deterministic prefixes.
The production notebook is patched only in the diagnostic's in-memory namespace.

Each condition has a 20-second inference budget. Post-inference demonstration
and oracle analysis has a separate five-second budget. An analysis timeout
retains completed predictions and their actual score. Oracle coverage asks
whether a generated demo-valid program could match the known test answer;
it never selects either submitted attempt. Signal deadlines require the Linux
main thread. Resume checks freeze configuration, selected IDs, data and source
hashes, including the diagnostic implementation. A missing verified baseline
selects an explicitly recorded recomputation path.

The frozen diagnostic scored 0/36 outputs for baseline and expanded search, and
0/35 for deeper search after one inference timeout. Oracle coverage was also
zero in all three conditions. This bounded sample did not support a useful
larger-search prior, but cannot prove that a required primitive is absent.

## Classical per-task learning

### Experiment A: cell classification

Each puzzle gets a new decision tree and random forest fitted to its own
demonstrations. There is no cross-task model. Each input cell has a 126-value
feature vector: one-hot center and 3x3 neighborhood colors with an out-of-bounds
bucket; normalized coordinates, edge distances, and grid dimensions; and
background/component size, bounding-box, and relative-position features.
Features use inputs only; labels are demonstration output colors.

The tree has `max_depth=12` and `random_state=17`. The forest has 32 trees,
`max_depth=12`, `n_jobs=1`, and `random_state=17`. Both require shape-preserving
demonstrations and predict outputs at the query's input shape. Unsupported tasks
retain symbolic predictions. Combined selection uses the tree first and forest
second; if they agree, it tries a distinct symbolic prediction for attempt two.

The fixed Experiment A selection contains 64 iteration tasks and all 216 tasks
in the deterministic training holdout bucket. The holdout split is a reporting
partition, not an automatic guarantee of isolation from prior development.

### Experiment A2: whole-demo source selection

A2 holds out one complete demonstration per fold. It refits the remaining
demonstrations and predicts the held input without supplying its output. It
scores exact whole-grid matches for symbolic attempt one, symbolic attempt two,
tree, and forest. Unavailable or invalid sources receive a no-match fold.
Sources rank by match rate; ties follow that same fixed source order. The final
two distinct grids come from predictions fitted on all demonstrations, with
symbolic fill and duplication when needed. Insufficient-demo tasks use fallback.

A2 selects 128 SHA-256-ordered training IDs outside the union of the earlier
280-task classifier and 32-task diagnostic manifests. Those IDs were fresh for
these experiments, but already had symbolic baseline predictions. Manifests,
source/data/runtime fingerprints, and policies are frozen. Resume validates
saved predictions and recomputes their reporting scores. Changed state,
incompatible schemas, and unexpected saved tasks fail closed. Test solutions
never rank sources or choose predictions.

The public evaluation checkpoint returned zero for symbolic, A, and A2. Small
training-sample gains therefore did not justify classical production promotion.
Reordering candidates cannot recover an output that the candidate generators
never produced; post-hoc oracle selection would not be an admissible fix.

## GPU grid generation and test-time training

The NVARC-labelled GPU pilot adapts Sorokin's pinned Qwen3 ARC notebook. It uses
`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, a specialized checkpoint
with exactly 16 tokenizer entries: ten colors, newline, role tokens, padding,
and message boundary tokens. This path generates complete grids directly. The
program-generation hybrid discussed in the proposal is not implemented here.

The original formatter serializes demonstration grid pairs and each query with
Qwen role delimiters. Augmentation combines eight geometric orientations, random
color permutations, and demonstration-order shuffling. Inference calls
`augment(n=2, seed=2)`, producing 16 views per query. Contexts are shortened by
dropping demonstrations to fit 8,192 tokens while reserving output capacity.

The vendored decoder performs probability-bounded depth-first token search
(`inference_turbo_dfs`), with cumulative negative log probability below
`-log(0.2)`, an internal 540-second search bound per call, and the supplied arm
deadline. Generated grids are transformed back into the original coordinates
and colors. Duplicate grids share cached augmented rescoring, computed in
four-view microbatches. `score_kgmon` groups identical solutions and ranks them
by occurrence count minus mean augmented loss. It is not a learned DSL prior.

TTT updates adapters using task demonstrations only. Adapter state is restored
and checked before each task and arm, preventing adaptation from carrying into
the next puzzle. The measured policy is fixed:

| Setting | Implemented value |
| --- | --- |
| Adapter | Rank-stabilized LoRA (`use_rslora=True`), rank 256, alpha 32, dropout 0, no bias |
| Target modules | Attention q/k/v/o; MLP gate/up/down; embeddings and language-model head |
| Training | One epoch, batch size 1, gradient accumulation 1, learning rate `5e-5` |
| Optimizer/schedule | `adamw_torch`, cosine schedule, weight decay 0, warmup ratio 0.1, warmup steps 0, max gradient norm 1 |
| Precision/context | BF16, no 4-bit loading, maximum sequence length 8,192; gradient checkpointing disabled |
| Training augmentation | `n=16`, seed 1, shuffled keys: 128 geometric/color views |
| Model/trainer seed | 42, distinct from augmentation seeds |

GPU-2, GPU-3, and the full submission embed a hash-checked snapshot of GPU-1
definitions and payloads. They adapt its policy and orchestration instead of
maintaining independent worker implementations. Encoded vendor payloads are
decoded and hash-checked before use. They are trusted source, not user input.

### Offline environment and execution controls

The CPU runtime requires Python 3.11 or later; recorded local checks used 3.14.
The GPU path requires CPython 3.11 because the attached legacy dependency bundle
contains native extensions for that ABI. The weights themselves are not tied
to Python 3.11. Copy the upstream setup notebook with **Pin to original
environment**; notebook language metadata alone does not pin the image.

The captured image is
`gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`.
Attach the checkpoint version above and version 2 of the patched setup utility.
Its observed package root is
`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`.
Expected versions include PyTorch 2.8/CUDA 12.8, Unsloth 2025.9.7,
unsloth_zoo 2025.9.9, Flash Attention 2.8.2, and NumPy 2.2.6.
The notebooks use attached packages and local weights without online installs.

Each child imports Unsloth before the Transformers-based vendor loader, creates
a process-local `UNSLOTH_COMPILE_LOCATION`, and validates `TRITON_PTXAS_PATH`
(default `/usr/local/cuda/bin/ptxas`) before Triton compilation. Preflight loads
the model and validates the exact tokenizer in a child process. The Linux parent
does not probe CUDA before forking workers.

Four workers each load a model on one GPU; the four L4 memory spaces are not
pooled. The captured competition setup provides L4x4 and documents twice-normal
quota consumption. GPU-1 has 90/120-minute graceful/hard limits; GPU-2 and GPU-3
use 5,100/5,400 seconds. Each inference arm is capped at 1,200 seconds within
the shared budget; TTT training adds cost. Full real mode uses 41,400 seconds
graceful, a 42,480-second worker deadline, and a 42,600-second total watchdog,
reserving cleanup time within the captured 12-hour platform limit. Monotonic
deadlines are converted to epoch time only at the upstream decoder boundary.
Supervision escalates child process-group termination and preserves errors.

The full entrypoint validates strict input keys, rejects duplicate JSON keys and
ambiguous challenge paths, and writes a complete symbolic submission before GPU
setup. It caches that genuine fallback, validates model outputs per query, and
atomically persists replacements and diagnostics. Caught failures retain
completed outputs and fallback coverage. Inspect model/fallback counts and errors;
a valid JSON file or hosted `COMPLETE` status alone does not establish model use.

## Recorded results

These are historical measurements, not new runs. Exact counts refer to either
attempt unless stated otherwise. Samples differ between rows.

| Checkpoint | Scope | Correct outputs | Fully solved tasks | Elapsed time |
| --- | --- | --- | --- | --- |
| Frozen symbolic | Full training: 1,000 tasks | 28/1,076 | 26/1,000 | 233.38 s summed task time |
| Frozen symbolic | Public evaluation: 120 tasks | 0/172 | 0/120 | 49.79 s |
| Experiment A combined | Training holdout: 216 tasks | 7/231; symbolic 4/231 | See detailed checkpoint | 144.42 s for all 280 selected tasks |
| A2 fresh training comparison | Same 128 tasks for all methods | Symbolic 4/137; A 6/137; A2 8/137 | 4; 5; 7 of 128 | 119.79 s total; 410 LOO folds, 76.79 s LOO |
| Frozen classical public check | 120 tasks | Symbolic, A, A2 each 0/172 | Each 0/120 | 182.04 s summed task time |
| GPU-1 training pilot | 8 tasks, 9 outputs | Inference 9/9; TTT 9/9 | Both 8/8 | No added accuracy from TTT on this sample |
| GPU-2 public inference | 16 tasks, 22 outputs | 6/22; first attempt 5/22 | 5/16 | 1,167.00 s notebook body; 828.90 s pilot |
| GPU-3 public TTT | Same 16 tasks, 22 outputs | 9/22; first attempt 8/22 | 6/16 | 2,874.01 s notebook body |

GPU-2 used the model for 10 outputs and fallback for 12; reported peak device
memory was 14.22 GiB. GPU-3 used the model for 21 and fallback for one, at
14.60 GiB. All 16 GPU-3 tasks trained for one epoch without worker/task errors.
TTT added four correct outputs and lost one relative to inference. Its summed
worker training time was 2,673.18 seconds, distinct from elapsed wall time.

Whole-sample scaling of GPU-3 to 240 tasks gave about 11.98 hours. This conservative
runtime gate selected inference-only for full integration. Scaling repeats fixed
overhead and assumes similar tasks; it is not proof that full TTT cannot fit.
No full TTT timing result is recorded. Public tasks may have been exposed during
checkpoint training, so these samples do not establish unseen-task accuracy.

The integration smoke run took 487.96 seconds and produced two model outputs
plus 257 fallback outputs, covering all 240 task keys and 259 outputs. It tested
the two smallest contexts, not full model throughput or accuracy. The subsequent
full real run took 6,367.8 seconds, reported 16.7 GiB peak device memory, processed
all 240 tasks, and supplied 258 model outputs plus one fallback. All four worker
exit codes were zero and `errors` was empty. Application status was `incomplete`
because one output used fallback, not because execution failed. No accuracy
claim follows from this placeholder run.

Local evidence, when retained, includes `artifacts/phase1c/`,
`artifacts/search-diagnostic-final/`, `artifacts/experiment-a/`,
`artifacts/experiment-a2/`, `artifacts/a2-evaluation/`,
`artifacts/gpu1-hosted-complete-cpbqc344/`,
`artifacts/gpu2-kaggle-run/independent-verification.json`,
`artifacts/gpu3-kaggle-run/independent-comparison.json`,
`artifacts/gpu-submission-smoke-run/`, and `artifacts/gpu-submission-full-run/`.
The full notebook SHA-256 is
`d13c5864eba2f7ff3919d7949c3a19ba49fb0e5612f8cb094279c553ec13db56`.
The manifest's `source.notebook_sha256` identifies the embedded GPU-1 snapshot,
not this outer notebook. Compare source, input, model, dependency, and output
hashes when connecting a saved version to its run. An unexecuted saved template
and a successful smoke run do not substitute for full-run evidence.

## Install

Run from the repository root. Reuse an existing project environment; otherwise
create one. These commands describe setup and are not part of notebook inference:

```bash
test -d .venv || uv venv .venv
type python python3
./.venv/bin/python -V
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python -r requirements-experiments.txt
```

The second requirements file is needed for classifier runs and the complete
test suite, not symbolic submission inference. Put approved competition JSON
files under `data/`, or use `ARC_DATA_DIR` for the symbolic notebook. Training and
evaluation modes also need their matching solutions files. The optional download
helper requires KaggleHub and account setup outside these runtime dependencies;
it downloads data immediately and has no argument parser.

## Usage and local verification

The commands below are intentional execution examples. Use new output locations
for fresh runs and preserve existing evidence. Do not run experimental entrypoints
merely to inspect their options: `experiment_a.py` has no `--help` handler and
starts its fixed experiment. `diagnose_search.py` and `experiment_a2.py` support
`--output-dir` and normal argparse help.

| Notebook | Environment controls |
| --- | --- |
| Symbolic | `ARC_MODE=submission` (default), `training`, or `evaluation`; `ARC_SOLVER_MODE=symbolic` (default) or `identity`; `ARC_DATA_DIR`; `ARC_OUTPUT_PATH` |
| GPU-1/2/3 | `GPU1_MODE`, `GPU2_MODE`, `GPU3_MODE`: `real` or `cpu-check`; corresponding `GPU1_INPUT_ROOT`/`GPU2_INPUT_ROOT`/`GPU3_INPUT_ROOT` and output-directory variables |
| Full GPU | `GPU_SUBMISSION_MODE=real` (default), `smoke`, or `cpu-check`; `GPU_SUBMISSION_INPUT_ROOT`; `GPU_SUBMISSION_OUTPUT_DIR` (default `/kaggle/working`) |
| Shared GPU discovery | `GPU1_MODEL_DIR`, `GPU1_DEPENDENCY_ROOT`, `TRITON_PTXAS_PATH` |

Full `real` mode schedules every incoming test task, without assuming 240 IDs.
Full `smoke` mode schedules only the two smallest demonstration/query contexts
and retains fallback for the rest. Full `cpu-check` writes genuine symbolic
fallback only. CPU checks do not establish CUDA, training, inference, or accuracy.

For a deliberate symbolic run, execute all notebook cells in a fresh kernel
bound to the intended interpreter. This example writes to a new directory:

```bash
ARC_MODE=submission ./.venv/bin/python - <<'PY'
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

root = Path.cwd()
output = Path(tempfile.mkdtemp(prefix="arc-symbolic-")) / "submission.json"
notebook = nbformat.read(root / "notebooks/submission.ipynb", as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
notebook.cells.insert(0, nbformat.v4.new_code_cell(
    f"import sys; assert sys.executable == {sys.executable!r}"
))
env = dict(os.environ, ARC_OUTPUT_PATH=str(output))
env.pop("PYTHONPATH", None)
NotebookClient(notebook, km=manager, timeout=1200).execute(cwd=str(root), env=env)
for cell in notebook.cells:
    for item in cell.get("outputs", []):
        if item.get("output_type") == "stream":
            print(item.text, end="")
print(f"Retained output: {output}")
PY
```

Use `ARC_MODE=training` or `evaluation` only for an intentional scoring checkpoint.
Inspect the artifact's actual task/output counts as well as cell execution status.
For standalone diagnostics and A2, choose a fresh `--output-dir`; A2 also requires
the frozen exclusion manifests and hash-verified Phase 1C baseline. Experiment A
supports an output override through its Python API:
`from pathlib import Path; from scripts.experiment_a import run; run(Path("artifacts/experiment-a-new"))`.
It does not have a CLI output flag. Historical schema-1 A artifacts cannot resume
under the current schema-2 runner; preserve them and choose a separate directory.

Run quality gates in format, lint, typecheck, test order. Ruff supports notebooks;
ty needs extracted Python. Extract one file per notebook, including entrypoint
cells, without executing them:

```bash
check_dir=$(mktemp -d)
./.venv/bin/python - "$check_dir" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
for path in Path("notebooks").glob("*.ipynb"):
    cells = json.loads(path.read_text())["cells"]
    source = "\n\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    (target / f"{path.stem}.py").write_text(source + "\n")
PY
./.venv/bin/ruff format --check scripts tests notebooks
./.venv/bin/ruff check scripts tests notebooks
./.venv/bin/ty check scripts/diagnose_search.py scripts/experiment_a.py scripts/experiment_a2.py tests "$check_dir"
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The optional KaggleHub downloader is outside the default typecheck scope because
its external dependency is not in the local requirements. Embedded immutable
payloads are checked by source-hash tests; extracted code must retain those
payloads. The recorded integration suite passed 104 tests. That historical count
is software verification, not model accuracy or a claim that this documentation
edit reran the suite.

## Kaggle execution and submission

Use the pinned Python 3.11 environment, attached model and utility versions,
competition input, L4x4 acceleration, Internet off, and a maximum 12-hour budget.
Replace online installer cells when importing the notebook. Keep full inference
as the default; do not leave a smoke or CPU-check override in the submitted code.

Save a version and use **Save & Run All** once for the intended full run. Inspect
its logs, `/kaggle/working/submission.json`, and
`/kaggle/working/gpu-submission/manifest.json`. Match the executed code/output
hashes and actual version number. A quick-save version can report `COMPLETE`
without establishing that the intended full code ran. An interactive full run
followed by Save & Run All repeats the work and consumes additional quota.

This is a code competition: submit the executed notebook version. Kaggle reruns
its code on hidden inputs. Downloading a local JSON file does not bypass that
rerun. For an explicitly authorized submission, the recorded CLI form is:

```bash
uvx --python /usr/bin/python3.12 --from kaggle==2.2.5 --with six kaggle competitions submit \
  -f submission.json -k kevinmhk/arc-gpu-full-submission -v 2 \
  -m "Full inference-only submission" arc-prize-2026-arc-agi-2
```

Use an installed compatible Python 3.11+ interpreter in place of that machine's
`/usr/bin/python3.12` path. Verify the notebook version and hashes before using
the example; version 2 must refer to the intended full execution. Retain the
returned submission reference and inspect that submission rather than submitting
duplicates while it is pending. No live competition status or leaderboard score
is asserted by this README.

## Attribution and license

The GPU notebooks preserve `arc_loader.py`, `arc_decoder.py`, `arc_solver.py`,
and `starter.py` from [Sorokin's Qwen3 ARC notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue),
version 10, and use version 2 of the
[patched offline setup utility](https://www.kaggle.com/code/sorokin/pip-install-unsloth-flash-patch).
The selected [grid checkpoint](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1)
and captured upstream notebook declare Apache-2.0; the full notice is retained
in GPU-1 and its embedded snapshots. Source and payload hashes are recorded.

There is no repository-wide `LICENSE` file. Do not infer a blanket license from
the vendor notice. The complete upstream code, checkpoint, and training-data
license/exposure chain has not been exhaustively audited. The
[competition reference](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
and local overview identify the task; captured rules and environment details
should be rechecked before a separately authorized external run or release.
