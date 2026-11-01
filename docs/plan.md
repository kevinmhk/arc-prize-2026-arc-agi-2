# ARC-AGI-2 implementation plan

Status: Phase 1C is frozen and verified locally; the updated-solver Kaggle run
and user PR review/merge remain pending. Final frozen metrics supersede earlier
pre-fix measurements. User-supplied Phase 1B evidence is
recorded below and is not independently live-verified in this workspace.
Scope: all phases in [the proposal](proposal.md). The proposal remains unchanged.

## Working order

Build the smallest local notebook → run it on local data → upload that notebook
to Kaggle → verify an offline run → improve the symbolic solver → experiment
with GPU, ML, and LLM components separately → promote only measured improvements.

Test Kaggle packaging early. Do not wait for the full symbolic solver. Prefer
uploading the versioned `.ipynb` over copy/paste. If notebook cells call local
Python modules, attach those exact modules too; uploading the notebook alone
will not transfer the repository. Keep one implementation, not separate local
and Kaggle copies.

Check boxes only after the stated check passes. Record the commit, configuration,
input split, exact score, runtime, and artifact or notebook version at milestones.
Do not create implementation files for later phases before starting those phases.

## 0. Shared setup and evaluation rules

- [x] Inspect the existing environment, download script, and supplied data.
  Check: identify dependencies and data paths without exposing credentials.
  Completed for Phase 1A: project `.venv` uses Python 3.14.8; local data has
  120 evaluation tasks/172 outputs and 240 placeholder tasks/259 outputs.
- [ ] Verify current competition rules, account entry status, notebook limits,
  accelerator availability, quota, and deadlines in Kaggle. Ask the user to
  accept rules or register if needed; do not do so without authorization.
  Check: distinguish verified account settings from proposal estimates.
- [x] Reserve a fixed development/validation split within the training tasks.
  Keep related augmentations and synthetic variants in their source split.
  Check: evaluation solutions are not used for training or checkpoint selection.
  Phase 1C uses SHA-256 task IDs modulo 5: bucket 0 is held out and the other
  buckets are iteration data. Solutions are excluded from inference; the loader
  reads them before inference only for validation and later scoring diagnostics.
- [x] Define one submission validator and one scoring implementation for every
  phase. Check task IDs, test-input order/count, both attempt keys, rectangular
  nonempty grids, dimensions, integer color range, and exact grid equality.
  Check: small hand-constructed cases include multiple test inputs, either
  attempt correct, malformed output, and neither attempt correct. Completed in
  `notebooks/submission.ipynb` and `tests/test_submission.py`.
- [x] Confirm the official aggregation formula before implementing scoring.
  Report output accuracy and task solve rate separately; if official scoring
  weights tasks rather than outputs, also report that official metric explicitly.
  Check: a multi-output example exposes any difference between these metrics.
  The parent-verified live overview confirms output-weighted scoring:
  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/evaluation

## 1. Phase 1 — local symbolic baseline and Kaggle notebook

### 1A. Smallest runnable notebook

- [x] Create `notebooks/submission.ipynb` with explanation, path configuration,
  a minimal identity predictor, validation, and JSON writing. Use Python stdlib
  for this first slice. Put reusable logic in a small module only as needed.
  Check: a fresh kernel can execute every cell in order without manual state.
  Completed with a self-contained notebook; runtime code has no third-party
  imports.
- [x] Separate evaluation mode from submission mode. Evaluation may read known
  solutions for scoring only. Submission mode reads the supplied test challenge
  file, discovers task IDs and test counts dynamically, and never loads solutions.
  Check: no hard-coded placeholder IDs or output counts. Submission mode defaults
  to `/kaggle/working/submission.json` on Kaggle.
- [x] Execute locally against evaluation challenges and solutions. Reproduce
  the proposal's identity baseline or investigate any discrepancy.
  Check: report the measured numerator, denominator, and runtime, not a copied
  expected score. Invalid submissions fail validation before scoring. The fresh
  evaluation run measured 0/172 output accuracy and 0/120 task solve rate.
- [x] Generate a local placeholder `submission.json`, then read it back and
  validate it against the test challenges. Run again from a fresh kernel.
  Check: every required output has two valid attempts; repeated runs agree.
  Two fresh submission runs produced 240 tasks/259 outputs each and identical
  JSON bytes.

### 1B. Early Kaggle smoke test

- [x] Upload the same notebook to a competition-attached Kaggle notebook.
  Attach competition data and any imported source files. Keep the notebook
  private unless the user authorizes publication.
  Check: record the notebook URL/version and confirm actual mounted input paths.
-  User-supplied evidence, not independently live-verified: saved Kaggle
  version `https://www.kaggle.com/code/kevinmhk/notebook33c1344ea5?scriptVersionId=361056044`,
  screenshot reported success in 21.6 seconds on CPU with no accelerator;
  interactive output reported 240 tasks/259 outputs, no solutions loaded, and
  0.525350 seconds. The user stated that internet was off.
- [x] Select CPU and disable internet. Read input from the mounted competition
  directory and write `/kaggle/working/submission.json`.
  Check: no dependency download, local absolute path, or repository-only import.
- [x] Save and execute a full notebook version, not only interactive cells.
  Check: the offline run completes; its saved output contains a parseable,
  validated `submission.json`. This is a packaging check, not a hidden-test score.
  The saved version evidence above is user-supplied; the updated solver has not
  yet been run on Kaggle.
- [ ] Ask the user before a competition submission. If authorized, submit the
  saved notebook version and inspect Kaggle's final status and score.
  Check: record the returned submission/version identifier and final result;
  do not treat a successful upload as a successful submission.

### 1C. Improve the solver incrementally

- [x] Add geometry/color rules and shape inference from proposal §4.2 A–B.
  Check: a runnable regression check for each nontrivial behavior; candidates
  must reproduce all demonstration outputs before being treated as exact fits.
- [x] Add object segmentation and selection (§4.2 C), starting with one useful
  rule family. Check: controlled object examples and training-set results agree.
- [x] Add symmetry/periodic completion (§4.2 D) only where training failures
  justify it. Check: do not overwrite conflicting observed cells silently.
- [x] Add bounded composition (§4.2 E), stable ranking, output deduplication,
  and fallback predictions. Check: candidate/time limits, multiple test inputs,
  ambiguous demo fits, and two-attempt behavior are covered.
- [x] Iterate on training tasks; run full training checks at milestones.
  Check: record coverage, exact score, runtime, and representative failures.
- [x] Freeze a candidate before the planned held-out evaluation. Record exact
  metrics and failure analysis. Check: do not tune repeatedly against that set.
- [ ] Repeat the offline Kaggle run with the selected solver and archive baseline
  predictions/configuration. Check: local/Kaggle behavior agrees on identical
  accessible inputs; the placeholder is not treated as hidden-test evidence.
  Updated solver Kaggle run is pending.

**Phase 1 completion:** a reproducible offline notebook, valid submission format,
measured local results, and a frozen CPU fallback. A nonzero score is a target,
not a guarantee or a prerequisite for learning how GPUs work.

## 2. GPU learning sandbox — independent of solver accuracy

This can start after the early packaging check. It does not require a successful
learned prior and does not change the submission notebook.

- [ ] Create `notebooks/gpu-lab.ipynb` when this work starts. Attach it to the
  competition if using restricted L4 hardware; keep internet disabled for L4.
  Check: use only competition-related experiments and comply with current rules.
- [ ] Inspect PyTorch/CUDA versions, GPU count, each device's memory, and supported
  precision. Check: execute a tensor operation on an actual GPU and inspect its
  result; detecting a GPU alone is not execution proof.
- [ ] Run one small ARC-related training exercise on the training split, save a
  checkpoint, reload it, and reproduce inference. Check: finite loss, correct
  tensor shapes, measured runtime, and no evaluation-label leakage.
- [ ] Use an available T4/P100 for small experiments if suitable. Use L4×4 when
  the experiment warrants it and quota permits. Stop idle sessions.
  Check: use the account's displayed quota, not an assumed weekly allowance.
- [ ] Inspect multi-GPU behavior before choosing a model. Four GPUs do not
  automatically act as one contiguous memory pool; select explicit sharding or
  replication only if needed. Check: measure per-device peak memory.

**Sandbox completion:** a saved offline GPU run and a reloadable checkpoint.
A negative accuracy result can still complete the learning objective.

## 3. Phase 2 — ML-guided symbolic search

- [ ] Use training-only diagnostics to test the proposal's search-prior premise:
  a deeper/larger search can discover useful programs missed by the baseline cap.
  Check: otherwise, record the limitation and do not claim ordering fixes a
  missing primitive. Keep the GPU sandbox available regardless of this gate.
- [ ] Collect demonstration-fitting programs on training tasks. Record actual
  label yield and ambiguity; do not assume hundreds of labels exist. If needed,
  generate bounded synthetic DSL examples and document their distribution gap.
  Check: source-task and program-family leakage are controlled across splits.
- [ ] Train the smallest useful operator/rule-family predictor. Compare it with
  a simple frequency-based ordering before using a CNN or transformer.
  Check: validation performance and search savings beat the simple comparator,
  or document a negative result.
- [ ] Integrate optional model-guided ordering without changing the candidate
  validator. Retain enumeration as an explicit fallback.
  Check: identical budgets and inputs for comparisons. A bad prior can reduce
  accuracy under a capped search; passing demos does not guarantee a test answer.
- [ ] Measure nodes, coverage, exact accuracy, inference overhead, and total time.
  Check: choose a checkpoint on training validation, then run the planned
  held-out checkpoint once. Restore the frozen CPU path when disabled.

**Phase 2 completion:** measured keep/revert decision plus model/configuration
artifacts if trained. Submit only if the promotion checks below pass.

## 4. Phase 3 — per-task test-time training

- [ ] After the proposal's Phase 2 gate is met, choose a small pretrained or
  cross-task-trained grid model. Check: a documented starting checkpoint and
  training data; do not assume training from scratch on a few demos will work.
- [ ] Adapt independently on each task's demonstration pairs, with valid grid
  rotations/reflections and consistent color augmentation. Reset model state
  between tasks. Check: test outputs never enter adaptation or stopping logic.
- [ ] Set adaptation and inference budgets from measured runs, including load
  time and output writing. Check: project full-run cost with safety margin and
  a fallback for every task, rather than assuming a per-task timing.
- [ ] Decode valid grids and measure demonstration reconstruction. Treat that
  fit as a diagnostic, not a proof that a neural test prediction is correct.
  Check: actual generalization is measured on training validation outputs first.
- [ ] Compare symbolic-only, TTT-only, and a fixed hybrid selection policy using
  the same two-attempt budget. Check: freeze selection before held-out evaluation.

**Phase 3 completion:** measured adaptation cost, exact scores, and a promotion
or stop decision. Do not block a valid competition submission on this phase;
continue after the deadline as a learning experiment if needed.

## 5. Phase 4 — offline LLM program proposals

- [ ] Choose one initial model with verified license and usable offline weights.
  Check: actual per-GPU memory, quantization support, loading time, context length,
  and package compatibility. Aggregate 96 GB is not single-device capacity.
- [ ] Attach approved weights and offline dependencies as Kaggle inputs. Run a
  minimal load-and-generate experiment on competition tasks.
  Check: no network access, missing downloads, or unlicensed assets.
- [ ] Define a small DSL text grammar and an allowlisted interpreter. Reject
  unknown operations, invalid arguments, and oversized programs; cap execution
  resources. Do not run generated text with Python `eval` or `exec`.
  Check: malformed/unsafe proposals fail safely without breaking fallback output.
- [ ] Prompt with demos only; generate bounded candidate programs and test each
  against all demos before applying it to test inputs.
  Check: track parse rate, demo-fit rate, test accuracy, and generation/runtime
  cost separately. Demo agreement is not proof of generalization.
- [ ] Compare enumeration and LLM proposals with fixed compute budgets and a
  frozen two-attempt selection policy. Check: proposal count is independent of
  the submission's two output attempts; do not assume one generation per attempt.
- [ ] Pin model/version/settings and test repeatability in the actual environment.
  Check: fixed seeds alone do not guarantee deterministic GPU execution.

**Phase 4 completion:** an offline measured comparison and keep/drop decision,
not a promise of novelty, universality, or a higher score. Follow the proposal's
ordering before competition promotion; post-deadline study has no deadline gate.

## 6. Promotion, final delivery, and mobile review

- [ ] For each proposed integration, open a focused PR with exact validation
  results, costs, limitations, and a notebook link/version when applicable.
  Check: user can inspect changes and comment in GitHub iOS; address comments
  with follow-up commits and leave merge control with the user.
- [ ] Enforce proposal §17 before promoting a model into the submission: held-out
  improvement, offline runtime margin, repeatability, and an intact CPU fallback.
  Check: speed-only or educational successes may stay in the lab without promotion.
- [ ] Run format → lint → typecheck → tests for implemented Python changes, then
  execute the notebook from a fresh kernel. Use minimal local development tools.
  For documentation-only changes, check links/structure and `git diff --check`;
  do not add an ML stack or code-test tooling solely to validate Markdown.
- [ ] Freeze the selected configuration; repeat an offline end-to-end run and
  validate saved output. Ask the user before submission or publication.
  Check: inspect Kaggle's final run/submission status, not just local logs.
- [ ] Write the concise solution explanation with measured successes, failures,
  and learning outcomes. Verify current publication rules if pursuing a prize.
  Check: do not claim that planned phases or unsuccessful experiments were deployed.
- [ ] After the user merges, fetch/prune, update local `main`, and remove only
  fully merged local branches. Check: clean working tree and matching remote head.

## Immediate next task

Review the uncommitted Phase 1C changes, then run the updated solver notebook on
Kaggle with the user. Do not treat the placeholder or the local 0.000000
evaluation score as a hidden-test result. Do not build ML, TTT, or LLM
scaffolding yet.

## Phase 1A checkpoint (2026-10-18)

Completed artifacts:

- `notebooks/submission.ipynb`: self-contained stdlib identity baseline,
  strict validator, evaluation/submission modes, scoring, and JSON readback.
- `tests/test_submission.py`: six unittest cases covering local path resolution,
  identity output ordering and copy isolation, multiple outputs,
  either-attempt matching, invalid output rejection, scoring distinctions, and
  no solution loading in submission mode.
- `requirements-dev.txt` and `pyproject.toml`: reproducible local notebook
  execution and quality-gate configuration. Runtime dependencies remain empty.
- Ignored local artifacts: `artifacts/evaluation-submission.json` and two
  deterministic placeholder submissions.

Quality gates and exact reproducible local commands:

- Format: `ruff format --check tests notebooks/submission.ipynb`.
- Lint: `ruff check tests notebooks/submission.ipynb`.
- Test typecheck: `ty check tests`.
- Trusted runtime typecheck:
  `runtime_scratch=$(mktemp -d "${TMPDIR:?}/arc-agi-runtime.XXXXXX"); jq -r '[.cells[] | select(.cell_type == "code")][0].source[]' notebooks/submission.ipynb > "$runtime_scratch/runtime.py"; ty check "$runtime_scratch/runtime.py"`.
- Tests: `./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`.
- Fresh evaluation from the repository root:
  `ARC_MODE=evaluation ./.venv/bin/python -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("notebooks/submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh root evaluation: passed")'`.
- Fresh evaluation from `notebooks/`:
  `repo_root=$(pwd); (cd notebooks && ARC_MODE=evaluation "$repo_root/.venv/bin/python" -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh notebooks evaluation: passed")')`.
- Fresh submission from the repository root:
  `ARC_MODE=submission ./.venv/bin/python -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("notebooks/submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh root submission: passed")'`.
- Fresh submission from `notebooks/`:
  `repo_root=$(pwd); (cd notebooks && ARC_MODE=submission "$repo_root/.venv/bin/python" -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh notebooks submission: passed")')`.
- Deterministic root submission rerun:
  `first=$(mktemp "${TMPDIR:?}/arc-agi-submission.XXXXXX"); cp artifacts/submission.json "$first"; ARC_MODE=submission ./.venv/bin/python -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("notebooks/submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh root submission rerun: passed")'; cmp "$first" artifacts/submission.json`.

Recorded results:

- Both local path regression cases resolve `data/` and `artifacts/` at the
  repository root; Kaggle path precedence remains unchanged.
- Format reported `2 files already formatted`; lint, both typechecks, and all
  six unittest cases passed.
- Fresh evaluation from root and `notebooks/` reported 120 tasks/172 outputs,
  0/172 accuracy, 0/120 solved tasks, and 0.000000 official score; elapsed
  times were 0.247999 and 0.214600 seconds.
- Fresh submission from root and `notebooks/` reported 240 tasks/259 outputs,
  validated readback, and `solutions loaded: no`; elapsed times were 0.191490
  and 0.183649 seconds. The deterministic root rerun reported identical bytes
  and elapsed 0.176874 seconds.

The proposal remains unchanged. Kaggle account entry, quota, registration,
credentials, and independent live verification of the user-supplied 1B
evidence have not been performed in this workspace. Human checkpoint: review
the uncommitted Phase 1C changes, run the updated solver on Kaggle, and merge
only after that check.

## Phase 1C frozen local checkpoint (2026-10-22)

Implemented: eight D4 transforms, bijective recoloring including mappings learned
from transformed demos, shape inference/resizing, crop/border removal,
4/8-connected object selection, symmetry/periodic completion, exact-all-demo
filtering, deduplication and deterministic fallback. The notebook remains
self-contained with stdlib runtime. `ARC_SOLVER_MODE=identity` preserves baseline.

Search bounds are **1024 candidates, depth two**, not a wall-clock timeout.
Known ceilings: modal-color background; largest/smallest objects; bijective
color maps; periodic completion needs an intact leading template (periods 1–6);
only structural-prefix recolor learning; later depth-two combinations may be
truncated. These limitations are not evidence of general reasoning.

### Review and evidence

Independent re-review passed after correcting singleton program encoding,
premature completion filtering, transformed recolor inference, missing direct
anti-diagonal reflection, and failure diagnostics. All **22 tests**, format,
lint, and typechecks of tests and full extracted notebook runtime passed.
The proposal remains unchanged.

Earlier 26/1076 training and 0/172 evaluation results came from defective
pre-freeze implementations and are superseded. Corrections followed controlled
regressions, not evaluation-answer tuning. After the final freeze, one fresh
training and one fresh evaluation run produced:

- Full training: **28/1076 outputs**, **26/1000 solved tasks**.
- Iteration bucket: **24/845 outputs**, **22/784 solved tasks**.
- Training held-out bucket: **4/231 outputs**, **4/216 solved tasks**.
- Candidate coverage: **28/1000** tasks with an exact demo fit; mean best demo
  coverage `0.045771428571428574`.
- Training task runtime: `233.381519` seconds total, `3.518807` seconds maximum;
  fresh-kernel wall time `235.023325` seconds.
- Frozen evaluation: **0/172 outputs**, **0/120 solved tasks**, elapsed
  `49.789325` seconds. No evaluation accuracy gain over identity.
- Symbolic submissions: **240 tasks / 259 outputs**, validated JSON readback,
  no solution loading. Root/notebooks-dir elapsed times: `58.534660` and
  `58.351615` seconds; repeat `58.365901` seconds. All three files were
  byte-identical. Identity output matched a separate identity prediction reference.
- Each fresh kernel asserted the project `.venv/bin/python` executable.
- Representative training failures: `00576224`, `007bbfb7`, `009d5c81`,
  `00d62c1b`, `00dbd492` (content/placement). All unsolved test outputs are
  considered, not only tasks without an exact demo fit.

Solutions are loaded/validated before inference in training/evaluation modes,
but excluded from candidate inference and used only for scoring. Submission
loads no solutions. The deterministic hash split partitions reporting; it does
not enforce development isolation by itself.

Ignored evidence: `artifacts/phase1c/verification.json`, executed `.ipynb` files,
and prediction JSON files. Frozen notebook SHA-256:
`6866cf2b7c69ce38a9de2dea1bcb225cd5a39d20c8342c26f08e5c494c766028`.
The outer verification command timed out after recording training, evaluation,
and both submission-directory runs. Resume reused those completed artifacts
and finished the repeat/identity checks without rerunning training/evaluation.

### Search diagnostic blocker-fix checkpoint (2026-10-23)

The independent-review blockers were fixed only in
`scripts/diagnose_search.py` and `tests/test_search_diagnostic.py`.
`notebooks/submission.ipynb` and `proposal.md` were not changed. Inference now
has the frozen 20.0 second task limit; post-inference demo/oracle analysis has
an independent 5.0 second bound and explicit status. Validated predictions and
the actual score are retained when later analysis times out. Resume state
includes the diagnostic implementation SHA-256:
`c6535a8ff52f2a4eea25ea0869fb4232a9fe027feaa449b7cf0d69d73e34d797`.
Missing `verification.json` now selects the documented recomputed-baseline
path instead of failing during metadata access.

Quality gates and exact counts:

- `ruff format --check scripts/diagnose_search.py tests/test_search_diagnostic.py`:
  2 files already formatted.
- `ruff check scripts/diagnose_search.py tests/test_search_diagnostic.py`:
  passed.
- `ty check scripts/diagnose_search.py tests/test_search_diagnostic.py`:
  passed.
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 28 tests passed.

The fresh command was
`./.venv/bin/python scripts/diagnose_search.py --output-dir artifacts/search-diagnostic-final`.
It ran 32 frozen task IDs x 3 conditions = 96 condition runs. The final
results contain 95 inference-complete conditions and one inference timeout
(`deeper_depth3`, task `f9d67f8b`, at 20.0 seconds). Analysis statuses are
95 `completed`, 0 `timeout`, 0 `error`, and 1 `not_started` because inference
did not complete. Completed-only actual scoring was:

- `baseline`: 0/36 outputs, 0/32 solved tasks.
- `expanded_depth2`: 0/36 outputs, 0/32 solved tasks.
- `deeper_depth3`: 0/35 outputs, 0/31 solved tasks.

The actual score compares retained attempts with the known solution. The
separate oracle field asks whether any demo-valid candidate could have produced
the known test output after inference; it does not change attempts or actual
scoring. Oracle candidate output coverage was 0 for all three conditions and
for the three deterministic structural failure summaries: `8b28cd80`,
`cbded52d`, and `705a3229`. These summaries report shapes, color counts, and
bounded predictions only; they do not infer task rules. The bounded sample
therefore provides no support for a useful larger-search prior, but cannot
prove that the required primitive is absent. Judge the search-prior premise
conservatively.

Artifacts: `artifacts/search-diagnostic-final/config.json`,
`selection.json`, `results.json`, `summary.json`, `tables.json`, and
`failure-inspection.json`. The final selection IDs were byte-for-byte equal to
the previous manifest; no previous results were reused.

### Reproduce a fresh run

Use existing quality-gate commands from Phase 1A; there are now 28 tests,
including 6 search-diagnostic regressions.
From the repository root, set `ARC_MODE` to `training`, `evaluation`, or
`submission`. Evaluation is a planned checkpoint, not an iteration loop:

```bash
ARC_MODE=training ./.venv/bin/python - <<'PY'
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path.cwd()
notebook = nbformat.read(root / "notebooks/submission.ipynb", as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
notebook.cells.insert(0, nbformat.v4.new_code_cell(
    f"import sys; assert sys.executable == {sys.executable!r}; print(sys.executable)"
))
mode = os.environ["ARC_MODE"]
env = dict(os.environ, ARC_SOLVER_MODE="symbolic",
           ARC_OUTPUT_PATH=str(root / "artifacts" / f"{mode}.json"))
NotebookClient(notebook, km=manager, timeout=1200).execute(cwd=str(root), env=env)
for cell in notebook.cells:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            print(output.text, end="")
PY
```

### Human checkpoint

Review the implementation PR, then upload its notebook to Kaggle and repeat the
saved CPU/offline smoke test. The earlier saved Kaggle version proves Phase 1A
packaging only, not this updated solver. No competition submission, publication,
or merge has been performed for Phase 1C.

## Experiment A checkpoint (2026-10-26)

- [x] Implement and run the frozen CPU cell-classifier experiment on branch
  `experiment/cell-classifiers`. `scripts/experiment_a.py` uses the trusted
  notebook runtime through `scripts/diagnose_search.py` for validation, scoring,
  background, components, and symbolic fallback. The production notebook and
  proposal were unchanged.
- [x] Add focused tests and pinned local dependencies. Format, lint, typecheck,
  and all **38 unittest tests** passed after the reporting corrections (35 passed
  at the original measurement freeze). Dependencies were installed only in
  `.venv` from `requirements-experiments.txt`.
- [x] Freeze and run exactly **280 tasks**: deterministic 64-task iteration
  sample plus all **216 heldout** tasks, each ordered by SHA-256 task ID. The
  selection hash is `b4c21035624615f8eb7e454f8b7c47ab1c14d892880b61403a310a5b4c38b2da`;
  iteration and heldout ID hashes are recorded in `selection.json`.

Exact commands:

```bash
uv pip install --python .venv/bin/python -r requirements-experiments.txt
./.venv/bin/ruff format scripts/experiment_a.py tests/test_experiment_a.py
./.venv/bin/ruff check scripts/experiment_a.py tests/test_experiment_a.py
./.venv/bin/ty check scripts/experiment_a.py tests/test_experiment_a.py
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
TMPDIR="${TMPDIR:?}" ./.venv/bin/python -c 'from pathlib import Path; from scripts.experiment_a import run; run(Path("artifacts/experiment-a-reporting-v2"))'
```

The last command starts a new schema-2 run in a separate directory, leaving
`artifacts/experiment-a/` and its measured schema-1 artifacts intact. Use it only
for an intentional rerun; no additional fitting was needed for the reporting
corrections below.

The run took **144.420331 seconds** for the task loop and completed **280/280**
tasks: **190 supported**, **90 unsupported**, and **302 outputs**. Results below
are `correct_outputs/total_outputs`; the official score is output-weighted.

| Scope | Fallback | Tree | Forest | Combined |
| --- | ---: | ---: | ---: | ---: |
| Heldout, complete: 216 tasks / 231 outputs | 4/231 | 6/231 | 5/231 | **7/231** |
| Heldout, supported: 150 tasks / 158 outputs | 2/158 | 4/158 | 3/158 | **5/158** |
| Iteration, complete: 64 tasks / 71 outputs | 2/71 | 0/71 | 1/71 | 1/71 |
| Iteration, supported: 40 tasks / 44 outputs | 2/44 | 0/44 | 1/44 | 1/44 |
| All selected, complete: 280 tasks / 302 outputs | 6/302 | 6/302 | 6/302 | **8/302** |
| All selected, supported: 190 tasks / 202 outputs | 4/202 | 4/202 | 4/202 | **6/202** |

Heldout paired output deltas were tree→forest **+1/-2** added/lost, tree→combined
**+1/-0**, and forest→combined **+2/-0**. The verified Phase 1C symbolic
artifact scored **4/231** on the heldout complete set, so the combined result
has a net gain of three outputs in this fixed offline comparison. This is evidence of a
small training-heldout gain, not evidence for hidden evaluation improvement or
production promotion.

Artifacts are ignored under `artifacts/experiment-a/`: `config.json`,
`selection.json`, `results.json` (all predictions and per-task scores), and
`summary.json`. The run recorded source hashes for the notebook, diagnostic
script, experiment script, requirements, training challenges, and solutions.
The notebook hash was verified; the Phase 1C baseline was accepted only as the
hash-checked `artifacts/phase1c/training.json` artifact. No Kaggle, GPU, model
network call, commit, push, publication, or notebook deployment was performed.

The measurements above were made with the preserved ignored snapshot
`artifacts/experiment-a/frozen_script.py`, SHA-256
`09a09366a8bfb4d50b6180806c7036da984e27812503705b71c70e60a88f3a6a`.
The final-review revision is reporting/provenance-only: it did not retrain or
reevaluate the 280 tasks, overwrite the original config, or overwrite any
predictions. It writes a new artifact schema and refuses the old measured
schema when resuming.

The parent independently rescored the exact saved predictions. Against the
Phase 1C baseline, heldout results had five new outputs and two lost outputs
(net **+3**); iteration results had zero new outputs and one lost output (net
**-1**). This tiny gain is not monotonic improvement: the paired comparison
includes losses and is not evidence of general improvement or promotion.

Limitations: the models are independent per puzzle and cellwise; they do not
learn cross-task rules, cannot represent shape-changing demonstrations, and
unsupported tasks retain the two frozen symbolic attempts. The fallback and
model outputs were validated by the trusted runtime. Solutions were loaded and
validated before fitting, excluded from feature construction, fitting,
inference, and variant selection, then used only for scoring after prediction.
The result does not justify changing the production notebook yet; review the
ignored artifacts and promote only after independent review.

## Experiment A2 checkpoint

- [x] Freeze one policy before running: score symbolic attempt 1, symbolic
  attempt 2, tree, and forest by exact whole-demo leave-one-out matches. Rank
  descending by match rate; ties prefer symbolic 1, symbolic 2, tree, forest.
  Select two distinct full-demo predictions. Unsupported classifiers and
  insufficient-demo tasks retain symbolic fallback. No test answers select outputs.
- [x] Validate both exclusion manifests (280 A IDs, 32 diagnostic IDs), each
  with unique valid training IDs. Their overlap is legitimate: use the union,
  not a cross-manifest uniqueness constraint. Select 128 SHA-256-ordered tasks
  absent from both previous experiments.
- [x] Resolve pre-run review findings: resume validates predictions and
  recomputes scores; unknown saved tasks and malformed manifests fail closed.
  All 51 tests and format/lint/typecheck gates pass.
- [x] Parent executed the frozen run once, then independently rescored all
  384 task/variant prediction sets without refitting or tuning. The original
  A implementation, production notebook, and proposal remain unchanged.

Command (resumes only compatible frozen artifacts):

```bash
TMPDIR="${TMPDIR:?}" ./.venv/bin/python scripts/experiment_a2.py \
  --output-dir artifacts/experiment-a2
```

Results on the same 128 tasks / 137 test outputs:

- Symbolic baseline: **4/137 outputs**, **4/128 fully solved tasks**.
- Experiment A: **6/137 outputs**, **5/128 fully solved tasks**.
- A2: **8/137 outputs**, **7/128 fully solved tasks**.
- A2 versus baseline: **4 added outputs, 0 lost**; net +4 outputs/+3 tasks.
- A2 versus A: **2 added outputs, 0 lost**; it recovers both baseline successes
  lost by A on this sample. This is not a guarantee of monotonic improvement.
- Supported subset: 98 tasks / 104 outputs. Baseline **3/104**, A **5/104**,
  A2 **7/104**; fully solved tasks 3, 4, and 6 respectively.
- 410 whole-demo folds; leave-one-out inference `76.790169` seconds.
  Full task-loop wall time `119.788484` seconds.

Ignored evidence: `artifacts/experiment-a2/{config,selection,results,summary}.json`.
Source/data/runtime fingerprints and exact IDs are frozen in those artifacts.
Solutions are loaded for validation/reporting, but held demonstration outputs
are absent from each fold's fitting task, and task test answers never enter
candidate fitting, ranking, or attempt selection.

These tasks were unused by the previous classifier experiments, not completely
unseen: symbolic baseline predictions already existed for all training tasks.
The 120-task evaluation set and Kaggle hidden tasks were not used. Small local
sample gains warrant further validation, not a competition accuracy claim.
No Kaggle notebook integration, upload, or submission is part of A2.

## GPU-1 NVARC pilot checkpoint

- [x] Add the complete self-contained `notebooks/gpu1_nvarc_pilot.ipynb`.
  It embeds the pinned upstream formatter, augmentation, DFS decoder, and
  `score_kgmon` ranking, plus the production validator/scorer/symbolic fallback.
  It selects one smoke task and eight training-only pilot tasks by fixed SHA256
  order, preserves multi-output order, resets adapters between both arms and
  tasks, and records per-task predictions, timing, memory, truncation, invalid,
  fallback, and coverage diagnostics.
- [x] Add offline input/dependency/model/tokenizer preflight and a parent
  supervisor with 90-minute graceful and 120-minute hard total deadlines.
  The real path refuses CPU fallback, network/package downloads, missing or
  ambiguous attachments, and existing output directories. Training solutions
  are loaded only after all smoke/pilot workers terminate.
- [x] Add focused notebook RED-GREEN unittest checks for source drift, vendor
  compilation, deterministic selection, missing-input failure, adapter reset,
  fixed arm order, multi-output preservation, and a synthetic child timeout
  that preserves a completed partial result.
- [x] CPU-checked locally from a fresh kernel. Command:

  ```bash
  GPU1_MODE=cpu-check \
    GPU1_OUTPUT_DIR=artifacts/gpu1-local-checks \
    ./.venv/bin/python -c 'from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("notebooks/gpu1_nvarc_pilot.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd())'
  ```

  The executed notebook is preserved as `artifacts/gpu1-local-checks-executed.ipynb`.
  Status: implemented and CPU-checked; GPU execution, CUDA/model loading,
  training, inference, accuracy, and peak GPU memory remain unverified.
- [ ] Run the real path only after parent review with the exact model and
  versioned patched dependency-tree Inputs, GPU-enabled Internet-off Kaggle
  configuration, and no submission action. Preserve the manifest and partial
  metrics for review; do not modify `notebooks/submission.ipynb`.

### GPU-1 parent-review blocker fix checkpoint (2026-10-28)

- [x] Fix the concrete review blockers in the uploadable notebook. The runtime
  now converts monotonic supervisor deadlines to epoch deadlines only at the
  upstream decoder boundary, applies the independent per-arm inference cap,
  validates the exact reduced16 tokenizer map and role prefixes, preserves
  partial decoded candidates, validates every model output before per-output
  fallback, persists completed arms before TTT, and reports incomplete/error
  status honestly.
- [x] Harden child supervision. A child-owned process group is tracked only
  when it is safe to signal; termination escalates from SIGTERM to SIGKILL,
  waits for the group to disappear, and cleans up queue/callback failures.
  Offline environment flags, CPython 3.11 ABI checks, attached dependency
  origins, patched-source metadata hashes, and full Apache-2.0 notice material
  are recorded or enforced.
- [x] Add RED-GREEN regressions through `run_gpu_task`, `_decode_arm`,
  `_fill_result`, `score_results`, and real subprocess supervision. The tests
  cover expired and future deadlines, tokenizer IDs, empty/wrong-count/
  malformed/bool/ragged grids, mixed multi-output fallback, arm persistence,
  and SIGTERM-ignoring descendants.
- [x] Run the final local quality gates and fresh-kernel CPU checks after this
  checkpoint. `ruff format --check`, `ruff check`, `ty check`, notebook
  compilation, `git diff --check`, and `python -m unittest discover -s tests
  -p 'test_*.py'` passed; the full repository suite is **67 tests passed**
  after the parent’s final corrections. The own-code gate includes all four
  code cells, including the final `main()` call; only immutable encoded
  payload lines have a narrow line-length exemption. Vendor hashes are checked.

  The parent added and observed failing regressions for three remaining issues,
  then fixed them: scoring no longer relabels symbolic fallback as model output;
  candidate rescoring retains the upstream four-view microbatches; and a real-mode
  total watchdog bounds synchronous setup/reporting as well as GPU work. Worker
  termination starts 30 seconds before the total 120-minute cap to reserve cleanup.

  Parent-owned fresh kernels passed from the repository root and a detached
  directory under the active profile’s scratch, with `PYTHONPATH` removed.
  `KernelManager.kernel_spec.argv[0]` was set to the project interpreter and the
  actual kernel executable was asserted. Final evidence is preserved under
  `artifacts/gpu1-parent-checks-kgpxx3zq/`: root/detached executed notebooks,
  logs, and CPU-check reports. Both reports explicitly say `mode=cpu-check`
  and `status=passed`; they do not establish GPU/model loading/training/
  inference/accuracy/memory results.
- [x] Independent final read-only review passed for notebook SHA-256
  `c5ffe668c7a54aad76abcab8df5be3115af067176506f0fe667aba7af5032381`.
  The reviewer ran all 16 focused CPU tests and checked the actual preflight,
  smoke, paired-worker, fallback, and scoring callers. No concrete security or
  logic blockers remained. The complete repository suite has 67 passing tests;
  real GPU validation is still the next unchecked human checkpoint.

### GPU-1 hosted dependency discovery correction

- User-reported hosted evidence: the fresh Python 3.13.15 session failed after
  attaching the legacy utility bundle. Copying the setup notebook with its
  original environment retained Python 3.11.13 and survived the attachment.
  Package metadata then matched the reference: torch 2.8.0, Unsloth 2025.9.7,
  unsloth_zoo 2025.9.9, Flash Attention 2.8.2, NumPy 2.2.6. This establishes
  startup and package discovery, not successful CUDA imports or inference.
- Confirmed root:
  `/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`.
  The old resolver searched only `/kaggle/input` at a limited depth. The fix
  locates the top-level Unsloth package without importing it, checks local
  `sys.path` roots and nested ordinary Inputs, deduplicates resolved paths,
  preserves the explicit override, and rejects ambiguous matches.
- Real mode now checks CPython 3.11 before creating the output directory or
  scanning inputs. The introduction describes preserving the upstream image;
  importing notebook language metadata alone does not pin that image.
- RED-GREEN regressions cover the observed Utility Script layout, package
  discovery by import origin, deeply nested Inputs, ambiguity, explicit override,
  and early ABI failure without output-directory mutation. Format, lint,
  typecheck, and all **69 tests** pass. Every cell also passed CPU-check mode
  in fresh project kernels from root and detached scratch working directories;
  evidence is ignored under `artifacts/gpu1-mount-fix-dbsp7v_8/`.
- Model/training/decoding/ranking code and pinned vendor payloads are unchanged.
  No hosted GPU run, weight download, commit, push, or submission was performed
  for this correction. The next GPU run remains unverified.
- Independent focused review passed for corrected notebook SHA-256
  `6e3dd61c995ddfdfcde1ebc39e83125c8ac92b2b026540159e830aaacf169067`.

### GPU-1 worker startup/reporting fix checkpoint (2026-11-02)

- [x] Isolate each child process's Unsloth compiler cache before importing
  Unsloth or the vendored Transformers loader. The loader validates the
  executable `TRITON_PTXAS_PATH`, defaulting to `/usr/local/cuda/bin/ptxas`
  while preserving a valid explicit override, and uses a process-local
  temporary `UNSLOTH_COMPILE_LOCATION`.
- [x] Preserve completed arm predictions when a later task/worker fails.
  Missing arms become `error` entries with the original error and traceback;
  symbolic fallbacks remain marked as fallback and repeated scoring preserves
  provenance.
- [x] Add CPU-only RED-GREEN checks for loader ordering/cache separation and
  the actual `run_real` event-to-task/summary reporting path. Final local
  gates passed: the two new tests failed before implementation and passed
  afterward; `ruff format --check` reported 11 files already formatted,
  `ruff check` passed, `ty check` passed on the extracted trusted notebook
  code plus `tests` and `scripts`, and the unittest suite passed **71 tests**.
  `git diff --check` is run as the final handoff check.
- [x] Verify the startup correction from the user's saved Kaggle outputs.
  Preflight and smoke passed; all four workers completed all eight pilot tasks
  in both arms, with no error events, model-arm fallbacks or hard timeout.
  Independent exact-grid rescoring against hash-matched training solutions
  confirms **9/9 outputs and 8/8 tasks** for inference-only and TTT, all correct
  in attempt 1. TTT added no correct outputs and used 594.55 summed worker
  training seconds. Reported peak device memory was 10.42 GiB inference-only
  and 12.71 GiB with TTT. This is public training-task reproduction evidence,
  not clean unseen generalization or a leaderboard score.
  Original uploads and hashes are ignored under
  `artifacts/gpu1-hosted-complete-cpbqc344/`; the exact hosted notebook source
  was not supplied. No competition submission was made.
