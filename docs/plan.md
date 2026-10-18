# ARC-AGI-2 implementation plan

Status: proposed checklist. No implementation or Kaggle execution is claimed.
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
- [ ] Reserve a fixed development/validation split within the training tasks.
  Keep related augmentations and synthetic variants in their source split.
  Check: evaluation solutions are not used for training or checkpoint selection.
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

- [ ] Upload the same notebook to a competition-attached Kaggle notebook.
  Attach competition data and any imported source files. Keep the notebook
  private unless the user authorizes publication.
  Check: record the notebook URL/version and confirm actual mounted input paths.
- [ ] Select CPU and disable internet. Read input from the mounted competition
  directory and write `/kaggle/working/submission.json`.
  Check: no dependency download, local absolute path, or repository-only import.
- [ ] Save and execute a full notebook version, not only interactive cells.
  Check: the offline run completes; its saved output contains a parseable,
  validated `submission.json`. This is a packaging check, not a hidden-test score.
- [ ] Ask the user before a competition submission. If authorized, submit the
  saved notebook version and inspect Kaggle's final status and score.
  Check: record the returned submission/version identifier and final result;
  do not treat a successful upload as a successful submission.

### 1C. Improve the solver incrementally

- [ ] Add geometry/color rules and shape inference from proposal §4.2 A–B.
  Check: a runnable regression check for each nontrivial behavior; candidates
  must reproduce all demonstration outputs before being treated as exact fits.
- [ ] Add object segmentation and selection (§4.2 C), starting with one useful
  rule family. Check: controlled object examples and training-set results agree.
- [ ] Add symmetry/periodic completion (§4.2 D) only where training failures
  justify it. Check: do not overwrite conflicting observed cells silently.
- [ ] Add bounded composition (§4.2 E), stable ranking, output deduplication,
  and fallback predictions. Check: candidate/time limits, multiple test inputs,
  ambiguous demo fits, and two-attempt behavior are covered.
- [ ] Iterate on training tasks; run full training checks at milestones.
  Check: record coverage, exact score, runtime, and representative failures.
- [ ] Freeze a candidate before the planned held-out evaluation. Record exact
  metrics and failure analysis. Check: do not tune repeatedly against that set.
- [ ] Repeat the offline Kaggle run with the selected solver and archive baseline
  predictions/configuration. Check: local/Kaggle behavior agrees on identical
  accessible inputs; the placeholder is not treated as hidden-test evidence.

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

Implement **1A only**, including the shared validator/scoring checks needed by
that slice. Execute the notebook on local data. Review that change in a PR,
then perform **1B** with the user. Do not build ML, TTT, or LLM scaffolding yet.

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
  `runtime_scratch=$(mktemp -d "${TMPDIR:-/tmp}/arc-agi-runtime.XXXXXX"); jq -r '[.cells[] | select(.cell_type == "code")][0].source[]' notebooks/submission.ipynb > "$runtime_scratch/runtime.py"; ty check "$runtime_scratch/runtime.py"`.
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
  `first=$(mktemp "${TMPDIR:-/tmp}/arc-agi-submission.XXXXXX"); cp artifacts/submission.json "$first"; ARC_MODE=submission ./.venv/bin/python -c 'import sys; from pathlib import Path; import nbformat; from nbclient import NotebookClient; notebook=nbformat.read(Path("notebooks/submission.ipynb"), as_version=4); NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=Path.cwd()); print("sys.executable:", sys.executable); print("fresh root submission rerun: passed")'; cmp "$first" artifacts/submission.json`.

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

The proposal remains unchanged. Kaggle account entry, quota, mounted input
paths, upload, registration, credentials, and 1B have not been verified or
performed. Human checkpoint: review the implementation PR, then perform the
Phase 1B Kaggle packaging check with the user.
