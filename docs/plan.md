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
