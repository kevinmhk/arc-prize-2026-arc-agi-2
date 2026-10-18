# ARC-AGI-2: two-week working proposal

Status: proposal only. No solver code has been written yet.

## 1. Objective and honest scope

Build a **CPU-first symbolic solver** for ARC-AGI-2 that produces a valid
Kaggle submission and gives us a useful learning artifact.

The realistic goal is **not** to win the competition. ARC-AGI-2 is designed to
resist memorization and simple pattern matching. With roughly two weeks, the
valuable outcome is:

- A working, reproducible solver.
- A valid `submission.json` for all hidden-test tasks.
- Measured results on the local evaluation set.
- A clear failure analysis and solution writeup.
- Hands-on learning in program search, grid transformations, exact-match
  evaluation, and offline Kaggle packaging.

We should treat any non-zero score as progress, not as evidence that the
approach is competitive.

## 2. What the supplied data actually says

Local data:

| File | Tasks | Notes |
|---|---:|---|
| `arc-agi_training_challenges.json` | 1,000 | 3,232 train pairs; test outputs not included |
| `arc-agi_training_solutions.json` | 1,000 | ground truth for training test outputs |
| `arc-agi_evaluation_challenges.json` | 120 | 358 train pairs |
| `arc-agi_evaluation_solutions.json` | 120 | 172 test outputs |
| `arc-agi_test_challenges.json` | 240 | placeholder tasks; hidden rerun replaces them |
| `sample_submission.json` | 240 | submission-format example |

Important details:

- Evaluation has **172 scored test outputs**: 71 tasks with one output, 46 with
  two, and 3 with three.
- Test challenge has **259 placeholder outputs** across 240 tasks.
- Grids are rectangular, 1x1 to 30x30, with colors `0` through `9`.
- Train pairs per task range from 2 to 10; most tasks have 3.
- About **65% of training pairs** and **69% of evaluation pairs** have the same
  input and output shape. Shape preservation is common but not universal.
- The local `arc-agi_test_challenges.json` is a placeholder built from
  evaluation tasks. It is **not** the hidden rerun data.

Exploratory baseline, measured locally:

- Identity prediction: **0/172** evaluation outputs correct.
- Constant-background prediction with input shape: **0/172**.
- A small set of simple rules (rotations, flips, transpose, crop, border
  removal, tiling, global color mapping) matched all train pairs on only
  **14/1,000** training tasks and solved **0/172** evaluation outputs.

Conclusion: trivial transformations are not enough. The solver needs
**object-level reasoning and composition**, and we must expect exact-match
scoring to be harsh.

## 3. Submission constraints

The competition is a **Code Competition**:

- Submission must be a Kaggle Notebook, not a locally uploaded prediction file.
- Notebook runtime must be at most **12 hours** on CPU or GPU.
- **Internet must be disabled** during the scoring run.
- `submission.json` must be produced by the notebook.
- Every task output needs **two predictions**, `attempt_1` and `attempt_2`.
- All task IDs from the hidden test challenges must be present.
- The hidden rerun has 240 unseen tasks; the local placeholder cannot be used
  to infer them.
- Freely/publicly available external data and pretrained models are allowed,
  subject to the rules. Our default is no external model dependency.

Scoring: for each test output, a task output scores 1 if **either** attempt
matches the ground truth exactly, otherwise 0. Final score is the average over
all test outputs.

## 4. Recommended approach

### 4.1 Hybrid candidate generation, not one fixed rule

Use a library of small, interpretable transformations, then compose and
validate them against the demonstration pairs.

Pipeline per task:

1. Parse and normalize grids.
2. Infer background color and object structure.
3. Generate candidate output grids from multiple rule families.
4. Keep only candidates that reproduce **all** training outputs exactly.
5. Rank survivors by simplicity and coverage.
6. Use the top two distinct candidates as `attempt_1` and `attempt_2`.
7. Fall back to a shape/color heuristic if no candidate survives.

This is a search problem, not a trained model. It is CPU-friendly, offline,
explainable, and testable.

### 4.2 Rule families

Start with the highest-value families, then expand only where error analysis
shows a gap.

**A. Global geometry and color**

- Identity, rotations, reflections, transpose.
- Global color bijection learned from aligned training pairs.
- Crop to non-background bounding box.
- Remove a uniform border.
- Tile/scale the input to the output shape.

These are cheap and already measured as insufficient alone, but they are useful
building blocks inside compositions.

**B. Shape inference**

- Predict output height and width from training-pair relationships:
  same shape, scaled shape, transposed shape, cropped shape, constant shape.
- Generate candidates using the predicted shape and a simple fill rule.
- This is also the fallback when no semantic rule applies.

**C. Object-level operations**

- Segment same-color connected components (4- and 8-connectivity).
- Select objects by size, color, shape, position, or uniqueness.
- Crop the selected object, move it to the origin, or extract its bounding box.
- Delete, keep, recolor, or overlay objects.
- Handle grids made of repeated subgrids by splitting into blocks.

**D. Symmetry and pattern completion**

- Complete a grid using horizontal, vertical, or rotational symmetry.
- Detect periodic patterns and extend them.
- Fill missing regions from the surrounding pattern.

**E. Composition**

- Apply up to two operations in sequence.
- Cap candidate count per task.
- Prefer fewer operations when two candidates both match all training pairs.
- Deduplicate identical output grids.

The initial implementation should stay small. Add a rule only when the failure
analysis shows repeated tasks that need it.

### 4.3 Ranking and two attempts

For each candidate, record:

- Number of training pairs reproduced exactly.
- Number of output cells covered by the prediction.
- Number of operations used.
- Whether the predicted shape matches the inferred shape.

Require all training pairs to match whenever possible. If no candidate matches
all pairs, use the best partial candidate but label it as weak internally.

Attempt 1 is the best candidate. Attempt 2 is the best **different** candidate;
if none exists, use the shape/color fallback. Never emit the same grid twice
unless the format requires a second entry.

## 5. Code layout

Keep the first version to a few files:

```text
2-arc-agi-2/
├── data/                       # supplied competition data
├── docs/
│   ├── overview.md
│   ├── proposal.md             # this file
│   └── solution.md             # final writeup, later
├── scripts/
│   └── download_data.py
├── arc_solver/
│   ├── __init__.py
│   ├── grids.py                # grid helpers, objects, shape utilities
│   ├── rules.py                # candidate generators
│   ├── solver.py               # search, ranking, attempts
│   └── io.py                   # load tasks, validate/write submissions
├── evaluate.py                 # local scoring and reports
├── make_submission.py          # local end-to-end submission generator
├── tests/
│   └── test_solver.py
└── notebooks/
    └── submission.ipynb        # offline Kaggle notebook
```

Do not add a web service, database, CLI framework, or experiment tracker. A
plain Python package, one evaluation script, one test file, and one notebook are
enough.

## 6. Evaluation protocol

### Local metrics

- **Output accuracy:** correct outputs / total outputs, using two attempts.
- **Task solve rate:** tasks where every test output is correct.
- **Candidate coverage:** fraction of tasks where at least one rule survives
  training-pair validation.
- **Runtime:** total seconds and maximum per-task seconds.
- **Failure categories:** wrong shape, wrong object, wrong color, wrong
  placement, incomplete pattern, unsupported composition, multiple outputs.

### Data use

- Use a fixed subset of training tasks for fast iteration.
- Run the full 1,000 training tasks at milestones.
- Keep the 120 evaluation tasks as the held-out local check.
- Do not tune repeatedly against the evaluation set; use it at planned checkpoints.
- Never use the hidden test set for development, even if a copy appears later.

Report the exact number, not a rounded impression. “0/172” and “3/172” are both
useful results.

## 7. Two-week plan

Dates assume Hong Kong time. The final deadline is **2 November 2026, 23:59 UTC**
(3 November, 07:59 HKT), so submit before the final day.

| Day | Work | Check |
|---|---|---|
| Oct 19 | Scaffold package; loader; submission validator; evaluation harness; reproduce 0 baseline | Local evaluation runs end to end |
| Oct 20–21 | Grid helpers, objects, geometry/color/shape rules; candidate validation | First non-zero candidate coverage measured |
| Oct 22–23 | Object selection, block splitting, symmetry, composition; first real solver score | Score and failure categories recorded |
| Oct 24 | Error analysis; add only the highest-value missing rules | Coverage and score improve or the gap is explained |
| Oct 25 | Full training run; ablation of rule families; runtime check | Stable configuration selected |
| Oct 26–27 | Kaggle notebook; offline dependency check; generate a complete placeholder submission | Notebook completes without internet |
| Oct 28–29 | Held-out evaluation; writeup draft; code cleanup | Reproducible result table |
| Oct 30–31 | Fix highest-impact failures; final validation | No regressions |
| Nov 1 | Freeze solver; final submission file and notebook check | All 240 tasks, two attempts each |
| Nov 2 | Submit early; verify Kaggle score/status | Submission confirmed |
| Nov 9 | Open-source artifacts and solution writeup deadline (if pursuing Grand Prize) | Public repository and writeup |

This is a target schedule, not a promise. If a milestone slips, cut rule
families—not validation, offline packaging, or the writeup.

## 8. Definition of done

**Must have:**

- A valid `submission.json` with every required task and two attempts per output.
- Notebook runs offline within the 12-hour limit.
- Deterministic output for the same input.
- Local evaluation report with exact scores and failure categories.
- Tests for grid operations, rule validation, submission formatting, and the
  two-attempt behavior.
- Public, documented code if we pursue a prize.

**Should have:**

- A non-zero evaluation score.
- At least one measured improvement over the simple-rule baseline.
- A concise `docs/solution.md` explaining what works, what fails, and why.

**Not required:**

- A state-of-the-art score.
- A trained neural model.
- A grand-prize writeup.
- Any claim that the solver “reasons” beyond what the measured results show.

## 9. Risks and stop rules

- **ARC-AGI-2 is hard.** Expect low exact-match scores. Do not confuse a working
  pipeline with a competitive solver.
- **No partial credit.** A nearly correct grid scores zero.
- **Hidden-test mismatch.** Local evaluation is a guide, not a guarantee.
- **Offline notebook failures.** Verify imports and file paths in a fresh
  offline run before the deadline.
- **Overfitting to the evaluation set.** Limit evaluation runs and keep a
  written record of changes.
- **Time.** If the solver is not producing any non-zero result by the end of
  Oct 23, stop expanding the DSL and focus on a correct submission, an honest
  baseline, and the writeup.
- **Originality and licensing.** Do not copy public solutions. Check the
  competition’s open-source and licensing requirements before publishing.
- **No hidden automation.** Do not submit, register, or spend money without the
  user’s explicit action.

## 10. What the user should personally do

To make this an education project rather than only agent output:

1. Implement at least two rule families and their tests.
2. Run the evaluation harness and explain one failure category.
3. Inspect a sample of failed tasks and propose one targeted rule.
4. Package and run the offline notebook once.
5. Write the “why it works / why it fails” section of the solution writeup.

Agent assistance can accelerate the code, but the learning comes from
implementing, measuring, and explaining the failures.

## 11. Immediate next step

Implement the thinnest vertical slice:

```text
load tasks → validate submission format → score identity baseline → report 0/172
```

Then add one rule family and measure whether candidate coverage changes. Do not
build the full DSL before the evaluation loop is trustworthy.

---

# Part II — Later phases: ML, GPU, and LLM

Sections 1–11 above are unchanged and remain the delivery plan for **Phase 1**.
Everything below is additive. Phase 1 is still the only committed deliverable;
Phases 2–4 are gated experiments that must earn their place with a measured
number, and each has a rollback to the Phase 1 submission.

The reason for splitting them this way: the Phase 1 solver is exact-match
symbolic search, and the measured baseline is 0/172. Adding a model to a system
that cannot yet solve anything gives you two unknowns instead of one. Add the
model once the symbolic loop is trustworthy and its failure modes are named.

Also note what the competition rules actually permit. The line
`GPU Notebook <= 12 hours run-time` means a GPU is **allowed**, not required.
Accelerator choice is a per-notebook setting and is independent of the solver.
Selecting GPU does not make serial Python search faster; it only rents hardware
you may not use and consumes quota.

## 12. Phase 1 hardening (unchanged scope)

Before any model work:

- Valid `submission.json`, every hidden-test task present, two attempts each.
- Offline notebook verified with internet disabled.
- Exact local score recorded, plus the failure-category breakdown from §6.
- A saved `baseline.json` of per-task predictions, so later phases can be
  compared against a fixed artifact rather than a moving solver.

Gate to Phase 2: Phase 1 score is **> 0/172**, and the failure analysis shows
cases where a correct program exists in the DSL but the enumerator does not
reach it within the candidate cap. That is the specific gap a learned prior
closes. If Phase 1 is at 0/172 and the failures are semantic rather than
search-order, fix the DSL first — a model will not rescue a missing operator.

## 13. Phase 2 — learned search priors (ML)

**Goal.** Same DSL, same validator, same output format. Replace part of the
blind enumeration order with a model that scores which primitives are plausible
for a given task. This is the [DreamCoder](https://doi.org/10.1145/3453483.3454080)
idea (Ellis et al., PLDI 2021: *DreamCoder: Bootstrapping Inductive Program
Synthesis with Wake-Sleep Library Learning*;
[open-access published paper at MIT](https://dspace.mit.edu/handle/1721.1/145949)),
reduced to the one piece that fits two weeks: a *recognition model*, not a
learned library.

**Training data comes free from Phase 1.** Run the Phase 1 solver over the
1,000 training tasks and keep the program that matched all demo pairs. Each such
program is a labeled example: `(task grids) → sequence of DSL operators`. That
gives a few hundred labeled programs without any hand annotation.

**Model shape, deliberately small.**

- Input: the demo pairs, capped to a fixed canvas (30×30, 10 colors), one-hot or
  embedding over color.
- Encoder: a small CNN or grid transformer, a few million parameters. Nothing
  that needs 96 GB.
- Output: a distribution over DSL operators, conditioned on the operators
  already chosen — i.e. a next-operator predictor used to order the search.
- Cost: minutes per epoch on a T4×2. This is a laptop-scale model that happens
  to train faster on a GPU.

**Wiring.** Keep it behind one switch:

```python
ORDER_STRATEGY = "enum"      # Phase 1 behaviour, always available
               # "model"     # Phase 2, requires a checkpoint
```

The validator, ranking, and two-attempt logic do not change. If the model orders
badly, search is slower, not wrong. That property is why this is the safe first
model to add.

**Success measure, not vibes.** Report median search nodes expanded per solved
task against Phase 1, and any exact-match gain on the held-out 120. A prior that
saves nodes but adds no accuracy is still a win for the writeup; a prior that
adds accuracy on train and loses on eval is overfitting, and gets reverted.

## 14. GPU usage rules

Applies to every phase after 1.

- **Submission notebook runs CPU.** Phase 1's solver is search-bound, not
  matrix-bound. Flipping the submission to GPU buys nothing and costs quota.
- **One separate lab notebook.** `notebooks/gpu-lab.ipynb`, accelerator L4×4,
  never committed as a submission version. Use it for torch environment checks,
  dataset loading, training runs, and breaking things.
- **Quota budget.** Kaggle grants roughly 30 GPU hours per week per account
  (Kaggle adjusts this; verify in account settings). L4×4 consumes at **2×**
  the rate, so it is effectively ~15 h/week. Plan around the halved number.
- **Learn on the cheap accelerator.** T4×2 and P100 run at 1× and are ample for
  models of this size. Reserve L4×4 for what its 96 GB is actually for: holding
  a small LLM, or very large batch runs.
- **Rules compliance.** L4 instances are restricted to notebooks attached to
  this competition. No attempts to use them elsewhere, no quota circumvention.
  Violations risk competition and account bans.
- **No internet.** Every dependency and every weight file must arrive as a
  Kaggle dataset or a pre-installed package. Test the offline path before
  trusting it.
- **Determinism.** Fix seeds, pin versions, and record them in the notebook.
  Phase 1 output must stay reproducible even after models exist.

## 15. Phase 3 — test-time training (ML, GPU-dependent)

**Idea.** Fine-tune a small grid model per task on that task's 2–10 demo pairs,
then predict the test output. This is the strongest known symbolic-plus-learning
lineage on ARC (Icecuber, ARChitects, and the 2024 winner's test-time-training
approach), and it is where a GPU genuinely changes what is possible.

**Honest cost.** TTT needs a per-task training loop, augmentation over the D4
group and color permutations, a careful early-stopping rule, and a
prediction→grid decoder. That is weeks of work at competition quality, not days.
At 240 hidden tasks inside a 12-hour cap, the loop also has to be fast: budget
roughly 30–60 seconds per task, which forces a small model and few epochs.

**Positioning.** Phase 3 is a candidate generator like any other. Its output
goes through the same exact validation against demo pairs; a TTT prediction that
contradicts the demos is discarded. It competes for `attempt_1` / `attempt_2`
against the symbolic candidates rather than replacing them.

**Gate.** Only start Phase 3 if Phase 2 shows the model's predictions are
useful at all, and only if the timeline still has a frozen Phase 1 submission to
fall back to.

## 16. Phase 4 — offline LLM as a program proposer (LLM, highest risk)

**Idea.** Instead of enumerating the DSL, ask a small language model to write a
DSL program for the task from the demo pairs, then validate it. This is the
highest-novelty option and the one that best matches the competition's
"universality" criterion, because a program-space prior generalizes further than
a hand-written grammar.

**The hard constraints, stated plainly.**

- **No internet during scoring.** The model weights must ship as a Kaggle
  dataset. A 7–8B model quantized to 4-bit fits comfortably in 96 GB; a larger
  one fits too, but every extra GB is upload and load time inside the 12-hour
  budget.
- **Output must be machine-checkable.** The LLM emits text in a strict DSL
  grammar. A parser, not the model's confidence, decides validity. Anything
  unparseable is discarded.
- **Validation is the safety net.** A proposed program still has to reproduce
  every demo pair exactly. A hallucinated program fails fast and costs little.
- **Determinism.** Greedy decoding or fixed seeds; otherwise the submission
  changes between runs and cannot be debugged.
- **Runtime.** Model load plus 2 attempts × 240 tasks of generation is the main
  risk. Measure it on the lab notebook before it goes anywhere near the
  submission path.

**Why it is last.** It has the longest feedback loop, the largest packaging
surface, the most ways to fail offline, and no measured evidence yet that a
small model's program proposals beat plain enumeration on ARC-AGI-2. Treat it
as a stretch experiment with its own branch.

## 17. Promotion rules

A later phase reaches the submission only when all of these hold:

1. It improves the held-out 120-task exact-match score over the frozen
   Phase 1 baseline, using the same two-attempt rule.
2. The offline notebook completes well inside 12 hours, with margin.
3. Output is deterministic for a fixed input and seed.
4. The fallback path still exists: with the model disabled, the notebook
   reproduces the Phase 1 submission exactly.

Anything that fails 1–4 stays in the lab notebook and becomes a writeup
section, not a submission change.

## 18. Revised timeline

Phase 1 dates from §7 are unchanged. The competition's final deadline is
2 November 2026, 23:59 UTC (3 November, 07:59 HKT), so Phase 1 must be submitted
early and treated as the floor.

| Window | Work | Output |
|---|---|---|
| Oct 19–24 | Phase 1 solver, as originally planned | Local score, failure analysis |
| Oct 25–26 | Phase 1 notebook, offline check, submit | Frozen baseline submission |
| Oct 26–27 | Lab notebook setup; torch/CUDA check; load train data on GPU | Working GPU sandbox |
| Oct 27–29 | Phase 2 prior: trace collection from Phase 1, small model, training | Node-savings and accuracy numbers |
| Oct 29–30 | Phase 2 promotion test, or documented negative result | Decision, recorded |
| Oct 30–31 | Optional Phase 3 prototype, timeboxed and abandonable | Prototype or dropped |
| Nov 1 | Freeze. Best validated configuration only | Final submission |
| Nov 2 | Submit early, verify status | Confirmed |

Phase 4 does not appear in this table on purpose. It runs only if Phases 1–3
finish early, and it does not block the deadline.

## 19. Additional risks

- **Two unknowns at once.** A model plus an unvalidated solver produces
  unattributable failures. Mitigation: Phase 1 freezes first.
- **Quota exhaustion.** L4×4 at 2× rate plus long training runs can burn a
  week's quota in two sessions. Mitigation: train small models on T4×2, keep
  checkpoints in a dataset, never re-download weights.
- **Offline packaging surprises.** A model that imports cleanly online can fail
  on a no-internet machine. Mitigation: one full offline dry run per phase.
- **Rule compliance.** L4 access is competition-restricted and monitored.
  Mitigation: use it only in notebooks attached to this competition.
- **Overfitting to the 120 evaluation tasks.** With a learnable component, the
  temptation to tune on eval grows. Mitigation: fixed checkpoint schedule, at
  most one eval run per phase, changes recorded before the run.
- **Scope creep against the deadline.** Mitigation: the promotion rules in §17
  are the only path to the submission.

## 20. Definition of done, extended

Phase 1 must-haves from §8 are unchanged. In addition:

**Phase 2 done:** a trained checkpoint, a `model` ordering strategy behind a
switch, a measured node-savings table, and a clear keep/revert decision.

**Phase 3 done:** a per-task TTT loop that fits the runtime budget on a sample
of tasks, with its measured accuracy and its measured cost per task.

**Phase 4 done:** a documented set of model candidates that fit the 96 GB
budget offline, a parsing layer for DSL text, and a measured proposal-quality
number — or a documented reason it was dropped.

**Across all phases:** the submission remains reproducible, offline, and
deterministic, with the Phase 1 path always available.
