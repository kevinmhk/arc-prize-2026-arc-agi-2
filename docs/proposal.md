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
