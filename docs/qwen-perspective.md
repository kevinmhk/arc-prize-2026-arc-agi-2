# Qwen for ARC-AGI-2: perspective and experiment options

Status: explanatory note, not an implemented or benchmarked Qwen solution.
This records the discussion alongside the [working proposal](proposal.md).
No particular Qwen competition result, checkpoint, GPU requirement, or training
cost has been independently verified. No model download or training is authorized
by this document.

## 1. Generative AI is still a neural network

“Generative AI” and “traditional neural networks” are not mutually exclusive
categories. A Qwen language model is a neural network, typically based on a
Transformer, trained to predict and generate token sequences. A generated
sequence can represent prose, code, or a grid of integers.

ARC asks for an output grid conditioned on demonstration input/output pairs
and a new input. That is structured prediction. We can model the answer as an
image-like array, as a sequence, or as the result of executing an inferred
program. The grid format does not force a particular architecture.

A convolutional network, encoder-decoder Transformer, or other grid-native
network is possible. Variable dimensions do not make these approaches
impossible: padding, masks, explicit dimensions, or variable-length decoding
can handle them. The difficult part is learning a new transformation from a few
examples and generalizing it exactly, not merely choosing an output tensor size.

## 2. How a language model can represent the problem

A text-only model does not require screenshots. Serialize each grid with
unambiguous row boundaries and put the demonstration pairs in the prompt.

Hypothetical toy example, not a measured competition result:

```text
Example input:
0 1
2 3

Example output:
2 0
3 1

Test input:
4 5
6 7

Predicted output:
6 4
7 5
```

The proposed rule here is a clockwise rotation. In a real task, the model must
use all available demonstrations; a single example can support many different
rules. Choose the serialization after measuring context length and parsing
reliability. Do not assume each cell corresponds to exactly one model token.

There are two distinct ways to use Qwen.

### A. Generate the output grid directly

The prompt contains the demonstrations and the test input. The model generates
a serialized output grid, which the application parses and validates.

Potential benefit: learn a shared prior over ARC-like transformations without
manually enumerating every rule. Output length and row structure can represent
variable grid dimensions.

Limitations: malformed grids, truncated output, incorrect dimensions, and
plausible but wrong cells. Syntactic validity does not establish correctness.
Direct generation is a legitimate approach; it does not need to generate code.

### B. Generate a transformation program

The model proposes a small program that maps an input grid to an output grid.
Execute that same program on every demonstration input. Keep candidates that
reproduce the demonstration outputs exactly, then apply them to the test inputs.

Potential benefit: use a coding model's prior to search a much larger space of
transformations than a short handwritten rule list. Execution handles exact
arithmetic, indexing, counting, and grid construction.

Limitations: generated code may be unsafe, slow, invalid, or overfit to the
examples. A candidate can match every demonstration and still fail on the test
input. Passing examples is evidence of consistency, not proof of the intended
rule. A code-oriented Qwen checkpoint is an option for this route, not evidence
that other participants necessarily used it this way.

## 3. Prompting, fine-tuning, and test-time adaptation

These mechanisms are different and must be evaluated separately.

**In-context learning:** Give the pretrained model the current task's
examples in a prompt. Its weights do not change. This is the smallest initial
model experiment.

**Supervised fine-tuning:** Update model weights using ARC-formatted training
examples. For direct prediction, the target is an output grid. For program
generation, the target is a correct transformation program; the grid dataset
does not supply those programs automatically. Program targets require a separate
source or generation-and-verification process, with provenance and license
checks.

Fine-tuning can improve format adherence and the model's prior over useful
transformations. A standard next-token training objective can train on
serialized grids; no image classifier is required. Training loss improvement
does not establish improvement in exact-match ARC accuracy.

**Parameter-efficient fine-tuning, such as LoRA:** Train adapter parameters
rather than all model weights. This can reduce trainable-state memory compared
with full fine-tuning, but base weights, activations, context length, and inference
still consume resources. Actual feasibility depends on the chosen model and
hardware. Adapters must be packaged with a compatible base model or merged
correctly for offline inference.

**Test-time adaptation:** Update weights separately for the current task using
its visible demonstration pairs. This does not permit use of the unknown test
outputs. Reset to the intended base/adapters between tasks unless the experiment
explicitly defines otherwise. Adaptation adds runtime, overfitting risk, and
reproducibility requirements.

Augmentation can create rotated, reflected, or consistently color-permuted
versions of a task. Transform the inputs and outputs consistently, and invert
the transformation when mapping predictions back. More augmented examples are
not equivalent to more independent reasoning tasks.

## 4. Why Qwen could help, and what remains uncertain

Pretraining can supply priors about code, counting, composition, and spatial
relationships represented in text. Fine-tuning can adapt those priors to ARC's
representation. Neither guarantees that a particular model will infer the
correct task rule.

The [proposal](proposal.md) records a local simple-rule experiment with no
correct evaluation outputs. That shows that the tested primitive set was weak;
it does not prove that symbolic methods are best, that Qwen will improve it,
or that neural approaches cannot work.

Both model-based and symbolic approaches can overfit. Public ARC tasks may
already occur in a pretrained model's training data, so a high public score can
also reflect contamination. Model training provenance is often incomplete;
disclose that uncertainty rather than claiming unseen-task generalization.

## 5. Proposed hybrid: propose, execute, verify

```text
current task's demonstrations and test inputs
    -> bounded Qwen candidate generation
    -> parse grids or validate proposed programs
    -> check consistency against demonstrations
    -> deduplicate and rank distinct test predictions
    -> choose attempt_1 and attempt_2
    -> validate and write submission.json
```

For programs, require one consistent transformation across the demonstrations.
For direct-grid generation, demonstration reconstruction or leave-one-pair-out
checks can provide supporting evidence, but do not directly verify the unknown
answer. Ranking must not consult evaluation solution files during prediction.

Prefer a constrained transformation representation if it is sufficient. If
arbitrary generated Python is tested, treat it as untrusted code. Do not use
unrestricted `exec` in the solver process. A timeout or a subprocess alone is
not a security boundary: restrict filesystem access, network, process creation,
CPU, and memory with an appropriate sandbox. If safe execution is unavailable,
use direct grids or a constrained interpreter instead.

Generate only a bounded number of candidates. Select distinct, valid outputs
when available; duplicate attempts are permitted by the documented format but
provide no extra coverage. Define a deterministic fallback when no candidate
survives. Agreement and model confidence are ranking signals, not correctness
proofs.

## 6. Fair comparison and competition constraints

Use one evaluation harness for the symbolic baseline, pretrained Qwen, and any
fine-tuned variant. Record exact two-attempt output accuracy, full-task solve
rate, invalid-output rate, demonstration-fit coverage, runtime, peak memory, and
which tasks each method adds or loses. Do not invent expected score gains.

Split by original task before fine-tuning or augmentation. Keep all variants of
a task in the same partition. Fit task-specific mappings from that task's
visible demonstrations only. Never train on validation answers and then report
those same tasks as held out. The earlier exploratory evaluation has already
been inspected; a new untouched training-task holdout provides an additional
check, not a guarantee against pretraining contamination.

According to the saved [overview](overview.md) and
[dataset description](../data/dataset-description.md):

- Kaggle reruns the submission notebook with internet disabled.
- CPU and GPU scoring notebooks have a twelve-hour runtime limit.
- Hidden test challenges replace the public placeholder file during rerun.
- Each test input needs two attempts, in order, with exact grid matching.
- All task IDs must appear in `submission.json`.

Package model weights, tokenizer, adapters, and dependencies for offline use.
Do not rely on a hosted Qwen API or download weights during scoring. Benchmark
the complete pipeline on the actual permitted hardware, including adaptation
and multiple candidates; do not infer feasibility from model size alone.
Check the chosen checkpoint's license and the current competition rules before
using or publishing it. General permission for public pretrained models does
not establish permission to disclose competition data to an external service.

## 7. Recommendation for our two-week learning project

Keep the symbolic baseline as a cheap, explainable comparison, not an assertion
of the winning architecture. First build the loader, exact-match scorer,
submission validator, and offline execution path shared by either approach.

Then run a small pretrained-Qwen experiment if approved compute is available.
Start with direct-grid generation, or constrained program proposals when safe
execution is ready. Pin the checkpoint revision, prompt, decoding settings,
candidate budget, and task split. Compare measured results with the baseline.

Consider a small adapter fine-tune only after that experiment identifies a
specific weakness and establishes a feasible memory/runtime budget. Do not
start full-model training as the default. This recommendation reflects our
time and learning objectives, not a verified claim that fine-tuning is inferior.

Stop or reduce the model experiment if it threatens offline packaging, runtime
limits, or the final reproducible result. Useful learning outcomes include
explaining tokenization, next-token fine-tuning, augmentation leakage, LoRA,
and the difference between fitting demonstrations and generalizing to a new
input—even if the score remains low.

Bottom line: **Qwen generates a structured answer or a program. That is a valid
neural approach to ARC. Whether it helps us is an empirical question, and
fine-tuning is one optional mechanism—not a prerequisite or a guarantee.**
