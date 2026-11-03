"""Independent frozen TTT comparison; invoke only after saved run ends."""

import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root / "artifacts/gpu3-kaggle-run"
paths = list((run / "output-v3").rglob("manifest.json"))
assert len(paths) == 1
folder = paths[0].parent
manifest = json.loads(paths[0].read_text())
assert manifest["mode"] == "real" and manifest["answers_loaded"] is False
assert manifest["stage"] == "complete"
predictions = json.loads((folder / "predictions.json").read_text())
old = root / "artifacts/gpu2-kaggle-run/output-v3/gpu2-public-evaluation"
baseline_manifest = json.loads((old / "manifest.json").read_text())
for frozen_input in ("model", "dependencies"):
    assert (
        manifest["inputs"][frozen_input] == baseline_manifest["inputs"][frozen_input]
    ), frozen_input
baseline = json.loads((old / "predictions.json").read_text())
ids = baseline_manifest["configuration"]["selection"]["pilot"]
assert len(ids) == len(predictions) == 16 and set(ids) == set(predictions)
for filename, key in [
    ("arc-agi_evaluation_challenges.json", "evaluation_challenges"),
    ("arc-agi_evaluation_solutions.json", "evaluation_solutions"),
]:
    data = (root / "data" / filename).read_bytes()
    assert (
        hashlib.sha256(data).hexdigest() == baseline_manifest["inputs"][key]["sha256"]
    )
    if key == "evaluation_challenges":
        assert hashlib.sha256(data).hexdigest() == manifest["inputs"][key]["sha256"]
challenges = json.loads((root / "data/arc-agi_evaluation_challenges.json").read_text())
solutions = json.loads((root / "data/arc-agi_evaluation_solutions.json").read_text())
assert sum(len(challenges[k]["test"]) for k in ids) == 22


def grid_ok(grid):
    assert isinstance(grid, list) and 1 <= len(grid) <= 30
    assert isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
    assert all(
        isinstance(row, list)
        and len(row) == len(grid[0])
        and all(type(v) is int and 0 <= v <= 9 for v in row)
        for row in grid
    )


def score(rows):
    either, first, solved = [], [], []
    for key in ids:
        assert len(rows[key]) == len(challenges[key]["test"]) == len(solutions[key])
        hits = []
        for i, (guess, answer) in enumerate(
            zip(rows[key], solutions[key], strict=True)
        ):
            assert set(guess) == {"attempt_1", "attempt_2"}
            for grid in guess.values():
                grid_ok(grid)
            hit = answer in guess.values()
            hits.append(hit)
            if hit:
                either.append(f"{key}:{i}")
            if guess["attempt_1"] == answer:
                first.append(f"{key}:{i}")
        if all(hits):
            solved.append(key)
    return {
        "correct_outputs": len(either),
        "first_correct": len(first),
        "solved_tasks": len(solved),
        "total_outputs": 22,
        "total_tasks": 16,
        "correct_ids": either,
        "first_ids": first,
    }


scores = {
    "inference_only": score({k: baseline[k]["inference_only"] for k in ids}),
    "ttt": score({k: predictions[k]["ttt"] for k in ids}),
}
assert scores["inference_only"]["correct_outputs"] == 6
results = {k: json.loads((folder / "tasks" / (k + ".json")).read_text()) for k in ids}
errors, trained, model_outputs, fallbacks = [], 0, 0, 0
entries = []
for key in ids:
    assert predictions[key]["symbolic"] == baseline[key]["symbolic"]
    entry = results[key]["arms"]["ttt"]
    entries.append(entry)
    diag = entry["diagnostics"]
    masks = diag["fallback_mask"]
    assert len(masks) == len(challenges[key]["test"])
    for i, fallback in enumerate(masks):
        if fallback:
            assert predictions[key]["ttt"][i] == baseline[key]["symbolic"][i]
    model_outputs += sum(not flag for flag in masks)
    fallbacks += sum(masks)
    errors.extend(results[key].get("worker_errors", []))
    if entry["status"] == "error":
        errors.append({"task_id": key, "error": entry.get("error")})
    stats = diag.get("train_stats")
    epoch = (
        stats[2].get("epoch", 0)
        if isinstance(stats, list) and len(stats) > 2 and isinstance(stats[2], dict)
        else 0
    )
    trained += bool(
        entry.get("model_loaded")
        and entry.get("training_succeeded")
        and diag.get("adapter_reset_verified")
        and entry.get("timing", {}).get("train_seconds", 0) > 0
        and epoch >= 0.999999
    )
events = [
    json.loads(line)
    for line in (folder / "events.jsonl").read_text().splitlines()
    if line.strip()
]
seen = collections.Counter()
for event in events:
    if event.get("kind") == "arm":
        assert event["arm"] == "ttt"
        key = event["task_id"]
        assert key in ids
        seen[key] += 1
        for i, fallback in enumerate(
            results[key]["arms"]["ttt"]["diagnostics"]["fallback_mask"]
        ):
            if not fallback:
                assert event["result"]["predictions"][i] == predictions[key]["ttt"][i]
assert all(count == 1 for count in seen.values())
assert model_outputs + fallbacks == 22
added = sorted(
    set(scores["ttt"]["correct_ids"]) - set(scores["inference_only"]["correct_ids"])
)
lost = sorted(
    set(scores["inference_only"]["correct_ids"]) - set(scores["ttt"]["correct_ids"])
)
peak = max(e.get("memory", {}).get("arm_peak_device_mb", 0) for e in entries) / 1024
elapsed = manifest["timing"]["total_seconds"]
projection = elapsed * 240 / 16
clean = (
    not errors
    and trained == 16
    and manifest["status"] not in {"error", "deadline_partial"}
)
selected = (
    "ttt"
    if scores["ttt"]["correct_outputs"] > scores["inference_only"]["correct_outputs"]
    and clean
    and 0 < peak < 22
    and projection < 41400
    else "inference_only"
)
report = {
    "kernel_ref": "kevinmhk/arc-gpu3-ttt-evaluation/3",
    "status": manifest["status"],
    "scores": scores,
    "added_outputs": len(added),
    "lost_outputs": len(lost),
    "added_ids": added,
    "lost_ids": lost,
    "model_outputs": model_outputs,
    "fallback_outputs": fallbacks,
    "trained_full_epoch_tasks": trained,
    "error_events": errors,
    "elapsed_seconds": elapsed,
    "summed_training_seconds": sum(
        e.get("timing", {}).get("train_seconds", 0) for e in entries
    ),
    "summed_inference_seconds": sum(
        e.get("timing", {}).get("infer_seconds", 0) for e in entries
    ),
    "reported_peak_GiB": peak,
    "naive_full240_seconds": projection,
    "projection_caveat": (
        "Simple sample scaling including fixed overhead; not full-run timing proof."
    ),
    "selected_policy": selected,
    "validation": (
        "actual data hashes, all22 grids/counts, cached fallback, "
        "raw arm events and paired score checks passed"
    ),
}
(run / "independent-comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
