"""Independently validate the single frozen GPU-2 run; no model execution."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict


class Score(TypedDict):
    either_correct: int
    first_correct: int
    total_outputs: int
    solved_tasks: int
    total_tasks: int
    either_correct_output_ids: list[str]
    first_correct_output_ids: list[str]


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/gpu2-kaggle-run"
found = list((RUN / "output-v3").rglob("summary.json"))
assert len(found) == 1, f"Expected one saved summary, found {found}"
folder = found[0].parent
summary = json.loads(found[0].read_text())
manifest = json.loads((folder / "manifest.json").read_text())
predictions = json.loads((folder / "predictions.json").read_text())
events = [
    json.loads(line)
    for line in (folder / "events.jsonl").read_text().splitlines()
    if line.strip()
]
assert manifest["answers_loaded_stage"] == "after_predictions_persisted"
assert manifest["configuration"]["arms"] == ["inference_only"]
selection = manifest["configuration"]["selection"]
ids = selection["pilot"]
assert len(ids) == 16 and len(set(ids)) == 16
assert set(predictions) == set(ids + selection["smoke"])
for name, key in [
    ("arc-agi_evaluation_challenges.json", "evaluation_challenges"),
    ("arc-agi_evaluation_solutions.json", "evaluation_solutions"),
]:
    assert (
        hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
        == manifest["inputs"][key]["sha256"]
    )
challenges = json.loads((ROOT / "data/arc-agi_evaluation_challenges.json").read_text())
solutions = json.loads((ROOT / "data/arc-agi_evaluation_solutions.json").read_text())
assert sum(len(challenges[key]["test"]) for key in ids) == 22


def validate_grid(grid):
    assert isinstance(grid, list) and 1 <= len(grid) <= 30
    assert isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
    assert all(
        isinstance(row, list)
        and len(row) == len(grid[0])
        and all(type(value) is int and 0 <= value <= 9 for value in row)
        for row in grid
    )


scores: dict[str, Score] = {}
for arm in ("symbolic", "inference_only"):
    either_ids, first_ids = [], []
    solved, first_solved = [], []
    for task_id in ids:
        assert set(predictions[task_id]) == {"symbolic", "inference_only"}
        outputs = predictions[task_id][arm]
        assert (
            len(outputs) == len(challenges[task_id]["test"]) == len(solutions[task_id])
        )
        either_hits, first_hits = [], []
        for index, (prediction, answer) in enumerate(
            zip(outputs, solutions[task_id], strict=True)
        ):
            assert set(prediction) == {"attempt_1", "attempt_2"}
            for grid in prediction.values():
                validate_grid(grid)
            either = any(grid == answer for grid in prediction.values())
            first = prediction["attempt_1"] == answer
            either_hits.append(either)
            first_hits.append(first)
            if either:
                either_ids.append(f"{task_id}:{index}")
            if first:
                first_ids.append(f"{task_id}:{index}")
        if all(either_hits):
            solved.append(task_id)
        if all(first_hits):
            first_solved.append(task_id)
    for key, correct, solved_ids in [
        ("either_attempt", either_ids, solved),
        ("first_attempt", first_ids, first_solved),
    ]:
        reported = summary["metrics"]["full_frozen"][arm][key]
        assert (
            reported["correct_outputs"] == len(correct)
            and reported["total_outputs"] == 22
        )
        assert (
            reported["solved_tasks"] == len(solved_ids)
            and reported["total_tasks"] == 16
        )
    scores[arm] = {
        "either_correct": len(either_ids),
        "first_correct": len(first_ids),
        "total_outputs": 22,
        "solved_tasks": len(solved),
        "total_tasks": 16,
        "either_correct_output_ids": either_ids,
        "first_correct_output_ids": first_ids,
    }

for key, id_key in (
    ("either_attempt", "either_correct_output_ids"),
    ("first_attempt", "first_correct_output_ids"),
):
    added = set(scores["inference_only"][id_key]) - set(scores["symbolic"][id_key])
    lost = set(scores["symbolic"][id_key]) - set(scores["inference_only"][id_key])
    reported = summary["comparisons"]["inference_only_vs_symbolic"][key]
    assert (
        set(reported["added_output_ids"]) == added
        and set(reported["lost_output_ids"]) == lost
    )
    assert reported["added_outputs"] == len(added) and reported["lost_outputs"] == len(
        lost
    )

for task_id in ids:
    diagnostics = summary["by_task"][task_id]["inference_only"]["diagnostics"]
    assert len(diagnostics["fallback_mask"]) == len(challenges[task_id]["test"])
    for index, fallback in enumerate(diagnostics["fallback_mask"]):
        if fallback:
            assert (
                predictions[task_id]["inference_only"][index]
                == predictions[task_id]["symbolic"][index]
            )
for event in events:
    if event.get("kind") == "arm" and event.get("task_id") in ids:
        assert event["arm"] == "inference_only"
        task_id = event["task_id"]
        masks = summary["by_task"][task_id]["inference_only"]["diagnostics"][
            "fallback_mask"
        ]
        for index, fallback in enumerate(masks):
            if not fallback:
                assert (
                    event["result"]["predictions"][index]
                    == predictions[task_id]["inference_only"][index]
                )

entries = [summary["by_task"][key]["inference_only"] for key in ids]
result = {
    "kernel_ref": "kevinmhk/arc-gpu2-public-evaluation/3",
    "status": summary["status"],
    "scores": scores,
    "coverage": summary["coverage"],
    "timing": summary["timing"],
    "summed_inference_seconds": sum(
        entry["timing"].get("infer_seconds", 0) for entry in entries
    ),
    "reported_peak_GiB": max(
        entry["memory"].get("arm_peak_device_mb", 0) for entry in entries
    )
    / 1024,
    "errors": summary["errors"],
    "validation": "grid/count/data-hash/score/delta/fallback/event checks passed",
}
(RUN / "independent-verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
