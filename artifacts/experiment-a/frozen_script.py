"""Run the frozen CPU per-puzzle cell-classifier experiment.

The classifier sees input cells only.  The trusted notebook runtime supplies
validation, scoring, background inference, connected components, and the
symbolic fallback.  Solutions are read by the runner only after inference for
scoring.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from scripts import diagnose_search as diagnostic
except ModuleNotFoundError:  # Direct execution: python scripts/experiment_a.py.
    import diagnose_search as diagnostic

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CHALLENGES_PATH = REPO_ROOT / "data" / "arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_PATH = REPO_ROOT / "data" / "arc-agi_training_solutions.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements-experiments.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "experiment-a"
EXPERIMENT_PATH = Path(__file__).resolve()
RANDOM_STATE = 17
COLOR_BUCKETS = 11
OUT_OF_BOUNDS_BUCKET = 10
ITERATION_SAMPLE_SIZE = 64
MODEL_NAMES = ("tree", "forest")


def feature_schema() -> list[str]:
    def color_name(color: int) -> str:
        return "oob" if color == OUT_OF_BOUNDS_BUCKET else str(color)

    names = [f"center_color_{color_name(color)}" for color in range(COLOR_BUCKETS)]
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            prefix = f"neighbor_{row_delta}_{column_delta}"
            names.extend(
                f"{prefix}_color_{color_name(color)}" for color in range(COLOR_BUCKETS)
            )
    names.extend(
        (
            "row_norm",
            "column_norm",
            "top_edge_norm",
            "bottom_edge_norm",
            "left_edge_norm",
            "right_edge_norm",
            "height_norm",
            "width_norm",
            "is_background",
            "component_size_norm",
            "component_bbox_height_norm",
            "component_bbox_width_norm",
            "component_relative_row",
            "component_relative_column",
            "component_top_norm",
            "component_left_norm",
        )
    )
    return names


FEATURE_SCHEMA = tuple(feature_schema())


def _normalised(value: int, denominator: int) -> float:
    return value / denominator if denominator else 0.0


def _one_hot(value: int | None) -> list[float]:
    bucket = OUT_OF_BOUNDS_BUCKET if value is None else value
    if not 0 <= bucket < COLOR_BUCKETS:
        raise ValueError(f"invalid color bucket: {bucket}")
    return [float(index == bucket) for index in range(COLOR_BUCKETS)]


def _component_features(
    grid: list[list[int]], runtime: dict[str, Any]
) -> dict[tuple[int, int], tuple[float, ...]]:
    height, width = runtime["grid_shape"](grid)
    background = runtime["infer_background"](grid)
    values: dict[tuple[int, int], tuple[float, ...]] = {}
    for component in runtime["connected_components"](grid, connectivity=4):
        cells = component["cells"]
        rows = [cell[0] for cell in cells]
        columns = [cell[1] for cell in cells]
        top, bottom = min(rows), max(rows)
        left, right = min(columns), max(columns)
        size_norm = len(cells) / (height * width)
        bbox_height = bottom - top + 1
        bbox_width = right - left + 1
        for row, column in cells:
            values[(row, column)] = (
                0.0,
                size_norm,
                bbox_height / height,
                bbox_width / width,
                _normalised(row - top, max(bbox_height - 1, 1)),
                _normalised(column - left, max(bbox_width - 1, 1)),
                _normalised(top, max(height - 1, 1)),
                _normalised(left, max(width - 1, 1)),
            )
    return {
        (row, column): values.get(
            (row, column), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        for row in range(height)
        for column in range(width)
        if grid[row][column] == background or (row, column) in values
    }


def feature_matrix(grid: list[list[int]], runtime: dict[str, Any]) -> list[list[float]]:
    """Return fixed-schema, input-only features in row-major cell order."""

    runtime["validate_grid"](grid)
    height, width = runtime["grid_shape"](grid)
    component_values = _component_features(grid, runtime)
    rows: list[list[float]] = []
    for row in range(height):
        for column in range(width):
            values = _one_hot(grid[row][column])
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    neighbor = (
                        grid[neighbor_row][neighbor_column]
                        if 0 <= neighbor_row < height and 0 <= neighbor_column < width
                        else None
                    )
                    values.extend(_one_hot(neighbor))
            component = component_values[(row, column)]
            values.extend(
                (
                    _normalised(row, max(height - 1, 1)),
                    _normalised(column, max(width - 1, 1)),
                    _normalised(row, max(height - 1, 1)),
                    _normalised(height - 1 - row, max(height - 1, 1)),
                    _normalised(column, max(width - 1, 1)),
                    _normalised(width - 1 - column, max(width - 1, 1)),
                    height / 30.0,
                    width / 30.0,
                    *component,
                )
            )
            rows.append(values)
    return rows


def shape_supported(task: dict[str, Any], runtime: dict[str, Any]) -> bool:
    return all(
        runtime["grid_shape"](demo["input"]) == runtime["grid_shape"](demo["output"])
        for demo in task["train"]
    )


def _training_arrays(
    task: dict[str, Any], runtime: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    for demo in task["train"]:
        if runtime["grid_shape"](demo["input"]) != runtime["grid_shape"](
            demo["output"]
        ):
            raise ValueError("cell classifier requires shape-preserving demonstrations")
        feature_rows.extend(feature_matrix(demo["input"], runtime))
        labels.extend(color for row in demo["output"] for color in row)
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(
        labels, dtype=np.int64
    )


def _new_model(name: str) -> Any:
    if name == "tree":
        return DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE)
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=32,
            max_depth=12,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown model: {name}")


def _model_submission(
    model: Any, task: dict[str, Any], runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = []
    for test in task["test"]:
        grid = test["input"]
        height, width = runtime["grid_shape"](grid)
        values = model.predict(np.asarray(feature_matrix(grid, runtime)))
        prediction = [
            [int(values[row * width + column]) for column in range(width)]
            for row in range(height)
        ]
        predictions.append(
            {"attempt_1": prediction, "attempt_2": copy.deepcopy(prediction)}
        )
    return predictions


def _validate_prediction(
    task: dict[str, Any], predictions: list[dict[str, Any]], runtime: dict[str, Any]
) -> None:
    runtime["validate_submission"]({"00000000": task}, {"00000000": predictions})


def _symbolic_predictions(
    task: dict[str, Any], runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = runtime["solve_task"](task)
    _validate_prediction(task, predictions, runtime)
    return copy.deepcopy(predictions)


def _duplicate_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "attempt_1": copy.deepcopy(prediction["attempt_1"]),
            "attempt_2": copy.deepcopy(prediction["attempt_1"]),
        }
        for prediction in predictions
    ]


def combined_predictions(
    tree: list[dict[str, Any]],
    forest: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = []
    for tree_prediction, forest_prediction, fallback_prediction in zip(
        tree, forest, fallback, strict=True
    ):
        first = tree_prediction["attempt_1"]
        second = forest_prediction["attempt_1"]
        if first == second:
            second = next(
                (
                    candidate
                    for candidate in (
                        fallback_prediction["attempt_1"],
                        fallback_prediction["attempt_2"],
                    )
                    if candidate != first
                ),
                copy.deepcopy(first),
            )
        combined.append(
            {"attempt_1": copy.deepcopy(first), "attempt_2": copy.deepcopy(second)}
        )
    return combined


def fit_predict_variants(
    task: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    """Fit independent models for one task and return fixed attempt variants."""

    fallback = _symbolic_predictions(task, runtime)
    supported = shape_supported(task, runtime)
    status: dict[str, Any] = {}
    model_predictions: dict[str, list[dict[str, Any]]] = {}
    if not supported:
        status = {name: "unsupported_shape" for name in MODEL_NAMES}
    else:
        features, labels = _training_arrays(task, runtime)
        for name in MODEL_NAMES:
            started = time.perf_counter()
            try:
                model = _new_model(name)
                model.fit(features, labels)
                predictions = _model_submission(model, task, runtime)
                _validate_prediction(task, predictions, runtime)
                model_predictions[name] = predictions
                status[name] = "fitted"
            except (ValueError, RuntimeError) as exc:
                model_predictions[name] = copy.deepcopy(fallback)
                status[name] = f"fallback:{type(exc).__name__}"
            status[f"{name}_elapsed_seconds"] = time.perf_counter() - started
    tree = (
        _duplicate_predictions(model_predictions["tree"])
        if supported and status["tree"] == "fitted"
        else copy.deepcopy(fallback)
    )
    forest = (
        _duplicate_predictions(model_predictions["forest"])
        if supported and status["forest"] == "fitted"
        else copy.deepcopy(fallback)
    )
    combined = (
        combined_predictions(tree, forest, fallback)
        if supported
        else copy.deepcopy(fallback)
    )
    _validate_prediction(task, tree, runtime)
    _validate_prediction(task, forest, runtime)
    _validate_prediction(task, combined, runtime)
    return {
        "supported": supported,
        "status": status,
        "fallback": fallback,
        "tree": tree,
        "forest": forest,
        "combined": combined,
    }


def fit_predict(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Fit one task without receiving solutions and return combined attempts."""

    runtime = diagnostic.load_runtime_namespace()
    return fit_predict_variants(task, runtime)["combined"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=scratch_dir,
        prefix="experiment-a-",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_metadata() -> dict[str, Any]:
    source = diagnostic.training_source_metadata()
    source.update(
        {
            "experiment_script_sha256": _sha256_file(EXPERIMENT_PATH),
            "requirements_sha256": _sha256_file(REQUIREMENTS_PATH),
        }
    )
    return source


def _selection(runtime: dict[str, Any], challenges: dict[str, Any]) -> dict[str, Any]:
    by_split = {
        split: sorted(
            (
                task_id
                for task_id in challenges
                if runtime["training_split"](task_id) == split
            ),
            key=lambda task_id: hashlib.sha256(task_id.encode("ascii")).hexdigest(),
        )
        for split in ("iteration", "heldout")
    }
    selected = by_split["iteration"][:ITERATION_SAMPLE_SIZE] + by_split["heldout"]
    return {
        "schema_version": 1,
        "order": "iteration first 64, then all heldout; each sorted by SHA256(taskID)",
        "iteration_all_count": len(by_split["iteration"]),
        "iteration_selected_count": min(
            ITERATION_SAMPLE_SIZE, len(by_split["iteration"])
        ),
        "heldout_count": len(by_split["heldout"]),
        "selected_ids": selected,
        "selected_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "split_ids": {
            "iteration": by_split["iteration"][:ITERATION_SAMPLE_SIZE],
            "heldout": by_split["heldout"],
        },
    }


def _frozen_config(source: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "A",
        "random_state": RANDOM_STATE,
        "feature_schema_sha256": hashlib.sha256(
            "\n".join(FEATURE_SCHEMA).encode()
        ).hexdigest(),
        "feature_count": len(FEATURE_SCHEMA),
        "models": {
            "tree": {"class": "DecisionTreeClassifier", "max_depth": 12},
            "forest": {
                "class": "RandomForestClassifier",
                "n_estimators": 32,
                "max_depth": 12,
                "n_jobs": 1,
            },
        },
        "fallback": (
            "trusted symbolic solve_task; unsupported tasks retain both symbolic "
            "attempts"
        ),
        "inference_contract": {
            "fit_predict_receives_solutions": False,
            "solutions_used_only_for_scoring_after_fit": True,
            "answer_based_selection": False,
        },
        "source": source,
        "selected_ids_sha256": selection["selected_ids_sha256"],
    }


def _score_variant(
    runtime: dict[str, Any],
    task_id: str,
    task: dict[str, Any],
    solutions: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    score = runtime["score_submission"](
        {task_id: task}, {task_id: solutions[task_id]}, {task_id: predictions}
    )
    return {
        **score,
        "correct_by_output": [
            prediction["attempt_1"] == expected or prediction["attempt_2"] == expected
            for prediction, expected in zip(
                predictions, solutions[task_id], strict=True
            )
        ],
    }


def _scope_ids(
    results: dict[str, Any], selection: dict[str, Any]
) -> dict[str, list[str]]:
    selected = selection["selected_ids"]
    supported = [
        task_id for task_id in selected if results["tasks"][task_id]["supported"]
    ]
    return {
        "supported": supported,
        "complete": selected,
        "iteration_supported": [
            task_id
            for task_id in supported
            if task_id in selection["split_ids"]["iteration"]
        ],
        "iteration_complete": [
            task_id
            for task_id in selected
            if task_id in selection["split_ids"]["iteration"]
        ],
        "heldout_supported": [
            task_id
            for task_id in supported
            if task_id in selection["split_ids"]["heldout"]
        ],
        "heldout_complete": [
            task_id
            for task_id in selected
            if task_id in selection["split_ids"]["heldout"]
        ],
    }


def _metrics_for_scope(
    results: dict[str, Any], ids: list[str], variant: str
) -> dict[str, Any]:
    scores = [
        results["tasks"][task_id]["variants"][variant]["score"] for task_id in ids
    ]
    total_outputs = sum(score["total_outputs"] for score in scores)
    correct_outputs = sum(score["correct_outputs"] for score in scores)
    return {
        "tasks": len(ids),
        "outputs": total_outputs,
        "correct_outputs": correct_outputs,
        "output_accuracy": correct_outputs / total_outputs if total_outputs else None,
        "solved_tasks": sum(score["solved_tasks"] for score in scores),
        "task_solve_rate": (
            sum(score["solved_tasks"] for score in scores) / len(scores)
            if scores
            else None
        ),
        "task_averaged_pass_at_2": (
            sum(score["task_averaged_pass_at_2"] for score in scores) / len(scores)
            if scores
            else None
        ),
    }


def _comparison(
    results: dict[str, Any], ids: list[str], before: str, after: str
) -> dict[str, int]:
    before_flags = [
        flag
        for task_id in ids
        for flag in results["tasks"][task_id]["variants"][before]["score"][
            "correct_by_output"
        ]
    ]
    after_flags = [
        flag
        for task_id in ids
        for flag in results["tasks"][task_id]["variants"][after]["score"][
            "correct_by_output"
        ]
    ]
    added = sum(
        not old and new for old, new in zip(before_flags, after_flags, strict=True)
    )
    lost = sum(
        old and not new for old, new in zip(before_flags, after_flags, strict=True)
    )
    return {"added_outputs": added, "lost_outputs": lost, "net_outputs": added - lost}


def _aggregate(
    results: dict[str, Any],
    selection: dict[str, Any],
    baseline: dict[str, Any] | None,
    runtime: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> dict[str, Any]:
    scopes = _scope_ids(results, selection)
    report: dict[str, Any] = {"scopes": {}, "comparisons": {}}
    for scope, ids in scopes.items():
        report["scopes"][scope] = {
            "task_ids": ids,
            "variants": {
                variant: _metrics_for_scope(results, ids, variant)
                for variant in ("fallback", "tree", "forest", "combined")
            },
            "comparisons": {
                "tree_to_forest": _comparison(results, ids, "tree", "forest"),
                "tree_to_combined": _comparison(results, ids, "tree", "combined"),
                "forest_to_combined": _comparison(results, ids, "forest", "combined"),
            },
        }
    if baseline is not None:
        baseline_scores = {}
        for task_id in selection["selected_ids"]:
            baseline_scores[task_id] = _score_variant(
                runtime, task_id, challenges[task_id], solutions, baseline[task_id]
            )
        report["phase1c_baseline"] = {
            scope: {
                "tasks": len(ids),
                "outputs": sum(
                    baseline_scores[task_id]["total_outputs"] for task_id in ids
                ),
                "correct_outputs": sum(
                    baseline_scores[task_id]["correct_outputs"] for task_id in ids
                ),
                "solved_tasks": sum(
                    baseline_scores[task_id]["solved_tasks"] for task_id in ids
                ),
            }
            for scope, ids in scopes.items()
        }
    return report


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    runtime = diagnostic.load_runtime_namespace()
    challenges, solutions = diagnostic.load_training_data(runtime)
    source = _source_metadata()
    selection = _selection(runtime, challenges)
    config = _frozen_config(source, selection)
    output_dir = output_dir.resolve()
    config_path = output_dir / "config.json"
    selection_path = output_dir / "selection.json"
    if config_path.exists() or selection_path.exists():
        if not config_path.exists() or not selection_path.exists():
            raise RuntimeError("frozen Experiment A state is incomplete")
        if _read_json(config_path) != config or _read_json(selection_path) != selection:
            raise RuntimeError("frozen Experiment A config, source, or IDs changed")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config_path, config)
        _write_json_atomic(selection_path, selection)
    result_path = output_dir / "results.json"
    expected_header = {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "selected_ids_sha256": selection["selected_ids_sha256"],
    }
    results: dict[str, Any] = cast(
        dict[str, Any],
        _read_json(result_path)
        if result_path.exists()
        else {**expected_header, "tasks": {}},
    )
    if any(results.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError("incremental Experiment A results do not match frozen state")
    _write_json_atomic(result_path, results)
    baseline, baseline_source = diagnostic.verified_baseline(runtime, challenges)
    started = time.perf_counter()
    for index, task_id in enumerate(selection["selected_ids"], start=1):
        if results["tasks"].get(task_id, {}).get("complete") is True:
            print(
                json.dumps(
                    {"event": "task_skipped_resume", "task_id": task_id}, sort_keys=True
                ),
                flush=True,
            )
            continue
        task_started = time.perf_counter()
        variants = fit_predict_variants(challenges[task_id], runtime)
        scored_variants = {
            name: {
                "predictions": variants[name],
                "score": _score_variant(
                    runtime, task_id, challenges[task_id], solutions, variants[name]
                ),
            }
            for name in ("fallback", "tree", "forest", "combined")
        }
        results["tasks"][task_id] = {
            "complete": True,
            "split": runtime["training_split"](task_id),
            "supported": variants["supported"],
            "model_status": variants["status"],
            "variants": scored_variants,
            "elapsed_seconds": time.perf_counter() - task_started,
        }
        _write_json_atomic(result_path, results)
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "index": index,
                    "total": len(selection["selected_ids"]),
                    "task_id": task_id,
                    "supported": variants["supported"],
                    "elapsed_seconds": results["tasks"][task_id]["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if any(
        task_id not in results["tasks"] or not results["tasks"][task_id].get("complete")
        for task_id in selection["selected_ids"]
    ):
        raise RuntimeError("Experiment A did not complete every frozen task")
    summary = {
        "schema_version": 1,
        "experiment": "A",
        "source": source,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "selection": selection,
        "baseline_source": baseline_source,
        "inference_contract": config["inference_contract"],
        "wall_clock_seconds": time.perf_counter() - started,
        "report": _aggregate(
            results, selection, baseline, runtime, challenges, solutions
        ),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "experiment_complete",
                "summary": str(output_dir / "summary.json"),
                "baseline_source": baseline_source,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
