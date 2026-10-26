"""Run the frozen Experiment A2 demo-based source selector.

A2 selects between the existing symbolic and cell-classifier outputs using
whole-demonstration leave-one-out evidence. Each fold fits on the remaining
input/output pairs; the held demonstration output is used only to score that
fold. Training-task test solutions are used only for final reporting.

The entrypoint is intentionally fixed to 128 training tasks.  It freezes task
IDs, source hashes, runtime versions, and the policy before the first task,
then writes one validated JSON result after each completed task.  Do not run
the entrypoint before parent review of this policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

try:
    from scripts import diagnose_search as diagnostic
    from scripts import experiment_a
except ModuleNotFoundError:  # Direct execution: python scripts/experiment_a2.py.
    import diagnose_search as diagnostic
    import experiment_a

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "experiment-a2"
ALL_A_SELECTION_PATH = REPO_ROOT / "artifacts" / "experiment-a" / "selection.json"
DIAGNOSTIC_SELECTION_PATH = (
    REPO_ROOT / "artifacts" / "search-diagnostic-final" / "selection.json"
)
EXPERIMENT_A2_PATH = Path(__file__).resolve()
SAMPLE_SIZE = 128
ALL_A_EXPECTED_COUNT = 280
DIAGNOSTIC_EXPECTED_COUNT = 32
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_ORDER = ("sym1", "sym2", "tree", "forest")
MODEL_SOURCES = {"tree": "tree", "forest": "forest"}
TASK_ID_PATTERN = re.compile(r"[0-9a-f]{8}")


def _grid_key(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def _read_json(path: Path) -> Any:
    return diagnostic.read_json(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    diagnostic.write_json_atomic(path, value)


def _sha256_file(path: Path) -> str:
    return diagnostic.sha256_file(path)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _read_exclusion_manifest(
    path: Path,
    label: str,
    challenges: dict[str, Any],
    *,
    expected_count: int,
) -> tuple[set[str], dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"Experiment A2 requires the {label} ID manifest at {path}; "
            "restore the reviewed Experiment A artifacts before running."
        )
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read the {label} ID manifest at {path}: {exc}"
        ) from exc
    ids = value.get("selected_ids") if isinstance(value, dict) else None
    if not isinstance(ids, list) or not all(
        isinstance(task_id, str) for task_id in ids
    ):
        raise RuntimeError(
            f"{label} ID manifest at {path} has no valid selected_ids list"
        )
    if len(ids) != expected_count:
        raise RuntimeError(
            f"{label} ID manifest at {path} has {len(ids)} IDs; "
            f"expected {expected_count} IDs"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{label} ID manifest at {path} contains duplicate IDs")
    invalid_ids = [
        task_id for task_id in ids if TASK_ID_PATTERN.fullmatch(task_id) is None
    ]
    if invalid_ids:
        raise RuntimeError(
            f"{label} ID manifest at {path} contains invalid 8-hex task IDs: "
            f"{invalid_ids[:3]}"
        )
    unknown_ids = sorted(set(ids) - challenges.keys())
    if unknown_ids:
        raise RuntimeError(
            f"{label} ID manifest at {path} contains IDs not present in loaded "
            f"training tasks: {unknown_ids[:3]}"
        )
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)
    return set(ids), {
        "path": display_path,
        "sha256": _sha256_file(path),
        "count": len(ids),
    }


def select_task_ids(
    challenges: dict[str, Any],
    excluded_ids: set[str],
    sample_size: int | None = None,
) -> list[str]:
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    candidates = sorted(
        (task_id for task_id in challenges if task_id not in excluded_ids),
        key=lambda task_id: hashlib.sha256(task_id.encode("ascii")).hexdigest(),
    )
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Experiment A2 found only {len(candidates)} eligible training tasks; "
            f"need fixed sample size {sample_size} after exclusions."
        )
    return candidates[:sample_size]


def _source_predictions(variants: dict[str, Any], source: str) -> list[dict[str, Any]]:
    if source in ("sym1", "sym2"):
        return variants["fallback"]
    return variants[MODEL_SOURCES[source]]


def _source_grid(variants: dict[str, Any], source: str, output_index: int) -> Any:
    try:
        predictions = _source_predictions(variants, source)
    except (KeyError, TypeError):
        return None
    if not isinstance(predictions, list):
        return None
    if output_index >= len(predictions):
        return None
    try:
        if source == "sym1":
            return predictions[output_index]["attempt_1"]
        if source == "sym2":
            return predictions[output_index]["attempt_2"]
        return predictions[output_index]["attempt_1"]
    except (KeyError, TypeError, IndexError):
        return None


def _source_available(variants: dict[str, Any], source: str) -> bool:
    if source in ("sym1", "sym2"):
        return isinstance(variants.get("fallback"), list)
    return variants.get("status", {}).get(MODEL_SOURCES[source]) == "fitted"


def _grid_is_valid(runtime: dict[str, Any], grid: Any) -> bool:
    if grid is None:
        return False
    try:
        runtime["validate_grid"](grid)
    except Exception:
        return False
    return True


def _new_source_stats() -> dict[str, Any]:
    return {
        "matches": 0,
        "folds": 0,
        "invalid_or_unavailable_folds": 0,
        "match_rate": None,
        "statuses": [],
    }


def _finalise_source_stats(stats: dict[str, Any]) -> None:
    stats["match_rate"] = stats["matches"] / stats["folds"] if stats["folds"] else None


def leave_one_demo_out(
    task: dict[str, Any],
    runtime: dict[str, Any],
    *,
    fit_predict: Any = experiment_a.fit_predict_variants,
) -> dict[str, Any]:
    """Score four sources on complete held-out demonstrations.

    The callback receives a task containing only the remaining demonstrations
    and the held demonstration input as a test input.  The held output is not
    present in that task.
    """

    source_stats = {source: _new_source_stats() for source in SOURCE_ORDER}
    if len(task["train"]) < 2:
        return {
            "reason": "insufficient_demos",
            "fold_count": 0,
            "loo_inference_count": 0,
            "elapsed_seconds": 0.0,
            "folds": [],
            "sources": source_stats,
        }

    started = time.perf_counter()
    folds = []
    for held_index, held_demo in enumerate(task["train"]):
        fold_task = {
            "train": [
                copy.deepcopy(demo)
                for index, demo in enumerate(task["train"])
                if index != held_index
            ],
            "test": [{"input": copy.deepcopy(held_demo["input"])}],
        }
        fold_started = time.perf_counter()
        fold = {"held_demo_index": held_index, "sources": {}}
        try:
            variants = fit_predict(fold_task, runtime)
        except Exception as exc:  # A failed source is a recorded no-match fold.
            error = f"{type(exc).__name__}: {exc}"
            for source in SOURCE_ORDER:
                stats = source_stats[source]
                stats["folds"] += 1
                stats["invalid_or_unavailable_folds"] += 1
                stats["statuses"].append("unavailable:" + type(exc).__name__)
                fold["sources"][source] = {"status": "unavailable", "error": error}
        else:
            for source in SOURCE_ORDER:
                stats = source_stats[source]
                stats["folds"] += 1
                available = _source_available(variants, source)
                prediction = _source_grid(variants, source, 0)
                if not available or not _grid_is_valid(runtime, prediction):
                    status = "unavailable"
                    stats["invalid_or_unavailable_folds"] += 1
                    fold["sources"][source] = {"status": status}
                elif prediction == held_demo["output"]:
                    status = "matched"
                    stats["matches"] += 1
                    fold["sources"][source] = {"status": status}
                else:
                    status = "available_no_match"
                    fold["sources"][source] = {"status": status}
                stats["statuses"].append(status)
        fold["elapsed_seconds"] = time.perf_counter() - fold_started
        folds.append(fold)

    for stats in source_stats.values():
        _finalise_source_stats(stats)
    return {
        "reason": "completed",
        "fold_count": len(folds),
        "loo_inference_count": len(folds),
        "elapsed_seconds": time.perf_counter() - started,
        "folds": folds,
        "sources": source_stats,
    }


def rank_sources(reliability: dict[str, dict[str, Any]]) -> list[str]:
    def rate(source: str) -> float:
        stored_rate = reliability[source].get("match_rate")
        if isinstance(stored_rate, (int, float)):
            return float(stored_rate)
        folds = reliability[source].get("folds", 0)
        matches = reliability[source].get("matches", 0)
        return float(matches) / float(folds) if folds else 0.0

    return sorted(
        SOURCE_ORDER,
        key=lambda source: (-rate(source), SOURCE_ORDER.index(source)),
    )


def _append_distinct(
    grids: list[list[list[int]]], labels: list[str], grid: Any, source: str
) -> bool:
    if not isinstance(grid, list):
        return False
    key = _grid_key(grid)
    if any(_grid_key(existing) == key for existing in grids):
        return False
    grids.append(copy.deepcopy(grid))
    labels.append(source)
    return True


def select_predictions(
    variants: dict[str, Any], reliability: dict[str, Any]
) -> dict[str, Any]:
    """Select two full-demo grids per test, with deterministic baseline fill."""

    source_stats = reliability.get("sources", reliability)
    ranked = rank_sources(source_stats)
    fallback = variants["fallback"]
    if reliability.get("reason") == "insufficient_demos":
        chosen_by_test = [
            ["sym1", "sym2"]
            if fallback[index]["attempt_1"] != fallback[index]["attempt_2"]
            else ["sym1"]
            for index in range(len(fallback))
        ]
        return {
            "predictions": copy.deepcopy(fallback),
            "ranked_sources": ranked,
            "chosen_sources": list(
                dict.fromkeys(label for labels in chosen_by_test for label in labels)
            ),
            "chosen_sources_by_test": chosen_by_test,
        }
    predictions = []
    chosen_by_test = []
    for output_index in range(len(fallback)):
        grids: list[list[list[int]]] = []
        labels: list[str] = []
        for source in ranked:
            if not _source_available(variants, source):
                continue
            _append_distinct(
                grids, labels, _source_grid(variants, source, output_index), source
            )
            if len(grids) == 2:
                break
        for source, attempt in (("sym1", "attempt_1"), ("sym2", "attempt_2")):
            if len(grids) == 2:
                break
            if output_index < len(fallback):
                _append_distinct(grids, labels, fallback[output_index][attempt], source)
        if not grids:
            grids.append(copy.deepcopy(fallback[output_index]["attempt_1"]))
            labels.append("fallback")
        if len(grids) == 1:
            grids.append(copy.deepcopy(grids[0]))
            labels.append("duplicate_fill")
        predictions.append({"attempt_1": grids[0], "attempt_2": grids[1]})
        chosen_by_test.append(labels)
    return {
        "predictions": predictions,
        "ranked_sources": ranked,
        "chosen_sources": list(
            dict.fromkeys(label for labels in chosen_by_test for label in labels)
        ),
        "chosen_sources_by_test": chosen_by_test,
    }


def _score_prediction(
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


def _runtime_metadata() -> dict[str, str]:
    return {
        "python": sys.version,
        "numpy": experiment_a.np.__version__,
        "sklearn": experiment_a.sklearn.__version__,
        "platform": platform.platform(),
    }


def _source_metadata() -> dict[str, Any]:
    try:
        source = diagnostic.training_source_metadata()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Experiment A2 cannot fingerprint its training inputs; "
            f"missing file: {exc.filename}"
        ) from exc
    source.update(
        {
            "experiment_a2_sha256": _sha256_file(EXPERIMENT_A2_PATH),
            "experiment_a_sha256": _sha256_file(experiment_a.EXPERIMENT_PATH),
            "requirements_sha256": _sha256_file(experiment_a.REQUIREMENTS_PATH),
        }
    )
    return source


def _selection(
    challenges: dict[str, Any],
    exclusion_manifests: dict[str, dict[str, Any]],
    excluded_ids: set[str],
) -> dict[str, Any]:
    selected_ids = select_task_ids(challenges, excluded_ids, sample_size=SAMPLE_SIZE)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "sample_size": SAMPLE_SIZE,
        "order": "all eligible training task IDs sorted by SHA256(taskID)",
        "selected_ids": selected_ids,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "excluded_ids_count": len(excluded_ids),
        "exclusion_manifests": exclusion_manifests,
    }


def _config(
    source: dict[str, Any],
    selection: dict[str, Any],
    runtime: dict[str, str],
    baseline_source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment": "A2",
        "sample_size": SAMPLE_SIZE,
        "policy": {
            "candidate_sources": list(SOURCE_ORDER),
            "tie_order": list(SOURCE_ORDER),
            "holdout": (
                "one complete demonstration per fold; held input only is test input"
            ),
            "fold_score": "exact held-grid match; invalid or unavailable is no-match",
            "selection": (
                "descending fold match rate, then fixed tie order; distinct grids"
            ),
            "fallback": "symbolic attempt order, then duplicate first grid",
        },
        "models": {
            "inherited_from": "scripts/experiment_a.py",
            "random_state": experiment_a.RANDOM_STATE,
            "feature_schema_sha256": hashlib.sha256(
                "\n".join(experiment_a.FEATURE_SCHEMA).encode("utf-8")
            ).hexdigest(),
        },
        "inference_contract": {
            "solutions_passed_to_fit": False,
            "solutions_used_only_after_full_prediction_for_reporting": True,
            "test_answers_used_for_selection": False,
        },
        "baseline_note": (
            "baseline is the hash-verified pre-measured Phase 1C artifact; "
            "A2 does not fit or recompute it"
        ),
        "runtime": runtime,
        "source": source,
        "baseline_source": baseline_source,
        "selected_ids_sha256": selection["selected_ids_sha256"],
    }


def _read_frozen(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"{path.name} uses an incompatible Experiment A2 artifact schema; "
            f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
    return value


def _validate_saved_task(
    runtime: dict[str, Any],
    task_id: str,
    task: dict[str, Any],
    entry: dict[str, Any],
    solutions: dict[str, Any],
) -> None:
    if entry.get("complete") is not True:
        raise RuntimeError(f"saved Experiment A2 task {task_id} is not complete")
    if entry.get("task_id") != task_id:
        raise RuntimeError(
            f"saved Experiment A2 task {task_id} has a mismatched task ID"
        )
    predictions = entry.get("predictions")
    if not isinstance(predictions, dict):
        raise RuntimeError(f"saved Experiment A2 task {task_id} has no predictions")
    scores = entry.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError(f"saved Experiment A2 task {task_id} has no scores")
    for name in ("baseline", "experiment_a", "experiment_a2"):
        try:
            variant_predictions = predictions[name]
        except KeyError as exc:
            raise RuntimeError(
                f"saved Experiment A2 task {task_id} has no {name} predictions"
            ) from exc
        runtime["validate_submission"]({task_id: task}, {task_id: variant_predictions})
        expected_score = _score_prediction(
            runtime, task_id, task, solutions, variant_predictions
        )
        if scores.get(name) != expected_score:
            raise RuntimeError(
                f"saved Experiment A2 task {task_id} {name} score mismatch"
            )


def _metric_scope(
    results: dict[str, Any], task_ids: list[str], variant: str
) -> dict[str, Any]:
    scores = [results["tasks"][task_id]["scores"][variant] for task_id in task_ids]
    outputs = sum(score["total_outputs"] for score in scores)
    correct = sum(score["correct_outputs"] for score in scores)
    solved = sum(score["solved_tasks"] for score in scores)
    return {
        "tasks": len(task_ids),
        "outputs": outputs,
        "correct_outputs": correct,
        "output_accuracy": correct / outputs if outputs else None,
        "solved_tasks": solved,
        "task_solve_rate": solved / len(task_ids) if task_ids else None,
    }


def _paired(
    results: dict[str, Any], task_ids: list[str], before: str, after: str
) -> dict[str, Any]:
    added_outputs = []
    lost_outputs = []
    added_tasks = []
    lost_tasks = []
    for task_id in task_ids:
        before_flags = results["tasks"][task_id]["scores"][before]["correct_by_output"]
        after_flags = results["tasks"][task_id]["scores"][after]["correct_by_output"]
        for index, (old, new) in enumerate(zip(before_flags, after_flags, strict=True)):
            output_id = f"{task_id}:{index}"
            if not old and new:
                added_outputs.append(output_id)
            elif old and not new:
                lost_outputs.append(output_id)
        before_solved = results["tasks"][task_id]["scores"][before]["solved_tasks"] == 1
        after_solved = results["tasks"][task_id]["scores"][after]["solved_tasks"] == 1
        if not before_solved and after_solved:
            added_tasks.append(task_id)
        elif before_solved and not after_solved:
            lost_tasks.append(task_id)
    return {
        "before": before,
        "after": after,
        "added_output_ids": added_outputs,
        "lost_output_ids": lost_outputs,
        "net_outputs": len(added_outputs) - len(lost_outputs),
        "added_task_ids": added_tasks,
        "lost_task_ids": lost_tasks,
        "net_tasks": len(added_tasks) - len(lost_tasks),
    }


def _summary(
    results: dict[str, Any], selection: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    task_ids = selection["selected_ids"]
    supported_ids = [
        task_id for task_id in task_ids if results["tasks"][task_id]["supported"]
    ]
    variants = ("baseline", "experiment_a", "experiment_a2")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment": "A2",
        "selection": selection,
        "source": config["source"],
        "policy": config["policy"],
        "views": {
            view: {
                "task_ids": ids,
                "variants": {
                    variant: _metric_scope(results, ids, variant)
                    for variant in variants
                },
            }
            for view, ids in (("full", task_ids), ("supported", supported_ids))
        },
        "paired": {
            "baseline_to_experiment_a": _paired(
                results, task_ids, "baseline", "experiment_a"
            ),
            "baseline_to_experiment_a2": _paired(
                results, task_ids, "baseline", "experiment_a2"
            ),
            "experiment_a_to_experiment_a2": _paired(
                results, task_ids, "experiment_a", "experiment_a2"
            ),
        },
        "loo": {
            "tasks": len(task_ids),
            "total_folds": sum(
                results["tasks"][task_id]["loo"]["fold_count"] for task_id in task_ids
            ),
            "total_inference_count": sum(
                results["tasks"][task_id]["loo"]["loo_inference_count"]
                for task_id in task_ids
            ),
            "elapsed_seconds": sum(
                results["tasks"][task_id]["loo"]["elapsed_seconds"]
                for task_id in task_ids
            ),
        },
    }


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    runtime = diagnostic.load_runtime_namespace()
    try:
        challenges, solutions = diagnostic.load_training_data(runtime)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Experiment A2 training data is unavailable; download the approved "
            f"training challenge and solution files first: {exc.filename}"
        ) from exc
    source = _source_metadata()
    all_a_ids, all_a_manifest = _read_exclusion_manifest(
        ALL_A_SELECTION_PATH,
        "all Experiment A",
        challenges,
        expected_count=ALL_A_EXPECTED_COUNT,
    )
    diagnostic_ids, diagnostic_manifest = _read_exclusion_manifest(
        DIAGNOSTIC_SELECTION_PATH,
        "previous 32-task diagnostic",
        challenges,
        expected_count=DIAGNOSTIC_EXPECTED_COUNT,
    )
    exclusion_manifests = {
        "experiment_a": all_a_manifest,
        "search_diagnostic": diagnostic_manifest,
    }
    selection = _selection(
        challenges,
        exclusion_manifests,
        all_a_ids | diagnostic_ids,
    )
    baseline, baseline_source = diagnostic.verified_baseline(runtime, challenges)
    if baseline is None:
        raise RuntimeError(
            "Experiment A2 requires the hash-verified 1,000-task baseline artifact; "
            "refusing to recompute or silently replace it."
        )
    output_dir = output_dir.resolve()
    config = _config(source, selection, _runtime_metadata(), baseline_source)
    config_path = output_dir / "config.json"
    selection_path = output_dir / "selection.json"
    result_path = output_dir / "results.json"
    if config_path.exists() or selection_path.exists():
        if not config_path.exists() or not selection_path.exists():
            raise RuntimeError("frozen Experiment A2 state is incomplete")
        if (
            _read_frozen(config_path) != config
            or _read_json(selection_path) != selection
        ):
            raise RuntimeError("frozen Experiment A2 config, source, or IDs changed")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic.write_json_atomic(config_path, config)
        diagnostic.write_json_atomic(selection_path, selection)

    header = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": _json_hash(config),
        "selected_ids_sha256": selection["selected_ids_sha256"],
        "source": source,
    }
    if result_path.exists():
        results = _read_frozen(result_path)
        if any(results.get(key) != value for key, value in header.items()):
            raise RuntimeError(
                "incremental Experiment A2 results do not match frozen state"
            )
        if not isinstance(results.get("tasks"), dict):
            raise RuntimeError("incremental Experiment A2 results have invalid tasks")
    else:
        results = {**header, "tasks": {}}
        diagnostic.write_json_atomic(result_path, results)
    tasks = cast(dict[str, Any], results["tasks"])
    unexpected_task_ids = sorted(set(tasks) - set(selection["selected_ids"]))
    if unexpected_task_ids:
        raise RuntimeError(
            f"unexpected saved Experiment A2 task IDs: {unexpected_task_ids[:3]}"
        )

    started = time.perf_counter()
    for index, task_id in enumerate(selection["selected_ids"], start=1):
        saved = tasks.get(task_id)
        if saved is not None:
            _validate_saved_task(
                runtime, task_id, challenges[task_id], saved, solutions
            )
            print(
                json.dumps(
                    {"event": "task_skipped_resume", "task_id": task_id}, sort_keys=True
                ),
                flush=True,
            )
            continue
        task_started = time.perf_counter()
        task = challenges[task_id]
        variants = experiment_a.fit_predict_variants(task, runtime)
        loo = leave_one_demo_out(
            task, runtime, fit_predict=experiment_a.fit_predict_variants
        )
        selected = select_predictions(variants, loo)
        prediction_sets = {
            "baseline": baseline[task_id],
            "experiment_a": variants["combined"],
            "experiment_a2": selected["predictions"],
        }
        for predictions in prediction_sets.values():
            runtime["validate_submission"]({task_id: task}, {task_id: predictions})
        scores = {
            name: _score_prediction(runtime, task_id, task, solutions, predictions)
            for name, predictions in prediction_sets.items()
        }
        tasks[task_id] = {
            "complete": True,
            "task_id": task_id,
            "supported": variants["supported"],
            "model_status": {
                name: status
                for name, status in variants["status"].items()
                if not name.endswith("_elapsed_seconds")
            },
            "loo": loo,
            "ranked_sources": selected["ranked_sources"],
            "chosen_sources_by_test": selected["chosen_sources_by_test"],
            "predictions": prediction_sets,
            "scores": scores,
            "elapsed_seconds": time.perf_counter() - task_started,
        }
        diagnostic.write_json_atomic(result_path, results)
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "index": index,
                    "total": len(selection["selected_ids"]),
                    "task_id": task_id,
                    "supported": variants["supported"],
                    "loo_folds": loo["fold_count"],
                    "elapsed_seconds": tasks[task_id]["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    missing = [
        task_id
        for task_id in selection["selected_ids"]
        if task_id not in tasks or tasks[task_id].get("complete") is not True
    ]
    if missing:
        raise RuntimeError(
            f"Experiment A2 did not complete frozen tasks: {missing[:3]}"
        )
    summary = _summary(results, selection, config)
    summary["wall_clock_seconds"] = time.perf_counter() - started
    diagnostic.write_json_atomic(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "experiment_complete",
                "summary": str(output_dir / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
