"""Run the frozen bounded-search diagnostic on a training-only sample.

The production notebook is loaded as trusted source.  This module patches its
runtime namespace only while a diagnostic task is running; it does not modify
the notebook or its solver definitions on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import itertools
import json
import os
import platform
import signal
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
TRAINING_CHALLENGES_PATH = REPO_ROOT / "data" / "arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_PATH = REPO_ROOT / "data" / "arc-agi_training_solutions.json"
VERIFICATION_PATH = REPO_ROOT / "artifacts" / "phase1c" / "verification.json"
BASELINE_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "phase1c" / "training.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "search-diagnostic"
DIAGNOSTIC_SCRIPT_PATH = Path(__file__).resolve()

SAMPLE_SIZE = 32
TASK_TIMEOUT_SECONDS = 20.0
ANALYSIS_TIMEOUT_SECONDS = 5.0
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "name": "baseline",
        "generator": "production",
        "max_programs": 1024,
        "max_depth": 2,
    },
    "expanded_depth2": {
        "name": "expanded_depth2",
        "generator": "diagnostic_prefix_preserving",
        "max_programs": 4096,
        "max_depth": 2,
    },
    "deeper_depth3": {
        "name": "deeper_depth3",
        "generator": "diagnostic_prefix_preserving",
        "max_programs": 8192,
        "max_depth": 3,
    },
}
CONDITION_ORDER = tuple(CONFIGS)


class DiagnosticError(RuntimeError):
    """Raised when frozen diagnostic inputs or state are inconsistent."""


class TaskTimeout(TimeoutError):
    """Raised when a diagnostic task exceeds its wall-clock limit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=scratch_dir,
        prefix="arc-search-diagnostic-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def load_runtime_namespace() -> dict[str, Any]:
    """Executes only the notebook's trusted definition cell."""

    notebook = read_json(NOTEBOOK_PATH)
    definition_cell = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    source = "".join(definition_cell["source"])
    namespace: dict[str, Any] = {
        "__name__": "submission_runtime_search_diagnostic",
        "__file__": str(NOTEBOOK_PATH),
    }
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@contextlib.contextmanager
def patched_runtime(
    runtime: dict[str, Any],
    config: dict[str, Any],
    generator: Any,
) -> Generator[None, None, None]:
    """Patch diagnostic globals and restore every value on exit."""

    old_values = {
        "generate_programs": runtime["generate_programs"],
        "MAX_PROGRAMS_PER_TASK": runtime["MAX_PROGRAMS_PER_TASK"],
        "MAX_COMPOSITION_DEPTH": runtime["MAX_COMPOSITION_DEPTH"],
    }
    runtime["generate_programs"] = generator
    runtime["MAX_PROGRAMS_PER_TASK"] = config["max_programs"]
    runtime["MAX_COMPOSITION_DEPTH"] = config["max_depth"]
    try:
        yield
    finally:
        runtime.update(old_values)


def _production_singletons(
    runtime: dict[str, Any], production_generator: Any, task: dict[str, Any]
) -> list[tuple[Any, ...]]:
    old_cap = runtime["MAX_PROGRAMS_PER_TASK"]
    old_depth = runtime["MAX_COMPOSITION_DEPTH"]
    runtime["MAX_PROGRAMS_PER_TASK"] = 1024
    runtime["MAX_COMPOSITION_DEPTH"] = 2
    try:
        production_programs = production_generator(task)
    finally:
        runtime["MAX_PROGRAMS_PER_TASK"] = old_cap
        runtime["MAX_COMPOSITION_DEPTH"] = old_depth

    primitives: list[tuple[Any, ...]] = []
    seen_primitives: set[str] = set()
    seen_programs: set[str] = set()
    for program in production_programs:
        if len(program) != 1:
            continue
        operation = program[0]
        primitive = (operation,)
        primitive_key = repr(primitive)
        program_key = repr(primitive)
        if primitive_key in seen_primitives or program_key in seen_programs:
            continue
        seen_primitives.add(primitive_key)
        seen_programs.add(program_key)
        primitives.append(primitive)
    return primitives


def make_diagnostic_generator(
    runtime: dict[str, Any],
    production_generator: Any,
    config: dict[str, Any],
) -> Any:
    """Build a prefix-preserving generator from production singleton programs."""

    def generate(task: dict[str, Any]) -> list[tuple[Any, ...]]:
        primitives = _production_singletons(runtime, production_generator, task)
        programs: list[tuple[Any, ...]] = []
        seen_programs: set[str] = set()

        def append(program: tuple[Any, ...]) -> bool:
            key = repr(program)
            if key in seen_programs:
                return True
            if len(programs) >= config["max_programs"]:
                return False
            seen_programs.add(key)
            programs.append(program)
            return True

        for depth in range(1, config["max_depth"] + 1):
            for operations in itertools.product(primitives, repeat=depth):
                program = tuple(
                    operation for primitive in operations for operation in primitive
                )
                if not append(program):
                    return programs
        return programs

    return generate


@contextlib.contextmanager
def task_deadline(seconds: float) -> Generator[None, None, None]:
    if not sys.platform.startswith("linux") or threading_is_not_main():
        raise DiagnosticError("the per-task signal timeout requires Linux main thread")

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TaskTimeout(f"task exceeded {seconds:.3f}s wall limit")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def threading_is_not_main() -> bool:
    import threading

    return threading.current_thread() is not threading.main_thread()


def _score_flags(predictions: list[dict[str, Any]], solutions: list[Any]) -> list[bool]:
    return [
        prediction["attempt_1"] == expected or prediction["attempt_2"] == expected
        for prediction, expected in zip(predictions, solutions, strict=True)
    ]


def _demo_survivors(
    runtime: dict[str, Any], task: dict[str, Any], programs: list[tuple[Any, ...]]
) -> tuple[list[tuple[Any, ...]], int]:
    survivors: list[tuple[Any, ...]] = []
    best_coverage = 0
    for program in programs:
        coverage = 0
        valid = True
        for demo in task["train"]:
            try:
                predicted = runtime["apply_program"](demo["input"], program)
            except (runtime["ValidationError"], ValueError):
                valid = False
                continue
            coverage += predicted == demo["output"]
        best_coverage = max(best_coverage, coverage)
        if valid and coverage == len(task["train"]):
            survivors.append(program)
    return survivors, best_coverage


def _oracle_candidate_coverage(
    runtime: dict[str, Any],
    task: dict[str, Any],
    survivors: list[tuple[Any, ...]],
    solutions: list[Any],
) -> dict[str, Any]:
    output_hits = []
    for test, expected in zip(task["test"], solutions, strict=True):
        output_hit = False
        for program in survivors:
            try:
                if runtime["apply_program"](test["input"], program) == expected:
                    output_hit = True
                    break
            except (runtime["ValidationError"], ValueError):
                continue
        output_hits.append(output_hit)
    return {
        "oracle_analysis": True,
        "description": (
            "Known solutions are used only after inference to ask whether any "
            "demo-valid survivor could have produced the test output."
        ),
        "outputs_with_any_demo_valid_program": sum(output_hits),
        "total_outputs": len(output_hits),
        "task_has_any_candidate": any(output_hits),
        "task_is_fully_covered": all(output_hits),
    }


def _incomplete_result(
    status: str, inference_elapsed: float, total_elapsed: float, error: str
) -> dict[str, Any]:
    return {
        "status": status,
        "inference_completed": False,
        "complete": False,
        "inference_elapsed_seconds": inference_elapsed,
        "analysis_elapsed_seconds": 0.0,
        "total_diagnostic_elapsed_seconds": total_elapsed,
        "analysis_status": "not_started",
        "error": error,
        "predictions": None,
        "actual_score": None,
        "actual_correct_by_output": None,
        "demo_fit": None,
        "oracle_candidate_coverage": None,
    }


def run_runtime_task(
    runtime: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
    timeout_seconds: float,
    *,
    task_id: str | None = None,
    solutions: list[Any] | None = None,
    analysis_timeout_seconds: float = ANALYSIS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run inference, then bounded post-inference analysis with explicit status."""

    if analysis_timeout_seconds <= 0:
        raise ValueError("analysis_timeout_seconds must be positive")

    production_generator = runtime["generate_programs"]
    generator = production_generator
    if config["generator"] != "production":
        generator = make_diagnostic_generator(runtime, production_generator, config)

    captured_programs: list[list[tuple[Any, ...]]] = []

    def capture_generator(current_task: dict[str, Any]) -> list[tuple[Any, ...]]:
        programs = generator(current_task)
        captured_programs.append(programs)
        return programs

    started = time.perf_counter()
    inference_started = started
    try:
        with task_deadline(timeout_seconds):
            with patched_runtime(runtime, config, capture_generator):
                predictions, diagnostics = runtime["solve_task"](
                    task, return_diagnostics=True
                )
                runtime["validate_submission"](
                    {task_id or "00000000": task},
                    {task_id or "00000000": predictions},
                )
    except TaskTimeout as exc:
        now = time.perf_counter()
        return _incomplete_result(
            "timeout", now - inference_started, now - started, str(exc)
        )
    except Exception as exc:  # Report incomplete coverage instead of a silent score.
        now = time.perf_counter()
        return _incomplete_result(
            "error",
            now - inference_started,
            now - started,
            f"{type(exc).__name__}: {exc}",
        )

    inference_elapsed = time.perf_counter() - inference_started
    programs = captured_programs[-1] if captured_programs else []
    actual_score = None
    actual_correct_by_output = None
    survivors = None
    best_coverage = None
    oracle_coverage = None
    analysis_status = "completed"
    analysis_error = None
    analysis_started = time.perf_counter()
    try:
        with task_deadline(analysis_timeout_seconds):
            # Score immediately after validation. Later bounded analysis must not
            # discard a valid prediction or score that has already been produced.
            if solutions is not None:
                if task_id is None:
                    raise DiagnosticError("task_id is required when scoring")
                actual_score = runtime["score_submission"](
                    {task_id: task},
                    {task_id: solutions},
                    {task_id: predictions},
                )
                actual_correct_by_output = _score_flags(predictions, solutions)
            survivors, best_coverage = _demo_survivors(runtime, task, programs)
            if solutions is not None:
                oracle_coverage = _oracle_candidate_coverage(
                    runtime, task, survivors, solutions
                )
    except TaskTimeout as exc:
        analysis_status = "timeout"
        analysis_error = str(exc)
    except Exception as exc:  # Preserve inference and any score already computed.
        analysis_status = "error"
        analysis_error = f"{type(exc).__name__}: {exc}"

    analysis_elapsed = time.perf_counter() - analysis_started
    total_elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "inference_completed": True,
        "complete": True,
        "inference_elapsed_seconds": inference_elapsed,
        "analysis_elapsed_seconds": analysis_elapsed,
        "total_diagnostic_elapsed_seconds": total_elapsed,
        "analysis_status": analysis_status,
        "analysis_error": analysis_error,
        "predictions": predictions,
        "actual_score": actual_score,
        "actual_correct_by_output": actual_correct_by_output,
        "programs_tested": diagnostics["program_count"],
        "demo_fit": {
            "exact_fit": diagnostics["exact_fit"],
            "best_demo_coverage": diagnostics["train_demo_coverage"],
            "demo_valid_survivor_count": (
                len(survivors) if survivors is not None else None
            ),
        }
        if survivors is not None
        else None,
        "oracle_candidate_coverage": oracle_coverage,
    }


def training_source_metadata() -> dict[str, Any]:
    verification_available = VERIFICATION_PATH.exists()
    verification = read_json(VERIFICATION_PATH) if verification_available else {}
    notebook_hash = sha256_file(NOTEBOOK_PATH)
    expected_hash = verification.get("notebook_sha256")
    return {
        "diagnostic_script_sha256": sha256_file(DIAGNOSTIC_SCRIPT_PATH),
        "notebook_path": str(NOTEBOOK_PATH.relative_to(REPO_ROOT)),
        "notebook_sha256": notebook_hash,
        "verification_notebook_sha256": expected_hash,
        "notebook_hash_verified": notebook_hash == expected_hash,
        "verification_metadata_available": verification_available,
        "training_challenges_sha256": sha256_file(TRAINING_CHALLENGES_PATH),
        "training_solutions_sha256": sha256_file(TRAINING_SOLUTIONS_PATH),
    }


def verified_baseline(
    runtime: dict[str, Any], challenges: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load the frozen training artifact only when its source hash is verified."""

    source = training_source_metadata()
    if not source["verification_metadata_available"]:
        return None, {
            "kind": "recomputed",
            "reason": "verification metadata unavailable",
        }
    if not source["notebook_hash_verified"]:
        return None, {"kind": "recomputed", "reason": "notebook hash mismatch"}
    if not VERIFICATION_PATH.exists() or not BASELINE_ARTIFACT_PATH.exists():
        return None, {"kind": "recomputed", "reason": "baseline artifact unavailable"}

    verification = read_json(VERIFICATION_PATH)
    training_run = next(
        (run for run in verification.get("runs", []) if run.get("name") == "training"),
        None,
    )
    artifact_hash = sha256_file(BASELINE_ARTIFACT_PATH)
    if training_run is None or training_run.get("json_sha256") != artifact_hash:
        return None, {"kind": "recomputed", "reason": "baseline artifact hash mismatch"}

    baseline = read_json(BASELINE_ARTIFACT_PATH)
    runtime["validate_submission"](challenges, baseline)
    return baseline, {
        "kind": "verified_artifact",
        "path": str(BASELINE_ARTIFACT_PATH.relative_to(REPO_ROOT)),
        "sha256": artifact_hash,
        "verification_run": "training",
    }


def load_training_data(
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenges = read_json(TRAINING_CHALLENGES_PATH)
    solutions = read_json(TRAINING_SOLUTIONS_PATH)
    runtime["validate_challenges"](challenges)
    runtime["validate_solutions"](challenges, solutions)
    return challenges, solutions


def deterministic_iteration_ids(
    runtime: dict[str, Any], challenges: dict[str, Any]
) -> list[str]:
    return sorted(
        (
            task_id
            for task_id in challenges
            if runtime["training_split"](task_id) == "iteration"
        ),
        key=lambda task_id: hashlib.sha256(task_id.encode("ascii")).hexdigest(),
    )


def task_is_solved(
    runtime: dict[str, Any],
    task_id: str,
    task: dict[str, Any],
    solutions: dict[str, Any],
    predictions: dict[str, Any],
) -> bool:
    score = runtime["score_submission"](
        {task_id: task},
        {task_id: solutions[task_id]},
        {task_id: predictions[task_id]},
    )
    return score["solved_tasks"] == 1


def select_sample(
    runtime: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    iteration_ids = deterministic_iteration_ids(runtime, challenges)
    if baseline is None:
        baseline = {}
        for index, task_id in enumerate(iteration_ids, start=1):
            result = run_runtime_task(
                runtime,
                challenges[task_id],
                CONFIGS["baseline"],
                TASK_TIMEOUT_SECONDS,
                task_id=task_id,
            )
            if not result["complete"]:
                raise DiagnosticError(
                    f"baseline recomputation incomplete for sample selection: {task_id}"
                )
            baseline[task_id] = result["predictions"]
            print(
                json.dumps(
                    {
                        "event": "baseline_selection_recompute",
                        "completed": index,
                        "total": len(iteration_ids),
                        "task_id": task_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    unsolved = [
        task_id
        for task_id in iteration_ids
        if not task_is_solved(
            runtime, task_id, challenges[task_id], solutions, baseline
        )
    ]
    if len(unsolved) < SAMPLE_SIZE:
        raise DiagnosticError(
            f"only {len(unsolved)} unsolved iteration tasks; need {SAMPLE_SIZE}"
        )
    selected_ids = unsolved[:SAMPLE_SIZE]
    return selected_ids, {
        "selection_rule": (
            "unsolved TRAIN tasks with training_split == iteration, ordered by "
            "sha256(task_id).hexdigest()"
        ),
        "iteration_task_count": len(iteration_ids),
        "iteration_unsolved_task_count": len(unsolved),
        "selected_task_count": len(selected_ids),
        "selected_ids_sha256": sha256_text("\n".join(selected_ids)),
    }


def frozen_config(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "sample_size": SAMPLE_SIZE,
        "task_timeout_seconds": TASK_TIMEOUT_SECONDS,
        "analysis_timeout_seconds": ANALYSIS_TIMEOUT_SECONDS,
        "conditions": [copy.deepcopy(CONFIGS[name]) for name in CONDITION_ORDER],
        "source": source,
        "inference_contract": {
            "runtime": "trusted notebook definition cell",
            "training_solutions_passed_to_inference": False,
            "oracle_candidate_coverage_is_post_inference_only": True,
        },
    }


def load_or_freeze(
    output_dir: Path,
    runtime: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    source = training_source_metadata()
    if (
        source["verification_metadata_available"]
        and not source["notebook_hash_verified"]
    ):
        raise DiagnosticError(
            "notebook source hash does not match verification.json; "
            "diagnostic is frozen"
        )
    config = frozen_config(source)
    config_path = output_dir / "config.json"
    selection_path = output_dir / "selection.json"

    if config_path.exists() != selection_path.exists():
        raise DiagnosticError("config.json and selection.json must be created together")
    if config_path.exists():
        saved_config = read_json(config_path)
        if saved_config != config:
            raise DiagnosticError("frozen config/source differs from current runtime")
        selection = read_json(selection_path)
        selected_ids = selection["selected_ids"]
        if len(selected_ids) != SAMPLE_SIZE:
            raise DiagnosticError("frozen sample does not contain 32 task IDs")
        if selection["selected_ids_sha256"] != sha256_text("\n".join(selected_ids)):
            raise DiagnosticError("frozen sample ID hash is inconsistent")
        if any(
            runtime["training_split"](task_id) != "iteration"
            for task_id in selected_ids
        ):
            raise DiagnosticError("frozen sample contains a non-iteration task")
        baseline, baseline_source = verified_baseline(runtime, challenges)
        return config, selection, baseline, baseline_source

    baseline, baseline_source = verified_baseline(runtime, challenges)
    selected_ids, selection_details = select_sample(
        runtime, challenges, solutions, baseline
    )
    selection = {
        "schema_version": 1,
        "selected_ids": selected_ids,
        "selected_ids_sha256": selection_details["selected_ids_sha256"],
        "selection": selection_details,
        "baseline_source": baseline_source,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config_path, config)
    write_json_atomic(selection_path, selection)
    return config, selection, baseline, baseline_source


def initial_results(
    output_dir: Path, config: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    path = output_dir / "results.json"
    if path.exists():
        results = read_json(path)
        expected = {
            "schema_version": 1,
            "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
            "selected_ids_sha256": selection["selected_ids_sha256"],
        }
        if any(results.get(key) != value for key, value in expected.items()):
            raise DiagnosticError(
                "incremental results do not match frozen config/sample"
            )
        return results
    results = {
        **{
            "schema_version": 1,
            "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
            "selected_ids_sha256": selection["selected_ids_sha256"],
        },
        "tasks": {},
    }
    write_json_atomic(path, results)
    return results


def _completed(results: dict[str, Any], task_id: str, condition: str) -> bool:
    entry = results.get("tasks", {}).get(task_id, {}).get(condition, {})
    return entry.get("inference_completed", entry.get("complete", False)) is True


def run_experiment(
    output_dir: Path,
    runtime: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    config: dict[str, Any],
    selection: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    results = initial_results(output_dir, config, selection)
    for task_index, task_id in enumerate(selection["selected_ids"], start=1):
        task_results = results["tasks"].setdefault(task_id, {})
        for condition_index, condition_name in enumerate(CONDITION_ORDER, start=1):
            if _completed(results, task_id, condition_name):
                print(
                    json.dumps(
                        {
                            "event": "task_skipped_resume",
                            "condition": condition_name,
                            "task_id": task_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            result = run_runtime_task(
                runtime,
                challenges[task_id],
                CONFIGS[condition_name],
                TASK_TIMEOUT_SECONDS,
                task_id=task_id,
                solutions=solutions[task_id],
            )
            if (
                condition_name == "baseline"
                and baseline is not None
                and result["complete"]
            ):
                result["baseline_artifact_match"] = (
                    result["predictions"] == baseline[task_id]
                )
            task_results[condition_name] = result
            write_json_atomic(output_dir / "results.json", results)
            print(
                json.dumps(
                    {
                        "event": "task_complete",
                        "task_index": task_index,
                        "task_total": len(selection["selected_ids"]),
                        "condition_index": condition_index,
                        "condition_total": len(CONDITION_ORDER),
                        "condition": condition_name,
                        "task_id": task_id,
                        "status": result["status"],
                        "programs_tested": result.get("programs_tested"),
                        "inference_elapsed_seconds": result[
                            "inference_elapsed_seconds"
                        ],
                        "total_diagnostic_elapsed_seconds": result[
                            "total_diagnostic_elapsed_seconds"
                        ],
                        "analysis_status": result["analysis_status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return results


def aggregate_results(
    results: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = results["tasks"]
    condition_rows: list[dict[str, Any]] = []
    for condition_name in CONDITION_ORDER:
        entries = [
            tasks.get(task_id, {}).get(condition_name)
            for task_id in selection["selected_ids"]
        ]
        completed = [
            entry
            for entry in entries
            if entry and entry.get("inference_completed", entry.get("complete", False))
        ]
        scored = [entry for entry in completed if entry.get("actual_score")]
        demo_analyzed = [entry for entry in completed if entry.get("demo_fit")]
        oracle_analyzed = [
            entry for entry in completed if entry.get("oracle_candidate_coverage")
        ]
        timed_out = sum(
            entry is not None and entry.get("status") == "timeout" for entry in entries
        )
        errors = sum(
            entry is not None and entry.get("status") == "error" for entry in entries
        )
        analysis_timeouts = sum(
            entry is not None and entry.get("analysis_status") == "timeout"
            for entry in entries
        )
        analysis_errors = sum(
            entry is not None and entry.get("analysis_status") == "error"
            for entry in entries
        )
        total_outputs = sum(entry["actual_score"]["total_outputs"] for entry in scored)
        correct_outputs = sum(
            entry["actual_score"]["correct_outputs"] for entry in scored
        )
        solved_tasks = sum(entry["actual_score"]["solved_tasks"] for entry in scored)
        demo_fit_tasks = sum(entry["demo_fit"]["exact_fit"] for entry in demo_analyzed)
        oracle_outputs = sum(
            entry["oracle_candidate_coverage"]["outputs_with_any_demo_valid_program"]
            for entry in oracle_analyzed
        )
        condition_rows.append(
            {
                "condition": condition_name,
                "sample_tasks": len(entries),
                "completed_tasks": len(completed),
                "scored_tasks": len(scored),
                "score_unavailable_tasks": len(completed) - len(scored),
                "timeout_tasks": timed_out,
                "error_tasks": errors,
                "incomplete_tasks": len(entries) - len(completed),
                "analysis_completed_tasks": sum(
                    entry is not None and entry.get("analysis_status") == "completed"
                    for entry in entries
                ),
                "analysis_timeout_tasks": analysis_timeouts,
                "analysis_error_tasks": analysis_errors,
                "total_outputs_completed_denominator": total_outputs,
                "correct_outputs": correct_outputs,
                "output_accuracy_completed_only": (
                    correct_outputs / total_outputs if total_outputs else None
                ),
                "solved_tasks": solved_tasks,
                "task_solve_rate_completed_only": (
                    solved_tasks / len(scored) if scored else None
                ),
                "demo_fit_tasks": demo_fit_tasks,
                "demo_valid_survivors": sum(
                    entry["demo_fit"]["demo_valid_survivor_count"] or 0
                    for entry in demo_analyzed
                ),
                "oracle_candidate_outputs_completed_only": oracle_outputs,
                "total_inference_elapsed_seconds": sum(
                    entry["inference_elapsed_seconds"] for entry in entries if entry
                ),
                "max_inference_elapsed_seconds": max(
                    (entry["inference_elapsed_seconds"] for entry in entries if entry),
                    default=None,
                ),
                "total_diagnostic_elapsed_seconds": sum(
                    entry["total_diagnostic_elapsed_seconds"]
                    for entry in entries
                    if entry
                ),
                "max_total_diagnostic_elapsed_seconds": max(
                    (
                        entry["total_diagnostic_elapsed_seconds"]
                        for entry in entries
                        if entry
                    ),
                    default=None,
                ),
                "tested_programs_total": sum(
                    entry.get("programs_tested", 0) for entry in completed
                ),
                "baseline_artifact_matches": (
                    sum(
                        entry.get("baseline_artifact_match", False)
                        for entry in completed
                    )
                    if condition_name == "baseline"
                    else None
                ),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for condition_name in CONDITION_ORDER[1:]:
        paired_tasks = [
            task_id
            for task_id in selection["selected_ids"]
            if tasks.get(task_id, {})
            .get("baseline", {})
            .get(
                "inference_completed",
                tasks.get(task_id, {}).get("baseline", {}).get("complete", False),
            )
            and tasks.get(task_id, {})
            .get(condition_name, {})
            .get(
                "inference_completed",
                tasks.get(task_id, {}).get(condition_name, {}).get("complete", False),
            )
        ]
        baseline_correct = 0
        condition_correct = 0
        additional_outputs = 0
        lost_outputs = 0
        baseline_unsolved = 0
        condition_solves_from_baseline_unsolved = 0
        for task_id in paired_tasks:
            baseline_entry = tasks[task_id]["baseline"]
            condition_entry = tasks[task_id][condition_name]
            baseline_flags = baseline_entry["actual_correct_by_output"]
            condition_flags = condition_entry["actual_correct_by_output"]
            baseline_correct += sum(baseline_flags)
            condition_correct += sum(condition_flags)
            additional_outputs += sum(
                not before and after
                for before, after in zip(baseline_flags, condition_flags, strict=True)
            )
            lost_outputs += sum(
                before and not after
                for before, after in zip(baseline_flags, condition_flags, strict=True)
            )
            baseline_solved = baseline_entry["actual_score"]["solved_tasks"] == 1
            condition_solved = condition_entry["actual_score"]["solved_tasks"] == 1
            if not baseline_solved:
                baseline_unsolved += 1
                condition_solves_from_baseline_unsolved += condition_solved
        paired_rows.append(
            {
                "baseline": "baseline",
                "condition": condition_name,
                "paired_completed_tasks": len(paired_tasks),
                "paired_total_outputs": sum(
                    len(tasks[task_id]["baseline"]["actual_correct_by_output"])
                    for task_id in paired_tasks
                ),
                "baseline_correct_outputs": baseline_correct,
                "condition_correct_outputs": condition_correct,
                "additional_correct_outputs": additional_outputs,
                "lost_correct_outputs": lost_outputs,
                "net_correct_outputs": additional_outputs - lost_outputs,
                "paired_baseline_unsolved_tasks": baseline_unsolved,
                "condition_solved_tasks_from_baseline_unsolved": (
                    condition_solves_from_baseline_unsolved
                ),
            }
        )

    task_rows = []
    for task_id in selection["selected_ids"]:
        row: dict[str, Any] = {"task_id": task_id}
        for condition_name in CONDITION_ORDER:
            entry = tasks.get(task_id, {}).get(condition_name)
            row[condition_name] = {
                "status": entry.get("status") if entry else "missing",
                "inference_completed": (
                    entry.get("inference_completed", entry.get("complete", False))
                    if entry
                    else False
                ),
                "analysis_status": entry.get("analysis_status") if entry else None,
                "inference_elapsed_seconds": (
                    entry.get("inference_elapsed_seconds") if entry else None
                ),
                "total_diagnostic_elapsed_seconds": (
                    entry.get("total_diagnostic_elapsed_seconds") if entry else None
                ),
                "tested_programs": entry.get("programs_tested") if entry else None,
                "actual_correct_outputs": (
                    entry["actual_score"]["correct_outputs"]
                    if entry
                    and entry.get("inference_completed", entry.get("complete", False))
                    and entry.get("actual_score")
                    else None
                ),
                "actual_solved_task": (
                    entry["actual_score"]["solved_tasks"] == 1
                    if entry
                    and entry.get("inference_completed", entry.get("complete", False))
                    and entry.get("actual_score")
                    else None
                ),
                "demo_valid_survivors": (
                    entry["demo_fit"]["demo_valid_survivor_count"]
                    if entry
                    and entry.get("inference_completed", entry.get("complete", False))
                    and entry.get("demo_fit")
                    else None
                ),
                "oracle_candidate_outputs": (
                    entry["oracle_candidate_coverage"][
                        "outputs_with_any_demo_valid_program"
                    ]
                    if entry
                    and entry.get("inference_completed", entry.get("complete", False))
                    and entry.get("oracle_candidate_coverage")
                    else None
                ),
            }
        task_rows.append(row)

    summary = {
        "schema_version": 1,
        "scope": {
            "training_split": "iteration",
            "sample_size": len(selection["selected_ids"]),
            "evaluation_or_placeholder_data_read": False,
            "inference_received_training_solutions": False,
        },
        "conditions": condition_rows,
        "paired_comparisons": paired_rows,
        "coverage_note": (
            "Completed-task denominators exclude timeout/error tasks. Paired output "
            "comparisons use only tasks completed under both conditions."
        ),
        "oracle_note": (
            "oracle_candidate_coverage is a post-inference analysis of known test "
            "solutions; it did not affect ranking or either actual attempt."
        ),
    }
    tables = {
        "condition_table": condition_rows,
        "paired_comparison_table": paired_rows,
        "task_table": task_rows,
    }
    return summary, tables


def structural_grid(grid: list[list[int]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in grid:
        for color in row:
            key = str(color)
            counts[key] = counts.get(key, 0) + 1
    return {
        "shape": [len(grid), len(grid[0])],
        "rows": ["".join(str(color) for color in row) for row in grid],
        "color_counts": counts,
    }


def failure_inspection(
    output_dir: Path,
    results: dict[str, Any],
    selection: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        task_id
        for task_id in selection["selected_ids"]
        if results["tasks"].get(task_id, {}).get("deeper_depth3", {}).get("complete")
        and not results["tasks"][task_id]["deeper_depth3"]["actual_score"][
            "solved_tasks"
        ]
    ]
    candidates.sort(
        key=lambda task_id: hashlib.sha256(task_id.encode("ascii")).hexdigest()
    )
    selected_failures = candidates[:3]
    records = []
    for task_id in selected_failures:
        task = challenges[task_id]
        task_record: dict[str, Any] = {
            "task_id": task_id,
            "selection_rule": (
                "first three deeper-depth3-unsolved sample tasks by sha256(task_id)"
            ),
            "train": [
                {
                    "input": structural_grid(pair["input"]),
                    "output": structural_grid(pair["output"]),
                }
                for pair in task["train"]
            ],
            "test_inputs": [structural_grid(test["input"]) for test in task["test"]],
            "known_test_solutions_for_posthoc_inspection_only": [
                structural_grid(output) for output in solutions[task_id]
            ],
            "conditions": {},
            "limits_with_uncertainty": [
                "The bounded grammar may omit the task's required primitive or "
                "composition.",
                "A demo-valid survivor may still be ranked outside the two "
                "retained attempts.",
                "These are search-limit hypotheses, not an inferred ground-truth rule.",
            ],
        }
        for condition_name in CONDITION_ORDER:
            entry = results["tasks"][task_id].get(condition_name)
            task_record["conditions"][condition_name] = {
                "status": entry.get("status") if entry else "missing",
                "predictions": (
                    [
                        {
                            "attempt_1": structural_grid(prediction["attempt_1"]),
                            "attempt_2": structural_grid(prediction["attempt_2"]),
                        }
                        for prediction in entry["predictions"]
                    ]
                    if entry and entry.get("complete")
                    else None
                ),
                "demo_fit": entry.get("demo_fit") if entry else None,
                "oracle_candidate_coverage": (
                    entry.get("oracle_candidate_coverage") if entry else None
                ),
            }
        records.append(task_record)
    inspection = {
        "schema_version": 1,
        "description": (
            "Deterministic structural/grid inspection of three remaining "
            "sample failures."
        ),
        "task_count": len(records),
        "tasks": records,
    }
    write_json_atomic(output_dir / "failure-inspection.json", inspection)
    return inspection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not sys.platform.startswith("linux"):
        raise DiagnosticError("this diagnostic requires Linux signal/setitimer limits")

    runtime = load_runtime_namespace()
    challenges, solutions = load_training_data(runtime)
    config, selection, baseline, baseline_source = load_or_freeze(
        output_dir, runtime, challenges, solutions
    )
    metadata = {
        "schema_version": 1,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "source": config["source"],
        "config": config,
        "selection": selection,
        "baseline_source": baseline_source,
        "execution": {
            "output_dir": str(output_dir),
            "scratch_dir": os.environ.get("TMPDIR", tempfile.gettempdir()),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        saved_metadata = read_json(metadata_path)
        if (
            saved_metadata["source"] != metadata["source"]
            or saved_metadata["config"] != config
        ):
            raise DiagnosticError("metadata source/config differs from frozen state")
    else:
        write_json_atomic(metadata_path, metadata)

    print(
        json.dumps(
            {
                "event": "experiment_start",
                "selected_task_count": len(selection["selected_ids"]),
                "selected_ids": selection["selected_ids"],
                "baseline_source": baseline_source,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    results = run_experiment(
        output_dir,
        runtime,
        challenges,
        solutions,
        config,
        selection,
        baseline,
    )
    summary, tables = aggregate_results(results, selection)
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "tables.json", tables)
    inspection = failure_inspection(
        output_dir, results, selection, challenges, solutions
    )
    print(
        json.dumps(
            {
                "event": "experiment_complete",
                "summary_path": str(output_dir / "summary.json"),
                "tables_path": str(output_dir / "tables.json"),
                "failure_inspection_path": str(output_dir / "failure-inspection.json"),
                "failure_inspection_task_ids": [
                    task["task_id"] for task in inspection["tasks"]
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
