import copy
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gpu2_public_evaluation.ipynb"


def load_runtime() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "test_gpu2_notebook",
        "__file__": str(NOTEBOOK),
    }
    source = "".join(notebook["cells"][1]["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


class GPU2NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_runtime()

    def test_reference_is_hash_verified_and_policy_is_inference_only(self) -> None:
        runtime = self.runtime
        self.assertEqual(
            runtime["GPU2_NOTEBOOK_SHA256"],
            "463829abfea83fb95d7336dc6a4ffb8a7f652e3bcdb3947e840cf502bc7b70e4",
        )
        self.assertEqual(
            runtime["REFERENCE"]["runtime_source_fingerprint"](),
            "bbbd6ca85cc3a763b53b691e9068cc40b10049b410430afe54daa011e89ff1c2",
        )
        self.assertEqual(runtime["REFERENCE"]["arm_order"](), ("inference_only",))
        self.assertEqual(
            runtime["validate_inference_only_policy"](), ("inference_only",)
        )

    def test_gpu2_retry_caps_match_parent_launch_budget(self) -> None:
        runtime = self.runtime
        self.assertEqual(runtime["GPU2_GRACEFUL_SECONDS"], 5100)
        self.assertEqual(runtime["GPU2_HARD_SECONDS"], 5400)
        self.assertEqual(runtime["GPU2_INFERENCE_SECONDS"], 1200)

    def test_frozen_selection_uses_explicit_reference_arguments_and_output_order(
        self,
    ) -> None:
        runtime = self.runtime
        selection = runtime["GPU2_FROZEN_SELECTION"]
        challenges = {
            task_id: {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[0]]}]
                * (
                    3
                    if task_id == "d35bdbdc"
                    else 2
                    if task_id in {"4c3d4a41", "446ef5d2", "269e22fb", "65b59efc"}
                    else 1
                ),
            }
            for task_id in [*selection["smoke"], *selection["pilot"]]
        }
        calls: list[tuple[object, object, object]] = []

        def fake_selector(
            _challenges: dict[str, object],
            *,
            seed_label: str,
            smoke_count: int,
            pilot_count: int,
        ) -> dict[str, list[str]]:
            calls.append((seed_label, smoke_count, pilot_count))
            return copy.deepcopy(selection)

        with patch.dict(runtime["REFERENCE"], {"select_task_ids": fake_selector}):
            actual = runtime["validate_public_selection"](challenges)

        self.assertEqual(actual, selection)
        self.assertEqual(
            calls,
            [
                (
                    runtime["GPU2_SEED_LABEL"],
                    runtime["GPU2_SMOKE_COUNT"],
                    runtime["GPU2_PILOT_COUNT"],
                )
            ],
        )
        self.assertEqual(
            [*actual["smoke"], *actual["pilot"]],
            [
                "581f7754",
                "16b78196",
                "31f7f899",
                "4c3d4a41",
                "89565ca0",
                "53fb4810",
                "d35bdbdc",
                "6e453dd6",
                "7491f3cf",
                "446ef5d2",
                "7b3084d4",
                "1818057f",
                "80a900e0",
                "269e22fb",
                "65b59efc",
                "dfadab01",
                "a251c730",
            ],
        )

    def test_missing_evaluation_input_fails_before_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(self.runtime["MissingInputError"]):
                self.runtime["discover_evaluation_challenges"](Path(directory))

    def test_cpu_check_runs_real_adapter_path_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.runtime["run_cpu_check"](Path(directory) / "run")

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["orchestration"]["arms"], ["inference_only"])
        self.assertFalse(result["orchestration"]["training_called"])
        self.assertTrue(result["orchestration"]["adapter_reset_verified"])

    def test_inference_adapter_does_not_call_reference_training(self) -> None:
        runtime = self.runtime
        task_id = "00000001"
        task = {
            task_id: {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]]}],
            }
        }

        def decode(*_args: object) -> tuple[list[dict[str, object]], dict[str, object]]:
            return [{"attempt_1": [[2]], "attempt_2": [[2]]}], {"status": "complete"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenges.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            modules = runtime["_cpu_modules"]()
            with patch.dict(
                runtime["REFERENCE"],
                {
                    "_train_arm": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError()
                    )
                },
            ):
                with patch.dict(runtime["REFERENCE"], {"_decode_arm": decode}):
                    result = runtime["REFERENCE"]["run_gpu_task"](
                        types.SimpleNamespace(adapter_state={"value": 0}),
                        object(),
                        modules,
                        {"value": 0},
                        path,
                        task_id,
                        10**12,
                    )

        arm = result["arms"]["inference_only"]
        self.assertTrue(arm["training_succeeded"])
        self.assertTrue(arm["diagnostics"]["adapter_reset_verified"])

    def test_reference_timing_boundaries_keep_decode_out_of_setup(self) -> None:
        runtime = self.runtime
        task_id = "00000001"
        task = {
            task_id: {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]]}],
            }
        }

        def decode(*_args: object) -> tuple[list[dict[str, object]], dict[str, object]]:
            time.sleep(0.03)
            return [{"attempt_1": [[2]], "attempt_2": [[2]]}], {"status": "complete"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenges.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            modules = runtime["_cpu_modules"]()
            with patch.dict(
                runtime["REFERENCE"],
                {
                    "_train_arm": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError()
                    ),
                    "_decode_arm": decode,
                },
            ):
                result = runtime["REFERENCE"]["run_gpu_task"](
                    types.SimpleNamespace(adapter_state={"value": 0}),
                    object(),
                    modules,
                    {"value": 0},
                    path,
                    task_id,
                    time.monotonic() + 60,
                )

        timing = result["arms"]["inference_only"]["timing"]
        self.assertGreater(timing["infer_seconds"], 0.02)
        self.assertLess(timing["setup_seconds"], timing["infer_seconds"])
        self.assertGreaterEqual(timing["total_seconds"], timing["infer_seconds"])

    def test_report_timing_is_recorded_after_scoring(self) -> None:
        source = "".join(json.loads(NOTEBOOK.read_text())["cells"][1]["source"])
        self.assertLess(
            source.index("summary = score_results("),
            source.index('manifest["timing"]["report_seconds"] ='),
        )

    def test_error_traceback_and_symbolic_fallback_provenance_are_preserved(
        self,
    ) -> None:
        runtime = self.runtime
        symbolic = runtime["REFERENCE"]["load_symbolic_runtime"]()
        task_id = "00000001"
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        result: dict[str, Any] = {
            "task_id": task_id,
            "arms": {
                "inference_only": {
                    "status": "error",
                    "predictions": [],
                    "error": "RuntimeError: worker failed",
                    "traceback": "Traceback: worker failed",
                }
            },
        }

        runtime["REFERENCE"]["_fill_result"](symbolic, task_id, task, result)
        arm: dict[str, Any] = result["arms"]["inference_only"]
        self.assertEqual(arm["status"], "error")
        self.assertEqual(
            arm["diagnostics"]["original_error"], "RuntimeError: worker failed"
        )
        self.assertEqual(
            arm["diagnostics"]["original_traceback"], "Traceback: worker failed"
        )
        self.assertEqual(arm["diagnostics"]["output_sources"], ["symbolic_fallback"])
        self.assertEqual(arm["diagnostics"]["fallback_mask"], [True])

    def test_run_real_merges_worker_error_by_task_ids_without_losing_predictions(
        self,
    ) -> None:
        runtime = self.runtime
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        smoke_id = "00000000"
        pilot_id = "cafebabe"
        prediction = {"attempt_1": [[0]], "attempt_2": [[0]]}

        def fake_supervise(
            assignments: list[list[str]],
            _worker: Any,
            _config: dict[str, Any],
            _clock: Any,
            on_event: Any = None,
        ) -> dict[str, Any]:
            if assignments == [[]]:
                return {
                    "events": [{"kind": "preflight", "status": "complete"}],
                    "hard_timeout": False,
                    "exitcodes": [0],
                }
            if assignments == [[smoke_id]]:
                event = {
                    "kind": "task",
                    "task_id": smoke_id,
                    "status": "complete",
                    "arms": {
                        "inference_only": {
                            "status": "complete",
                            "model_loaded": True,
                            "training_succeeded": True,
                            "predictions": [prediction],
                            "diagnostics": {"adapter_reset_verified": True},
                        }
                    },
                }
                on_event(event)
                return {
                    "events": [event],
                    "hard_timeout": False,
                    "exitcodes": [0],
                }
            on_event(
                {
                    "kind": "arm",
                    "task_id": pilot_id,
                    "arm": "inference_only",
                    "result": {
                        "status": "complete",
                        "predictions": [prediction],
                    },
                }
            )
            on_event(
                {
                    "kind": "worker_error",
                    "task_ids": [pilot_id],
                    "error": "RuntimeError: worker failed",
                    "traceback": "Traceback: worker failed",
                }
            )
            return {"events": [], "hard_timeout": False, "exitcodes": [1]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "arc-agi_evaluation_challenges.json").write_text(
                json.dumps({smoke_id: task, pilot_id: task}), encoding="utf-8"
            )
            (root / "arc-agi_evaluation_solutions.json").write_text(
                json.dumps({smoke_id: [[[0]]], pilot_id: [[[0]]]}), encoding="utf-8"
            )
            (root / "model").mkdir()
            (root / "dependencies").mkdir()
            output = root / "run"
            replacements = {
                "check_python_abi": lambda: None,
                "select_task_ids": lambda *_args, **_kwargs: {
                    "smoke": [smoke_id],
                    "pilot": [pilot_id],
                },
                "discover_model_dir": lambda _root: root / "model",
                "discover_dependency_root": lambda _root: root / "dependencies",
                "model_fingerprint": lambda _path: {"directory": str(root)},
                "dependency_fingerprint": lambda _path: {"root": str(root)},
                "compile_vendored_sources": lambda: {},
                "runtime_source_fingerprint": lambda: "runtime",
                "supervise_workers": fake_supervise,
            }
            with (
                patch.dict(
                    os.environ,
                    {
                        "GPU2_INPUT_ROOT": str(root),
                        "GPU2_MODE": "real",
                    },
                ),
                patch.dict(
                    runtime,
                    {
                        "validate_public_selection": lambda _challenges: {
                            "smoke": [smoke_id],
                            "pilot": [pilot_id],
                        }
                    },
                ),
                patch.dict(runtime["REFERENCE"], replacements),
            ):
                summary = runtime["run_real"](output)

            saved_task = json.loads((output / "tasks" / f"{pilot_id}.json").read_text())
            self.assertEqual(summary["status"], "error")
            self.assertEqual(
                saved_task["arms"]["inference_only"]["predictions"], [prediction]
            )
            self.assertEqual(
                saved_task["worker_errors"][0]["error"],
                "RuntimeError: worker failed",
            )

    def test_run_real_rejects_smoke_error_after_valid_arm_before_pilot(self) -> None:
        runtime = self.runtime
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        prediction = {"attempt_1": [[0]], "attempt_2": [[0]]}

        for error_kind in ("worker_error", "task"):
            with self.subTest(error_kind=error_kind):
                smoke_id = "00000000"
                pilot_id = "cafebabe"
                worker_calls: list[list[list[str]]] = []
                error = f"RuntimeError: smoke {error_kind}"
                traceback = f"Traceback: smoke {error_kind}"
                selection = {"smoke": [smoke_id], "pilot": [pilot_id]}

                def fake_supervise(
                    assignments: list[list[str]],
                    _worker: Any,
                    _config: dict[str, Any],
                    _clock: Any,
                    on_event: Any = None,
                    *,
                    worker_calls: list[list[list[str]]] = worker_calls,
                    smoke_id: str = smoke_id,
                    error_kind: str = error_kind,
                    error: str = error,
                    traceback: str = traceback,
                ) -> dict[str, Any]:
                    worker_calls.append(assignments)
                    if assignments == [[]]:
                        return {
                            "events": [{"kind": "preflight", "status": "complete"}],
                            "hard_timeout": False,
                            "exitcodes": [0],
                        }
                    if assignments != [[smoke_id]]:
                        raise AssertionError("pilot worker must not start")

                    on_event(
                        {
                            "kind": "arm",
                            "task_id": smoke_id,
                            "arm": "inference_only",
                            "result": {
                                "status": "complete",
                                "model_loaded": True,
                                "training_succeeded": True,
                                "predictions": [prediction],
                                "diagnostics": {"adapter_reset_verified": True},
                            },
                        }
                    )
                    if error_kind == "worker_error":
                        on_event(
                            {
                                "kind": "worker_error",
                                "task_ids": [smoke_id],
                                "error": error,
                                "traceback": traceback,
                            }
                        )
                    else:
                        on_event(
                            {
                                "kind": "task",
                                "task_id": smoke_id,
                                "status": "error",
                                "error": error,
                                "traceback": traceback,
                            }
                        )
                    return {"events": [], "hard_timeout": False, "exitcodes": [1]}

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "arc-agi_evaluation_challenges.json").write_text(
                        json.dumps({smoke_id: task, pilot_id: task}), encoding="utf-8"
                    )
                    (root / "model").mkdir()
                    (root / "dependencies").mkdir()
                    output = root / "run"

                    def fail_solution_loading(_root: Path) -> Path:
                        raise AssertionError("solution loading must not start")

                    def fake_select_task_ids(
                        _challenges: dict[str, Any],
                        selection: dict[str, list[str]] = selection,
                    ) -> dict[str, list[str]]:
                        return selection

                    replacements = {
                        "check_python_abi": lambda: None,
                        "select_task_ids": fake_select_task_ids,
                        "discover_model_dir": lambda _root, root=root: root / "model",
                        "discover_dependency_root": lambda _root, root=root: (
                            root / "dependencies"
                        ),
                        "model_fingerprint": lambda _path, root=root: {
                            "directory": str(root)
                        },
                        "dependency_fingerprint": lambda _path, root=root: {
                            "root": str(root)
                        },
                        "compile_vendored_sources": lambda: {},
                        "runtime_source_fingerprint": lambda: "runtime",
                        "supervise_workers": fake_supervise,
                    }
                    with (
                        patch.dict(
                            os.environ,
                            {
                                "GPU2_INPUT_ROOT": str(root),
                                "GPU2_MODE": "real",
                            },
                        ),
                        patch.dict(
                            runtime,
                            {
                                "validate_public_selection": fake_select_task_ids,
                                "discover_evaluation_solutions": fail_solution_loading,
                            },
                        ),
                        patch.dict(runtime["REFERENCE"], replacements),
                    ):
                        with self.assertRaises(runtime["GPU2Error"]):
                            runtime["run_real"](output)

                    manifest = json.loads((output / "manifest.json").read_text())
                    saved_task = json.loads(
                        (output / "tasks" / f"{smoke_id}.json").read_text()
                    )
                    self.assertEqual(worker_calls, [[[]], [[smoke_id]]])
                    self.assertEqual(manifest["stage"], "smoke_error")
                    self.assertEqual(manifest["smoke"]["status"], "error")
                    self.assertEqual(manifest["smoke"]["error"], error)
                    self.assertEqual(
                        saved_task["arms"]["inference_only"]["predictions"],
                        [prediction],
                    )
                    if error_kind == "worker_error":
                        self.assertEqual(saved_task["worker_errors"][0]["error"], error)

    def test_run_real_proceeds_after_completed_zero_candidate_smoke(self) -> None:
        runtime = self.runtime
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}, {"input": [[1]]}],
        }
        smoke_id = "00000000"
        pilot_id = "cafebabe"
        pilot_predictions = [
            {"attempt_1": [[0]], "attempt_2": [[0]]},
            {"attempt_1": [[1]], "attempt_2": [[1]]},
        ]
        worker_calls: list[list[list[str]]] = []

        def fake_supervise(
            assignments: list[list[str]],
            _worker: Any,
            _config: dict[str, Any],
            _clock: Any,
            on_event: Any = None,
        ) -> dict[str, Any]:
            worker_calls.append(assignments)
            if assignments == [[]]:
                return {
                    "events": [{"kind": "preflight", "status": "complete"}],
                    "hard_timeout": False,
                    "exitcodes": [0],
                }
            if assignments == [[smoke_id]]:
                event = {
                    "kind": "task",
                    "task_id": smoke_id,
                    "status": "complete",
                    "arms": {
                        "inference_only": {
                            "status": "complete",
                            "model_loaded": True,
                            "training_succeeded": True,
                            "predictions": [{}, {}],
                            "diagnostics": {
                                "status": "complete",
                                "candidate_count": 0,
                                "invalid_candidates": 0,
                                "fallback_outputs": 2,
                                "adapter_reset_verified": True,
                            },
                        }
                    },
                }
                on_event(event)
                return {"events": [event], "hard_timeout": False, "exitcodes": [0]}
            if assignments == [[pilot_id], [], [], []]:
                event = {
                    "kind": "task",
                    "task_id": pilot_id,
                    "status": "complete",
                    "arms": {
                        "inference_only": {
                            "status": "complete",
                            "model_loaded": True,
                            "training_succeeded": True,
                            "predictions": pilot_predictions,
                            "diagnostics": {
                                "status": "complete",
                                "candidate_count": 1,
                                "invalid_candidates": 0,
                                "fallback_outputs": 0,
                                "adapter_reset_verified": True,
                            },
                        }
                    },
                }
                on_event(event)
                return {"events": [event], "hard_timeout": False, "exitcodes": [0]}
            raise AssertionError(f"unexpected worker assignments: {assignments}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "arc-agi_evaluation_challenges.json").write_text(
                json.dumps({smoke_id: task, pilot_id: task}), encoding="utf-8"
            )
            (root / "arc-agi_evaluation_solutions.json").write_text(
                json.dumps({smoke_id: [[[0]], [[1]]], pilot_id: [[[0]], [[1]]]}),
                encoding="utf-8",
            )
            (root / "model").mkdir()
            (root / "dependencies").mkdir()
            output = root / "run"
            replacements = {
                "check_python_abi": lambda: None,
                "select_task_ids": lambda *_args, **_kwargs: {
                    "smoke": [smoke_id],
                    "pilot": [pilot_id],
                },
                "discover_model_dir": lambda _root: root / "model",
                "discover_dependency_root": lambda _root: root / "dependencies",
                "model_fingerprint": lambda _path: {"directory": str(root)},
                "dependency_fingerprint": lambda _path: {"root": str(root)},
                "compile_vendored_sources": lambda: {},
                "runtime_source_fingerprint": lambda: "runtime",
                "supervise_workers": fake_supervise,
            }
            with (
                patch.dict(
                    os.environ,
                    {"GPU2_INPUT_ROOT": str(root), "GPU2_MODE": "real"},
                ),
                patch.dict(
                    runtime,
                    {
                        "validate_public_selection": lambda _challenges: {
                            "smoke": [smoke_id],
                            "pilot": [pilot_id],
                        }
                    },
                ),
                patch.dict(runtime["REFERENCE"], replacements),
            ):
                summary = runtime["run_real"](output)

            manifest = json.loads((output / "manifest.json").read_text())
            saved_smoke = json.loads(
                (output / "tasks" / f"{smoke_id}.json").read_text()
            )

        self.assertEqual(worker_calls, [[[]], [[smoke_id]], [[pilot_id], [], [], []]])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(manifest["stage"], "complete")
        self.assertTrue(manifest["smoke"]["no_candidate"])
        self.assertTrue(manifest["smoke"]["zero_coverage"])
        self.assertEqual(
            manifest["smoke"]["coverage"],
            {"candidate_count": 0, "invalid_candidates": 0, "fallback_outputs": 2},
        )
        self.assertEqual(
            saved_smoke["arms"]["inference_only"]["diagnostics"]["output_sources"],
            ["symbolic_fallback", "symbolic_fallback"],
        )
        self.assertNotEqual(
            saved_smoke["arms"]["inference_only"]["predictions"], [{}, {}]
        )

    def test_scores_either_and_first_attempt_deltas_with_exact_ids(self) -> None:
        runtime = self.runtime
        symbolic_runtime = runtime["REFERENCE"]["load_symbolic_runtime"]()
        task_id = "00000001"
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}, {"input": [[3]]}],
        }
        challenges = {task_id: task}
        solutions = {task_id: [[[1]], [[3]]]}
        results = {
            task_id: {
                "arms": {
                    "symbolic": {
                        "status": "symbolic_fallback",
                        "predictions": [
                            {"attempt_1": [[1]], "attempt_2": [[1]]},
                            {"attempt_1": [[0]], "attempt_2": [[0]]},
                        ],
                    },
                    "inference_only": {
                        "status": "complete",
                        "predictions": [
                            {"attempt_1": [[0]], "attempt_2": [[0]]},
                            {"attempt_1": [[3]], "attempt_2": [[0]]},
                        ],
                    },
                }
            }
        }

        summary = runtime["score_results"](
            symbolic_runtime, challenges, solutions, results
        )
        delta = summary["comparisons"]["inference_only_vs_symbolic"]
        self.assertEqual(delta["either_attempt"]["added_output_ids"], ["00000001:1"])
        self.assertEqual(delta["either_attempt"]["lost_output_ids"], ["00000001:0"])
        self.assertEqual(delta["first_attempt"]["added_output_ids"], ["00000001:1"])
        self.assertEqual(delta["first_attempt"]["lost_output_ids"], ["00000001:0"])
        self.assertEqual(summary["denominator"], {"tasks": 1, "outputs": 2})

    def test_predictions_are_persisted_before_solution_loading_in_source(self) -> None:
        source = "".join(json.loads(NOTEBOOK.read_text())["cells"][1]["source"])
        self.assertLess(
            source.index("predictions_path ="), source.index("solutions_path =")
        )
        self.assertNotIn("submission.json", source)


if __name__ == "__main__":
    unittest.main()
