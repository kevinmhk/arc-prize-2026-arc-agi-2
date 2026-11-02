import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gpu3_ttt_evaluation.ipynb"


def execute_trusted_cells(namespace: dict[str, Any], stop: int | None = None) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"][1:stop]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def load_runtime() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__name__": "test_gpu3_notebook",
        "__file__": str(NOTEBOOK),
    }
    execute_trusted_cells(namespace, stop=3)
    return namespace


class GPU3NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_runtime()

    def test_ttt_only_policy_and_frozen_budget(self) -> None:
        runtime = self.runtime
        self.assertEqual(runtime["REFERENCE"]["arm_order"](), ("ttt",))
        self.assertEqual(runtime["validate_ttt_policy"](), ("ttt",))
        self.assertEqual(runtime["GPU3_GRACEFUL_SECONDS"], 5100)
        self.assertEqual(runtime["GPU3_HARD_SECONDS"], 5400)
        self.assertEqual(runtime["REFERENCE"]["GRACEFUL_SECONDS"], 5100)
        self.assertEqual(runtime["REFERENCE"]["HARD_SECONDS"], 5400)
        self.assertEqual(runtime["GPU3_CLEANUP_SECONDS"], 30)
        self.assertEqual(runtime["GPU3_GPU_COUNT"], 4)
        self.assertEqual(runtime["GPU3_INFERENCE_SECONDS"], 1200)
        self.assertEqual(len(runtime["GPU3_PILOT_IDS"]), 16)
        self.assertNotIn("581f7754", runtime["GPU3_PILOT_IDS"])

    def test_reference_snapshot_is_hash_verified(self) -> None:
        runtime = self.runtime
        self.assertEqual(
            runtime["GPU3_NOTEBOOK_SHA256"],
            "463829abfea83fb95d7336dc6a4ffb8a7f652e3bcdb3947e840cf502bc7b70e4",
        )
        self.assertEqual(
            runtime["REFERENCE"]["runtime_source_fingerprint"](),
            "bbbd6ca85cc3a763b53b691e9068cc40b10049b410430afe54daa011e89ff1c2",
        )

    def test_clock_reserves_cleanup_from_worker_deadline(self) -> None:
        runtime = self.runtime
        started = 100.0
        clock = runtime["_clock"](started)
        self.assertEqual(clock.graceful, started + 5100)
        self.assertEqual(clock.hard, started + 5400 - 30)

    def test_real_watchdog_uses_reduced_hard_cap(self) -> None:
        runtime = self.runtime
        with (
            patch.dict(os.environ, {"GPU3_MODE": "real"}),
            patch.dict(runtime, {"run_real": lambda _output: {"status": "stubbed"}}),
            patch.object(runtime["signal"], "signal"),
            patch.object(
                runtime["signal"], "setitimer", return_value=(0.0, 0.0)
            ) as watchdog,
        ):
            self.assertEqual(runtime["main"](), {"status": "stubbed"})
        self.assertEqual(
            [call.args for call in watchdog.call_args_list],
            [(runtime["signal"].ITIMER_REAL, 5400), (runtime["signal"].ITIMER_REAL, 0)],
        )

    def test_frozen_baseline_validates_with_actual_symbolic_runtime(self) -> None:
        runtime = self.runtime
        symbolic_runtime = runtime["REFERENCE"]["load_symbolic_runtime"]()
        baseline = runtime["load_symbolic_baseline"]()
        challenges = {
            task_id: {
                "train": [{"input": [[0]], "output": [[0]]}],
                "test": [{"input": [[0]]} for _ in predictions],
            }
            for task_id, predictions in baseline.items()
        }
        self.assertNotIn("validate_submission", runtime["REFERENCE"])
        symbolic_runtime["validate_submission"](challenges, baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenges.json"
            path.write_text(json.dumps(challenges), encoding="utf-8")
            with patch.dict(
                runtime,
                {
                    "GPU3_BASELINE_CHALLENGES_SHA256": runtime["REFERENCE"][
                        "sha256_file"
                    ](path),
                },
            ):
                actual = runtime["validate_baseline_predictions"](
                    symbolic_runtime, challenges, path, list(baseline)
                )
                invalid_challenges = copy.deepcopy(challenges)
                invalid_challenges[next(iter(baseline))]["test"].append(
                    {"input": [[0]]}
                )
                with self.assertRaises(runtime["GPU3Error"]) as caught:
                    runtime["validate_baseline_predictions"](
                        symbolic_runtime, invalid_challenges, path, list(baseline)
                    )
                self.assertIsInstance(
                    caught.exception.__cause__, symbolic_runtime["ValidationError"]
                )
        self.assertEqual(actual, baseline)

    def test_literal_reference_lookups_exist_in_actual_reference(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        names = {
            node.slice.value
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for node in ast.walk(ast.parse("".join(cell["source"])))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "REFERENCE"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        self.assertEqual(names - self.runtime["REFERENCE"].keys(), set())

    def test_cpu_check_does_not_claim_gpu_or_ttt_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.runtime["run_cpu_check"](Path(directory) / "run")
        self.assertEqual(result["mode"], "cpu-check")
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["claims"]["model_loaded"])
        self.assertFalse(result["claims"]["ttt_training"])

    def test_all_notebook_code_cells_materialize_cpu_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            namespace: dict[str, Any] = {
                "__name__": "__main__",
                "__file__": str(NOTEBOOK),
            }
            with patch.dict(
                os.environ,
                {
                    "GPU3_MODE": "cpu-check",
                    "GPU3_OUTPUT_DIR": str(output),
                },
            ):
                execute_trusted_cells(namespace)

            check_path = output / "cpu-check.json"
            self.assertTrue(check_path.exists())
            result = json.loads(check_path.read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "cpu-check")
        self.assertEqual(result["status"], "passed")
        self.assertFalse(any(result["claims"].values()))
        self.assertIn("expected_real_outputs", result)
        self.assertNotIn("schema", result)
        limitations = " ".join(result["limitations"]).lower()
        self.assertIn("output schema", limitations)
        self.assertIn("model", limitations)
        self.assertIn("gpu", limitations)

    def test_real_orchestrator_reuses_first_task_and_persists_fallback_errors(
        self,
    ) -> None:
        runtime = self.runtime
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        ids = list(runtime["GPU3_PILOT_IDS"])
        fallback = runtime["load_symbolic_baseline"]()
        challenges = {
            task_id: {
                **copy.deepcopy(task),
                "test": [{"input": [[0]]} for _ in fallback[task_id]],
            }
            for task_id in ids
        }
        worker_calls: list[list[list[str]]] = []
        error_id = ids[-1]

        def fake_supervise(
            assignments: list[list[str]],
            worker: Any,
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
            if assignments == [[ids[0]]]:
                event = {
                    "kind": "task",
                    "task_id": ids[0],
                    "status": "complete",
                    "arms": {
                        "ttt": {
                            "status": "complete",
                            "model_loaded": True,
                            "training_succeeded": True,
                            "predictions": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
                            "diagnostics": {
                                "adapter_reset_verified": True,
                                "train_stats": {"loss": 1.0},
                            },
                        }
                    },
                }
                on_event(event)
                return {"events": [event], "hard_timeout": False, "exitcodes": [0]}
            self.assertEqual(assignments, [ids[1::4], ids[2::4], ids[3::4], ids[4::4]])
            on_event(
                {
                    "kind": "worker_error",
                    "task_ids": [error_id],
                    "error": "RuntimeError: worker failed",
                    "traceback": "Traceback: worker failed",
                }
            )
            for task_ids in assignments:
                for task_id in task_ids:
                    if task_id == error_id:
                        continue
                    on_event(
                        {
                            "kind": "task",
                            "task_id": task_id,
                            "status": "complete",
                            "arms": {
                                "ttt": {
                                    "status": "complete",
                                    "model_loaded": True,
                                    "training_succeeded": True,
                                    "predictions": copy.deepcopy(fallback[task_id]),
                                    "diagnostics": {
                                        "adapter_reset_verified": True,
                                        "train_stats": {"loss": 1.0},
                                    },
                                }
                            },
                        }
                    )
            return {"events": [], "hard_timeout": False, "exitcodes": [1]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenge_path = root / "arc-agi_evaluation_challenges.json"
            challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
            output = root / "run"
            with (
                patch.dict(
                    os.environ,
                    {"GPU3_INPUT_ROOT": str(root), "GPU3_MODE": "real"},
                ),
                patch.dict(
                    runtime,
                    {
                        "GPU3_BASELINE_CHALLENGES_SHA256": runtime["REFERENCE"][
                            "sha256_file"
                        ](challenge_path),
                        "validate_public_selection": lambda _challenges: {
                            "smoke": [ids[0]],
                            "pilot": ids[1:],
                        },
                        "discover_model_dir": lambda _root: root / "model",
                        "discover_dependency_root": lambda _root: root / "dependencies",
                    },
                ),
                patch.dict(
                    runtime["REFERENCE"],
                    {
                        "check_python_abi": lambda: None,
                        "model_fingerprint": lambda _path: {"directory": str(root)},
                        "dependency_fingerprint": lambda _path: {"root": str(root)},
                        "compile_vendored_sources": lambda: {},
                        "runtime_source_fingerprint": lambda: "runtime",
                        "discover_model_dir": lambda _root: root / "model",
                        "discover_dependency_root": lambda _root: root / "dependencies",
                        "supervise_workers": fake_supervise,
                    },
                ),
            ):
                result = runtime["run_real"](output)

            self.assertEqual(worker_calls[0], [[]])
            self.assertEqual(worker_calls[1], [[ids[0]]])
            self.assertNotIn(ids[0], sum(worker_calls[2], []))
            self.assertFalse(result["answers_loaded"])
            self.assertFalse((output / "submission.json").exists())
            saved = json.loads((output / "tasks" / f"{error_id}.json").read_text())
            self.assertEqual(
                saved["worker_errors"][0]["error"], "RuntimeError: worker failed"
            )
            self.assertEqual(
                saved["arms"]["ttt"]["diagnostics"]["output_sources"],
                ["symbolic_fallback"],
            )
            self.assertTrue((output / "predictions.json").exists())
            self.assertTrue((output / "events.jsonl").exists())

    def test_solution_discovery_is_not_part_of_real_runner(self) -> None:
        source = "".join(
            line
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][1:4]
            for line in (
                cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
            )
        )
        self.assertNotIn("discover_evaluation_solutions", source)


if __name__ == "__main__":
    unittest.main()
