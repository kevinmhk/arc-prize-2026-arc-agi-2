import ast
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gpu_submission.ipynb"
OLD_HASHES = {
    "submission.ipynb": (
        "6866cf2b7c69ce38a9de2dea1bcb225cd5a39d20c8342c26f08e5c494c766028"
    ),
    "gpu1_nvarc_pilot.ipynb": (
        "463829abfea83fb95d7336dc6a4ffb8a7f652e3bcdb3947e840cf502bc7b70e4"
    ),
    "gpu2_public_evaluation.ipynb": (
        "a1faac5fc44cde367fe8675aa001e9cf4da0c7711b4eb08fa072e547607b6bef"
    ),
    "gpu3_ttt_evaluation.ipynb": (
        "f44b8b763684687c55347b71b6683fad72b879eabded6f37c7fe4b2f0dfa7399"
    ),
}


def execute_cells(entrypoint: bool = False) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells if entrypoint else cells[:-1]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def task(count: int = 1, width: int = 1) -> dict[str, Any]:
    return {
        "train": [{"input": [[1] * width], "output": [[1] * width]}],
        "test": [{"input": [[2] * width]} for _ in range(count)],
    }


def write_inputs(root: Path, challenges: dict[str, Any]) -> Path:
    path = root / "competitions/arc-prize-2025/arc-agi_test_challenges.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(challenges))
    (path.parent / "arc-agi_test_solutions.json").write_text("DO NOT READ")
    (path.parent / "arc-agi_evaluation_challenges.json").write_text("DO NOT READ")
    return path


def gpu_assets(root: Path) -> dict[str, str]:
    model = root / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "weights.safetensors"):
        (model / name).write_text("{}")
    dependencies = root / "dependencies"
    (dependencies / "unsloth/models").mkdir(parents=True)
    (dependencies / "unsloth/models/qwen3.py").write_text("# fixture\n")
    (dependencies / "unsloth_zoo").mkdir()
    return {
        "GPU1_MODEL_DIR": str(model),
        "GPU1_DEPENDENCY_ROOT": str(dependencies),
        "GPU_SUBMISSION_INPUT_ROOT": str(root),
    }


class GPUSubmissionNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = execute_cells()

    def test_frozen_reference_policy_caps_and_old_notebooks(self) -> None:
        runtime = self.runtime
        self.assertEqual(runtime["SELECTED_ARM"], "inference_only")
        reference = runtime["REFERENCE"]
        self.assertEqual(reference["arm_order"](), ("inference_only",))
        self.assertNotIn("validate_submission", reference)
        self.assertEqual(
            reference["runtime_source_fingerprint"](),
            "bbbd6ca85cc3a763b53b691e9068cc40b10049b410430afe54daa011e89ff1c2",
        )
        reference["compile_vendored_sources"]()
        for mode, expected in (
            ("real", (41400, 42600, 120)),
            ("smoke", (900, 1200, 30)),
        ):
            with self.subTest(mode=mode):
                clock = runtime["configure_deadline"](mode, 100.0)
                graceful, hard, reserve = expected
                self.assertEqual(clock.graceful, 100 + graceful)
                self.assertEqual(clock.hard, 100 + hard - reserve)
                self.assertEqual(reference["GRACEFUL_SECONDS"], graceful)
                self.assertEqual(reference["HARD_SECONDS"], hard - reserve)
                self.assertEqual(reference["INFERENCE_SECONDS"], 1200)
                self.assertEqual(reference["GPU_COUNT"], 4)
        for name, digest in OLD_HASHES.items():
            self.assertEqual(
                hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest(),
                digest,
            )
        sources = [
            "".join(c["source"])
            for c in json.loads(NOTEBOOK.read_text())["cells"]
            if c["cell_type"] == "code"
        ]
        lookups = {
            node.slice.value
            for source in sources
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "REFERENCE"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        self.assertEqual(lookups - reference.keys(), set())
        self.assertEqual(
            ast.dump(ast.parse(sources[-1])), ast.dump(ast.parse("main()"))
        )

    def test_all_cells_cpu_entrypoint_nested_full_schema_without_answers(self) -> None:
        challenges = {"abcdef01": task(2), "abcdef02": task()}
        original_open = Path.open
        original_rglob = Path.rglob

        def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            if "solutions" in path.name:
                raise AssertionError("solutions must not be opened")
            return original_open(path, *args, **kwargs)

        def guarded_rglob(path: Path, pattern: str) -> Any:
            self.assertEqual(pattern, "arc-agi_test_challenges.json")
            return original_rglob(path, pattern)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, challenges)
            output = root / "output"
            with (
                redirect_stdout(io.StringIO()) as progress,
                patch.dict(
                    os.environ,
                    {
                        "GPU_SUBMISSION_MODE": "cpu-check",
                        "GPU_SUBMISSION_INPUT_ROOT": str(root),
                        "GPU_SUBMISSION_OUTPUT_DIR": str(output),
                    },
                ),
                patch.object(Path, "open", guarded_open),
                patch.object(Path, "rglob", guarded_rglob),
            ):
                namespace = execute_cells(entrypoint=True)
            self.assertNotIn("arms: inference_only, ttt", progress.getvalue())
            submission = json.loads((output / "submission.json").read_text())
            manifest = json.loads((output / "gpu-submission/manifest.json").read_text())
        symbolic = namespace["REFERENCE"]["load_symbolic_runtime"]()
        symbolic["validate_submission"](challenges, submission)
        self.assertEqual(set(submission), set(challenges))
        self.assertEqual(len(submission["abcdef01"]), 2)
        self.assertEqual(manifest["mode"], "cpu-check")
        self.assertFalse(any(manifest["claims"].values()))
        self.assertEqual(manifest["model_outputs"], 0)
        self.assertEqual(manifest["fallback_outputs"], 3)

    def test_rejects_missing_ambiguous_invalid_and_duplicate_input(self) -> None:
        runtime = self.runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "arc-agi_evaluation_challenges.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "expected one"):
                runtime["discover_test_challenges"](root)
            path = write_inputs(root, {"abcdef01": task()})
            duplicate = root / path.name
            duplicate.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "expected one"):
                runtime["discover_test_challenges"](root)
            duplicate.unlink()
            for invalid in (
                {},
                {"wrong-id": task()},
                {"abcdef01": {**task(), "test": [{"input": [[True]]}]}},
                {"abcdef01": {**task(), "test": [{"input": [[0]], "output": [[0]]}]}},
            ):
                with self.subTest(invalid=invalid):
                    path.write_text(json.dumps(invalid))
                    with patch.dict(
                        os.environ, {"GPU_SUBMISSION_INPUT_ROOT": str(root)}
                    ):
                        with self.assertRaises(ValueError):
                            runtime["run_submission"]("cpu-check", root / "output")
            path.write_text(
                '{"abcdef01":'
                + json.dumps(task())
                + ',"abcdef01":'
                + json.dumps(task())
                + "}"
            )
            with patch.dict(os.environ, {"GPU_SUBMISSION_INPUT_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    runtime["run_submission"]("cpu-check", root / "output")
            self.assertFalse((root / "output/submission.json").exists())

    def test_real_orchestrator_atomic_updates_cached_fallback_and_later_error(
        self,
    ) -> None:
        runtime = self.runtime
        reference = runtime["REFERENCE"]
        challenges = {f"{i:08x}": task(2 if i == 0 else 1) for i in range(5)}
        predictions = [{"attempt_1": [[8]], "attempt_2": [[9]]}, {}]
        calls = []
        original_fallback = reference["fallback_predictions"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, challenges)
            output = root / "output"
            symbolic = reference["load_symbolic_runtime"]()

            def read_submission() -> dict[str, Any]:
                value = json.loads((output / "submission.json").read_text())
                symbolic["validate_submission"](challenges, value)
                return value

            def supervise(
                assignments: Any,
                worker: Any,
                config: Any,
                clock: Any,
                on_event: Any = None,
            ) -> dict[str, Any]:
                calls.append(assignments)
                before = read_submission()
                if worker is reference["gpu_preflight_worker"]:
                    self.assertEqual(assignments, [[]])
                    self.assertTrue(
                        all(p["attempt_1"] != [[8]] for v in before.values() for p in v)
                    )
                    return {
                        "events": [{"kind": "preflight", "status": "complete"}],
                        "hard_timeout": False,
                        "exitcodes": [0],
                    }
                self.assertEqual(
                    assignments,
                    [
                        ["00000000", "00000004"],
                        ["00000001"],
                        ["00000002"],
                        ["00000003"],
                    ],
                )
                self.assertEqual(
                    config["training_path"],
                    str(runtime["discover_test_challenges"](root)),
                )
                on_event(
                    {
                        "kind": "arm",
                        "task_id": "00000000",
                        "arm": "inference_only",
                        "result": {
                            "status": "complete",
                            "model_loaded": True,
                            "predictions": copy.deepcopy(predictions),
                            "memory": {"arm_peak_device_mb": 12},
                        },
                    }
                )
                updated = read_submission()
                self.assertEqual(updated["00000000"][0], predictions[0])
                self.assertEqual(updated["00000000"][1], before["00000000"][1])
                on_event(
                    {
                        "kind": "worker_error",
                        "task_ids": ["00000000", "00000004"],
                        "error": "RuntimeError: later failure",
                        "traceback": "trace",
                    }
                )
                self.assertEqual(read_submission(), updated)
                on_event(
                    {
                        "kind": "task",
                        "task_id": "00000001",
                        "status": "error",
                        "error": "task failed",
                        "arms": {},
                    }
                )
                return {"events": [], "hard_timeout": False, "exitcodes": [1, 0, 0, 0]}

            with (
                patch.dict(os.environ, gpu_assets(root)),
                patch.dict(reference, {"supervise_workers": supervise}),
                patch.dict(
                    reference, {"fallback_predictions": Mock(wraps=original_fallback)}
                ),
            ):
                fallback_spy = reference["fallback_predictions"]
                result = runtime["run_submission"]("real", output)
                self.assertEqual(fallback_spy.call_count, len(challenges))
            final = read_submission()
            saved = json.loads(
                (output / "gpu-submission/tasks/00000000.json").read_text()
            )
            self.assertEqual(
                saved["arms"]["inference_only"]["diagnostics"]["fallback_mask"],
                [False, True],
            )
            self.assertEqual(
                saved["worker_errors"][0]["error"], "RuntimeError: later failure"
            )
            self.assertEqual(final["00000000"][0], predictions[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model_outputs"], 1)
        self.assertEqual(result["fallback_outputs"], 5)
        self.assertEqual(result["peak_device_mb"], 12)
        self.assertTrue(result["errors"])

    def test_preflight_failure_or_deadline_stops_model_workers(self) -> None:
        runtime = self.runtime
        reference = runtime["REFERENCE"]
        for failure in ("error", "deadline", "exception"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                challenges = {"abcdef01": task()}
                write_inputs(root, challenges)
                calls = []

                def supervise(
                    assignments: Any,
                    worker: Any,
                    *args: Any,
                    calls: list[Any] = calls,
                    failure: str = failure,
                ) -> dict[str, Any]:
                    calls.append(assignments)
                    self.assertIs(worker, reference["gpu_preflight_worker"])
                    if failure == "exception":
                        raise TimeoutError("synchronous setup deadline")
                    return {
                        "events": [
                            {
                                "kind": "preflight",
                                "status": "error" if failure == "error" else "complete",
                            }
                        ],
                        "hard_timeout": failure == "deadline",
                        "exitcodes": [0],
                    }

                with (
                    patch.dict(os.environ, gpu_assets(root)),
                    patch.dict(reference, {"supervise_workers": supervise}),
                ):
                    result = runtime["run_submission"]("real", root / "output")
                value = json.loads((root / "output/submission.json").read_text())
                reference["load_symbolic_runtime"]()["validate_submission"](
                    challenges, value
                )
                self.assertEqual(calls, [[[]]])
                self.assertNotEqual(result["status"], "complete")
                self.assertFalse(result["claims"]["model_loaded"])
                self.assertTrue(result["errors"])

    def test_fresh_project_kernels_from_root_and_detached_directory(self) -> None:
        import nbformat
        from jupyter_client import KernelManager
        from nbclient import NotebookClient

        challenges = {"abcdef01": task(2)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, challenges)
            for index, cwd in enumerate((ROOT, root)):
                with self.subTest(cwd=cwd):
                    output = root / f"output-{index}"
                    notebook = nbformat.read(NOTEBOOK, as_version=4)
                    expected = sys.executable
                    prefix = str(Path(expected).parent.parent)
                    notebook.cells.append(
                        nbformat.v4.new_code_cell(
                            f"assert sys.executable == {expected!r}\n"
                            f"assert sys.prefix == {prefix!r}\n"
                            "assert 'PYTHONPATH' not in os.environ\n"
                            f"output = Path({str(output)!r})\n"
                            "path = output / 'submission.json'\n"
                            "saved = json.loads(path.read_text())\n"
                            "validator = REFERENCE['load_symbolic_runtime']()\n"
                            f"validator['validate_submission']({challenges!r}, saved)\n"
                            "assert len(saved['abcdef01']) == 2"
                        )
                    )
                    manager = KernelManager(kernel_name="python3")
                    spec = manager.kernel_spec
                    assert spec is not None
                    spec.argv[0] = sys.executable
                    env = dict(os.environ)
                    env.pop("PYTHONPATH", None)
                    env.update(
                        {
                            "GPU_SUBMISSION_MODE": "cpu-check",
                            "GPU_SUBMISSION_INPUT_ROOT": str(root),
                            "GPU_SUBMISSION_OUTPUT_DIR": str(output),
                        }
                    )
                    try:
                        manager.start_kernel(cwd=str(cwd), env=env)
                        NotebookClient(notebook, km=manager, timeout=60).execute(
                            cleanup_kc=True
                        )
                    finally:
                        if manager.has_kernel:
                            manager.shutdown_kernel(now=True)
                    self.runtime["REFERENCE"]["load_symbolic_runtime"]()[
                        "validate_submission"
                    ](challenges, json.loads((output / "submission.json").read_text()))

    def test_smoke_uses_two_smallest_contexts_and_timeout_keeps_full_output(
        self,
    ) -> None:
        runtime = self.runtime
        reference = runtime["REFERENCE"]
        challenges = {
            "00000004": task(width=4),
            "00000002": task(),
            "00000003": task(2),
            "00000001": task(),
        }
        calls = []

        def supervise(
            assignments: Any, worker: Any, config: Any, clock: Any, on_event: Any = None
        ) -> dict[str, Any]:
            calls.append(assignments)
            if worker is reference["gpu_preflight_worker"]:
                return {
                    "events": [{"kind": "preflight", "status": "complete"}],
                    "hard_timeout": False,
                    "exitcodes": [0],
                }
            self.assertEqual(assignments, [["00000001"], ["00000002"], [], []])
            return {"events": [], "hard_timeout": True, "exitcodes": [-15] * 4}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, challenges)
            with (
                patch.dict(os.environ, gpu_assets(root)),
                patch.dict(reference, {"supervise_workers": supervise}),
            ):
                result = runtime["run_submission"]("smoke", root / "output")
            submission = json.loads((root / "output/submission.json").read_text())
        reference["load_symbolic_runtime"]()["validate_submission"](
            challenges, submission
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["selected_task_ids"], ["00000001", "00000002"])
        self.assertEqual(result["status"], "deadline_partial")
        self.assertEqual(result["fallback_outputs"], 5)
        self.assertTrue(result["errors"])

    def test_setup_failure_retains_fallback_and_reports_no_model_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenges = {"abcdef01": task(2)}
            write_inputs(root, challenges)
            with patch.dict(
                os.environ,
                {
                    "GPU_SUBMISSION_INPUT_ROOT": str(root),
                    "GPU1_MODEL_DIR": str(root / "missing"),
                },
            ):
                result = self.runtime["run_submission"]("real", root / "output")
            value = json.loads((root / "output/submission.json").read_text())
        self.runtime["REFERENCE"]["load_symbolic_runtime"]()["validate_submission"](
            challenges, value
        )
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["claims"]["model_loaded"])
        self.assertTrue(result["errors"])

    def test_real_default_and_watchdog_includes_setup(self) -> None:
        runtime = self.runtime
        for mode, seconds in (("real", 42600), ("smoke", 1200)):
            with (
                patch.dict(os.environ, {"GPU_SUBMISSION_MODE": mode}),
                patch.dict(
                    runtime,
                    {"run_submission": lambda *_args, mode=mode: {"mode": mode}},
                ),
                patch.object(runtime["signal"], "signal"),
                patch.object(
                    runtime["signal"], "setitimer", return_value=(0.0, 0.0)
                ) as timer,
            ):
                self.assertEqual(runtime["main"]()["mode"], mode)
            self.assertEqual(
                [c.args for c in timer.call_args_list],
                [
                    (runtime["signal"].ITIMER_REAL, seconds),
                    (runtime["signal"].ITIMER_REAL, 0),
                ],
            )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                runtime, {"run_submission": lambda mode, *_args: {"mode": mode}}
            ),
            patch.object(runtime["signal"], "signal"),
            patch.object(runtime["signal"], "setitimer", return_value=(0.0, 0.0)),
        ):
            self.assertEqual(runtime["main"]()["mode"], "real")
        with patch.dict(os.environ, {"GPU_SUBMISSION_MODE": "evaluation"}):
            with self.assertRaises(ValueError):
                runtime["main"]()


if __name__ == "__main__":
    unittest.main()
