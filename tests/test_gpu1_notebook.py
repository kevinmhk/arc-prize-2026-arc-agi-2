import base64
import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gpu1_nvarc_pilot.ipynb"


def load_definitions() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "test_gpu1_notebook",
        "__file__": str(NOTEBOOK),
    }
    for cell in notebook["cells"][1:4]:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


class GPU1NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_definitions()

    def test_embedded_runtime_matches_production_notebook(self) -> None:
        production = next(
            cell
            for cell in json.loads(
                (ROOT / "notebooks" / "submission.ipynb").read_text(encoding="utf-8")
            )["cells"]
            if cell["cell_type"] == "code"
        )
        source = production["source"]
        if isinstance(source, list):
            source = "".join(source)
        expected = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(self.runtime["runtime_source_fingerprint"](), expected)

    def test_vendored_sources_compile_and_match_captured_hashes(self) -> None:
        hashes = self.runtime["compile_vendored_sources"]()
        self.assertEqual(
            set(hashes),
            {"arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py"},
        )
        for source in self.runtime["VENDORED_SOURCES_B64"].values():
            compile(base64.b64decode(source).decode(), "vendor.py", "exec")
        self.assertEqual(hashes, self.runtime["VENDORED_SOURCE_SHA256"])

    def test_tokenizer_manifest_checks_exact_vocab_and_role_prefixes(self) -> None:
        expected = self.runtime["EXPECTED_TOKENIZER_VOCAB"]

        class Tokenizer:
            pad_token_id = 13
            eos_token_id = 15

            def get_vocab(self) -> dict[str, int]:
                return expected.copy()

            def encode(self, value: str) -> list[int]:
                return {
                    "<|im_start|>user\n": [14, 11, 10],
                    "<|im_start|>assistant\n": [14, 12, 10],
                }[value]

        manifest = self.runtime["_tokenizer_manifest"](Tokenizer())
        self.assertEqual(manifest["encoded_prefixes"]["user"], [14, 11, 10])
        self.assertEqual(manifest["encoded_prefixes"]["assistant"], [14, 12, 10])

        class WrongTokenizer(Tokenizer):
            def encode(self, value: str) -> list[int]:
                return [11, 10] if "user" in value else [14, 12, 10]

        with self.assertRaises(self.runtime["GPU1Error"]):
            self.runtime["_tokenizer_manifest"](WrongTokenizer())

    def test_selection_is_seeded_and_keeps_smoke_separate(self) -> None:
        challenges = {f"{index:08x}": {} for index in range(12)}
        first = self.runtime["select_task_ids"](challenges)
        second = self.runtime["select_task_ids"](challenges)
        self.assertEqual(first, second)
        self.assertEqual(len(first["smoke"]), 1)
        self.assertEqual(len(first["pilot"]), 8)
        self.assertNotIn(first["smoke"][0], first["pilot"])

    def test_missing_input_fails_before_gpu_work(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                self.runtime["MissingInputError"], "training_challenges"
            ):
                self.runtime["discover_training_challenges"](Path(directory))

    def test_dependency_discovery_handles_utility_paths_and_deep_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            inputs = base / "kaggle/input"
            inputs.mkdir(parents=True)
            utility = (
                base
                / "kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch"
            )
            nested = inputs / "notebooks/sorokin/setup/versions/2"
            for root in (utility, nested):
                (root / "unsloth/models").mkdir(parents=True)
                (root / "unsloth/models/qwen3.py").write_text("# discovery fixture\n")
                (root / "unsloth_zoo").mkdir()
            with (
                patch.dict(os.environ, {"GPU1_DEPENDENCY_ROOT": ""}),
                patch.dict(self.runtime, {"sys": types.SimpleNamespace(path=[])}),
                patch.object(
                    self.runtime["importlib"].util,
                    "find_spec",
                    return_value=types.SimpleNamespace(
                        origin=str(utility / "unsloth/__init__.py")
                    ),
                ),
            ):
                self.assertEqual(
                    self.runtime["discover_dependency_root"](base / "no-inputs"),
                    utility,
                )
            with (
                patch.dict(os.environ, {"GPU1_DEPENDENCY_ROOT": ""}),
                patch.dict(
                    self.runtime, {"sys": types.SimpleNamespace(path=[str(utility)])}
                ),
            ):
                empty_inputs = base / "empty-inputs"
                empty_inputs.mkdir()
                self.assertEqual(
                    self.runtime["discover_dependency_root"](empty_inputs), utility
                )
                with self.assertRaisesRegex(
                    self.runtime["MissingInputError"], "expected one"
                ):
                    self.runtime["discover_dependency_root"](inputs)
                with patch.dict(os.environ, {"GPU1_DEPENDENCY_ROOT": str(nested)}):
                    self.assertEqual(
                        self.runtime["discover_dependency_root"](inputs), nested
                    )
            with (
                patch.dict(os.environ, {"GPU1_DEPENDENCY_ROOT": ""}),
                patch.dict(self.runtime, {"sys": types.SimpleNamespace(path=[])}),
            ):
                self.assertEqual(
                    self.runtime["discover_dependency_root"](inputs), nested
                )

    def test_gpu_loader_sets_ptxas_and_process_local_unsloth_cache_before_imports(
        self,
    ) -> None:
        runtime = self.runtime
        ptxas_dirs: list[str] = []
        with TemporaryDirectory() as directory:
            ptxas = Path(directory) / "ptxas"
            ptxas.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ptxas.chmod(0o755)
            dependency_root = Path(directory) / "fake-dependency-root"
            (dependency_root / "unsloth").mkdir(parents=True)
            (dependency_root / "unsloth/__init__.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "assert Path(os.environ['UNSLOTH_COMPILE_LOCATION']).is_dir()\n"
                f"assert os.environ['TRITON_PTXAS_PATH'] == {str(ptxas)!r}\n"
                "READY = True\n",
                encoding="utf-8",
            )
            source = base64.b64encode(
                b"import sys\n"
                b"if not getattr(sys.modules.get('unsloth'), 'READY', False):\n"
                b"    raise RuntimeError('unsloth was not imported first')\n"
            ).decode()
            encoded_sources = {
                f"{name}.py": source
                for name in ("arc_loader", "arc_decoder", "arc_solver")
            }
            original_sources = runtime["VENDORED_SOURCES_B64"]
            original_check_abi = runtime["check_python_abi"]

            def load_child(result_queue: Any) -> None:
                sys.modules.pop("unsloth", None)
                runtime["_load_gpu_modules"](dependency_root)
                result_queue.put(
                    (
                        os.environ["UNSLOTH_COMPILE_LOCATION"],
                        os.environ["TRITON_PTXAS_PATH"],
                    )
                )

            try:
                with (
                    patch.dict(
                        runtime,
                        {
                            "VENDORED_SOURCES_B64": encoded_sources,
                            "check_python_abi": lambda: None,
                        },
                    ),
                    patch.dict(
                        os.environ,
                        {"TRITON_PTXAS_PATH": str(ptxas)},
                        clear=False,
                    ),
                ):
                    with self.assertRaisesRegex(
                        self.runtime["GPU1Error"], "executable ptxas"
                    ):
                        with patch.dict(
                            os.environ,
                            {"TRITON_PTXAS_PATH": str(ptxas) + ".missing"},
                        ):
                            runtime["_load_gpu_modules"](
                                Path(directory) / "fake-dependency-root"
                            )

                    context = mp.get_context("fork")
                    result_queue = context.Queue()
                    processes = [
                        context.Process(target=load_child, args=(result_queue,))
                        for _ in range(2)
                    ]
                    for process in processes:
                        process.start()
                    values = [result_queue.get(timeout=5) for _ in processes]
                    for process in processes:
                        process.join(timeout=5)
                    self.assertTrue(all(process.exitcode == 0 for process in processes))
                    self.assertEqual({value[1] for value in values}, {str(ptxas)})
                    caches = {value[0] for value in values}
                    self.assertEqual(len(caches), 2)
                    self.assertTrue(all(Path(cache).is_dir() for cache in caches))
                    ptxas_dirs.extend(caches)
            finally:
                runtime["VENDORED_SOURCES_B64"] = original_sources
                runtime["check_python_abi"] = original_check_abi
        for cache in ptxas_dirs:
            shutil.rmtree(cache, ignore_errors=True)

    def test_run_real_records_worker_error_per_missing_arm(self) -> None:
        runtime = self.runtime
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        smoke_id = "00000000"
        pilot_id = "cafebabe"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            challenges_path = root / "arc-agi_training_challenges.json"
            solutions_path = root / "arc-agi_training_solutions.json"
            challenges_path.write_text(
                json.dumps({smoke_id: task, pilot_id: task}), encoding="utf-8"
            )
            solutions_path.write_text(
                json.dumps({smoke_id: [[[0]]], pilot_id: [[[0]]]}),
                encoding="utf-8",
            )
            output = root / "run"
            smoke_prediction = {
                "attempt_1": [[0]],
                "attempt_2": [[0]],
            }

            def fake_supervise(
                assignments: list[list[str]],
                worker: Any,
                _config: dict[str, Any],
                _clock: Any,
                on_event: Any = None,
            ) -> dict[str, Any]:
                if worker is runtime["gpu_preflight_worker"]:
                    return {
                        "events": [{"kind": "preflight", "status": "complete"}],
                        "hard_timeout": False,
                        "exitcodes": [0],
                    }
                if worker is runtime["gpu_worker"] and assignments == [[smoke_id]]:
                    return {
                        "events": [
                            {
                                "kind": "task",
                                "task_id": smoke_id,
                                "arms": {
                                    arm: {
                                        "status": "complete",
                                        "model_loaded": True,
                                        "training_succeeded": True,
                                        "predictions": [smoke_prediction],
                                        "diagnostics": {"adapter_reset_verified": True},
                                    }
                                    for arm in runtime["arm_order"]()
                                },
                            }
                        ],
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
                            "predictions": [smoke_prediction],
                        },
                    }
                )
                on_event(
                    {
                        "kind": "worker_error",
                        "task_ids": [pilot_id],
                        "error": "AttributeError: UnslothKTOTrainer",
                        "traceback": "Traceback (most recent call last):\nworker boom",
                    }
                )
                return {"events": [], "hard_timeout": False, "exitcodes": [1]}

            replacements = {
                "check_python_abi": lambda: None,
                "select_task_ids": lambda _challenges: {
                    "smoke": [smoke_id],
                    "pilot": [pilot_id],
                },
                "discover_model_dir": lambda _root: root,
                "discover_dependency_root": lambda _root: root,
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
                        "GPU1_INPUT_ROOT": str(root),
                        "GPU1_MODE": "real",
                    },
                ),
                patch.dict(runtime, replacements),
            ):
                summary = runtime["run_real"](output)

            saved_summary = json.loads((output / "summary.json").read_text())
            saved_task = json.loads((output / "tasks" / f"{pilot_id}.json").read_text())
            self.assertEqual(summary["status"], "error")
            self.assertEqual(saved_summary["status"], "error")
            inference = saved_task["arms"]["inference_only"]
            failed = saved_task["arms"]["ttt"]
            self.assertEqual(inference["status"], "complete")
            self.assertEqual(inference["predictions"], [smoke_prediction])
            self.assertEqual(failed["status"], "error")
            self.assertEqual(
                failed["diagnostics"]["original_error"],
                "AttributeError: UnslothKTOTrainer",
            )
            self.assertEqual(
                failed["diagnostics"]["original_traceback"],
                "Traceback (most recent call last):\nworker boom",
            )
            self.assertEqual(failed["diagnostics"]["model_outputs"], 0)
            self.assertTrue(failed["diagnostics"]["fallback_mask"])
            report = saved_summary["by_task"][pilot_id]["ttt"]["diagnostics"]
            self.assertEqual(
                report["original_error"], failed["diagnostics"]["original_error"]
            )
            self.assertEqual(
                report["original_traceback"],
                failed["diagnostics"]["original_traceback"],
            )

    def test_real_run_rejects_wrong_python_before_creating_output(self) -> None:
        def unsupported_python() -> None:
            raise self.runtime["GPU1Error"]("ABI mismatch")

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.dict(self.runtime, {"check_python_abi": unsupported_python}):
                with self.assertRaisesRegex(self.runtime["GPU1Error"], "ABI mismatch"):
                    self.runtime["run_real"](output)
            self.assertFalse(output.exists())

    def test_adapter_reset_copies_state_and_arm_switch_is_fixed(self) -> None:
        captured = []
        weights = {"adapter": {"value": 1}}
        self.runtime["reset_adapter_state"](captured.append, weights)
        captured[0]["adapter"]["value"] = 9
        self.assertEqual(weights["adapter"]["value"], 1)
        self.assertEqual(self.runtime["arm_order"](), ("inference_only", "ttt"))
        with self.assertRaises(self.runtime["GPU1Error"]):
            self.runtime["validate_condition_switch"](("ttt", "inference_only"))

    def test_cpu_checks_cover_multioutput_and_real_child_supervision(self) -> None:
        checks = self.runtime["cpu_transform_checks"]()
        self.assertEqual(checks["score_denominator"], 2)
        self.assertEqual(checks["arms"], ("inference_only", "ttt"))
        supervision = self.runtime["fake_supervision_check"]()
        self.assertEqual(supervision["completed"], ["fast"])
        self.assertTrue(supervision["hard_timeout"])

    def test_run_gpu_task_converts_monotonic_deadline_for_decode(self) -> None:
        runtime = self.runtime
        calls: list[float] = []

        class Cuda:
            def reset_peak_memory_stats(self) -> None:
                pass

            def max_memory_allocated(self) -> int:
                return 0

            def synchronize(self) -> None:
                pass

        class Torch:
            cuda = Cuda()

            @staticmethod
            def inference_mode() -> Any:
                class Context:
                    def __enter__(self) -> None:
                        return None

                    def __exit__(self, *_args: Any) -> None:
                        return None

                return Context()

        class Model:
            state = {"weight": 1}

        class Formatter:
            def __init__(self, _tokenizer: Any) -> None:
                pass

            def max_new_tokens(self) -> int:
                return 1

        class Dataset:
            keys = ["task_0"]
            queries = {
                "task_0": {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[0]]}],
                }
            }

            @classmethod
            def from_file(cls, _path: str, keys: list[str]) -> "Dataset":
                return cls()

            def change_keys(self, _keys: list[str]) -> "Dataset":
                return self

            def split_multi_replies(self) -> "Dataset":
                return self

            def augment(self, **_kwargs: Any) -> "Dataset":
                return self

            def cut_to_len(self, **_kwargs: Any) -> "Dataset":
                return self

            def get(self, _key: str, _formatter: Formatter) -> dict[str, str]:
                return {"input": "x"}

            def invert_mod(self, grid: Any, _key: str, inv_perm: bool = True) -> Any:
                return grid

        class Solver:
            torch = Torch()
            FastLanguageModel = types.SimpleNamespace(
                for_inference=staticmethod(lambda model: model)
            )

            @staticmethod
            def set_peft_model_state_dict(
                model: Model, value: dict[str, int], **_kwargs: Any
            ) -> None:
                model.state = value

            @staticmethod
            def get_peft_model_state_dict(
                model: Model, **_kwargs: Any
            ) -> dict[str, int]:
                return model.state

            @staticmethod
            def inference_turbo_dfs(*_args: Any) -> list[Any]:
                calls.append(time.time())
                return []

        modules = {
            "arc_loader": types.SimpleNamespace(
                ArcDataset=Dataset, QwenFormatter=Formatter
            ),
            "arc_solver": Solver,
            "arc_decoder": types.SimpleNamespace(score_kgmon=lambda values: values),
        }
        original_train = runtime["_train_arm"]
        runtime["_train_arm"] = lambda model, *_args: (model, {"train_seconds": 0.0})
        try:
            result = runtime["run_gpu_task"](
                Model(),
                types.SimpleNamespace(encode=lambda value: [1]),
                modules,
                {"weight": 1},
                Path("unused.json"),
                "task",
                time.monotonic() + 10,
            )
        finally:
            runtime["_train_arm"] = original_train
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(result["arms"]), {"inference_only", "ttt"})

        persisted: list[str] = []

        def fail_training(*_args: Any) -> Any:
            raise RuntimeError("synthetic TTT failure")

        runtime["_train_arm"] = fail_training
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic TTT failure"):
                runtime["run_gpu_task"](
                    Model(),
                    types.SimpleNamespace(encode=lambda value: [1]),
                    modules,
                    {"weight": 1},
                    Path("unused.json"),
                    "task",
                    time.monotonic() + 10,
                    on_arm=lambda arm, _output: persisted.append(arm),
                )
        finally:
            runtime["_train_arm"] = original_train
        self.assertEqual(persisted, ["inference_only"])

        calls.clear()
        runtime["_train_arm"] = lambda model, *_args: (model, {"train_seconds": 0.0})
        try:
            expired = runtime["run_gpu_task"](
                Model(),
                types.SimpleNamespace(encode=lambda value: [1]),
                modules,
                {"weight": 1},
                Path("unused.json"),
                "task",
                time.monotonic() - 1,
            )
        finally:
            runtime["_train_arm"] = original_train
        self.assertEqual(calls, [])
        self.assertEqual(
            [expired["arms"][arm]["status"] for arm in runtime["arm_order"]()],
            ["timeout", "timeout"],
        )

    def test_fill_result_validates_each_output_and_preserves_arm_status(self) -> None:
        runtime = self.runtime
        task_id = "cafebabe"
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}, {"input": [[3]]}],
        }
        valid = {"attempt_1": [[9]], "attempt_2": [[9]]}
        result = {
            "task_id": task_id,
            "arms": {
                "inference_only": {
                    "status": "timeout",
                    "reason": "deadline",
                    "predictions": [valid],
                },
                "ttt": {"status": "complete", "predictions": []},
            },
        }
        filled = runtime["_fill_result"](
            runtime["load_symbolic_runtime"](), task_id, task, result
        )
        inference = filled["arms"]["inference_only"]
        self.assertEqual(inference["status"], "timeout")
        self.assertEqual(
            inference.get("diagnostics", {}).get("fallback_mask"), [False, True]
        )
        self.assertEqual(inference["predictions"][0], valid)
        self.assertEqual(len(filled["arms"]["ttt"]["predictions"]), 2)

    def test_score_results_excludes_incomplete_arms_from_paired_denominator(
        self,
    ) -> None:
        runtime = self.runtime
        task_id = "cafebabe"
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        results = {
            task_id: {
                "arms": {
                    "symbolic": {
                        "status": "symbolic_fallback",
                        "predictions": [{"attempt_1": [[2]], "attempt_2": [[2]]}],
                    },
                    "inference_only": {"status": "timeout", "predictions": []},
                    "ttt": {"status": "error", "predictions": []},
                }
            }
        }
        summary = runtime["score_results"](
            runtime["load_symbolic_runtime"](),
            {task_id: task},
            {task_id: [[[2]]]},
            results,
        )
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["paired_completed"]["task_ids"], [])
        self.assertEqual(summary["full8"]["inference_only"]["total_outputs"], 1)

    def test_fill_result_rejects_wrong_count_colors_shapes_and_bool_grids(self) -> None:
        runtime = self.runtime
        task_id = "cafebabe"
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        invalid_predictions = (
            [],
            [{"attempt_1": [[10]], "attempt_2": [[0]]}],
            [{"attempt_1": [[0], [1, 0]], "attempt_2": [[0]]}],
            [{"attempt_1": [[True]], "attempt_2": [[0]]}],
        )
        for predictions in invalid_predictions:
            with self.subTest(predictions=predictions):
                filled = runtime["_fill_result"](
                    runtime["load_symbolic_runtime"](),
                    task_id,
                    task,
                    {
                        "arms": {
                            "inference_only": {
                                "status": "complete",
                                "predictions": predictions,
                            },
                            "ttt": {
                                "status": "complete",
                                "predictions": predictions,
                            },
                        }
                    },
                )
                for arm in runtime["arm_order"]():
                    entry = filled["arms"][arm]
                    self.assertEqual(entry["diagnostics"]["fallback_mask"], [True])
                    self.assertEqual(entry["diagnostics"]["model_outputs"], 0)

    def test_supervisor_kills_sigterm_ignoring_process_group(self) -> None:
        def stubborn_worker(
            _rank: int,
            _tasks: list[str],
            result_queue: Any,
            _config: dict[str, Any],
            _started: float,
        ) -> None:
            self.runtime["_child_session"]()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import signal,time; signal.signal(signal.SIGTERM, "
                        "signal.SIG_IGN); time.sleep(30)"
                    ),
                ]
            )
            result_queue.put({"kind": "child", "pid": child.pid})
            while True:
                time.sleep(0.1)

        started = time.monotonic()
        clock = self.runtime["Deadline"](started)
        clock.hard = started + 0.3
        result = self.runtime["supervise_workers"]([[]], stubborn_worker, {}, clock)
        child_pid = next(
            event["pid"] for event in result["events"] if event["kind"] == "child"
        )
        self.assertIsNotNone(result["events"])
        self.assertTrue(all(code is not None for code in result["exitcodes"]))
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_scoring_keeps_existing_fallback_provenance(self) -> None:
        task_id = "cafebabe"
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        runtime = self.runtime["load_symbolic_runtime"]()
        results: dict[str, Any] = {
            task_id: {
                "arms": {
                    "inference_only": {"status": "timeout", "predictions": []},
                    "ttt": {"status": "error", "predictions": []},
                }
            }
        }
        self.runtime["_fill_result"](runtime, task_id, task, results[task_id])
        before = copy.deepcopy(results[task_id]["arms"])
        self.runtime["score_results"](
            runtime, {task_id: task}, {task_id: [[[0]]]}, results
        )
        for arm in self.runtime["arm_order"]():
            self.assertEqual(
                results[task_id]["arms"][arm]["diagnostics"],
                before[arm]["diagnostics"],
            )
            self.assertEqual(
                results[task_id]["arms"][arm]["diagnostics"]["model_outputs"], 0
            )

    def test_reference_rescoring_keeps_microbatches_at_four(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__dict__["AutoTokenizer"] = object
        previous = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        loader = types.ModuleType("cpu_loader")
        decoder = types.ModuleType("cpu_decoder")
        try:
            for module, name in (
                (loader, "arc_loader.py"),
                (decoder, "arc_decoder.py"),
            ):
                source = base64.b64decode(self.runtime["VENDORED_SOURCES_B64"][name])
                exec(compile(source, name, "exec"), module.__dict__)
        finally:
            if previous is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = previous

        class Tokenizer:
            def encode(self, _value: str) -> list[int]:
                return [0]

            def decode(self, _tokens: list[int]) -> str:
                return "1\n2"

        sizes: list[int] = []

        def calc_scores(
            queries: list[str], answers: list[str], *_args: Any
        ) -> list[float]:
            self.assertEqual(len(queries), len(answers))
            sizes.append(len(queries))
            return [0.0] * len(queries)

        solver = types.SimpleNamespace(
            torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext),
            inference_turbo_dfs=lambda _model, tokens, *_args: [
                (index, [(0.0, [1, 15])]) for index in range(len(tokens))
            ],
            calc_scores=calc_scores,
        )
        task_id = "cafebabe"
        dataset = loader.__dict__["ArcDataset"](
            {
                task_id: {
                    "train": [{"input": [[1]], "output": [[1]]}],
                    "test": [{"input": [[1]]}],
                }
            }
        )
        predictions, _diagnostics = self.runtime["_decode_arm"](
            object(),
            dataset,
            Tokenizer(),
            {"arc_loader": loader, "arc_decoder": decoder, "arc_solver": solver},
            time.time() + 30,
        )
        self.assertTrue(predictions)
        self.assertTrue(sizes)
        self.assertTrue(all(size <= 4 for size in sizes), sizes)

    def test_total_watchdog_interrupts_synchronous_setup(self) -> None:
        original_seconds = self.runtime["HARD_SECONDS"]
        original_run = self.runtime["run_real"]
        old_mode = os.environ.get("GPU1_MODE")
        os.environ["GPU1_MODE"] = "real"
        self.runtime["HARD_SECONDS"] = 0.05
        self.runtime["run_real"] = lambda _output: time.sleep(0.5)
        try:
            with self.assertRaisesRegex(self.runtime["GPU1Error"], "hard deadline"):
                self.runtime["main"]()
        finally:
            self.runtime["HARD_SECONDS"] = original_seconds
            self.runtime["run_real"] = original_run
            if old_mode is None:
                os.environ.pop("GPU1_MODE", None)
            else:
                os.environ["GPU1_MODE"] = old_mode

    def test_score_report_includes_added_and_lost_arm_deltas(self) -> None:
        task_id = "cafebabe"
        task = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        solution = {task_id: [[[1]]]}

        def prediction(grid: list[list[int]]) -> list[dict[str, list[list[int]]]]:
            return [{"attempt_1": grid, "attempt_2": grid}]

        results = {
            task_id: {
                "arms": {
                    "symbolic": {
                        "status": "symbolic_fallback",
                        "predictions": prediction([[1]]),
                    },
                    "inference_only": {
                        "status": "complete",
                        "predictions": prediction([[0]]),
                    },
                    "ttt": {"status": "complete", "predictions": prediction([[1]])},
                }
            }
        }
        summary = self.runtime["score_results"](
            self.runtime["load_symbolic_runtime"](),
            {task_id: task},
            solution,
            results,
        )
        self.assertEqual(
            summary["full8_deltas"]["inference_only_vs_symbolic"],
            {"added_outputs": 0, "lost_outputs": 1, "added_tasks": 0, "lost_tasks": 1},
        )
        self.assertEqual(
            summary["full8_deltas"]["ttt_vs_inference_only"],
            {"added_outputs": 1, "lost_outputs": 0, "added_tasks": 1, "lost_tasks": 0},
        )


if __name__ == "__main__":
    unittest.main()
