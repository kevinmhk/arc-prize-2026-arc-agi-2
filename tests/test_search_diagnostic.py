import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import scripts.diagnose_search as diagnostic
from scripts.diagnose_search import (
    CONFIGS,
    load_or_freeze,
    load_runtime_namespace,
    make_diagnostic_generator,
    run_runtime_task,
)


class SearchDiagnosticTests(unittest.TestCase):
    def test_deeper_search_has_depth_three_and_preserves_prefix_order(self) -> None:
        runtime = load_runtime_namespace()
        task = {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }
        production_generator = runtime["generate_programs"]
        baseline = production_generator(task)
        deeper_generator = make_diagnostic_generator(
            runtime,
            production_generator,
            CONFIGS["deeper_depth3"],
        )

        deeper = deeper_generator(task)

        self.assertEqual(deeper[: len(baseline)], baseline)
        self.assertTrue(any(len(program) == 3 for program in deeper))
        lengths = [len(program) for program in deeper]
        self.assertEqual(
            lengths, sorted(lengths, key=lambda length: (length > 1, length))
        )
        self.assertEqual(
            lengths,
            sorted(lengths, key=lambda length: {1: 0, 2: 1, 3: 2}[length]),
        )

    def test_baseline_predictions_match_trusted_runtime(self) -> None:
        runtime = load_runtime_namespace()
        task = {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 0]]}],
        }
        expected = runtime["solve_task"](task)

        result = run_runtime_task(
            runtime,
            task,
            CONFIGS["baseline"],
            timeout_seconds=1.0,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["predictions"], expected)

    def test_timeout_is_reported_without_a_score(self) -> None:
        runtime = load_runtime_namespace()
        original_solve_task = runtime["solve_task"]

        def slow_solve_task(task, return_diagnostics=False):
            time.sleep(0.05)
            return original_solve_task(task, return_diagnostics=return_diagnostics)

        runtime["solve_task"] = slow_solve_task
        try:
            result = run_runtime_task(
                runtime,
                {
                    "train": [{"input": [[1]], "output": [[1]]}],
                    "test": [{"input": [[2]]}],
                },
                CONFIGS["baseline"],
                timeout_seconds=0.001,
            )
        finally:
            runtime["solve_task"] = original_solve_task

        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["predictions"])
        self.assertIsNone(result["actual_score"])

    def test_analysis_timeout_preserves_completed_inference_and_score(self) -> None:
        runtime = load_runtime_namespace()
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }

        def slow_analysis(*args, **kwargs):
            time.sleep(0.05)
            return [], 0

        with patch.object(diagnostic, "_demo_survivors", side_effect=slow_analysis):
            result = run_runtime_task(
                runtime,
                task,
                CONFIGS["baseline"],
                timeout_seconds=1.0,
                task_id="00000001",
                solutions=[[[2]]],
                analysis_timeout_seconds=0.001,
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["inference_completed"])
        self.assertEqual(result["analysis_status"], "timeout")
        self.assertIsNotNone(result["predictions"])
        self.assertEqual(result["actual_score"]["solved_tasks"], 1)
        self.assertEqual(result["actual_correct_by_output"], [True])
        self.assertGreaterEqual(
            result["total_diagnostic_elapsed_seconds"],
            result["inference_elapsed_seconds"],
        )

    def test_missing_verification_metadata_uses_recomputed_source(self) -> None:
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "verification.json"
            with patch.object(diagnostic, "VERIFICATION_PATH", missing_path):
                source = diagnostic.training_source_metadata()

        self.assertFalse(source["verification_metadata_available"])
        self.assertIsNone(source["verification_notebook_sha256"])
        self.assertFalse(source["notebook_hash_verified"])

    def test_resume_rejects_changed_diagnostic_script_hash(self) -> None:
        selected_ids = [f"{index:08x}" for index in range(32)]
        source = {
            "diagnostic_script_sha256": "script-hash-a",
            "notebook_path": "notebooks/submission.ipynb",
            "notebook_sha256": "notebook-hash",
            "verification_notebook_sha256": "notebook-hash",
            "notebook_hash_verified": True,
            "verification_metadata_available": True,
            "training_challenges_sha256": "challenges-hash",
            "training_solutions_sha256": "solutions-hash",
        }
        runtime = {"training_split": lambda task_id: "iteration"}

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch.object(
                    diagnostic, "training_source_metadata", return_value=source
                ),
                patch.object(
                    diagnostic,
                    "verified_baseline",
                    return_value=(None, {"kind": "recomputed"}),
                ),
                patch.object(
                    diagnostic,
                    "select_sample",
                    return_value=(
                        selected_ids,
                        {
                            "selected_ids_sha256": diagnostic.sha256_text(
                                "\n".join(selected_ids)
                            )
                        },
                    ),
                ),
            ):
                config, _, _, _ = load_or_freeze(output_dir, runtime, {}, {})

            self.assertEqual(
                config["source"]["diagnostic_script_sha256"], "script-hash-a"
            )
            changed_source = {**source, "diagnostic_script_sha256": "script-hash-b"}
            with patch.object(
                diagnostic, "training_source_metadata", return_value=changed_source
            ):
                with self.assertRaises(diagnostic.DiagnosticError):
                    load_or_freeze(output_dir, runtime, {}, {})


if __name__ == "__main__":
    unittest.main()
