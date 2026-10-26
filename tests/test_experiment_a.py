import inspect
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import experiment_a
from scripts.diagnose_search import load_runtime_namespace
from scripts.experiment_a import (
    COLOR_BUCKETS,
    combined_predictions,
    feature_matrix,
    feature_schema,
    fit_predict,
    fit_predict_variants,
    shape_supported,
)


def task(train, test):
    return {"train": train, "test": [{"input": grid} for grid in test]}


class ExperimentATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime_namespace()

    def test_categorical_schema_has_oob_sentinel_and_no_ordered_color_feature(self):
        names = feature_schema()
        self.assertIn("center_color_oob", names)
        self.assertIn("neighbor_-1_-1_color_oob", names)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(COLOR_BUCKETS, 11)
        self.assertFalse(any("raw_color" in name for name in names))
        values = feature_matrix([[1, 2], [3, 4]], self.runtime)
        self.assertEqual(len(values), 4)
        self.assertEqual(len(values[0]), len(names))
        oob_index = names.index("neighbor_-1_-1_color_oob")
        self.assertEqual(values[0][oob_index], 1.0)

    def test_schema_dimension_is_invariant_to_grid_shape_and_colors(self):
        first = feature_matrix([[0]], self.runtime)
        second = feature_matrix([[9, 8, 7], [6, 5, 4]], self.runtime)
        self.assertEqual(len(first[0]), len(second[0]))
        self.assertEqual(len(second), 6)

    def test_features_do_not_depend_on_demo_targets(self):
        inputs = [[0, 1], [1, 0]]
        first = task(
            [{"input": inputs, "output": [[2, 3], [3, 2]]}],
            [inputs],
        )
        second = task(
            [{"input": inputs, "output": [[8, 9], [9, 8]]}],
            [inputs],
        )
        self.assertEqual(
            feature_matrix(first["train"][0]["input"], self.runtime),
            feature_matrix(second["train"][0]["input"], self.runtime),
        )
        self.assertEqual(list(inspect.signature(fit_predict).parameters), ["task"])

    def test_shape_support_and_unsupported_symbolic_fallback(self):
        unsupported = task(
            [{"input": [[1, 0]], "output": [[2], [3]]}],
            [[[4, 5]]],
        )
        variants = fit_predict_variants(unsupported, self.runtime)
        self.assertFalse(variants["supported"])
        self.assertEqual(variants["tree"], variants["fallback"])
        self.assertEqual(variants["forest"], variants["fallback"])
        self.assertFalse(shape_supported(unsupported, self.runtime))

    def test_synthetic_recoloring_is_learnable(self):
        recoloring = task(
            [{"input": [[0, 1], [1, 0]], "output": [[2, 3], [3, 2]]}],
            [[[1, 0], [0, 1]]],
        )
        variants = fit_predict_variants(recoloring, self.runtime)
        self.assertTrue(variants["supported"])
        expected = [[3, 2], [2, 3]]
        self.assertEqual(variants["tree"][0]["attempt_1"], expected)
        self.assertEqual(variants["forest"][0]["attempt_1"], expected)

    def test_multioutput_order_determinism_and_validity(self):
        value = task(
            [
                {"input": [[0, 1]], "output": [[2, 3]]},
                {"input": [[1, 0]], "output": [[3, 2]]},
            ],
            [[[1, 0]], [[0, 1]]],
        )
        first = fit_predict(value)
        second = fit_predict(value)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.runtime["validate_submission"]({"00000000": value}, {"00000000": first})
        self.assertEqual(first[0]["attempt_1"], [[3, 2]])
        self.assertEqual(first[1]["attempt_1"], [[2, 3]])

    def test_combined_uses_deterministic_symbolic_fallback_on_duplicate_models(self):
        value = task(
            [{"input": [[0]], "output": [[1]]}],
            [[[0]]],
        )
        variants = fit_predict_variants(value, self.runtime)
        if variants["tree"] == variants["forest"]:
            self.assertEqual(
                variants["combined"],
                combined_predictions(
                    variants["tree"], variants["forest"], variants["fallback"]
                ),
            )

    def _tiny_run_patches(self):
        task_ids = ("tiny-a", "tiny-b")
        challenges = {
            task_id: {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[0]]}],
            }
            for task_id in task_ids
        }
        solutions = {task_id: [[[1]]] for task_id in task_ids}
        runtime = {
            "training_split": lambda task_id: "iteration",
            "score_submission": lambda _challenges, expected, predicted: {
                "total_outputs": 1,
                "correct_outputs": int(
                    predicted[next(iter(predicted))][0]["attempt_1"]
                    == expected[next(iter(expected))][0]
                    or predicted[next(iter(predicted))][0]["attempt_2"]
                    == expected[next(iter(expected))][0]
                ),
                "solved_tasks": int(
                    predicted[next(iter(predicted))][0]["attempt_1"]
                    == expected[next(iter(expected))][0]
                    or predicted[next(iter(predicted))][0]["attempt_2"]
                    == expected[next(iter(expected))][0]
                ),
                "task_averaged_pass_at_2": 1.0,
            },
        }
        predictions = [
            {"attempt_1": [[1]], "attempt_2": [[1]]},
        ]
        variants = {
            "supported": True,
            "status": {"tree": "fitted", "forest": "fitted"},
            "fallback": predictions,
            "tree": predictions,
            "forest": predictions,
            "combined": predictions,
        }
        baseline = {task_id: predictions for task_id in task_ids}
        baseline_source = {"kind": "tiny", "sha256": "tiny-baseline"}
        stack = ExitStack()
        stack.enter_context(
            patch.object(
                experiment_a.diagnostic,
                "load_runtime_namespace",
                return_value=runtime,
            )
        )
        stack.enter_context(
            patch.object(
                experiment_a.diagnostic,
                "load_training_data",
                return_value=(challenges, solutions),
            )
        )
        stack.enter_context(
            patch.object(
                experiment_a.diagnostic,
                "training_source_metadata",
                return_value={"training_challenges_sha256": "tiny-challenges"},
            )
        )
        stack.enter_context(
            patch.object(
                experiment_a.diagnostic,
                "verified_baseline",
                return_value=(baseline, baseline_source),
            )
        )
        stack.enter_context(
            patch.object(
                experiment_a,
                "fit_predict_variants",
                return_value=variants,
            )
        )
        return stack

    def test_tiny_run_persists_provenance_and_resume_invocation_counts(self):
        with TemporaryDirectory() as temporary_dir, self._tiny_run_patches():
            output_dir = Path(temporary_dir)
            first = experiment_a.run(output_dir)
            first_ids = first["selection"]["selected_ids"]
            self.assertEqual(first["invocation_count"], 1)
            self.assertEqual(first["invocations"][0]["resumed_task_ids"], [])
            self.assertEqual(
                first["invocations"][0]["newly_completed_task_ids"], first_ids
            )
            config = experiment_a._read_json(output_dir / "config.json")
            results = experiment_a._read_json(output_dir / "results.json")
            self.assertEqual(config["runtime"]["numpy"], experiment_a.np.__version__)
            self.assertEqual(
                config["runtime"]["sklearn"], experiment_a.sklearn.__version__
            )
            self.assertEqual(config["baseline_source"]["kind"], "tiny")
            self.assertEqual(results["runtime"], config["runtime"])
            self.assertEqual(results["baseline_source"], config["baseline_source"])

            del results["tasks"][first_ids[-1]]
            experiment_a._write_json_atomic(output_dir / "results.json", results)

            second = experiment_a.run(output_dir)
            self.assertEqual(second["invocation_count"], 2)
            self.assertEqual(
                second["invocations"][1]["resumed_task_ids"], first_ids[:-1]
            )
            self.assertEqual(
                second["invocations"][1]["newly_completed_task_ids"], [first_ids[-1]]
            )
            self.assertGreaterEqual(
                second["timing"]["total_invocation_wall_clock_seconds"],
                second["timing"]["latest_invocation_wall_clock_seconds"],
            )
            self.assertEqual(
                experiment_a._read_json(output_dir / "results.json")[
                    "invocation_count"
                ],
                2,
            )

    def test_resume_rejects_incompatible_runtime_provenance(self):
        with TemporaryDirectory() as temporary_dir, self._tiny_run_patches():
            output_dir = Path(temporary_dir)
            experiment_a.run(output_dir)
            with patch.object(experiment_a.np, "__version__", "0.0.0"):
                with self.assertRaisesRegex(
                    RuntimeError, "frozen Experiment A config, source, or IDs changed"
                ):
                    experiment_a.run(output_dir)

    def test_resume_rejects_old_artifact_schema(self):
        for artifact_name in ("config.json", "results.json"):
            with TemporaryDirectory() as temporary_dir, self._tiny_run_patches():
                output_dir = Path(temporary_dir)
                experiment_a._write_json_atomic(
                    output_dir / artifact_name, {"schema_version": 1}
                )
                with self.assertRaisesRegex(
                    RuntimeError, "incompatible Experiment A artifact schema"
                ):
                    experiment_a.run(output_dir)


if __name__ == "__main__":
    unittest.main()
