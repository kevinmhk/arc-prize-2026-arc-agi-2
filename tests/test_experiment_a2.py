import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import experiment_a2


def grid(value):
    return [[value]]


def variants(fallback, tree=None, forest=None, *, supported=True):
    tree = copy.deepcopy(tree if tree is not None else fallback)
    forest = copy.deepcopy(forest if forest is not None else fallback)
    return {
        "supported": supported,
        "status": {
            "tree": "fitted" if supported else "unsupported_shape",
            "forest": "fitted" if supported else "unsupported_shape",
        },
        "fallback": copy.deepcopy(fallback),
        "tree": tree,
        "forest": forest,
        "combined": copy.deepcopy(fallback),
    }


class ExperimentA2Tests(unittest.TestCase):
    def test_loo_holds_out_complete_demo_in_stable_order(self):
        task = {
            "train": [
                {"input": grid(index), "output": grid(100 + index)}
                for index in range(3)
            ],
            "test": [{"input": grid(0)}],
        }
        calls = []

        def fake_fit_predict(fold_task, _runtime):
            calls.append(copy.deepcopy(fold_task))
            held_value = fold_task["test"][0]["input"][0][0]
            return variants(
                [{"attempt_1": grid(100 + held_value), "attempt_2": grid(99)}],
                [{"attempt_1": grid(50), "attempt_2": grid(50)}],
                [{"attempt_1": grid(51), "attempt_2": grid(51)}],
            )

        evidence = experiment_a2.leave_one_demo_out(
            task, {"validate_grid": lambda _grid: None}, fit_predict=fake_fit_predict
        )

        self.assertEqual(evidence["fold_count"], 3)
        self.assertEqual(evidence["loo_inference_count"], 3)
        self.assertEqual(
            [fold["held_demo_index"] for fold in evidence["folds"]], [0, 1, 2]
        )
        self.assertEqual(
            [[demo["input"] for demo in fold["train"]] for fold in calls],
            [
                [grid(1), grid(2)],
                [grid(0), grid(2)],
                [grid(0), grid(1)],
            ],
        )
        for index, fold_task in enumerate(calls):
            self.assertEqual(fold_task["test"], [{"input": grid(index)}])
            self.assertNotIn("output", fold_task["test"][0])
            self.assertNotIn(
                {"input": grid(index), "output": grid(100 + index)},
                fold_task["train"],
            )
        self.assertEqual(evidence["sources"]["sym1"]["matches"], 3)
        self.assertEqual(evidence["sources"]["sym1"]["folds"], 3)
        self.assertEqual(evidence["sources"]["tree"]["matches"], 0)

    def test_ranks_follow_fold_evidence_then_fixed_tie_order(self):
        reliability = {
            "sym1": {"matches": 1, "folds": 4},
            "sym2": {"matches": 4, "folds": 4},
            "tree": {"matches": 4, "folds": 4},
            "forest": {"matches": 2, "folds": 4},
        }
        self.assertEqual(
            experiment_a2.rank_sources(reliability), ["sym2", "tree", "forest", "sym1"]
        )

    def test_symbolic_ties_preserve_baseline_attempts(self):
        fallback = [{"attempt_1": grid(1), "attempt_2": grid(2)}]
        result = experiment_a2.select_predictions(
            variants(fallback, tree=[{"attempt_1": grid(3), "attempt_2": grid(3)}]),
            {
                source: {"matches": 0, "folds": 2}
                for source in experiment_a2.SOURCE_ORDER
            },
        )
        self.assertEqual(result["predictions"], fallback)
        self.assertEqual(result["chosen_sources"], ["sym1", "sym2"])

    def test_distinct_sources_are_deduplicated_in_rank_order(self):
        fallback = [{"attempt_1": grid(1), "attempt_2": grid(2)}]
        fitted = variants(
            fallback,
            tree=[{"attempt_1": grid(2), "attempt_2": grid(2)}],
            forest=[{"attempt_1": grid(3), "attempt_2": grid(3)}],
        )
        evidence = {
            "tree": {"matches": 3, "folds": 3},
            "forest": {"matches": 2, "folds": 3},
            "sym1": {"matches": 1, "folds": 3},
            "sym2": {"matches": 0, "folds": 3},
        }
        result = experiment_a2.select_predictions(fitted, evidence)
        self.assertEqual(
            [prediction["attempt_1"] for prediction in result["predictions"]],
            [grid(2)],
        )
        self.assertEqual(result["chosen_sources"], ["tree", "forest"])

    def test_unsupported_models_keep_symbolic_outputs_and_multioutput_is_deterministic(
        self,
    ):
        fallback = [
            {"attempt_1": grid(1), "attempt_2": grid(2)},
            {"attempt_1": grid(3), "attempt_2": grid(4)},
        ]
        unsupported = variants(fallback, supported=False)
        evidence = {
            source: {"matches": 2, "folds": 2} for source in experiment_a2.SOURCE_ORDER
        }
        first = experiment_a2.select_predictions(unsupported, evidence)
        second = experiment_a2.select_predictions(unsupported, evidence)
        self.assertEqual(first, second)
        self.assertEqual(first["predictions"], fallback)
        self.assertEqual(first["chosen_sources"], ["sym1", "sym2"])

    def test_insufficient_demos_do_not_invoke_loo_and_use_symbolic_fallback(self):
        fallback = [{"attempt_1": grid(1), "attempt_2": grid(2)}]
        calls = []

        def fake_fit_predict(_task, _runtime):
            calls.append(True)
            return variants(fallback)

        evidence = experiment_a2.leave_one_demo_out(
            {
                "train": [{"input": grid(0), "output": grid(1)}],
                "test": [{"input": grid(0)}],
            },
            {"validate_grid": lambda _grid: None},
            fit_predict=fake_fit_predict,
        )
        self.assertEqual(calls, [])
        self.assertEqual(evidence["reason"], "insufficient_demos")
        self.assertEqual(
            experiment_a2.select_predictions(variants(fallback), evidence)[
                "predictions"
            ],
            fallback,
        )

    def test_invalid_model_fold_is_recorded_as_unavailable_no_match(self):
        task = {
            "train": [
                {"input": grid(0), "output": grid(1)},
                {"input": grid(2), "output": grid(3)},
            ],
            "test": [{"input": grid(0)}],
        }

        def invalid_fit_predict(_task, _runtime):
            return {
                "fallback": [{"attempt_1": grid(1), "attempt_2": grid(2)}],
                "tree": [],
                "forest": [],
                "status": {"tree": "fitted", "forest": "fitted"},
            }

        evidence = experiment_a2.leave_one_demo_out(
            task,
            {"validate_grid": lambda _grid: None},
            fit_predict=invalid_fit_predict,
        )
        for source in ("tree", "forest"):
            self.assertEqual(evidence["sources"][source]["matches"], 0)
            self.assertEqual(
                evidence["sources"][source]["invalid_or_unavailable_folds"], 2
            )

    def test_selection_excludes_manifests_and_uses_sha256_order(self):
        challenges = {task_id: {} for task_id in ("a", "b", "c", "d")}
        selected = experiment_a2.select_task_ids(challenges, {"b", "d"}, sample_size=2)
        expected = sorted(
            ("a", "c"),
            key=lambda task_id: experiment_a2.hashlib.sha256(
                task_id.encode("ascii")
            ).hexdigest(),
        )
        self.assertEqual(selected, expected)

    def test_exclusion_manifest_rejects_duplicate_ids(self):
        with TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "selection.json"
            path.write_text(
                json.dumps({"selected_ids": ["00000001", "00000001"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                experiment_a2._read_exclusion_manifest(
                    path,
                    "test",
                    {"00000001": {}},
                    expected_count=2,
                )

    def test_exclusion_manifest_rejects_wrong_count(self):
        with TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "selection.json"
            path.write_text(
                json.dumps({"selected_ids": ["00000001"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "expected 2 IDs"):
                experiment_a2._read_exclusion_manifest(
                    path,
                    "test",
                    {"00000001": {}},
                    expected_count=2,
                )

    def test_exclusion_manifest_rejects_unknown_id(self):
        with TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "selection.json"
            path.write_text(
                json.dumps({"selected_ids": ["ffffffff"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "not present"):
                experiment_a2._read_exclusion_manifest(
                    path,
                    "test",
                    {"00000001": {}},
                    expected_count=1,
                )

    def test_resume_rejects_tampered_saved_score(self):
        task_id = "00000001"
        task = {"train": [], "test": [{"input": grid(0)}]}
        solutions = {task_id: [grid(1)]}
        predictions = {
            name: [{"attempt_1": grid(1), "attempt_2": grid(2)}]
            for name in ("baseline", "experiment_a", "experiment_a2")
        }

        def score_submission(_challenges, expected, predicted):
            correct = int(predicted[task_id][0]["attempt_1"] == expected[task_id][0])
            return {
                "total_outputs": 1,
                "correct_outputs": correct,
                "solved_tasks": correct,
            }

        runtime = {
            "validate_submission": lambda _challenges, _submission: None,
            "score_submission": score_submission,
        }
        entry = {
            "complete": True,
            "task_id": task_id,
            "predictions": predictions,
            "scores": {
                name: experiment_a2._score_prediction(
                    runtime, task_id, task, solutions, variant_predictions
                )
                for name, variant_predictions in predictions.items()
            },
        }
        entry["scores"]["experiment_a2"]["correct_outputs"] = 0

        with self.assertRaisesRegex(RuntimeError, "score mismatch"):
            experiment_a2._validate_saved_task(runtime, task_id, task, entry, solutions)

    def test_resume_rejects_changed_frozen_config(self):
        tasks = {
            "a": {
                "train": [
                    {"input": grid(0), "output": grid(1)},
                    {"input": grid(2), "output": grid(3)},
                ],
                "test": [{"input": grid(0)}],
            },
            "b": {
                "train": [
                    {"input": grid(4), "output": grid(5)},
                    {"input": grid(6), "output": grid(7)},
                ],
                "test": [{"input": grid(4)}],
            },
        }
        tasks["cafebabe"] = copy.deepcopy(tasks["a"])
        solutions = {task_id: [grid(1)] for task_id in tasks}
        baseline = {
            task_id: [{"attempt_1": grid(1), "attempt_2": grid(1)}] for task_id in tasks
        }
        runtime = {
            "validate_grid": lambda _grid: None,
            "validate_submission": lambda _challenges, _submission: None,
            "score_submission": lambda _challenges, expected, predicted: {
                "total_outputs": 1,
                "correct_outputs": int(
                    predicted[next(iter(predicted))][0]["attempt_1"]
                    == expected[next(iter(expected))][0]
                ),
                "solved_tasks": int(
                    predicted[next(iter(predicted))][0]["attempt_1"]
                    == expected[next(iter(expected))][0]
                ),
            },
        }
        with TemporaryDirectory() as temporary_dir:
            manifest_a = Path(temporary_dir) / "a.json"
            manifest_d = Path(temporary_dir) / "d.json"
            manifest_a.write_text(
                json.dumps({"selected_ids": ["cafebabe"]}), encoding="utf-8"
            )
            manifest_d.write_text(
                json.dumps({"selected_ids": ["cafebabe"]}), encoding="utf-8"
            )
            output_dir = Path(temporary_dir) / "output"
            patches = (
                patch.object(experiment_a2, "SAMPLE_SIZE", 2),
                patch.object(experiment_a2, "ALL_A_SELECTION_PATH", manifest_a),
                patch.object(experiment_a2, "DIAGNOSTIC_SELECTION_PATH", manifest_d),
                patch.object(
                    experiment_a2.diagnostic,
                    "load_runtime_namespace",
                    return_value=runtime,
                ),
                patch.object(
                    experiment_a2.diagnostic,
                    "load_training_data",
                    return_value=(tasks, solutions),
                ),
                patch.object(
                    experiment_a2.diagnostic,
                    "verified_baseline",
                    return_value=(baseline, {"kind": "tiny"}),
                ),
                patch.object(
                    experiment_a2.experiment_a,
                    "fit_predict_variants",
                    return_value=variants(
                        [{"attempt_1": grid(1), "attempt_2": grid(1)}]
                    ),
                ),
            )
            with unittest.TestCase().subTest("run"):
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    patches[4],
                    patches[5],
                    patches[6],
                    patch.object(experiment_a2, "ALL_A_EXPECTED_COUNT", 1),
                    patch.object(experiment_a2, "DIAGNOSTIC_EXPECTED_COUNT", 1),
                ):
                    experiment_a2.run(output_dir)
                    results = experiment_a2._read_json(output_dir / "results.json")
                    results["tasks"]["cafebabe"] = {}
                    experiment_a2._write_json_atomic(
                        output_dir / "results.json", results
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "unexpected saved Experiment A2 task IDs"
                    ):
                        experiment_a2.run(output_dir)
                    results["tasks"].pop("cafebabe")
                    experiment_a2._write_json_atomic(
                        output_dir / "results.json", results
                    )
                    config = experiment_a2._read_json(output_dir / "config.json")
                    config["policy"]["tie_order"] = ["forest", "tree", "sym1", "sym2"]
                    experiment_a2._write_json_atomic(output_dir / "config.json", config)
                    with self.assertRaisesRegex(RuntimeError, "frozen Experiment A2"):
                        experiment_a2.run(output_dir)


if __name__ == "__main__":
    unittest.main()
