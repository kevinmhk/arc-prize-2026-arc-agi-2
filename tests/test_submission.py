import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "submission.ipynb"


def load_runtime() -> dict[str, Any]:
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    definition_cell = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    namespace: dict[str, Any] = {"__name__": "submission_runtime_test"}
    exec("".join(definition_cell["source"]), namespace)
    return namespace


def call(namespace: dict[str, Any], name: str) -> Callable[..., Any]:
    return cast(Callable[..., Any], namespace[name])


def sample_challenges() -> dict[str, Any]:
    return {
        "00000001": {
            "train": [
                {"input": [[0]], "output": [[1]]},
            ],
            "test": [
                {"input": [[2]]},
                {"input": [[3]]},
            ],
        },
        "00000002": {
            "train": [
                {"input": [[4]], "output": [[5]]},
            ],
            "test": [
                {"input": [[6]]},
            ],
        },
    }


class SubmissionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_runtime()
        cls.validate_grid = staticmethod(call(cls.runtime, "validate_grid"))
        cls.validate_submission = staticmethod(call(cls.runtime, "validate_submission"))
        cls.score_submission = staticmethod(call(cls.runtime, "score_submission"))
        cls.identity_submission = staticmethod(call(cls.runtime, "identity_submission"))
        cls.apply_geometry = staticmethod(call(cls.runtime, "apply_geometry"))
        cls.crop_to_content = staticmethod(call(cls.runtime, "crop_to_content"))
        cls.learn_color_map = staticmethod(call(cls.runtime, "learn_color_map"))
        cls.connected_components = staticmethod(
            call(cls.runtime, "connected_components")
        )
        cls.extract_selected_object = staticmethod(
            call(cls.runtime, "extract_selected_object")
        )
        cls.apply_program = staticmethod(call(cls.runtime, "apply_program"))
        cls.generate_programs = staticmethod(call(cls.runtime, "generate_programs"))
        cls.solve_task = staticmethod(call(cls.runtime, "solve_task"))
        cls.load_inputs = staticmethod(call(cls.runtime, "load_inputs"))
        cls.resolve_data_dir = staticmethod(call(cls.runtime, "resolve_data_dir"))
        cls.resolve_output_path = staticmethod(call(cls.runtime, "resolve_output_path"))

    def test_local_paths_work_from_repo_root_and_notebooks_dir(self) -> None:
        repo_root = NOTEBOOK_PATH.parents[1]
        for cwd in (repo_root, repo_root / "notebooks"):
            with self.subTest(cwd=cwd):
                with (
                    patch.dict(
                        os.environ,
                        {"ARC_DATA_DIR": "", "ARC_OUTPUT_PATH": ""},
                    ),
                    patch.object(Path, "cwd", return_value=cwd),
                ):
                    self.assertEqual(self.resolve_data_dir(), repo_root / "data")
                    self.assertEqual(
                        self.resolve_output_path(),
                        repo_root / "artifacts" / "submission.json",
                    )

    def test_kaggle_competitions_mount_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            input_root = Path(scratch) / "input"
            data_dir = input_root / "competitions" / "arc-prize-2026-arc-agi-2"
            data_dir.mkdir(parents=True)
            (data_dir / "arc-agi_test_challenges.json").write_text(
                json.dumps(sample_challenges())
            )

            def mounted_path(value: str) -> Path:
                path = Path(value)
                if path.is_relative_to("/kaggle/input"):
                    return input_root / path.relative_to("/kaggle/input")
                return path

            with (
                patch.dict(os.environ, {"ARC_DATA_DIR": ""}),
                patch.dict(
                    self.runtime, {"Path": Mock(wraps=Path, side_effect=mounted_path)}
                ),
            ):
                self.assertEqual(self.resolve_data_dir(), data_dir)

    def test_identity_preserves_multi_output_order_and_copies_grids(self) -> None:
        challenges = sample_challenges()

        submission = self.identity_submission(challenges)

        self.assertEqual(list(submission), ["00000001", "00000002"])
        self.assertEqual(
            [prediction["attempt_1"] for prediction in submission["00000001"]],
            [[[2]], [[3]]],
        )
        first_output = submission["00000001"][0]
        second_output = submission["00000001"][1]
        self.assertIsNot(first_output["attempt_1"], first_output["attempt_2"])
        self.assertIsNot(first_output["attempt_1"], second_output["attempt_1"])
        first_output["attempt_1"][0][0] = 9
        self.assertEqual(challenges["00000001"]["test"][0]["input"], [[2]])

    def test_multiple_outputs_and_either_attempt_match(self) -> None:
        challenges = sample_challenges()
        submission = {
            "00000001": [
                {"attempt_1": [[9]], "attempt_2": [[1]]},
                {"attempt_1": [[8]], "attempt_2": [[7]]},
            ],
            "00000002": [
                {"attempt_1": [[5]], "attempt_2": [[8]]},
            ],
        }
        solutions = {
            "00000001": [[[1]], [[2]]],
            "00000002": [[[5]]],
        }

        metrics = self.score_submission(challenges, solutions, submission)

        self.assertEqual(metrics["correct_outputs"], 2)
        self.assertEqual(metrics["total_outputs"], 3)
        self.assertEqual(metrics["solved_tasks"], 1)
        self.assertEqual(metrics["total_tasks"], 2)
        self.assertAlmostEqual(metrics["output_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["task_solve_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["task_averaged_pass_at_2"], 3 / 4)
        self.assertEqual(metrics["official_score"], metrics["output_accuracy"])

    def test_invalid_output_is_rejected_before_scoring(self) -> None:
        challenges = sample_challenges()
        valid = {
            "00000001": [
                {"attempt_1": [[0]], "attempt_2": [[0]]},
                {"attempt_1": [[0]], "attempt_2": [[0]]},
            ],
            "00000002": [
                {"attempt_1": [[0]], "attempt_2": [[0]]},
            ],
        }

        invalid_outputs = [
            {
                **valid,
                "00000001": [
                    {"attempt_1": [[True]], "attempt_2": [[0]]},
                    valid["00000001"][1],
                ],
            },
            {
                **valid,
                "00000001": [
                    {"attempt_1": [[0, 1], [2]], "attempt_2": [[0]]},
                    valid["00000001"][1],
                ],
            },
            {
                **valid,
                "00000001": [
                    {"attempt_1": [[0]]},
                    valid["00000001"][1],
                ],
            },
            {
                **valid,
                "00000001": valid["00000001"][:1],
            },
            {
                **valid,
                "00000003": valid["00000002"],
            },
        ]

        for invalid in invalid_outputs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.validate_submission(challenges, invalid)

    def test_strict_grid_validation_rejects_bool_and_bad_types(self) -> None:
        for invalid_grid in ([], [[1], []], [[1.0]], [[10]], [[None]]):
            with self.subTest(invalid_grid=invalid_grid):
                with self.assertRaises(ValueError):
                    self.validate_grid(invalid_grid)

    def test_submission_mode_does_not_load_solutions(self) -> None:
        calls: list[str] = []
        original_loader = self.runtime["read_json"]

        def fake_loader(path: Path) -> Any:
            calls.append(path.name)
            return sample_challenges()

        self.runtime["read_json"] = fake_loader
        try:
            challenges, solutions = self.load_inputs("submission", Path("."))
        finally:
            self.runtime["read_json"] = original_loader

        self.assertEqual(set(challenges), {"00000001", "00000002"})
        self.assertIsNone(solutions)
        self.assertEqual(calls, ["arc-agi_test_challenges.json"])

    def test_rectangular_geometry_and_shape_inference(self) -> None:
        grid = [[1, 2, 3], [4, 5, 6]]

        self.assertEqual(
            self.apply_geometry(grid, "rotate_90"),
            [[4, 1], [5, 2], [6, 3]],
        )
        self.assertEqual(
            self.apply_geometry(grid, "transpose"), [[1, 4], [2, 5], [3, 6]]
        )

    def test_d4_geometry_operations_cover_asymmetric_rectangular_grid(self) -> None:
        grid = [[1, 2, 3], [4, 5, 6]]
        expected = {
            "identity": [[1, 2, 3], [4, 5, 6]],
            "rotate_90": [[4, 1], [5, 2], [6, 3]],
            "rotate_180": [[6, 5, 4], [3, 2, 1]],
            "rotate_270": [[3, 6], [2, 5], [1, 4]],
            "flip_horizontal": [[3, 2, 1], [6, 5, 4]],
            "flip_vertical": [[4, 5, 6], [1, 2, 3]],
            "transpose": [[1, 4], [2, 5], [3, 6]],
            "anti_diagonal": [[6, 3], [5, 2], [4, 1]],
        }

        for operation, output in expected.items():
            with self.subTest(operation=operation):
                if operation == "identity":
                    actual = grid
                else:
                    actual = self.apply_geometry(grid, operation)
                self.assertEqual(actual, output)

    def test_anti_diagonal_can_compose_with_recolor(self) -> None:
        task = {
            "train": [
                {
                    "input": [[1, 2, 3], [4, 5, 6]],
                    "output": [[4, 7], [5, 8], [6, 9]],
                }
            ],
            "test": [{"input": [[1, 2, 3], [4, 5, 6]]}],
        }
        anti_diagonal_then_recolor = (
            ("anti_diagonal",),
            ("recolor", ((1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4))),
        )

        self.assertIn(anti_diagonal_then_recolor, self.generate_programs(task))
        predictions, diagnostics = self.solve_task(task, return_diagnostics=True)
        self.assertTrue(diagnostics["exact_fit"])
        self.assertEqual(predictions[0]["attempt_1"], task["train"][0]["output"])

    def test_all_background_crop_is_smallest_valid_grid(self) -> None:
        self.assertEqual(self.crop_to_content([[0, 0], [0, 0]], 0), [[0]])

    def test_recolor_mapping_requires_consistency_across_all_demos(self) -> None:
        consistent = [
            {"input": [[1, 2]], "output": [[3, 4]]},
            {"input": [[2, 1]], "output": [[4, 3]]},
        ]
        conflicting = consistent + [{"input": [[1]], "output": [[5]]}]

        self.assertEqual(self.learn_color_map(consistent), {1: 3, 2: 4})
        self.assertIsNone(self.learn_color_map(conflicting))

    def test_object_connectivity_and_tie_selection_are_deterministic(self) -> None:
        diagonal = [[1, 0], [0, 1]]
        self.assertEqual(len(self.connected_components(diagonal, 4)), 2)
        self.assertEqual(len(self.connected_components(diagonal, 8)), 1)
        tied = [[2, 0, 1]]
        self.assertEqual(
            self.extract_selected_object(tied, connectivity=4, selection="largest"),
            [[2]],
        )

    def test_composition_validates_only_the_final_grid(self) -> None:
        program = (
            ("crop_non_background",),
            ("recolor", {1: 2}),
        )
        self.assertEqual(
            self.apply_program([[0, 1, 0], [0, 0, 0]], program),
            [[2]],
        )

    def test_generator_learns_recolor_after_crop(self) -> None:
        task = {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    "output": [[2]],
                }
            ],
            "test": [{"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}],
        }
        crop_then_recolor = (
            ("crop_non_background",),
            ("recolor", ((1, 2),)),
        )

        programs = self.generate_programs(task)
        self.assertIn(crop_then_recolor, programs)

        predictions, diagnostics = self.solve_task(task, return_diagnostics=True)
        self.assertTrue(diagnostics["exact_fit"])
        self.assertEqual(predictions[0]["attempt_1"], [[2]])

    def test_border_removal_can_precede_symmetry_completion(self) -> None:
        task = {
            "train": [
                {
                    "input": [
                        [1, 1, 1, 1, 1],
                        [1, 2, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                    ],
                    "output": [[2, 0, 2], [0, 0, 0], [0, 0, 0]],
                }
            ],
            "test": [
                {
                    "input": [
                        [1, 1, 1, 1, 1],
                        [1, 3, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                    ]
                }
            ],
        }
        border_then_symmetry = (
            ("remove_uniform_border",),
            ("symmetry", "vertical"),
        )

        with self.assertRaises(ValueError):
            self.apply_program(task["train"][0]["input"], (("symmetry", "vertical"),))
        self.assertEqual(
            self.apply_program(task["train"][0]["input"], border_then_symmetry),
            task["train"][0]["output"],
        )
        self.assertIn(border_then_symmetry, self.generate_programs(task))

        predictions, diagnostics = self.solve_task(task, return_diagnostics=True)
        self.assertTrue(diagnostics["exact_fit"])
        self.assertEqual(predictions[0]["attempt_1"], [[3, 0, 3], [0, 0, 0], [0, 0, 0]])

    def test_generated_singletons_are_programs_and_primitives_can_fit(self) -> None:
        identity_task = {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 0]]}],
        }
        rotation_task = {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }

        for task, expected_program in (
            (identity_task, (("identity",),)),
            (rotation_task, (("rotate_90",),)),
        ):
            with self.subTest(expected_program=expected_program):
                programs = self.generate_programs(task)
                self.assertTrue(
                    all(
                        isinstance(program, tuple)
                        and all(isinstance(operation, tuple) for operation in program)
                        for program in programs
                    )
                )
                self.assertIn(expected_program, programs)

                predictions, diagnostics = self.solve_task(
                    task, return_diagnostics=True
                )
                self.assertTrue(diagnostics["exact_fit"])
                if expected_program == (("identity",),):
                    self.assertEqual(predictions[0]["attempt_1"], [[2, 0]])
                else:
                    self.assertEqual(predictions[0]["attempt_1"], [[7, 5], [8, 6]])

    def test_completion_can_precede_a_transform(self) -> None:
        task = {
            "train": [
                {
                    "input": [[0, 1, 0], [2, 0, 0], [0, 3, 0]],
                    "output": [[0, 2, 0], [3, 0, 1], [0, 2, 0]],
                }
            ],
            "test": [{"input": [[0, 4, 0], [5, 0, 0], [0, 6, 0]]}],
        }
        completion_then_rotation = (
            ("symmetry", "vertical"),
            ("rotate_90",),
        )

        programs = self.generate_programs(task)
        self.assertIn(completion_then_rotation, programs)
        self.assertEqual(
            self.apply_program(task["train"][0]["input"], completion_then_rotation),
            task["train"][0]["output"],
        )

        predictions, diagnostics = self.solve_task(task, return_diagnostics=True)
        self.assertTrue(diagnostics["exact_fit"])
        self.assertEqual(predictions[0]["attempt_1"], [[0, 5, 0], [6, 0, 4], [0, 5, 0]])

    def test_anti_diagonal_reflection_is_covered_by_bounded_composition(self) -> None:
        task = {
            "train": [
                {
                    "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "output": [[9, 6, 3], [8, 5, 2], [7, 4, 1]],
                }
            ],
            "test": [{"input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}],
        }
        anti_diagonal = (("rotate_180",), ("transpose",))

        programs = self.generate_programs(task)
        self.assertLessEqual(len(programs), self.runtime["MAX_PROGRAMS_PER_TASK"])
        self.assertIn(anti_diagonal, programs)
        self.assertEqual(
            self.apply_program(task["train"][0]["input"], anti_diagonal),
            task["train"][0]["output"],
        )

    def test_training_failures_report_only_unsolved_outputs(self) -> None:
        challenges = {
            "00000001": {
                "train": [{"input": [[0]], "output": [[0]]}],
                "test": [{"input": [[1]]}, {"input": [[2]]}],
            },
            "00000002": {
                "train": [{"input": [[0]], "output": [[0]]}],
                "test": [{"input": [[3]]}],
            },
            "00000003": {
                "train": [{"input": [[0]], "output": [[0]]}],
                "test": [{"input": [[4]]}],
            },
        }
        solutions = {
            "00000001": [[[1]], [[9, 9]]],
            "00000002": [[[3]]],
            "00000003": [[[8]]],
        }
        predictions = {
            "00000001": [
                {"attempt_1": [[1]], "attempt_2": [[0, 0]]},
                {"attempt_1": [[2]], "attempt_2": [[2]]},
            ],
            "00000002": [{"attempt_1": [[3]], "attempt_2": [[0, 0]]}],
            "00000003": [{"attempt_1": [[4]], "attempt_2": [[5]]}],
        }
        diagnostics = {
            "00000001": {
                "exact_fit": True,
                "train_demo_coverage": 1.0,
                "elapsed_seconds": 0.0,
            },
            "00000002": {
                "exact_fit": False,
                "train_demo_coverage": 0.0,
                "elapsed_seconds": 0.0,
            },
            "00000003": {
                "exact_fit": False,
                "train_demo_coverage": 0.0,
                "elapsed_seconds": 0.0,
            },
        }
        original_solve_task = self.runtime["solve_task"]

        def fake_solve_task(task: dict[str, Any], return_diagnostics: bool = False):
            task_id = next(
                task_id
                for task_id, challenge in challenges.items()
                if challenge is task
            )
            result = (predictions[task_id], diagnostics[task_id])
            return result if return_diagnostics else result[0]

        self.runtime["solve_task"] = fake_solve_task
        try:
            _submission, metrics = self.runtime["evaluate_training"](
                challenges, solutions
            )
        finally:
            self.runtime["solve_task"] = original_solve_task

        self.assertEqual(
            metrics["representative_failures"],
            [
                {
                    "task_id": "00000001",
                    "test_index": 1,
                    "category": "wrong_shape",
                    "train_demo_coverage": 1.0,
                },
                {
                    "task_id": "00000003",
                    "test_index": 0,
                    "category": "content_or_placement",
                    "train_demo_coverage": 0.0,
                },
            ],
        )

    def test_ambiguous_fits_preserve_test_order_and_distinct_attempts(self) -> None:
        task = {
            "train": [{"input": [[1, 1], [0, 0]], "output": [[1, 1], [0, 0]]}],
            "test": [{"input": [[1, 0], [0, 2]]}, {"input": [[3, 0], [0, 4]]}],
        }

        predictions = self.solve_task(task)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["attempt_1"], [[1, 0], [0, 2]])
        self.assertEqual(predictions[1]["attempt_1"], [[3, 0], [0, 4]])
        self.assertNotEqual(predictions[0]["attempt_1"], predictions[0]["attempt_2"])

    def test_program_budget_depth_and_fallback_are_bounded(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[2, 3]]}],
            "test": [{"input": [[3]]}],
        }

        programs = self.generate_programs(task)
        self.assertLessEqual(len(programs), self.runtime["MAX_PROGRAMS_PER_TASK"])
        self.assertTrue(
            all(
                not isinstance(program[0], tuple) or len(program) <= 2
                for program in programs
            )
        )
        predictions = self.solve_task(task)
        self.assertEqual(predictions[0]["attempt_1"], [[3]])
        self.assertEqual(predictions[0]["attempt_2"], [[3, 3]])
        self.validate_grid(predictions[0]["attempt_1"])
        self.validate_grid(predictions[0]["attempt_2"])


if __name__ == "__main__":
    unittest.main()
