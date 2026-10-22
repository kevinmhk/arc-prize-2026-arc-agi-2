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


if __name__ == "__main__":
    unittest.main()
