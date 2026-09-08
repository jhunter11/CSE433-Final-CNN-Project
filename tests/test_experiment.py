import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_experiment.py"


class ExperimentCliTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-S", str(LAUNCHER), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_presets_keep_seed_ranges_and_training_schedule(self):
        for count in (5, 8, 16):
            with self.subTest(count=count):
                result = self.run_cli("--seed-count", str(count), "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(result.stdout)
                self.assertEqual(config["seeds"], list(range(42, 42 + count)))
                self.assertEqual(config["epochs"], 750)
                self.assertFalse(config["eval_test_at_end"])
                self.assertEqual(config["parallel_seed_workers"], "auto")

    def test_dry_run_needs_no_packages_or_output_writes(self):
        with TemporaryDirectory() as directory:
            result = self.run_cli("--dry-run", cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_smoke_configuration_and_explicit_paths(self):
        result = self.run_cli(
            "--seed-count",
            "1",
            "--first-seed",
            "7",
            "--epochs",
            "1",
            "--max-train-batches",
            "1",
            "--num-workers",
            "0",
            "--parallel-seed-workers",
            "1",
            "--no-download",
            "--data-dir",
            "existing data",
            "--output-dir",
            "runs/smoke",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(config["seeds"], [7])
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["max_train_batches"], 1)
        self.assertEqual(config["num_workers"], 0)
        self.assertEqual(config["parallel_seed_workers"], 1)
        self.assertFalse(config["download"])
        self.assertEqual(config["data_dir"], "existing data")
        self.assertEqual(config["output_dir"], "runs/smoke")

    def test_invalid_arguments_fail_before_loading_training(self):
        invalid = (
            ("--seed-count", "0"),
            ("--epochs", "-1"),
            ("--num-workers", "-1"),
            ("--max-train-batches", "0"),
            ("--first-seed", "-1"),
            ("--first-seed", str(2**32 - 1)),
            ("--parallel-seed-workers", "0"),
        )
        for args in invalid:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
