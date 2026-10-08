import unittest
from unittest.mock import MagicMock, patch

import train_ultrawidescaledtail10_repro as training


class SeedSchedulingTests(unittest.TestCase):
    def test_multiple_waves_keep_one_task_per_gpu(self):
        config = training.TrainConfig(
            seeds=tuple(range(42, 48)), num_workers=8, parallel_seed_workers="auto"
        )
        submitted = []

        def submit(worker, args):
            submitted.append((worker, args))
            future = MagicMock()
            seeds = args[1]
            if isinstance(seeds, int):
                future.result.return_value = {"seed": seeds}
            else:
                future.result.return_value = [{"seed": seed} for seed in seeds]
            return future

        executor = MagicMock()
        executor.submit.side_effect = submit
        pool = MagicMock()
        pool.__enter__.return_value = executor
        with (
            patch.object(training, "ensure_cifar10_available"),
            patch.object(training, "UltraWideScaledTail10"),
            patch.object(training, "count_param_layers", return_value=10),
            patch.object(training, "n_params", return_value=1),
            patch.object(training.torch.cuda, "is_available", return_value=True),
            patch.object(training.torch.cuda, "device_count", return_value=2),
            patch.object(training, "ProcessPoolExecutor", return_value=pool),
            patch.object(training, "as_completed", side_effect=lambda items: reversed(items)),
            patch.object(training, "summarize_results", return_value={}),
            patch.object(training, "write_sweep_outputs"),
        ):
            results, _ = training.run_seed_sweep(config)

        self.assertEqual(len(submitted), 2)
        self.assertEqual([args[2] for _, args in submitted], [0, 1])
        self.assertEqual([args[1] for _, args in submitted], [(42, 44, 46), (43, 45, 47)])
        self.assertEqual([row["seed"] for row in results], list(config.seeds))
        self.assertEqual([args[3] for _, args in submitted], [4, 4])


if __name__ == "__main__":
    unittest.main()
