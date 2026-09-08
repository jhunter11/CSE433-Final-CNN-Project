import unittest

import torch

from train_ultrawidescaledtail10_repro import (
    UltraWideScaledTail10,
    count_param_layers,
    n_params,
)


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = UltraWideScaledTail10().eval()

    def test_assignment_budget_and_recorded_parameter_count(self):
        self.assertEqual(count_param_layers(self.model), 10)
        self.assertEqual(n_params(self.model), 21_337_656)

    def test_cifar_image_produces_finite_class_logits(self):
        with torch.inference_mode():
            output = self.model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
