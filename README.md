# UltraWideScaledTail10: Depth-Limited CIFAR-10

A CIFAR-10 classification project for CSE 433 at Miami University.
The assignment limits the forward path to ten weighted layers, counting `Conv1d`, `Conv2d`, and `Linear` modules.

The model uses wide feature blocks and a residual tail at reduced spatial resolution.
Normalization, activation, pooling, residual addition, stochastic depth, and channel gating do not consume the assignment layer budget.

## Recorded result

| Metric | Value |
| --- | --- |
| Mean selected validation accuracy | 96.84%, standard deviation 0.10 percentage points |
| Seeds | Eight, numbered 42 through 49 |
| Best selected validation accuracy | 96.98% |
| Weighted layers | 10 |
| Parameters | 21,337,656 |
| Training schedule | 750 epochs per seed |

The [per-seed CSV](results/eight-seed/summary.csv) records the selected checkpoint for each seed.
Selection uses validation accuracy across raw and exponential-moving-average checkpoints.
The final model has no recorded official test accuracy in that CSV.
Repeated model and checkpoint selection can bias validation results upward.

## Layer budget

| Count | Module | Type |
| --- | --- | --- |
| 1 | `stem` | Conv2d |
| 2, 3 | `block1.conv1`, `block1.conv2` | Conv2d |
| 4, 5 | `block2.conv1`, `block2.conv2` | Conv2d |
| 6 | `block3.conv` | Conv2d |
| 7, 8 | `tail.conv1`, `tail.conv2` | Grouped Conv2d |
| 9 | `tail.conv3` | Conv2d |
| 10 | `fc` | Linear |

The training recipe includes data augmentation, mixup, label smoothing, momentum SGD, a cosine schedule, and checkpoint averaging.
The [technical report](docs/technical-report.md) records the architecture comparisons and rejected variants.
Those comparisons describe this selection process. They do not isolate the causal effect of every training choice.

## Reproduce

The full run requires PyTorch, torchvision, access to CIFAR-10, and suitable GPU capacity.
Preview the launcher settings before starting a run:

```bash
python run_experiment.py --dry-run
python run_experiment.py --seed-count 8
```

The default runs seeds 42 through 49 for 750 epochs each, with the original training recipe.
Use `--seed-count 5` or `--seed-count 16` for the other recorded launcher configurations.
The preview prints overrides to `TrainConfig` without loading PyTorch, downloading data, or creating output files.

For a pipeline smoke check with cached CIFAR-10 data:

```bash
python run_experiment.py --seed-count 1 --epochs 1 --max-train-batches 1 --num-workers 0 --no-download --output-dir runs/smoke
```

This trains one batch and evaluates the validation split. It checks execution and cannot reproduce the reported accuracy.
The automatic worker setting assigns one seed per GPU when multiple GPUs are available.
New runs write under `runs/`. The committed evidence under `results/` stays separate.
The original training file still supports its top-of-file configuration for notebook use.

## Check the code

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m ruff check .
```

The tests check launcher arguments, offline preview, the ten-layer budget, parameter count, and finite logits for CIFAR-shaped inputs.
They use synthetic tensors and do not download CIFAR-10 or repeat GPU training.

## Files

| Path | Contents |
| --- | --- |
| `run_experiment.py` | Seed sweep CLI and offline preview |
| `train_ultrawidescaledtail10_repro.py` | Model and training implementation |
| `hunter29_top3_depth_limited_models.ipynb` | Model selection and layer audit |
| `results/` | Recorded runs, including failed runs and smoke checks |
| `docs/technical-report.md` | Technical report and earlier comparisons |
| `paper/summary.tex`, `paper/summary.pdf` | Report source and PDF |
| `docs/runpod.md` | GPU setup and launch commands |
| `tests/` | CLI and model contract tests |

The reports and logs preserve the original experiment record. This documentation update did not repeat GPU training.

## License

MIT. See [LICENSE](LICENSE).
