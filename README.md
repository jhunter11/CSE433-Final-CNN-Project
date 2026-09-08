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

The [per-seed CSV](summary.csv) records the selected checkpoint for each seed.
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
The [technical report](FINAL_REPORT_DRAFT.md) records the architecture comparisons and rejected variants.
Those comparisons describe this selection process. They do not isolate the causal effect of every training choice.

## Reproduce

The full run requires PyTorch, torchvision, access to CIFAR-10, and suitable GPU capacity.
Review the configuration at the top of the training script before running it:

```bash
python train_ultrawidescaledtail10_repro.py
```

The default configuration runs eight seeds for 750 epochs each.
For a pipeline smoke check, change the configuration to:

```python
RUN_SEEDS = (42,)
RUN_EPOCHS = 1
RUN_MAX_TRAIN_BATCHES = 1
```

This checks execution only. It cannot reproduce the reported accuracy.
The automatic worker setting assigns one seed per GPU when multiple GPUs are available.

## Files

| Path | Contents |
| --- | --- |
| `train_ultrawidescaledtail10_repro.py` | Training and run configuration |
| `hunter29_top3_depth_limited_models.ipynb` | Model selection and layer audit |
| `summary.csv` | Eight-seed validation results |
| `full_8seed_run.log` | Training log for that run |
| `FINAL_REPORT_DRAFT.md` | Technical report and earlier comparisons |
| `FINAL_REPORT_SUMMARY.tex` | Report source |
| `local_smoke_repro_runs/` | Earlier pipeline checks |
| `RUNPOD_README.md` | Recorded GPU setup |

The reports and logs preserve the original experiment record. This documentation update did not repeat GPU training.

## License

MIT. See [LICENSE](LICENSE).
