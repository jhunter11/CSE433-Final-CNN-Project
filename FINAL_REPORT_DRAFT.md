# UltraWideScaledTail10 CIFAR-10 Report Draft

_CSE 433 final project draft, focused on the selected best single depth-limited model._

---

## Rubric alignment

| Report requirement | Where addressed |
| --- | --- |
| Clear instructions/descriptions of the model and performance analysis, 15 pts | Model description, run instructions, performance analysis |
| Clear description and discussion of how and why the model was improved, 5 pts | Improvement rationale |
| Efforts that failed to improve accuracy, at least 10 methods, 10 pts | Failed and rejected improvement attempts |

## Model description

The final submitted single model is `UltraWideScaledTail10`, implemented in `train_ultrawidescaledtail10_repro.py`. The design follows the project constraint that the forward path must contain at most 10 weighted layers, where weighted layers are `Conv1d`, `Conv2d`, or `Linear` modules. Normalization, activation, pooling, residual addition, stochastic depth, and channel gating are not counted as weighted layers.

The architecture uses a very wide convolutional feature trunk and a residual tail at low spatial resolution. The intuition is that CIFAR-10 benefits from high channel capacity, but the 10-layer limit makes it expensive to spend many layers early. Therefore, the model first builds wide features, reduces spatial size with pooling, and then spends the last convolutional layers in the tail where computation is cheaper.

```mermaid
flowchart LR
    input["CIFAR-10 image<br/>3 x 32 x 32"]
    stem["Stem conv<br/>3 to 24"]
    b1["AirBlock<br/>24 to 384"]
    b2["AirBlock<br/>384 to 768"]
    b3["SingleConvBlock<br/>768 to 768"]
    tail["Scaled residual tail<br/>grouped convs + channel gate"]
    pool["Adaptive max pool"]
    fc["Linear classifier<br/>768 to 10"]
    out["Class logits"]

    input --> stem --> b1 --> b2 --> b3 --> tail --> pool --> fc --> out
```

The counted layer audit is:

| Count | Module | Type | Notes |
| ---: | --- | --- | --- |
| 1 | `stem` | Conv2d | Initial image projection |
| 2 | `block1.conv1` | Conv2d | Wide feature block |
| 3 | `block1.conv2` | Conv2d | Wide feature block |
| 4 | `block2.conv1` | Conv2d | 768-channel feature block |
| 5 | `block2.conv2` | Conv2d | 768-channel feature block |
| 6 | `block3.conv` | Conv2d | Single low-resolution block |
| 7 | `tail.conv1` | Conv2d | Grouped tail convolution |
| 8 | `tail.conv2` | Conv2d | Grouped tail convolution |
| 9 | `tail.conv3` | Conv2d | Dense tail convolution |
| 10 | `fc` | Linear | Final classifier |

The model has exactly 10 weighted layers and 21,337,656 trainable or stored parameters. A local dummy forward pass returned logits with shape `(2, 10)`, which confirms that the classifier output matches the 10 CIFAR-10 classes.

## Run instructions

The final reproducibility run is configured at the top of `train_ultrawidescaledtail10_repro.py`. The intended final command is:

```powershell
python train_ultrawidescaledtail10_repro.py
```

The final configuration trains 8 seeds, `42` through `49`, for 750 epochs per seed. This matches the 8x A40 RunPod setup by running one seed per GPU in a single wave. The script writes per-seed checkpoints and JSON logs into `repro_runs_ultrawidescaledtail10_8seed/`, then writes aggregate `summary.csv` and `summary.json`.

For a quick professor verification run, temporarily reduce the seed list in the top configuration:

```python
RUN_SEEDS = (42, 43, 44)
```

For a smoke test that only checks the training path, also temporarily reduce:

```python
RUN_EPOCHS = 1
RUN_MAX_TRAIN_BATCHES = 1
```

These quick settings should not be used as accuracy evidence. They only confirm that data loading, model construction, CUDA execution, checkpoint saving, and summary writing work.

## Local execution and parallelization

Before running locally, I checked the available hardware. The local machine has Windows 11, 16 logical CPU cores, 31.1 GB RAM, and one NVIDIA GeForce RTX 5070 with 12,227 MB VRAM. Because only one CUDA GPU is visible, the safest parallel strategy is to use the GPU for one seed at a time while parallelizing data loading. The script now includes `RUN_PARALLEL_SEED_WORKERS = "auto"`; on a one-GPU machine this resolves to one seed worker to avoid VRAM contention, while on a multi-GPU machine it can run one seed per GPU.

The final run configuration uses CUDA acceleration, AMP/bfloat16 autocast on CUDA, channels-last tensor memory format, and DataLoader workers. This is the practical maximum parallelism for the available local hardware without risking multiple full model copies competing for the same 12 GB GPU.

A local smoke run was completed with cached CIFAR-10 data at `../data`. The smoke run used 3 seeds, 1 epoch, and 1 training batch per seed. This run is not an accuracy result, but it confirms the end-to-end training pipeline.

| Seed | Epochs | Selected validation accuracy | Best raw | Best EMA | Wall minutes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1 | 10.60 | 10.60 | 10.60 | 0.038 |
| 43 | 1 | 7.18 | 7.18 | 7.18 | 0.022 |
| 44 | 1 | 10.34 | 10.34 | 10.34 | 0.023 |

The smoke summary was 9.37 mean selected validation accuracy with 1.90 standard deviation. This is near random chance because the smoke run only trained one mini-batch. The value of this run is verification, not model performance.

## Performance analysis

The selected model was chosen from the prior multi-seed evidence preserved in the notebook. Across the prior 500-epoch validation runs, `UltraWideScaledTail10` had the best single-model mean out-of-sample validation accuracy among the compared 10-layer candidates.

| Model | Seeds | Mean selected OOS validation | Std | Max | Layers |
| --- | ---: | ---: | ---: | ---: | ---: |
| UltraWideScaledTail10 | 3 | 96.97 | 0.08 | 97.06 | 10 |
| WideScaledPlus10 | 3 | 96.70 | 0.14 | 96.80 | 10 |
| DeepScaled10 | 3 | 96.37 | 0.14 | 96.52 | 10 |

The selected model was strongest by mean validation accuracy and also reached the best single-seed validation result among these three candidates. Its seed-level retained results were:

| Seed | Selected OOS validation | Best raw | Best EMA | Selected epoch |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 96.94 | 96.88 | 96.94 | 459 |
| 43 | 97.06 | 97.06 | 97.02 | 481 |
| 44 | 96.90 | 96.90 | 96.90 | 486 |

The selected epochs occurred late in the 500-epoch runs, between epochs 459 and 486. This supports increasing the final reproducibility schedule to 750 epochs because the previous model was still finding its best checkpoint late in training. The 750-epoch run is intended to test whether the same architecture remains stable across more seeds and a longer schedule.

## Improvement rationale

The main improvement was moving away from simply making the model deeper. Under a strict 10-weighted-layer budget, depth is expensive because each new convolution consumes a large fraction of the allowed model path. The selected architecture instead uses width aggressively, with 384 channels in the first block and 768 channels in later blocks. This gives the classifier a large feature basis while staying within the layer budget.

The second improvement was placing extra computation in the low-resolution tail. After the early blocks reduce spatial size, the tail can add residual refinement at lower computational cost. This matters because the tail has three counted convolutional layers, but those layers operate after pooling rather than at the original 32 by 32 resolution.

The third improvement was using parameter-light components that do not add weighted layers. `ChannelGate` recalibrates channel responses without adding a `Linear` or `Conv2d` layer. `DropPath` regularizes the residual tail without adding weighted depth. `BNBias` and GELU improve optimization and nonlinearity without consuming the layer budget.

The fourth improvement was initialization and training polish. The convolutional blocks use Dirac initialization where appropriate, which makes residual-style paths start from a stable mapping. The training recipe uses random crop, horizontal flip, RandAugment, random erasing, mixup, label smoothing, SGD with Nesterov momentum, cosine warmup scheduling, EMA checkpointing, AMP, and channels-last layout. These changes are intended to improve both final accuracy and run-to-run reproducibility.

## Failed and rejected improvement attempts

The project included more than 10 attempts or design directions that were not kept in the final single-model submission. Some failed because they had lower validation accuracy; others were rejected because they violated the single-model or 10-layer requirement.

| # | Attempt | Result and reason it was not selected |
| ---: | --- | --- |
| 1 | `DeepScaled10` | Lower prior 3-seed mean validation accuracy, 96.37 vs 96.97 for `UltraWideScaledTail10`. |
| 2 | `WideScaledPlus10` | Strong but still lower prior 3-seed mean validation accuracy, 96.70 vs 96.97. |
| 3 | Plain deeper scaling | Depth alone did not beat the width-first design under the 10-layer budget. `DeepScaled10` is the clearest example. |
| 4 | Parameter-efficient smaller model preference | `DeepScaled10` used far fewer parameters, about 5.20M, but the reduced capacity cost validation accuracy. |
| 5 | Historical `UltraWide7` | Older 7-layer style result reached 95.14 official test accuracy, below the later selected validation performance. |
| 6 | Historical `WideScaled10` | Earlier width-plus-tail version reached 94.83 official test accuracy, below the final selected model evidence. |
| 7 | `GhostScaled10` | Compact ghost-style module reached 94.52 official test accuracy, not enough to replace the wide model. |
| 8 | `CoordAtt10` | Coordinate-attention style module reached 94.52 official test accuracy, not enough to replace the wide model. |
| 9 | Learned feature-concatenation ensemble | A unified ensemble head over three feature extractors was not a valid single 10-layer model; the counted parameter modules reached 28 total modules. |
| 10 | Weighted logit ensemble | A Wilson-weighted ensemble improved complementarity but was rejected because it combined multiple models and exceeded the single-model depth budget. |
| 11 | Keeping all top-three models in the final notebook | This made the notebook harder to grade and did not answer the requirement to submit the best single model. It was removed. |
| 12 | Shorter training screens as final evidence | The selected prior checkpoints occurred around epochs 459 to 486, so short runs were useful for screening but not sufficient for final reproducibility evidence. |

These failed attempts were still useful. They showed that the best route was not maximum architectural novelty or smallest parameter count. The repeatable signal was that wide feature extraction plus a low-resolution residual tail worked best inside the 10-layer constraint.

## Final claim

The final single-model submission is `UltraWideScaledTail10`. It satisfies the 10-weighted-layer rule, has 21,337,656 parameters, and was selected because it had the best prior multi-seed out-of-sample validation performance among the tested single models. The final reproducibility script is configured for 8 seeds and 750 epochs per seed on the 8x A40 RunPod setup, with local smoke verification completed and saved under `local_smoke_repro_runs/`.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `hunter29_top3_depth_limited_models.ipynb` | Streamlined notebook showing the selected model, audit, and reproducibility configuration |
| `train_ultrawidescaledtail10_repro.py` | Single-model training script with top-of-file configuration |
| `.claude_resources.json` | Local resource detection used to choose the parallelization strategy |
| `local_smoke_repro_runs/summary.csv` | Local 3-seed smoke-test summary |
| `local_smoke_repro_runs/summary.json` | Local 3-seed smoke-test detailed summary |
