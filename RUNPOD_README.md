# RunPod Upload Instructions

Copy the project into `/workspace/finalvers` on a pod with PyTorch, torchvision, and access to CIFAR-10. Review GPU capacity and run configuration before starting a full sweep.

## Files included

| File | Purpose |
| --- | --- |
| `train_ultrawidescaledtail10_repro.py` | Main model and training implementation |
| `run_full_5seed.py` | 5-seed, 750-epoch launcher |
| `run_full_8seed.py` | 8-seed launcher for exactly one wave on an 8-GPU pod |
| `run_full_16seed.py` | 16-seed launcher for two waves on an 8-GPU pod |
| `FINAL_REPORT_DRAFT.md` | Report draft |

## Verify GPUs

```bash
cd /workspace/finalvers
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Run on 8x A40

Recommended 8-seed run, one seed per GPU:

```bash
cd /workspace/finalvers
python -u run_full_8seed.py 2>&1 | tee full_8seed_run.log
```

Larger 16-seed run, two waves of eight seeds:

```bash
cd /workspace/finalvers
python -u run_full_16seed.py 2>&1 | tee full_16seed_run.log
```

Smaller 5-seed run:

```bash
cd /workspace/finalvers
python -u run_full_5seed.py 2>&1 | tee full_5seed_run.log
```

On an eight-GPU pod, the eight-seed launcher runs one wave and the sixteen-seed launcher runs two waves. The five-seed launcher uses five GPUs. Automatic worker selection adapts to the available GPU count.

## Monitor

```bash
watch -n 2 nvidia-smi
```

```bash
tail -f full_8seed_run.log
```

## Outputs

Each launcher writes a separate output directory below the working directory:

| Launcher | Output directory |
| --- | --- |
| `run_full_5seed.py` | `repro_runs_ultrawidescaledtail10_5seed/` |
| `run_full_8seed.py` | `repro_runs_ultrawidescaledtail10_8seed/` |
| `run_full_16seed.py` | `repro_runs_ultrawidescaledtail10_16seed/` |

Inspect `summary.csv`, `summary.json`, and the per-seed JSON logs and checkpoint files.
The launchers select validation checkpoints with `eval_test_at_end=False`. Their summaries do not establish official test accuracy.
