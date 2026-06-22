# RunPod Upload Instructions

Upload this project zip with the RunPod custom uploader, then run the commands below inside the pod.

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

The 8-seed and 16-seed launchers use all 8 GPUs. The 5-seed launcher uses 5 GPUs.

## Monitor

```bash
watch -n 2 nvidia-smi
```

```bash
tail -f full_5seed_run.log
```

## Outputs

Results are written to:

```text
/workspace/finalvers/repro_runs_ultrawidescaledtail10_5seed/
```

The most important files are:

```text
summary.csv
summary.json
ultrawidescaledtail10_e750_s*.json
ultrawidescaledtail10_e750_s*.pt
```
