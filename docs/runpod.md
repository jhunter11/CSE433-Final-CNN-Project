# Run on RunPod

Copy the project to `/workspace/finalvers` on a pod with PyTorch, torchvision, and access to CIFAR-10.
Run these commands from the repository root. Review GPU capacity and the configuration preview before starting training.

## Check the environment

```bash
cd /workspace/finalvers
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
python run_experiment.py --seed-count 8 --dry-run
```

## Start a sweep

```bash
mkdir -p runs
set -o pipefail
python -u run_experiment.py --seed-count 8 2>&1 | tee runs/eight-seed.log
```

Use `--seed-count 5` or `--seed-count 16` to change the sweep size.
On eight GPUs, eight seeds use one wave and sixteen seeds use two waves.
Automatic worker selection adapts to the available GPU count.
The default schedule is 750 epochs per seed.

## Inspect progress and outputs

```bash
nvidia-smi
tail -f runs/eight-seed.log
```

The launcher writes checkpoints, per-seed JSON logs, `summary.csv`, and `summary.json` under `runs/ultrawidescaledtail10_8seed/`.
Other seed counts have separate output directories. Set `--output-dir` to keep repeated runs separate.
Checkpoint selection uses validation accuracy with `eval_test_at_end=False`. These summaries do not establish official test accuracy.

The earlier `run_full_5seed.py`, `run_full_8seed.py`, and `run_full_16seed.py` launchers now share this CLI.
Their original training settings remain the defaults. New output directories keep generated files outside the committed experiment record.
