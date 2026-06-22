from __future__ import annotations

import contextlib
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Must be set before importing torch when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_KEY = "ultrawidescaledtail10"
MODEL_NAME = "UltraWideScaledTail10"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# =============================================================================
# Run configuration
# =============================================================================
# Final reproducibility setting:
#   python train_ultrawidescaledtail10_repro.py
#
# The final RunPod run uses 8 seeds and 750 epochs per seed so an 8x A40 pod can
# train one seed per GPU in a single wave. If the professor only
# wants to verify that the script works, temporarily change RUN_SEEDS to three
# seeds, for example:
#   RUN_SEEDS = (42, 43, 44)
#
# For an even faster local smoke test, also set RUN_EPOCHS to a small number and
# RUN_MAX_TRAIN_BATCHES to 4 or 8. Restore the final settings before reporting
# reproducibility results.
#
# Colab note: this file also works if pasted into a Colab cell. Colab injects
# hidden runtime arguments, so this script intentionally ignores command-line
# arguments and reads only this RUN_* block.
#
# Parallelism note: "auto" uses all available CUDA GPUs for seed-level
# parallelism. On a one-GPU machine it runs seeds sequentially to avoid GPU
# memory contention, while still parallelizing data loading with RUN_NUM_WORKERS.
# On an 8-GPU pod, the default runs exactly 8 seeds, one seed per GPU. For a
# larger two-wave run, use tuple(range(42, 58)) for 16 seeds.
RUN_SEEDS = tuple(range(42, 50))
RUN_EPOCHS = 750
RUN_BATCH_SIZE = 128
RUN_NUM_WORKERS = 32
RUN_PARALLEL_SEED_WORKERS = "auto"
RUN_AUG = "heavy"
RUN_LR = 0.1
RUN_WD = 5e-4
RUN_MOMENTUM = 0.9
RUN_MIXUP = 0.2
RUN_LABEL_SMOOTHING = 0.1
RUN_EMA_DECAY = 0.999
RUN_CHECKPOINT_METRIC = "best"
RUN_PATIENCE = None
RUN_USE_AMP = True
RUN_CHANNELS_LAST = True
RUN_MAX_TRAIN_BATCHES = None
RUN_EVAL_TEST_AT_END = False
RUN_DETERMINISTIC = True
RUN_DOWNLOAD = True
RUN_LOG_EVERY = 1
RUN_DATA_DIR = "./data"
RUN_OUTPUT_DIR = "./repro_runs_ultrawidescaledtail10_8seed"

DEFAULT_SEEDS = RUN_SEEDS


class BNBias(nn.BatchNorm2d):
    def __init__(self, c: int, momentum: float = 0.4):
        super().__init__(c, momentum=momentum, affine=True)
        nn.init.ones_(self.weight)
        self.weight.requires_grad = False


class ChannelGate(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), init_scale))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3), keepdim=True)
        scale = torch.sigmoid(avg * self.weight + self.bias)
        return x * (1.0 + scale)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div(keep_prob)


class AirBlock(nn.Module):
    def __init__(self, cin: int, cout: int, extra: bool = False, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.bn1 = BNBias(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = BNBias(cout)
        self.extra = extra
        if extra:
            self.conv3 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.bn3 = BNBias(cout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.pool(self.conv1(x))))
        if not self.extra:
            return self.act(self.bn2(self.conv2(x)))
        residual = x
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act(x + residual)


class SingleConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.bn = BNBias(cout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pool(self.conv(x))))


def init_dirac_(conv: nn.Conv2d) -> None:
    nn.init.dirac_(conv.weight.data, groups=conv.groups)


class UltraWideScaledTail(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False)
        self.bn1 = BNBias(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False)
        self.bn2 = BNBias(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn3 = BNBias(channels)
        self.res_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.channel_gate = ChannelGate(channels)
        self.drop_path = DropPath(0.03)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.channel_gate(x)
        return self.act(residual + self.drop_path(self.res_scale * x))


class UltraWideScaledTail10(nn.Module):
    def __init__(self, num_classes: int = 10, scale: float = 1 / 9):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.stem_act = nn.GELU()
        self.block1 = AirBlock(24, 384, extra=False, pool=True)
        self.block2 = AirBlock(384, 768, extra=False, pool=True)
        self.block3 = SingleConvBlock(768, 768, pool=True)
        self.tail = UltraWideScaledTail(768)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(768, num_classes, bias=False)
        self.scale = scale
        self._apply_init()

    def _apply_init(self) -> None:
        for block in (self.block1, self.block2):
            init_dirac_(block.conv1)
            init_dirac_(block.conv2)
        init_dirac_(self.block3.conv)
        init_dirac_(self.tail.conv1)
        init_dirac_(self.tail.conv2)
        init_dirac_(self.tail.conv3)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.tail(x)
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.scale


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_param_layers(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear))
    )


def weighted_layer_rows(model: nn.Module) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            rows.append(
                {
                    "layer_index": len(rows) + 1,
                    "module": name or "<root>",
                    "type": module.__class__.__name__,
                    "weight_shape": tuple(module.weight.shape),
                    "groups": getattr(module, "groups", None),
                }
            )
    return rows


@dataclass
class TrainConfig:
    seeds: tuple[int, ...] = RUN_SEEDS
    epochs: int = RUN_EPOCHS
    batch_size: int = RUN_BATCH_SIZE
    num_workers: int = RUN_NUM_WORKERS
    parallel_seed_workers: int | str = RUN_PARALLEL_SEED_WORKERS
    aug: str = RUN_AUG
    lr: float = RUN_LR
    wd: float = RUN_WD
    momentum: float = RUN_MOMENTUM
    mixup: float = RUN_MIXUP
    label_smoothing: float = RUN_LABEL_SMOOTHING
    ema_decay: float = RUN_EMA_DECAY
    checkpoint_metric: str = RUN_CHECKPOINT_METRIC
    patience: int | None = RUN_PATIENCE
    use_amp: bool = RUN_USE_AMP
    channels_last: bool = RUN_CHANNELS_LAST
    max_train_batches: int | None = RUN_MAX_TRAIN_BATCHES
    eval_test_at_end: bool = RUN_EVAL_TEST_AT_END
    deterministic: bool = RUN_DETERMINISTIC
    download: bool = RUN_DOWNLOAD
    log_every: int = RUN_LOG_EVERY
    data_dir: str = RUN_DATA_DIR
    output_dir: str = RUN_OUTPUT_DIR
    cuda_device: int | None = None


def set_training_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_training_device(config: TrainConfig) -> torch.device:
    if torch.cuda.is_available():
        if config.cuda_device is not None:
            torch.cuda.set_device(config.cuda_device)
            return torch.device(f"cuda:{config.cuda_device}")
        return torch.device("cuda")
    return torch.device("cpu")


def make_transforms(aug: str):
    import torchvision.transforms as transforms

    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if aug == "light":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif aug == "heavy":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=14),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ]
        )
    else:
        raise ValueError(f"unknown augmentation preset: {aug}")
    return train_tf, eval_tf


def make_loaders(config: TrainConfig, seed: int):
    import torchvision
    from torch.utils.data import DataLoader, Subset

    train_tf, eval_tf = make_transforms(config.aug)
    data_root = Path(config.data_dir)
    full_train = torchvision.datasets.CIFAR10(
        data_root, train=True, download=config.download, transform=train_tf
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        data_root, train=True, download=config.download, transform=eval_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        data_root, train=False, download=config.download, transform=eval_tf
    )

    split_gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(50_000, generator=split_gen).tolist()
    train_idx, val_idx = perm[:45_000], perm[45_000:]
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    loader_gen = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    worker_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": pin,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_gen,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=512,
        shuffle=False,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        **worker_kwargs,
    )
    return train_loader, val_loader, test_loader


def ensure_cifar10_available(config: TrainConfig) -> None:
    if not config.download:
        return
    import torchvision
    import torchvision.transforms as transforms

    data_root = Path(config.data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] ensuring CIFAR-10 is available at {data_root}")
    placeholder_tf = transforms.ToTensor()
    torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=placeholder_tf
    )
    torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=placeholder_tf
    )


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cosine_warmup_lambda(total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_accuracy_and_loss(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with amp_context(device, use_amp):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, loss_sum / total


def selected_checkpoint(
    checkpoint_metric: str,
    best_val_raw: float,
    best_raw_epoch: int,
    best_raw_state: dict[str, torch.Tensor],
    best_val_ema: float,
    best_ema_epoch: int,
    best_ema_state: dict[str, torch.Tensor],
):
    if checkpoint_metric == "raw":
        return "raw", best_val_raw, best_raw_epoch, best_raw_state
    if checkpoint_metric == "ema":
        return "ema", best_val_ema, best_ema_epoch, best_ema_state
    if checkpoint_metric == "best":
        if best_val_ema > best_val_raw:
            return "ema", best_val_ema, best_ema_epoch, best_ema_state
        return "raw", best_val_raw, best_raw_epoch, best_raw_state
    raise ValueError(f"unknown checkpoint metric: {checkpoint_metric}")


def train_one_seed(config: TrainConfig, seed: int) -> dict:
    if config.checkpoint_metric not in {"raw", "ema", "best"}:
        raise ValueError("checkpoint_metric must be one of: raw, ema, best")

    set_training_seed(seed, deterministic=config.deterministic)
    device = get_training_device(config)
    train_loader, val_loader, test_loader = make_loaders(config, seed)

    model = UltraWideScaledTail10().to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    n_layers = count_param_layers(model)
    if n_layers > 10:
        raise RuntimeError(f"{MODEL_NAME} has {n_layers} weighted layers")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.wd,
        nesterov=True,
    )
    steps_per_epoch = (
        min(len(train_loader), config.max_train_batches)
        if config.max_train_batches is not None
        else len(train_loader)
    )
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = min(steps_per_epoch * 5, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps)
    )
    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay),
    )
    ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{MODEL_KEY}_e{config.epochs}_s{seed}"
    ckpt_path = output_dir / f"{tag}.pt"
    json_path = output_dir / f"{tag}.json"

    best_val_raw = 0.0
    best_val_ema = 0.0
    best_raw_epoch = 0
    best_ema_epoch = 0
    stale_raw = 0
    stale_ema = 0
    best_raw_state = None
    best_ema_state = None
    history = []
    start_time = time.time()

    print(
        f"[{tag}] model={MODEL_NAME} layers={n_layers} params={n_params(model):,} "
        f"device={device} epochs={config.epochs} seed={seed} batch_size={config.batch_size}"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_seen = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if config.channels_last and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            xm, ya, yb, lam = mixup_batch(x, y, config.mixup)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, config.use_amp):
                logits = model(xm)
                loss = lam * ce(logits.float(), ya) + (1.0 - lam) * ce(logits.float(), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)

            train_loss_sum += float(loss.detach().cpu()) * x.size(0)
            train_seen += x.size(0)

        val_raw, val_loss_raw = evaluate_accuracy_and_loss(
            model, val_loader, device, config.use_amp, config.channels_last
        )
        val_ema, val_loss_ema = evaluate_accuracy_and_loss(
            ema, val_loader, device, config.use_amp, config.channels_last
        )

        raw_improved = val_raw > best_val_raw
        ema_improved = val_ema > best_val_ema
        if raw_improved:
            best_val_raw = val_raw
            best_raw_epoch = epoch
            stale_raw = 0
            best_raw_state = clone_state_dict(model)
        else:
            stale_raw += 1
        if ema_improved:
            best_val_ema = val_ema
            best_ema_epoch = epoch
            stale_ema = 0
            best_ema_state = clone_state_dict(ema.module)
        else:
            stale_ema += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_seen),
            "val_raw": val_raw,
            "val_loss_raw": val_loss_raw,
            "val_ema": val_ema,
            "val_loss_ema": val_loss_ema,
            "lr": scheduler.get_last_lr()[0],
            "epoch_sec": time.time() - epoch_start,
        }
        history.append(row)

        should_log = (
            epoch == 1
            or epoch == config.epochs
            or raw_improved
            or ema_improved
            or (config.log_every > 0 and epoch % config.log_every == 0)
        )
        if should_log:
            print(
                f"[{tag}] ep {epoch:03d}/{config.epochs} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_raw={val_raw:.2f} val_ema={val_ema:.2f} "
                f"best_raw={best_val_raw:.2f}@{best_raw_epoch} "
                f"best_ema={best_val_ema:.2f}@{best_ema_epoch} "
                f"lr={row['lr']:.5f} time={row['epoch_sec']:.1f}s"
            )

        if (
            config.patience is not None
            and stale_raw >= config.patience
            and stale_ema >= config.patience
        ):
            print(f"[{tag}] early stop after {config.patience} stale epochs")
            break

    if best_raw_state is None:
        best_raw_state = clone_state_dict(model)
    if best_ema_state is None:
        best_ema_state = clone_state_dict(ema.module)

    selected_kind, selected_acc, selected_epoch, selected_state = selected_checkpoint(
        config.checkpoint_metric,
        best_val_raw,
        best_raw_epoch,
        best_raw_state,
        best_val_ema,
        best_ema_epoch,
        best_ema_state,
    )

    test_acc = None
    if config.eval_test_at_end:
        eval_model = UltraWideScaledTail10().to(device)
        eval_model.load_state_dict({k: v.to(device) for k, v in selected_state.items()})
        if config.channels_last and device.type == "cuda":
            eval_model = eval_model.to(memory_format=torch.channels_last)
        test_acc, _ = evaluate_accuracy_and_loss(
            eval_model, test_loader, device, config.use_amp, config.channels_last
        )

    result = {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "seed": seed,
        "config": asdict(config),
        "n_layers": n_layers,
        "n_params": n_params(model),
        "best_val_raw": best_val_raw,
        "best_raw_epoch": best_raw_epoch,
        "best_val_ema": best_val_ema,
        "best_ema_epoch": best_ema_epoch,
        "checkpoint_metric": config.checkpoint_metric,
        "selected_checkpoint": selected_kind,
        "selected_val_acc": selected_acc,
        "selected_epoch": selected_epoch,
        "test_acc": test_acc,
        "wall_sec": time.time() - start_time,
        "history": history,
        "checkpoint_path": str(ckpt_path),
        "json_path": str(json_path),
    }
    torch.save(
        {
            "model_key": MODEL_KEY,
            "model_name": MODEL_NAME,
            "config": asdict(config),
            "seed": seed,
            "raw_state_dict": best_raw_state,
            "ema_state_dict": best_ema_state,
            "selected_state_dict": selected_state,
            "result": result,
        },
        ckpt_path,
    )
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{tag}] done selected_{selected_kind}={selected_acc:.2f}@{selected_epoch} "
        f"wall={result['wall_sec'] / 60:.1f}min saved={ckpt_path}"
    )
    return result


def summarize_results(results: list[dict]) -> dict:
    selected = np.array([r["selected_val_acc"] for r in results], dtype=float)
    raw = np.array([r["best_val_raw"] for r in results], dtype=float)
    ema = np.array([r["best_val_ema"] for r in results], dtype=float)
    summary = {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "seeds": [r["seed"] for r in results],
        "num_seeds": len(results),
        "epochs": results[0]["config"]["epochs"] if results else None,
        "mean_selected_val_acc": float(selected.mean()) if len(selected) else None,
        "std_selected_val_acc": float(selected.std(ddof=1)) if len(selected) > 1 else 0.0,
        "min_selected_val_acc": float(selected.min()) if len(selected) else None,
        "max_selected_val_acc": float(selected.max()) if len(selected) else None,
        "mean_best_raw": float(raw.mean()) if len(raw) else None,
        "mean_best_ema": float(ema.mean()) if len(ema) else None,
    }
    return summary


def write_sweep_outputs(config: TrainConfig, results: list[dict], summary: dict) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model_key": r["model_key"],
            "seed": r["seed"],
            "epochs": r["config"]["epochs"],
            "selected_checkpoint": r["selected_checkpoint"],
            "selected_val_acc": r["selected_val_acc"],
            "selected_epoch": r["selected_epoch"],
            "best_val_raw": r["best_val_raw"],
            "best_raw_epoch": r["best_raw_epoch"],
            "best_val_ema": r["best_val_ema"],
            "best_ema_epoch": r["best_ema_epoch"],
            "test_acc": r["test_acc"],
            "wall_min": r["wall_sec"] / 60,
            "checkpoint_path": r["checkpoint_path"],
            "json_path": r["json_path"],
        }
        for r in results
    ]
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "runs": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] mean selected val={summary['mean_selected_val_acc']:.2f} "
          f"+/- {summary['std_selected_val_acc']:.2f}")


def resolve_parallel_seed_workers(config: TrainConfig) -> int:
    requested = config.parallel_seed_workers
    if requested == "auto":
        if torch.cuda.is_available():
            return max(1, min(len(config.seeds), torch.cuda.device_count()))
        cpu_count = os.cpu_count() or 1
        return max(1, min(len(config.seeds), max(1, cpu_count // 4)))

    workers = int(requested)
    if workers < 1:
        raise ValueError("parallel_seed_workers must be >= 1 or 'auto'")
    if torch.cuda.is_available():
        gpu_count = max(1, torch.cuda.device_count())
        if workers > gpu_count:
            print(
                f"[sweep] requested {workers} seed workers, but only {gpu_count} CUDA "
                "device(s) are visible; capping seed workers to avoid GPU contention"
            )
            workers = gpu_count
    return min(workers, len(config.seeds))


def train_seed_worker(args: tuple[TrainConfig, int, int | None, int]) -> dict:
    config, seed, cuda_device, loader_workers = args
    worker_config = replace(
        config,
        cuda_device=cuda_device,
        num_workers=loader_workers,
    )
    return train_one_seed(worker_config, seed)


def run_seed_sweep(config: TrainConfig) -> tuple[list[dict], dict]:
    ensure_cifar10_available(config)
    model = UltraWideScaledTail10()
    layers = count_param_layers(model)
    if layers != 10:
        raise RuntimeError(f"expected 10 weighted layers, found {layers}")
    seed_workers = resolve_parallel_seed_workers(config)
    print(
        f"[sweep] selected model={MODEL_NAME} layers={layers} "
        f"params={n_params(model):,} seeds={list(config.seeds)} epochs={config.epochs} "
        f"seed_workers={seed_workers} loader_workers={config.num_workers}"
    )
    if seed_workers == 1:
        results = [train_one_seed(config, seed) for seed in config.seeds]
    else:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        loader_workers = max(0, config.num_workers // seed_workers)
        jobs = []
        for job_idx, seed in enumerate(config.seeds):
            cuda_device = job_idx % gpu_count if gpu_count else None
            jobs.append((config, seed, cuda_device, loader_workers))
        print(
            f"[sweep] running {len(jobs)} seed jobs with {seed_workers} parallel "
            f"processes and {loader_workers} DataLoader workers per process"
        )
        results = []
        # Spawn avoids CUDA re-initialization problems that can happen with
        # forked subprocesses on Linux GPU pods.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=seed_workers, mp_context=context) as executor:
            futures = [executor.submit(train_seed_worker, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: row["seed"])
    summary = summarize_results(results)
    write_sweep_outputs(config, results, summary)
    return results, summary


def build_config() -> TrainConfig:
    return TrainConfig(
        seeds=RUN_SEEDS,
        epochs=RUN_EPOCHS,
        batch_size=RUN_BATCH_SIZE,
        num_workers=RUN_NUM_WORKERS,
        parallel_seed_workers=RUN_PARALLEL_SEED_WORKERS,
        aug=RUN_AUG,
        lr=RUN_LR,
        wd=RUN_WD,
        momentum=RUN_MOMENTUM,
        mixup=RUN_MIXUP,
        label_smoothing=RUN_LABEL_SMOOTHING,
        ema_decay=RUN_EMA_DECAY,
        checkpoint_metric=RUN_CHECKPOINT_METRIC,
        patience=RUN_PATIENCE,
        use_amp=RUN_USE_AMP,
        channels_last=RUN_CHANNELS_LAST,
        max_train_batches=RUN_MAX_TRAIN_BATCHES,
        eval_test_at_end=RUN_EVAL_TEST_AT_END,
        deterministic=RUN_DETERMINISTIC,
        download=RUN_DOWNLOAD,
        log_every=RUN_LOG_EVERY,
        data_dir=RUN_DATA_DIR,
        output_dir=RUN_OUTPUT_DIR,
    )


def main() -> None:
    config = build_config()
    print("Using top-of-file RUN_* configuration.")
    print("Run command for .py usage: python train_ultrawidescaledtail10_repro.py")
    run_seed_sweep(config)


if __name__ == "__main__":
    main()
