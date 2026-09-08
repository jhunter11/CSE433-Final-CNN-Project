"""Run a seed sweep or preview its configuration overrides without PyTorch."""

from __future__ import annotations

import argparse
import json


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def parallel_workers(value: str) -> int | str:
    return value if value == "auto" else positive_int(value)


def parse_configuration(argv: list[str] | None = None) -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=positive_int, default=8)
    parser.add_argument("--first-seed", type=nonnegative_int, default=42)
    parser.add_argument("--epochs", type=positive_int, default=750)
    parser.add_argument("--num-workers", type=nonnegative_int, default=32)
    parser.add_argument(
        "--parallel-seed-workers", type=parallel_workers, default="auto"
    )
    parser.add_argument("--max-train-batches", type=positive_int)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.first_seed + args.seed_count > 2**32:
        parser.error("seed range must fit unsigned 32-bit integers")
    config = {
        "seeds": tuple(range(args.first_seed, args.first_seed + args.seed_count)),
        "epochs": args.epochs,
        "batch_size": 128,
        "num_workers": args.num_workers,
        "parallel_seed_workers": args.parallel_seed_workers,
        "max_train_batches": args.max_train_batches,
        "eval_test_at_end": False,
        "use_amp": True,
        "channels_last": True,
        "download": not args.no_download,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir
        or f"./runs/ultrawidescaledtail10_{args.seed_count}seed",
        "log_every": 10,
    }
    return config, args.dry_run


def main(argv: list[str] | None = None) -> None:
    config, dry_run = parse_configuration(argv)
    if dry_run:
        print(json.dumps(config, indent=2))
        return

    from train_ultrawidescaledtail10_repro import TrainConfig, run_seed_sweep

    run_seed_sweep(TrainConfig(**config))


if __name__ == "__main__":
    main()
