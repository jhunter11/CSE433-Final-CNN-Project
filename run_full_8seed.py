from train_ultrawidescaledtail10_repro import TrainConfig, run_seed_sweep


if __name__ == "__main__":
    config = TrainConfig(
        seeds=(42, 43, 44, 45, 46, 47, 48, 49),
        epochs=750,
        batch_size=128,
        num_workers=32,
        parallel_seed_workers="auto",
        max_train_batches=None,
        eval_test_at_end=False,
        use_amp=True,
        channels_last=True,
        download=True,
        data_dir="./data",
        output_dir="./repro_runs_ultrawidescaledtail10_8seed",
        log_every=10,
    )
    run_seed_sweep(config)
