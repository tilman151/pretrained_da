import os

import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import datasets
from lightning import pretraining, loggers
import building.build as build


def tune_pretraining(config, arch_config, source, target, percent_broken):
    logger = loggers.MLTBLogger(
        _get_hyperopt_logdir(),
        loggers.pretraining_hyperopt_name(source, target, percent_broken),
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/checkpoint_score")
    tune_callback = TuneReportCallback(
        {
            "checkpoint_score": "val/checkpoint_score",
            "target_loss": "val/regression_loss",
            "domain_loss": "val/domain_loss",
        },
        on="validation_end",
    )
    trainer = build.build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=1,
        seed=42,
        callbacks=[tune_callback],
        check_sanity=False,
    )

    data = datasets.PretrainingAdaptionDataModule(
        fd_source=source,
        fd_target=target,
        num_samples=50000,
        batch_size=config["batch_size"],
        percent_broken=percent_broken,
        truncate_target_val=True,
    )
    model = pretraining.UnsupervisedPretraining(
        in_channels=14,
        seq_len=data.window_size,
        num_layers=arch_config["num_layers"],
        kernel_size=3,
        base_filters=arch_config["base_filters"],
        latent_dim=arch_config["latent_dim"],
        dropout=config["dropout"],
        domain_tradeoff=config["domain_tradeoff"],
        domain_disc_dim=arch_config["latent_dim"],
        num_disc_layers=arch_config["num_disc_layers"],
        lr=config["lr"],
        record_embeddings=False,
        weight_decay=0.0,
    )
    build.add_hparams(model, data, 42)

    trainer.fit(model, datamodule=data)


def _get_hyperopt_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", "hyperopt"))

    return log_dir


def tune_loop(source, target, percent_broken, num_trials):
    arch_config = {
        "num_layers": 8,
        "base_filters": 16,
        "latent_dim": 16,
        "num_disc_layers": 1,
    }
    config = {
        "domain_tradeoff": tune.loguniform(1e-3, 10),
        "dropout": tune.choice([0.0, 0.1, 0.3, 0.5]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([256, 512]),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=200, grace_period=1, reduction_factor=2
    )
    reporter = tune.CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["source_reg_loss", "target_loss", "domain_loss"],
    )

    analysis = tune.run(
        lambda c: tune_pretraining(
            c, arch_config, source=source, target=target, percent_broken=percent_broken
        ),
        resources_per_trial={"cpu": 6, "gpu": 1},
        metric="source_reg_loss",
        mode="min",
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for pretraining"
    )
    parser.add_argument("--source", type=int, required=True, help="source FD number")
    parser.add_argument("--target", type=int, required=True, help="target FD number")
    parser.add_argument(
        "--percent_broken", type=float, required=True, help="degradation in [0, 1]"
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of hyperopt trials"
    )
    opt = parser.parse_args()

    tune_loop(opt.source, opt.target, opt.percent_broken, opt.num_trials)
