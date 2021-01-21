import os
from functools import partial

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import building
import datasets
from lightning import loggers


def tune_transfer(config, source, target, percent_broken):
    logger = pl_loggers.TensorBoardLogger(
        _get_hyperopt_logdir(),
        loggers.transfer_hyperopt_name(source, target, percent_broken),
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/source_regression_loss"
    )
    tune_callback = TuneReportCallback(
        {
            "source_reg_loss": "val/source_regression_loss",
            "target_loss": "val/regression_loss",
            "domain_loss": "val/domain_loss",
        },
        on="validation_end",
    )
    trainer = building.build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=200,
        val_interval=1.0,
        gpu=1,
        seed=42,
        callbacks=[tune_callback],
        check_sanity=False,
    )

    data = datasets.DomainAdaptionDataModule(
        fd_source=source,
        fd_target=target,
        batch_size=config["batch_size"],
        percent_broken=percent_broken,
    )
    model = building.build_dann_from_config(
        config, data.window_size, pretrained_encoder_path=None, record_embeddings=False
    )
    building.add_hparams(model, data, 42)

    trainer.fit(model, datamodule=data)


def _get_hyperopt_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", "..", "hyperopt"))

    return log_dir


def optimize_transfer(source, target, percent_broken, num_trials):
    config = {
        "num_layers": tune.choice([4, 6, 8]),
        "base_filters": tune.choice([16, 32, 64]),
        "domain_tradeoff": tune.choice([0.5, 1.0, 2.0]),
        "latent_dim": tune.choice([16, 32, 64, 128]),
        "dropout": tune.choice([0.0, 0.1, 0.3, 0.5]),
        "num_disc_layers": tune.choice([1, 2]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([256, 512]),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=100, grace_period=10, reduction_factor=2
    )
    reporter = tune.CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["source_reg_loss", "target_loss", "domain_loss"],
    )

    tune_func = partial(
        tune_transfer,
        source=source,
        target=target,
        percent_broken=percent_broken,
    )
    analysis = tune.run(
        tune_func,
        resources_per_trial={"cpu": 6, "gpu": 1},
        metric="source_reg_loss",
        mode="min",
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_transfer_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    return analysis.best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter optimization for DANN")
    parser.add_argument("--source", type=int, required=True, help="source FD number")
    parser.add_argument("--target", type=int, required=True, help="target FD number")
    parser.add_argument(
        "--percent_broken", type=float, required=True, help="degradation in [0, 1]"
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of hyperopt trials"
    )
    opt = parser.parse_args()

    optimize_transfer(opt.source, opt.target, opt.percent_broken, opt.num_trials)
