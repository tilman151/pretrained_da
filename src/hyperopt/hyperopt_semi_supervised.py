import os
from functools import partial

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import building
import datasets
from lightning import loggers


def tune_pretraining(config, arch_config, source, percent_broken):
    logger = pl_loggers.TensorBoardLogger(
        _get_hyperopt_logdir(),
        loggers.semi_supervised_hyperopt_name(source, percent_broken),
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
    trainer = building.build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=1,
        seed=None,
        callbacks=[tune_callback],
        check_sanity=False,
    )

    data = datasets.PretrainingBaselineDataModule(
        fd_source=source,
        num_samples=50000,
        batch_size=config["batch_size"],
        percent_broken=percent_broken,
        percent_fail_runs=0.0,
        min_distance=config["min_distance"],
        truncate_val=True,
    )
    model = building.build_pretraining_from_config(
        arch_config, config, data.window_size, record_embeddings=False, use_adaption=False
    )
    building.add_hparams(model, data, 42)

    trainer.fit(model, datamodule=data)


def _get_hyperopt_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", "..", "hyperopt"))

    return log_dir


def optimize_pretraining(source, percent_broken, arch_config, num_trials):
    config = {
        "domain_tradeoff": tune.choice([0.0]),
        "dropout": tune.choice([0.0, 0.1, 0.3, 0.5]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([256, 512]),
        "min_distance": tune.choice([1, 10, 15, 30]),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=200, grace_period=10, reduction_factor=2
    )
    reporter = tune.CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["checkpoint_score", "target_loss", "domain_loss"],
    )

    tune_func = partial(
        tune_pretraining,
        arch_config=arch_config,
        source=source,
        percent_broken=percent_broken,
    )
    analysis = tune.run(
        tune_func,
        resources_per_trial={"cpu": 6, "gpu": 1},
        metric="checkpoint_score",
        mode="min",
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pretraining_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    return analysis.best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for pretraining"
    )
    parser.add_argument("--source", type=int, required=True, help="source FD number")
    parser.add_argument(
        "--percent_broken", type=float, required=True, help="degradation in [0, 1]"
    )
    parser.add_argument("arch_config_path", help="path to architecture config JSON")
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of hyperopt trials"
    )
    opt = parser.parse_args()

    _arch_config = building.load_config(opt.arch_config_path)
    optimize_pretraining(opt.source, opt.percent_broken, _arch_config, opt.num_trials)
