import random
from datetime import datetime

import mlflow
import numpy as np
import pytorch_lightning as pl

import building


def run(
    source,
    target,
    percent_broken,
    config,
    record_embeddings,
    seed,
    gpu,
    pretrained_encoder_path,
    version,
):
    trainer, data, model = building.build_transfer(
        source,
        target,
        percent_broken,
        config,
        pretrained_encoder_path,
        record_embeddings,
        gpu,
        seed,
        version,
    )
    trainer.fit(model, datamodule=data)
    _test_for_each_checkpoint_metric(data, trainer)


def _test_for_each_checkpoint_metric(data, trainer):
    metrics = ["regression_loss", "score", "source_regression_loss"]
    run_id = trainer.logger.run_id
    mlflow_client = trainer.logger.mlflow_experiment
    for metric in metrics:
        val_metric_name = f"val/{metric}"
        history = mlflow_client.get_metric_history(run_id, val_metric_name)
        min_epoch = np.argmin([step.value for step in history]).squeeze()
        checkpoint_path = sorted(trainer.checkpoint_callback.best_k_models.keys())[
            min_epoch
        ]
        trainer.model.test_tag = metric
        trainer.test(ckpt_path=checkpoint_path, test_dataloaders=data.val_dataloader())


def run_multiple(
    source,
    target,
    broken,
    config,
    record_embeddings,
    replications,
    gpu,
    pretrained_encoder_path,
    version=None,
):
    broken = broken if broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]
    if version is None:
        version = datetime.now().timestamp()

    for b in broken:
        for s in seeds:
            run(
                source,
                target,
                b,
                config,
                record_embeddings,
                s,
                gpu,
                pretrained_encoder_path,
                version,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run domain adaption experiment")
    parser.add_argument("--source", type=int, help="FD number of the source data")
    parser.add_argument("--target", type=int, help="FD number of the target data")
    parser.add_argument(
        "--pretrained_encoder",
        default=None,
        help="Path to checkpoint file form pretraining",
    )
    parser.add_argument(
        "-b", "--broken", nargs="*", type=float, help="percent broken to use"
    )
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--record_embeddings",
        action="store_true",
        help="whether to record embeddings of val data",
    )
    parser.add_argument(
        "-r", "--replications", type=int, default=3, help="replications for each run"
    )
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    opt = parser.parse_args()

    _config = building.load_config(opt.config)
    run_multiple(
        opt.source,
        opt.target,
        opt.broken,
        _config,
        opt.record_embeddings,
        opt.replications,
        opt.gpu,
        opt.pretrained_encoder,
    )
