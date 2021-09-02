import random
import re
from datetime import datetime

import mlflow
import numpy as np

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
    skip_epochs = trainer.checkpoint_callback.min_epochs_before_saving + 1
    mlflow_client: mlflow.tracking.MlflowClient = trainer.logger.mlflow_experiment
    for metric in metrics:
        val_metric_name = f"val/{metric}"
        history = mlflow_client.get_metric_history(run_id, val_metric_name)
        history = history[skip_epochs:]
        min_epoch = np.argmin([step.value for step in history]).squeeze() + skip_epochs
        checkpoint_path = _get_epoch2ckpt_dict(
            trainer.checkpoint_callback.best_k_models.keys()
        )[min_epoch]
        trainer.model.test_tag = metric
        mlflow_client.set_tag(run_id, f"{metric}_epoch", min_epoch)
        mlflow_client.set_tag(
            run_id, f"{metric}_step", history[min_epoch - skip_epochs].step
        )
        trainer.test(ckpt_path=checkpoint_path, test_dataloaders=data.val_dataloader())


def _get_epoch2ckpt_dict(checkpoint_paths):
    pattern = re.compile(r"epoch=(?P<epoch>\d+)-step=\d+.ckpt")
    d = {}
    for ckpt_path in checkpoint_paths:
        matcher = pattern.search(ckpt_path)
        epoch = int(matcher.group("epoch"))
        d[epoch] = ckpt_path

    return d


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
