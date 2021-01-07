import os
import random

import sklearn.model_selection

from building.build import build_transfer
from lightning import loggers as loggers


def run(
    source,
    target,
    percent_broken,
    domain_tradeoff,
    record_embeddings,
    seed,
    gpu,
    pretrained_encoder_path,
):
    logger = loggers.MLTBLogger(
        _get_logdir(),
        loggers.transfer_experiment_name(source, target),
        tensorboard_struct={"pb": percent_broken, "dt": domain_tradeoff},
    )
    trainer, data, model = build_transfer(
        source,
        target,
        percent_broken,
        pretrained_encoder_path,
        record_embeddings,
        domain_tradeoff,
        logger,
        gpu,
        seed,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def _get_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, ".."))

    return log_dir


def run_multiple(
    source,
    target,
    broken,
    domain_tradeoff,
    record_embeddings,
    replications,
    gpu,
    pretrained_encoder_path,
):
    broken = broken if broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    parameter_grid = {"domain_tradeoff": domain_tradeoff, "broken": broken}

    for params in sklearn.model_selection.ParameterGrid(parameter_grid):
        for s in seeds:
            run(
                source,
                target,
                params["broken"],
                params["domain_tradeoff"],
                record_embeddings,
                s,
                gpu,
                pretrained_encoder_path,
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
    parser.add_argument(
        "--domain_tradeoff",
        nargs="*",
        type=float,
        help="tradeoff for domain classification",
    )
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

    run_multiple(
        opt.source,
        opt.target,
        opt.broken,
        opt.domain_tradeoff,
        opt.record_embeddings,
        opt.replications,
        opt.gpu,
        opt.pretrained_encoder,
    )
