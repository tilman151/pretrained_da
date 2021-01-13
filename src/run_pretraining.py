import os
import random

import sklearn

from building.build import build_pretraining


def run(
    source,
    target,
    percent_broken,
    arch_config,
    config,
    record_embeddings,
    seed,
    gpu,
):
    trainer, data, model = build_pretraining(
        source,
        target,
        percent_broken,
        arch_config,
        config,
        record_embeddings,
        gpu,
        seed,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)

    checkpoint_path = _get_checkpoint_path(trainer.logger)
    best_score = trainer.checkpoint_callback.best_model_score

    return checkpoint_path, best_score


def _get_checkpoint_path(logger):
    checkpoints_path = logger.checkpoint_path
    *_, checkpoint = sorted(
        [f for f in os.listdir(checkpoints_path)]
    )  # get last checkpoint
    checkpoint_path = os.path.join(checkpoints_path, checkpoint)

    return checkpoint_path


def run_multiple(
    source,
    target,
    broken,
    arch_config,
    config,
    record_embeddings,
    replications,
    gpu,
):
    broken = broken if broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    checkpoints = {b: [] for b in broken}
    for b in broken:
        for s in seeds:
            checkpoint_path = run(
                source,
                target,
                b,
                arch_config,
                config,
                record_embeddings,
                s,
                gpu,
            )
            checkpoints[b].append(checkpoint_path)

    return checkpoints


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run domain adaption experiment")
    parser.add_argument(
        "--source", type=int, required=True, help="FD number of the source data"
    )
    parser.add_argument("--target", type=int, help="FD number of the target data")
    parser.add_argument(
        "-b", "--broken", nargs="+", type=float, help="percent broken to use"
    )
    parser.add_argument(
        "--arch_config",
        required=True,
        help="path to architecture config file",
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

    run_multiple(
        opt.source,
        opt.target,
        opt.broken,
        opt.arch_config,
        opt.config,
        opt.record_embeddings,
        opt.replications,
        opt.gpu,
    )
