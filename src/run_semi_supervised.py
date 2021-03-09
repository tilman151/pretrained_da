import random
from datetime import datetime

import building
from run_baseline import run as run_baseline
from run_pretraining import run_multiple as run_pretraining


def run(
    source,
    percent_broken,
    percent_fails,
    arch_config,
    pre_config,
    mode,
    record_embeddings,
    pretraining_reps,
    best_only,
    gpu,
):
    version = datetime.now().timestamp()

    pretrained_checkpoints = run_pretraining(
        source,
        None,
        [percent_broken],
        [percent_fails],
        arch_config,
        pre_config,
        mode,
        record_embeddings,
        pretraining_reps,
        gpu,
        version,
    )
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(pretraining_reps)]
    if best_only:
        _train_with_best_pretrained(
            source,
            percent_fails,
            arch_config,
            pretrained_checkpoints,
            seeds,
            gpu,
            version,
        )
    else:
        _train_with_all_pretrained(
            source,
            percent_fails,
            arch_config,
            pretrained_checkpoints,
            seeds,
            gpu,
            version,
        )


def _train_with_all_pretrained(
    source,
    fails,
    arch_config,
    pretrained_checkpoints,
    seeds,
    gpu,
    version,
):
    for _, checkpoints in pretrained_checkpoints.items():
        for (pretrained_checkpoint, best_val_score), s in zip(checkpoints, seeds):
            run_baseline(
                source,
                fails,
                arch_config,
                s,
                gpu,
                pretrained_checkpoint,
                version,
            )


def _train_with_best_pretrained(
    source,
    fails,
    arch_config,
    pretrained_checkpoints,
    seeds,
    gpu,
    version,
):
    for broken, checkpoints in pretrained_checkpoints.items():
        best_pretrained_path = _get_best_pretrained_checkpoint(checkpoints)
        for s in seeds:
            run_baseline(
                source,
                fails,
                arch_config,
                s,
                gpu,
                best_pretrained_path,
                version,
            )


def _get_best_pretrained_checkpoint(checkpoints):
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x[1])  # sort by val loss
    best_pretrained_path = sorted_checkpoints[0][0]

    return best_pretrained_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run semi-supervised experiment")
    parser.add_argument(
        "--source", required=True, type=int, help="FD number of the source data"
    )
    parser.add_argument(
        "-b", "--broken", required=True, type=float, help="percent broken to use"
    )
    parser.add_argument(
        "-f", "--fails", required=True, type=float, help="percent fail runs to use"
    )
    parser.add_argument(
        "--arch_config", required=True, help="path to architecture config file"
    )
    parser.add_argument(
        "--pre_config", required=True, help="path to pretraining config file"
    )
    parser.add_argument(
        "-p",
        "--pretraining_reps",
        type=int,
        default=1,
        help="replications for each pretraining run",
    )
    parser.add_argument(
        "--best_only", action="store_true", help="adapt only on best pretraining run"
    )
    parser.add_argument(
        "--mode",
        default="metric",
        choices=["metric", "autoencoder"],
        help="metric or autoencoder pre-training mode",
    )
    parser.add_argument(
        "--record_embeddings",
        action="store_true",
        help="whether to record embeddings of val data",
    )
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    opt = parser.parse_args()

    _arch_config = building.load_config(opt.arch_config)
    _pre_config = building.load_config(opt.pre_config)
    run(
        opt.source,
        opt.broken,
        opt.fails,
        _arch_config,
        _pre_config,
        opt.mode,
        opt.record_embeddings,
        opt.pretraining_reps,
        opt.best_only,
        opt.gpu,
    )
