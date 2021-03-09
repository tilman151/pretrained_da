import random
from datetime import datetime

from building import load_config
from run_dann import run as run_dann
from run_pretraining import run_multiple as run_pretraining


def run(
    source,
    target,
    percent_broken,
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
        target,
        percent_broken,
        None,
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
        _adapt_with_best_pretrained(
            source,
            target,
            arch_config,
            pretrained_checkpoints,
            record_embeddings,
            seeds,
            gpu,
            version,
        )
    else:
        _adapt_with_all_pretrained(
            source,
            target,
            arch_config,
            pretrained_checkpoints,
            record_embeddings,
            seeds,
            gpu,
            version,
        )


def _adapt_with_all_pretrained(
    source,
    target,
    arch_config,
    pretrained_checkpoints,
    record_embeddings,
    seeds,
    gpu,
    version,
):
    for broken, checkpoints in pretrained_checkpoints.items():
        for (pretrained_checkpoint, best_val_score), s in zip(checkpoints, seeds):
            run_dann(
                source,
                target,
                broken,
                arch_config,
                record_embeddings,
                s,
                gpu,
                pretrained_checkpoint,
                version,
            )


def _adapt_with_best_pretrained(
    source,
    target,
    arch_config,
    pretrained_checkpoints,
    record_embeddings,
    seeds,
    gpu,
    version,
):
    for broken, checkpoints in pretrained_checkpoints.items():
        best_pretrained_path = _get_best_pretrained_checkpoint(checkpoints)
        for s in seeds:
            run_dann(
                source,
                target,
                broken,
                arch_config,
                record_embeddings,
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

    parser = argparse.ArgumentParser(
        description="Run pretraining and domain adaption for one dataset pairing"
    )
    parser.add_argument(
        "--source", required=True, type=int, help="FD number of the source data"
    )
    parser.add_argument(
        "--target", required=True, type=int, help="FD number of the target data"
    )
    parser.add_argument(
        "-b", "--broken", nargs="*", type=float, help="percent broken to use"
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

    _arch_config = load_config(opt.arch_config)
    _pre_config = load_config(opt.pre_config)
    run(
        opt.source,
        opt.target,
        opt.broken,
        _arch_config,
        _pre_config,
        opt.mode,
        opt.record_embeddings,
        opt.pretraining_reps,
        opt.best_only,
        opt.gpu,
    )
