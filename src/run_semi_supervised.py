import random
from datetime import datetime

from sklearn.model_selection import ShuffleSplit

import building
from datasets.loader import CMAPSSLoader
from run_baseline import run as run_baseline
from run_pretraining import run as run_pretraining


def run(
    source,
    percent_broken,
    percent_fails,
    arch_config,
    pre_config,
    pretrain,
    mode,
    record_embeddings,
    replications,
    gpu,
):
    version = datetime.now().timestamp()
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    splitter = ShuffleSplit(
        n_splits=replications, train_size=percent_fails, random_state=42
    )
    run_idx = range(CMAPSSLoader.NUM_TRAIN_RUNS[source])
    for (failed_idx, _), s in zip(splitter.split(run_idx), seeds):
        if pretrain:
            checkpoint, _ = run_pretraining(
                source,
                None,
                percent_broken,
                failed_idx,
                arch_config,
                pre_config,
                mode,
                record_embeddings,
                s,
                gpu,
                version,
            )
        else:
            checkpoint = None
        run_baseline(
            source,
            failed_idx,
            arch_config,
            s,
            gpu,
            checkpoint,
            version,
        )


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
    parser.add_argument("--pretrain", action="store_true", help="use pre-training")
    parser.add_argument(
        "-r",
        "--replications",
        type=int,
        default=1,
        help="runs of the cross-validation",
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
        opt.pretrain,
        opt.mode,
        opt.record_embeddings,
        opt.replications,
        opt.gpu,
    )
