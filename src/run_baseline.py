import os
import random

from building.build import build_baseline


def run(source, fails, seed, gpu, pretrained_encoder_path):
    trainer, data, model = build_baseline(
        source, fails, pretrained_encoder_path, gpu, seed
    )
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(source, fails, replications, gpu, pretrained_encoder_path):
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    for f in fails:
        for s in seeds:
            run(source, f, s, gpu, pretrained_encoder_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument("--source", type=int, help="FD number of the source data")
    parser.add_argument(
        "-f", "--fails", nargs="+", type=float, help="percent fail runs to use"
    )
    parser.add_argument(
        "--pretrained_encoder",
        default=None,
        help="Path to checkpoint file form pretraining",
    )
    parser.add_argument(
        "-r", "--replications", type=int, default=3, help="replications for each run"
    )
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    opt = parser.parse_args()

    run_multiple(opt.source, opt.fails, opt.replications, opt.gpu, opt.pretrained_encoder)
