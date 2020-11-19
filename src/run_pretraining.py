import os
import random

import pytorch_lightning as pl
import sklearn

import datasets
from lightning import logger as loggers
from lightning import pretraining


def run(source, target, percent_broken, domain_tradeoff, record_embeddings, seed, gpu):
    pl.trainer.seed_everything(seed)
    logger = loggers.MLTBLogger(_get_logdir(), loggers.pretraining_experiment_name(source, target),
                                tensorboard_struct={'pb': percent_broken, 'dt': domain_tradeoff})
    trainer = pl.Trainer(gpus=[gpu], max_epochs=100, logger=logger,
                         deterministic=True, log_every_n_steps=10)
    data = _build_datamodule(percent_broken, source, target)
    model = pretraining.UnsupervisedPretraining(in_channels=14,
                                                seq_len=30,
                                                num_layers=4,
                                                kernel_size=3,
                                                base_filters=16,
                                                latent_dim=128,
                                                dropout=0.1,
                                                domain_tradeoff=domain_tradeoff,
                                                lr=0.01,
                                                weight_decay=0,
                                                record_embeddings=record_embeddings)
    model.add_data_hparams(data)
    model.hparams.update({'seed': seed})
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)

    return _get_checkpoint_path(logger)


def _build_datamodule(percent_broken, source, target):
    if target is None:
        return datasets.PretrainingBaselineDataModule(fd_source=source,
                                                      num_samples=25000,
                                                      batch_size=512,
                                                      window_size=30,
                                                      min_distance=1,
                                                      percent_broken=percent_broken)
    else:
        return datasets.PretrainingAdaptionDataModule(fd_source=source,
                                                      fd_target=target,
                                                      num_samples=50000,
                                                      batch_size=512,
                                                      window_size=30,
                                                      min_distance=1,
                                                      percent_broken=percent_broken)


def _get_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, '..'))

    return log_dir


def _get_checkpoint_path(logger):
    checkpoints_path = logger.checkpoint_path
    *_, checkpoint = sorted([f for f in os.listdir(checkpoints_path)])  # get last checkpoint
    checkpoint_path = os.path.join(checkpoints_path, checkpoint)

    return checkpoint_path


def run_multiple(source, target, broken, domain_tradeoff, record_embeddings, replications, gpu):
    broken = broken if broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    parameter_grid = {'domain_tradeoff': domain_tradeoff,
                      'broken': broken}

    checkpoints = {b: {dt: [] for dt in domain_tradeoff} for b in broken}
    for params in sklearn.model_selection.ParameterGrid(parameter_grid):
        for s in seeds:
            checkpoint_path = run(source, target, params['broken'], params['domain_tradeoff'],
                                  record_embeddings, s, gpu)
            checkpoints[params['broken']][params['domain_tradeoff']].append(checkpoint_path)

    return checkpoints


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run domain adaption experiment')
    parser.add_argument('--source', type=int, required=True, help='FD number of the source data')
    parser.add_argument('--target', type=int, help='FD number of the target data')
    parser.add_argument('-b', '--broken', nargs='+', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='+', type=float, help='tradeoff for domain classification')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run_multiple(opt.source, opt.target, opt.broken, opt.domain_tradeoff,
                 opt.record_embeddings, opt.replications, opt.gpu)
