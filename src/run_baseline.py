import os
import random

import pytorch_lightning as pl

import datasets
from lightning import baseline
from lightning import logger as loggers


def run(source, fails, seed, gpu, pretrained_encoder_path):
    pl.trainer.seed_everything(seed)
    logger = loggers.MLTBLogger(_get_logdir(), loggers.baseline_experiment_name(source))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/regression_loss')
    trainer = pl.Trainer(gpus=[gpu], max_epochs=100, logger=logger,
                         checkpoint_callback=checkpoint_callback, deterministic=True, log_every_n_steps=10)
    data = datasets.BaselineDataModule(fd_source=source,
                                       batch_size=512,
                                       window_size=30,
                                       percent_fail_runs=fails)
    model = baseline.Baseline(in_channels=14,
                              seq_len=30,
                              num_layers=4,
                              kernel_size=3,
                              base_filters=16,
                              latent_dim=128,
                              optim_type='adam',
                              lr=0.01,
                              record_embeddings=False)
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path)
    model.add_data_hparams(data)
    model.hparams.update({'seed': seed})
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def _get_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, '..'))

    return log_dir


def run_multiple(source, fails, replications, gpu, pretrained_encoder_path):
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    for f in fails:
        for s in seeds:
            run(source, f, s, gpu, pretrained_encoder_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run baseline experiment')
    parser.add_argument('--source', type=int, help='FD number of the source data')
    parser.add_argument('-f', '--fails', nargs='+', type=float, help='percent fail runs to use')
    parser.add_argument('--pretrained_encoder', default=None, help='Path to checkpoint file form pretraining')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run_multiple(opt.source, opt.fails, opt.replications, opt.gpu, opt.pretrained_encoder)
