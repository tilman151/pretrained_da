import random

import pytorch_lightning as pl
from pytorch_lightning import loggers

import lightning
import datasets


def run(percent_broken, seed):
    pl.trainer.seed_everything(seed)
    tf_logger = loggers.TensorBoardLogger('./results', name=f'percent_broken_{percent_broken:.0%}')
    trainer = pl.Trainer(gpus=1, max_epochs=200, logger=tf_logger, deterministic=True)
    data = datasets.DomainAdaptionDataModule(fd_source=3,
                                             fd_target=1,
                                             batch_size=512,
                                             window_size=30,
                                             percent_broken=percent_broken)
    model = lightning.AdaptiveAE(in_channels=14,
                                 seq_len=30,
                                 num_layers=4,
                                 kernel_size=3,
                                 base_filters=16,
                                 latent_dim=64,
                                 recon_trade_off=0.,
                                 domain_trade_off=1.,
                                 domain_disc_dim=64,
                                 num_disc_layers=2,
                                 lr=0.01)
    model.add_data_hparams(data)
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(broken):
    broken = broken if opt.broken is not None else [1.0]
    random.seed(42)
    seeds = [random.randint(0, 9999999) for _ in broken]
    for b in broken:
        for s in seeds:
            run(b, s)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run domain adaption experiment')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    opt = parser.parse_args()

    run_multiple(opt.broken)
