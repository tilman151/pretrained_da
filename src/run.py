import random

import sklearn.model_selection
import pytorch_lightning as pl
from pytorch_lightning import loggers

import lightning
import datasets


def run(percent_broken, domain_tradeoff, recon_tradeoff, seed):
    pl.trainer.seed_everything(seed)
    tf_logger = loggers.TensorBoardLogger('./tradeoffs',
                                          name=f'{percent_broken:.0%}pb_{domain_tradeoff:.1f}dt_{recon_tradeoff:.1f}rt')
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
                                 recon_trade_off=recon_tradeoff,
                                 domain_trade_off=domain_tradeoff,
                                 domain_disc_dim=64,
                                 num_disc_layers=2,
                                 lr=0.01)
    model.add_data_hparams(data)
    model.hparams.update({'seed': seed})
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(broken, domain_tradeoff, recon_tradeoff, replications):
    broken = broken if opt.broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    parameter_grid = {'domain_tradeoff': domain_tradeoff,
                      'recon_tradeoff': recon_tradeoff,
                      'broken': broken}

    for params in sklearn.model_selection.ParameterGrid(parameter_grid):
        for s in seeds:
            run(params['broken'], params['domain_tradeoff'], params['recon_tradeoff'], s)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run domain adaption experiment')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_trade_off', type=float, help='tradeoff for domain classification')
    parser.add_argument('--recon_trade_off', type=float, help='tradeoff for reconstruction')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    opt = parser.parse_args()

    run_multiple(opt.broken, opt.domain_tradeoff, opt.recon_tradeoff, opt.replications)
