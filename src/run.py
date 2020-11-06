import os
import random

import pytorch_lightning as pl
import sklearn.model_selection
from pytorch_lightning import loggers

import datasets
import lightning

ExperimentNaming = {1: 'one',
                    2: 'two',
                    3: 'three',
                    4: 'four'}
script_path = os.path.dirname(__file__)


def run(source, target, percent_broken, domain_tradeoff, recon_tradeoff, cap, record_embeddings, seed, gpu):
    pl.trainer.seed_everything(seed)
    tensorboard_path = os.path.join(script_path, '..', 'tensorboard', f'{ExperimentNaming[source]}2{ExperimentNaming[target]}')
    tf_logger = loggers.TensorBoardLogger(tensorboard_path,
                                          name=f'{percent_broken:.0%}pb_{domain_tradeoff:.1f}dt_{recon_tradeoff:.1f}rt')
    mlflow_logger = loggers.MLFlowLogger(f'{ExperimentNaming[source]}2{ExperimentNaming[target]}',
                                         tracking_uri=os.path.join('file:', script_path, '..', 'mlruns'))
    trainer = pl.Trainer(gpus=[gpu], max_epochs=200, logger=loggers.LoggerCollection([tf_logger, mlflow_logger]),
                         deterministic=True, log_every_n_steps=10)
    data = datasets.DomainAdaptionDataModule(fd_source=source,
                                             fd_target=target,
                                             batch_size=512,
                                             window_size=30,
                                             percent_broken=percent_broken)
    model = lightning.AdaptiveAE(in_channels=14,
                                 seq_len=30,
                                 num_layers=4,
                                 kernel_size=3,
                                 base_filters=16,
                                 latent_dim=128,
                                 recon_trade_off=recon_tradeoff,
                                 domain_trade_off=domain_tradeoff,
                                 domain_disc_dim=64,
                                 num_disc_layers=2,
                                 source_rul_cap=int((1 - percent_broken) * 125) if cap else None,
                                 optim_type='adam',
                                 lr=0.01,
                                 record_embeddings=record_embeddings)
    model.add_data_hparams(data)
    model.hparams.update({'seed': seed})
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(source, target, broken, domain_tradeoff, recon_tradeoff, cap, record_embeddings, replications, gpu):
    broken = broken if opt.broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    parameter_grid = {'domain_tradeoff': domain_tradeoff,
                      'recon_tradeoff': recon_tradeoff,
                      'broken': broken}

    for params in sklearn.model_selection.ParameterGrid(parameter_grid):
        for s in seeds:
            run(source, target, params['broken'], params['domain_tradeoff'], params['recon_tradeoff'], cap, s, gpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run domain adaption experiment')
    parser.add_argument('--source', type=int, help='FD number of the source data')
    parser.add_argument('--target', type=int, help='FD number of the target data')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='*', type=float, help='tradeoff for domain classification')
    parser.add_argument('--recon_tradeoff', nargs='*', type=float, help='tradeoff for reconstruction')
    parser.add_argument('-c', '--cap', action='store_true', help='cap the source data for adaption loss')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run_multiple(opt.source, opt.target, opt.broken, opt.domain_tradeoff,
                 opt.recon_tradeoff, opt.cap, opt.record_embeddings, opt.replications, opt.gpu)
