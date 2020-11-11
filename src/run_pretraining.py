import os
import random

import pytorch_lightning as pl
import sklearn
from pytorch_lightning import loggers

from datasets import cmapss
from lightning import pretraining

ExperimentNaming = {1: 'one',
                    2: 'two',
                    3: 'three',
                    4: 'four'}
script_path = '/home/tkrokots/repos/ae_adapt/src' if not os.path.dirname(__file__) else os.path.dirname(__file__)


def run(source, target, percent_broken, domain_tradeoff, record_embeddings, seed, gpu):
    pl.trainer.seed_everything(seed)
    tensorboard_path = os.path.join(script_path, '..', 'tensorboard', f'{ExperimentNaming[source]}2{ExperimentNaming[target]}')
    tf_logger = loggers.TensorBoardLogger(tensorboard_path,
                                          name=f'pretraining_{percent_broken:.0%}pb')
    mlflow_logger = loggers.MLFlowLogger(f'pretraining_{ExperimentNaming[source]}2{ExperimentNaming[target]}',
                                         tracking_uri=os.path.join('file:', script_path, '..', 'mlruns'))
    trainer = pl.Trainer(gpus=[gpu], max_epochs=100, logger=loggers.LoggerCollection([tf_logger, mlflow_logger]),
                         deterministic=True, log_every_n_steps=10)
    data = cmapss.PretrainingDataModule(fd_source=source,
                                        fd_target=target,
                                        num_samples=50000,
                                        batch_size=512,
                                        window_size=30,
                                        min_distance=1,
                                        percent_broken=percent_broken)
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


def run_multiple(source, target, broken, domain_tradeoff, record_embeddings, replications, gpu):
    broken = broken if opt.broken is not None else [1.0]
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    parameter_grid = {'domain_tradeoff': domain_tradeoff,
                      'broken': broken}

    for params in sklearn.model_selection.ParameterGrid(parameter_grid):
        for s in seeds:
            run(source, target, params['broken'], params['domain_tradeoff'], record_embeddings, s, gpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run domain adaption experiment')
    parser.add_argument('--source', type=int, help='FD number of the source data')
    parser.add_argument('--target', type=int, help='FD number of the target data')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='*', type=float, help='tradeoff for domain classification')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run_multiple(opt.source, opt.target, opt.broken, opt.domain_tradeoff,
                 opt.record_embeddings, opt.replications, opt.gpu)
