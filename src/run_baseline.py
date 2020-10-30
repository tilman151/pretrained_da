import os
import random

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

import datasets
import lightning

ExperimentNaming = {1: 'one',
                    2: 'two',
                    3: 'three',
                    4: 'four'}
script_path = os.path.dirname(__file__)


def run(source, seed, gpu):
    pl.trainer.seed_everything(seed)
    tensorboard_path = os.path.join(script_path, '..', 'tensorboard')
    tf_logger = loggers.TensorBoardLogger(tensorboard_path,
                                          name=f'cmapss_{ExperimentNaming[source]}_baseline')
    mlflow_logger = loggers.MLFlowLogger(f'cmapss_{ExperimentNaming[source]}_baseline',
                                         tracking_uri=os.path.join(script_path, '..', 'mlruns'))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/regression_loss')
    trainer = pl.Trainer(gpus=[gpu], max_epochs=100, logger=[tf_logger, mlflow_logger],
                         checkpoint_callback=checkpoint_callback, deterministic=True, log_every_n_steps=10)
    data = datasets.BaselineDataModule(fd_source=source,
                                       batch_size=512,
                                       window_size=30)
    model = lightning.Baseline(in_channels=14,
                               seq_len=30,
                               num_layers=4,
                               kernel_size=3,
                               base_filters=16,
                               latent_dim=128,
                               optim_type='adam',
                               lr=0.01,
                               record_embeddings=False)
    model.add_data_hparams(data)
    model.hparams.update({'seed': seed})
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(source, replications, gpu):
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    for s in seeds:
        run(source, s, gpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run baseline experiment')
    parser.add_argument('--source', type=int, help='FD number of the source data')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run_multiple(opt.source, opt.replications, opt.gpu)
