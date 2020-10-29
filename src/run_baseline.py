import random

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

import datasets
import lightning


def run(seed):
    pl.trainer.seed_everything(seed)
    tf_logger = loggers.TensorBoardLogger('./three2one',
                                          name='baseline')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/regression_loss')
    trainer = pl.Trainer(gpus=[0], max_epochs=100, logger=tf_logger, checkpoint_callback=checkpoint_callback,
                         deterministic=True, log_every_n_steps=10)
    data = datasets.BaselineDataModule(fd_source=3,
                                       fd_target=1,
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


def run_multiple(replications):
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    for s in seeds:
        run(s)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run baseline experiment')
    parser.add_argument('-r', '--replications', type=int, default=3, help='replications for each run')
    opt = parser.parse_args()

    run_multiple(opt.replications)
