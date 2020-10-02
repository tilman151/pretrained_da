from functools import partial

import pytorch_lightning as pl
from pytorch_lightning import loggers
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

import datasets
import lightning


def run(config, percent_broken, seed):
    domain_tradeoff = config['domain_tradeoff']
    recon_tradeoff = config['recon_tradeoff']

    pl.trainer.seed_everything(seed)
    tf_logger = loggers.TensorBoardLogger(tune.get_trial_dir(),
                                          name='', version='.')
    callbacks = [TuneReportCallback({'regression_loss': 'val_checkpoint_on'})]
    trainer = pl.Trainer(gpus=1, max_epochs=200, logger=tf_logger, deterministic=True,
                         progress_bar_refresh_rate=0, callbacks=callbacks)
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


def tune_hyperparameters():
    config = {'domain_tradeoff': tune.uniform(0.1, 10.),
              'recon_tradeoff': tune.uniform(0.0, 10.)}
    scheduler = ASHAScheduler(
            metric='loss',
            mode='min',
            max_t=200,
            grace_period=1,
            reduction_factor=2)
    reporter = CLIReporter(parameter_columns=['domain_tradeoff', 'recon_tradeoff'],
                           metric_columns=['regression_loss'])
    training_func = partial(run, percent_broken=1., seed=42)

    tune.run(training_func,
             resources_per_trial={"cpu": 6, 'gpu': 1},
             config=config,
             num_samples=100,
             scheduler=scheduler,
             progress_reporter=reporter,
             name='tune_tradeoffs')


if __name__ == '__main__':
    tune_hyperparameters()
