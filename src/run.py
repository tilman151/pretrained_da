import pytorch_lightning as pl
from pytorch_lightning import loggers

import lightning
import datasets


def run(percent_broken, seed):
    pl.trainer.seed_everything(seed)
    tf_logger = loggers.TensorBoardLogger('./results', name=f'percent_broken_{percent_broken:.0%}')
    trainer = pl.Trainer(gpus=1, max_epochs=200, logger=tf_logger, deterministic=True)
    data = datasets.DomainAdaptionDataModule(fd_source=3, fd_target=1, batch_size=512, window_size=30, percent_broken=percent_broken)
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


if __name__ == '__main__':
    broken = [1.0, 0.8, 0.6, 0.4, 0.2]
    seeds = [345, 754, 568, 290, 914]
    for b in broken:
        for s in seeds:
            run(b, s)
