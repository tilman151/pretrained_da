import pytorch_lightning as pl
from pytorch_lightning import loggers

import lightning
import datasets


def run():
    pl.trainer.seed_everything(42)
    trainer = pl.Trainer(gpus=1, max_epochs=2, logger=loggers.TensorBoardLogger('./results'), deterministic=True)
    data = datasets.DomainAdaptionDataModule(fd_source=3, fd_target=1, batch_size=512, window_size=30)
    model = lightning.AdaptiveAE(in_channels=14,
                                 seq_len=30,
                                 num_layers=4,
                                 kernel_size=3,
                                 base_filters=16,
                                 latent_dim=64,
                                 recon_trade_off=0.,
                                 domain_trade_off=10.,
                                 domain_disc_dim=64,
                                 num_disc_layers=2,
                                 lr=0.01)
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


if __name__ == '__main__':
    run()
