import pytorch_lightning as pl
from pytorch_lightning import loggers

import lightning
import datasets


def run():
    data = datasets.CMAPSSDataModule(fd=3, batch_size=512, window_size=30)
    model = lightning.AdaptiveAE(in_channels=14,
                                 seq_len=30,
                                 num_layers=4,
                                 kernel_size=3,
                                 base_filters=16,
                                 latent_dim=8)
    trainer = pl.Trainer(gpus=1, max_epochs=500, logger=loggers.TensorBoardLogger('./results'))
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


if __name__ == '__main__':
    run()
