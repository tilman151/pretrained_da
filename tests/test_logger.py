import os
import shutil
import tempfile
import unittest

import pytorch_lightning as pl

import datasets
import lightning.logger as loggers
from lightning import baseline


class TestMLTBLogger(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.logger = loggers.MLTBLogger(self.logdir, 'Test', {'pb': 0.1, 'foo': 'bar'})

    def test_save_path(self):
        self._run_dummy_training()
        artifact_path = self.logger.checkpoint_path
        self.assertListEqual(['epoch=0.ckpt'], os.listdir(artifact_path))

    def test_tensorboard_struct(self):
        self._run_dummy_training()
        expected_tf_logdir = os.path.join(self.logdir, 'tensorboard', 'Test', '0.1pb', 'foo:bar', 'version_0')
        self.assertTrue(os.path.exists(expected_tf_logdir))

    def _run_dummy_training(self):
        trainer = pl.Trainer(gpus=0, max_epochs=1, logger=self.logger,
                             deterministic=True, log_every_n_steps=10)
        data = datasets.BaselineDataModule(fd_source=1,
                                           batch_size=512,
                                           window_size=30)
        model = baseline.Baseline(in_channels=14,
                                  seq_len=30,
                                  num_layers=1,
                                  kernel_size=1,
                                  base_filters=1,
                                  latent_dim=16,
                                  optim_type='adam',
                                  lr=0.01,
                                  record_embeddings=False)
        trainer.fit(model, datamodule=data)

    def tearDown(self):
        shutil.rmtree(self.logdir)
