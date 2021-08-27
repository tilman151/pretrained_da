import os
import shutil
import tempfile
import unittest
from typing import List, Optional
from unittest import mock

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning.loggers as loggers
from lightning import baseline


class DummyDataModule(pl.LightningDataModule):
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._dummy_dataset(), batch_size=10, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._dummy_dataset(), batch_size=10)

    def test_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        return [DataLoader(self._dummy_dataset(), batch_size=10) for _ in range(4)]

    def _dummy_dataset(self):
        return TensorDataset(torch.randn(100, 14, 30), torch.rand(100))


class TestMLTBLogger(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.test_tag = "test_tag"
        self.logger = loggers.MLTBLogger(
            self.logdir, "Test", self.test_tag, {"pb": 0.1, "foo": "bar"}
        )

    def test_save_path(self):
        self._run_dummy_training()
        artifact_path = self.logger.checkpoint_path
        self.assertListEqual(["epoch=0.ckpt"], os.listdir(artifact_path))

    def test_tensorboard_struct(self):
        self._run_dummy_training()
        expected_tf_logdir = os.path.join(
            self.logdir, "tensorboard", "Test", "0.1pb", "foo:bar", "version_0"
        )
        self.assertTrue(os.path.exists(expected_tf_logdir))

    def test_version_tag(self):
        self._run_dummy_training()
        run_id = self.logger._mlflow_logger.run_id
        run = self.logger.mlflow_experiment.get_run(run_id)
        self.assertDictEqual({"version": "test_tag"}, run.data.tags)

    def _run_dummy_training(self):
        trainer = pl.Trainer(
            gpus=0,
            max_epochs=1,
            logger=self.logger,
            deterministic=True,
            log_every_n_steps=10,
        )
        data = DummyDataModule()
        model = baseline.Baseline(
            in_channels=14,
            seq_len=30,
            num_layers=1,
            kernel_size=1,
            base_filters=1,
            latent_dim=16,
            optim_type="adam",
            lr=0.01,
            record_embeddings=False,
        )
        trainer.fit(model, datamodule=data)

    def tearDown(self):
        shutil.rmtree(self.logdir)


class TestMinEpochModelCheckpoint(unittest.TestCase):
    @mock.patch("pytorch_lightning.callbacks.ModelCheckpoint.save_checkpoint")
    def test_min_epoch(self, mock_super_save_checkpoint):
        mock_trainer = mock.MagicMock("trainer")
        mock_pl_module = mock.MagicMock("pl_module")

        min_epoch = 1
        checkpoint_callback = loggers.MinEpochModelCheckpoint(
            "val/test_metric", min_epochs_before_saving=min_epoch
        )

        with self.subTest(case="before_min_epoch"):
            mock_trainer.current_epoch = 0
            checkpoint_callback.save_checkpoint(mock_trainer, mock_pl_module)
            mock_super_save_checkpoint.assert_not_called()

        with self.subTest(case="after_min_epoch"):
            mock_trainer.current_epoch = 2
            checkpoint_callback.save_checkpoint(mock_trainer, mock_pl_module)
            mock_super_save_checkpoint.assert_called_with(mock_trainer, mock_pl_module)
