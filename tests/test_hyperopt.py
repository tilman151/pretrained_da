import unittest

from unittest import mock

import hyperopt.hyperopt_transfer as hyperopt_transfer
import hyperopt.hyperopt_pretraining as hyperopt_pretraining


class TestTransferHyperopt(unittest.TestCase):
    def test_tune_function(self):
        config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }
        hyperopt_transfer.tune_transfer(config, 2, 1, 0.8)


class TestPretrainingHyperopt(unittest.TestCase):
    @mock.patch("lightning.pretraining.UnsupervisedPretraining")
    @mock.patch("pytorch_lightning.Trainer.fit")
    def test_model_creation(self, mock_fit, mock_pretraining_model):
        arch_config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }
        config = {
            "domain_tradeoff": 2.0,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 256,
        }
        hyperopt_pretraining.tune_pretraining(config, arch_config, 2, 1, 0.8)

        mock_pretraining_model.assert_called_with(
            in_channels=14,
            seq_len=30,
            num_layers=arch_config["num_layers"],
            kernel_size=3,
            base_filters=arch_config["base_filters"],
            latent_dim=arch_config["latent_dim"],
            dropout=config["dropout"],
            domain_tradeoff=config["domain_tradeoff"],
            domain_disc_dim=arch_config["latent_dim"],
            num_disc_layers=arch_config["num_disc_layers"],
            lr=config["lr"],
            record_embeddings=False,
            weight_decay=0.0,
        )

    @mock.patch("datasets.PretrainingAdaptionDataModule")
    @mock.patch("pytorch_lightning.Trainer.fit")
    def test_data_creation(self, mock_fit, mock_data):
        arch_config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }
        config = {
            "domain_tradeoff": 2.0,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 256,
        }
        mock_data.window_size = 30
        hyperopt_pretraining.tune_pretraining(config, arch_config, 2, 1, 0.8)

        mock_data.assert_called_with(
            fd_source=2,
            fd_target=1,
            num_samples=50000,
            batch_size=config["batch_size"],
            percent_broken=0.8,
            truncate_target_val=True,
        )
