import unittest

from unittest import mock

from building import build


class TestBuildingFunctions(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }

    @mock.patch("pytorch_lightning.callbacks.ModelCheckpoint")
    @mock.patch("building.build.build_dann_from_config")
    @mock.patch("datasets.DomainAdaptionDataModule")
    @mock.patch("building.build.build_trainer")
    def test_build_transfer(
        self, mock_build_trainer, mock_datamodule, mock_dann_from_config, mock_checkpoint
    ):
        mock_checkpoint.return_value = "mock_checkpoint"

        with self.subTest("no_encoder"):
            build.build_transfer(2, 1, 0.8, self.config, None, False, None, 1, 42)
            mock_build_trainer.assert_called_with(
                None,
                "mock_checkpoint",
                max_epochs=200,
                val_interval=1.0,
                gpu=1,
                seed=42,
            )
            mock_datamodule.assert_called_with(
                fd_source=2,
                fd_target=1,
                batch_size=self.config["batch_size"],
                percent_broken=0.8,
            )
            mock_dann_from_config.assert_called_with(
                self.config, mock_datamodule().window_size, None, False
            )
        with self.subTest("with_encoder"):
            build.build_transfer(
                2, 1, 0.8, self.config, "encoder_path", False, None, 1, 42
            )
            mock_build_trainer.assert_called_with(
                None,
                "mock_checkpoint",
                max_epochs=20,
                val_interval=0.1,
                gpu=1,
                seed=42,
            )
            mock_datamodule.assert_called_with(
                fd_source=2,
                fd_target=1,
                batch_size=self.config["batch_size"],
                percent_broken=0.8,
            )
            mock_dann_from_config.assert_called_with(
                self.config, mock_datamodule().window_size, "encoder_path", False
            )

    @mock.patch("lightning.dann.DANN")
    def test_build_dann_from_config(self, mock_dann):
        with self.subTest("no_encoder"):
            build.build_dann_from_config(self.config, 30, None, False)
            self._assert_dann_build_correctly(mock_dann)
            mock_dann().load_encoder.assert_not_called()
        with self.subTest("with_encoder"):
            build.build_dann_from_config(self.config, 30, "encoder_path", False)
            self._assert_dann_build_correctly(mock_dann)
            mock_dann().load_encoder.assert_called_with("encoder_path", load_disc=True)

    def _assert_dann_build_correctly(self, mock_dann):
        mock_dann.assert_called_with(
            in_channels=14,
            seq_len=30,
            num_layers=self.config["num_layers"],
            kernel_size=3,
            base_filters=self.config["base_filters"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
            domain_trade_off=self.config["domain_tradeoff"],
            domain_disc_dim=self.config["latent_dim"],
            num_disc_layers=self.config["num_disc_layers"],
            optim_type="adam",
            lr=self.config["lr"],
            record_embeddings=False,
        )
