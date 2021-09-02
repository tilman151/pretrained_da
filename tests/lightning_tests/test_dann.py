import unittest
from unittest import mock

import torch

from lightning import dann


class TestDAAN(unittest.TestCase):
    def setUp(self):
        self.trade_off = 0.5
        self.net = dann.DANN(
            in_channels=14,
            seq_len=30,
            num_layers=4,
            kernel_size=3,
            base_filters=16,
            latent_dim=64,
            dropout=0.1,
            domain_trade_off=self.trade_off,
            domain_disc_dim=32,
            num_disc_layers=2,
            optim_type="adam",
            lr=0.01,
        )

    @torch.no_grad()
    def test_encoder(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net.encoder(inputs)
        self.assertEqual(torch.Size((16, 64)), outputs.shape)

    @torch.no_grad()
    def test_regressor(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.regressor(inputs)
        self.assertEqual(torch.Size((16,)), outputs.shape)

    @torch.no_grad()
    def test_domain_disc(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.domain_disc(inputs)
        self.assertEqual(torch.Size((16,)), outputs.shape)

    def test_batch_independence(self):
        inputs = torch.randn(16, 14, 30)
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = outputs.shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        output = outputs * mask

        # Compute backward pass
        loss = output.mean()
        loss.backward(retain_graph=True)

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad[:batch_size]):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))
        inputs.grad = None

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        loss, *_ = self.net._get_losses(
            torch.randn(16, 14, 30),
            torch.ones(16),
            torch.randn(16, 14, 30),
            torch.ones(32),
        )
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad ** 2))

    @mock.patch("pytorch_lightning.LightningModule.log")
    @torch.no_grad()
    def test_train_metrics(self, mock_log):
        criterion = torch.nn.MSELoss()
        source = torch.zeros(16, 14, 30)
        target = torch.zeros(16, 14, 30)
        source_labels = torch.ones(16)

        source_embeddings = self.net.encoder(source)
        target_embeddings = self.net.encoder(target)
        source_prediction = self.net.regressor(source_embeddings)
        regression_loss = torch.sqrt(criterion(source_prediction, source_labels))
        source_domain_pred = self.net.domain_disc(source_embeddings)
        target_domain_pred = self.net.domain_disc(target_embeddings)
        source_domain_loss = self.net.criterion_domain(
            source_domain_pred, torch.ones_like(source_domain_pred)
        )
        target_domain_loss = self.net.criterion_domain(
            target_domain_pred, torch.zeros_like(target_domain_pred)
        )
        domain_loss = (source_domain_loss + target_domain_loss) / 2
        expected_overall_loss = (
            regression_loss + self.net.domain_trade_off * domain_loss
        )

        actual_overall_loss = self.net.training_step((source, source_labels, target), 0)
        self.assertAlmostEqual(
            expected_overall_loss.item(), actual_overall_loss.item(), delta=0.001
        )

        expected_logs = {
            "train/regression_loss": (regression_loss, 0.001),
            "train/loss": (expected_overall_loss, 0.001),
            "train/domain_loss": (domain_loss, 0.0001),
        }
        self._assert_logs(mock_log, expected_logs)

    def test_norm_output(self):
        with self.subTest(norm=False):
            inputs = torch.randn(10, 14, 30)
            outputs = self.net.encoder(inputs)
            for sample in outputs:
                self.assertNotEqual(1.0, torch.norm(sample, p=2))

        with self.subTest(norm=True):
            self.net.encoder.norm_outputs = True
            outputs = self.net.encoder(inputs)
            for sample in outputs:
                self.assertAlmostEqual(1.0, torch.norm(sample, p=2).item(), places=5)

    @mock.patch("torch.load")
    @mock.patch("lightning.mixins.LoadEncoderMixin._extract_state_dict")
    @mock.patch("torch.nn.Module.load_state_dict")
    def test_feature_norm_on_transferred_encoder(
        self, mock_load_state_dict, mock_extract_state_dict, mock_load
    ):
        self.assertFalse(self.net.encoder.norm_outputs)
        self.net.load_encoder("bogus", load_disc=False)
        self.assertTrue(self.net.encoder.norm_outputs)

    @mock.patch("pytorch_lightning.LightningModule.log")
    @torch.no_grad()
    def test_metric_reduction(self, mock_log):
        source_batches = 20
        target_batches = 10
        score_batches = target_batches
        source_prediction = torch.randn(source_batches, 32) + 30
        target_prediction = torch.randn(target_batches, 32) + 20
        score_prediction = torch.randn(2 * score_batches, 32) + 10
        prediction = torch.cat([source_prediction, target_prediction, score_prediction])
        domain_pred = torch.rand(source_batches + target_batches, 32)
        self._mock_predictions(prediction, domain_pred)

        self._feed_dummy_source(source_batches)
        self._feed_dummy_target(target_batches)
        self._feed_dummy_score(score_batches)

        domain_labels = torch.cat(
            [torch.ones_like(source_prediction), torch.zeros_like(target_prediction)]
        )
        source_regression_loss = torch.sqrt(torch.mean(source_prediction ** 2))
        target_regression_loss = torch.sqrt(torch.mean(target_prediction ** 2))
        domain_loss = self.net.criterion_domain(domain_pred, domain_labels)
        score = torch.sqrt(
            torch.mean((score_prediction[1::2] - score_prediction[::2]) ** 2)
        )
        rul_score = self.net.rul_score(
            target_prediction.view(-1), torch.zeros(target_batches * 32)
        )

        expected_logs = {
            "val/regression_loss": (target_regression_loss, 0.001),
            "val/source_regression_loss": (source_regression_loss, 0.001),
            "val/domain_loss": (domain_loss, 0.0001),
            "val/score": (score, 0.001),
            "val/rul_score": (rul_score, 0.1),
        }
        self.net.validation_epoch_end([])
        self._assert_logs(mock_log, expected_logs)

    def _assert_logs(self, mock_log, expected_logs):
        mock_log.assert_called()
        for call in mock_log.mock_calls:
            metric = call[1][0]
            self.assertIn(metric, expected_logs, "Unexpected logged metric found.")
            expected_value, delta = expected_logs[metric]
            expected_value = expected_value.item()
            actual_value = call[1][1].item()
            with self.subTest(metric):
                self.assertAlmostEqual(expected_value, actual_value, delta=delta)

    def test_metric_updates(self):
        source_batches = 20
        target_batches = 10
        batch_size = 32
        score_batches = target_batches
        num_batches = source_batches + target_batches
        source_prediction = torch.randn(source_batches, batch_size) + 30
        target_prediction = torch.randn(target_batches, batch_size) + 20
        score_prediction = torch.randn(2 * score_batches, batch_size) + 10
        prediction = torch.cat([source_prediction, target_prediction, score_prediction])
        domain_pred = torch.rand(source_batches + target_batches, batch_size)
        self._mock_predictions(prediction, domain_pred)

        with self.subTest("source_data"):
            self._feed_dummy_source(source_batches)
            self.assertEqual(
                batch_size * source_batches,
                self.net.source_regression_metric.num_elements,
            )
            self.assertEqual(source_batches, len(self.net.domain_loss_metric.sizes))
            self.assertEqual(0, self.net.target_regression_metric.num_elements)
            self.assertEqual(0, len(self.net.target_rul_score_metric.sizes))
            self.assertEqual(0, self.net.target_checkpoint_score_metric.num_elements)

        with self.subTest("target_data"):
            self._feed_dummy_target(target_batches)
            self.assertEqual(
                source_batches * batch_size,
                self.net.source_regression_metric.num_elements,
            )
            self.assertEqual(num_batches, len(self.net.domain_loss_metric.sizes))
            self.assertEqual(
                target_batches * batch_size,
                self.net.target_regression_metric.num_elements,
            )
            self.assertEqual(
                target_batches,
                len(self.net.target_rul_score_metric.sizes),
            )
            self.assertEqual(0, self.net.target_checkpoint_score_metric.num_elements)

        with self.subTest("score_data"):
            self._feed_dummy_score(score_batches)
            self.assertEqual(
                source_batches * batch_size,
                self.net.source_regression_metric.num_elements,
            )
            self.assertEqual(num_batches, len(self.net.domain_loss_metric.sizes))
            self.assertEqual(
                target_batches * batch_size,
                self.net.target_regression_metric.num_elements,
            )
            self.assertEqual(
                target_batches, len(self.net.target_rul_score_metric.sizes)
            )
            self.assertEqual(
                score_batches * batch_size,
                self.net.target_checkpoint_score_metric.num_elements,
            )

    def _mock_predictions(self, prediction, domain_pred):
        self.net.regressor.forward = mock.MagicMock(side_effect=prediction)
        self.net.domain_disc.forward = mock.MagicMock(side_effect=domain_pred)

    def _feed_dummy_source(self, num_batches):
        for i in range(num_batches):
            self.net.validation_step(
                (torch.zeros(32, 14, 30), torch.zeros(32)),
                batch_idx=i,
                dataloader_idx=0,
            )

    def _feed_dummy_target(self, num_batches):
        for i in range(num_batches):
            self.net.validation_step(
                (torch.zeros(32, 14, 30), torch.zeros(32)),
                batch_idx=i,
                dataloader_idx=1,
            )

    def _feed_dummy_score(self, num_batches):
        for i in range(num_batches):
            self.net.validation_step(
                (
                    torch.zeros(32, 14, 30),
                    torch.zeros(32, 14, 30),
                    torch.zeros(32),
                    torch.zeros(32),
                ),
                batch_idx=i,
                dataloader_idx=2,
            )
