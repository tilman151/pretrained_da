import unittest
from unittest import mock

import torch

from lightning import baseline, dann, pretraining


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
        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    @torch.no_grad()
    def test_domain_disc(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.domain_disc(inputs)
        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    def test_batch_independence(self):
        inputs = torch.randn(16, 14, 30)
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        for n, output in enumerate(outputs):
            with self.subTest(n_output=n):
                # Mask loss for certain samples in batch
                batch_size = output.shape[0]
                mask_idx = torch.randint(0, batch_size, ())
                mask = torch.ones_like(output)
                mask[mask_idx] = 0
                output = output * mask

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

        loss, *_ = self.net._train(
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

    @torch.no_grad()
    def test_train_metrics(self):
        criterion = torch.nn.MSELoss()
        target = torch.zeros(16, 14, 30)
        source = torch.zeros(16, 14, 30)
        target_labels = torch.ones(16)
        domain_labels = torch.cat(
            [torch.zeros_like(target_labels), torch.ones_like(target_labels)]
        )

        expected_prediction = self.net.regressor(self.net.encoder(target))
        expected_loss = torch.sqrt(
            criterion(expected_prediction.squeeze(), target_labels)
        )
        _, actual_loss, _ = self.net._train(target, target_labels, source, domain_labels)

        self.assertEqual(expected_loss, actual_loss)

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
    def test_metric_reduction(self, mock_log):
        loss_source = torch.randn(10000) + 5
        loss_target = torch.randn(10000) + 10

        expected_rmse_source = torch.sqrt(loss_source.mean())
        expected_rmse_target = torch.sqrt(loss_target.mean())
        expected_domain_loss = torch.cat([loss_source, loss_target]).mean()
        expected_rul_score = loss_target.sum()
        expected_score = torch.sqrt(loss_target.mean())

        batch_sizes = [3000, 3000, 3000, 1000]
        batched_rmse_source = [
            torch.sqrt(r.mean()) for r in torch.split(loss_source, batch_sizes)
        ]
        batched_rmse_target = [
            torch.sqrt(r.mean()) for r in torch.split(loss_target, batch_sizes)
        ]
        batched_loss_source = [r.mean() for r in torch.split(loss_source, batch_sizes)]
        batched_loss_target = [r.mean() for r in torch.split(loss_target, batch_sizes)]
        batched_rul_score = [r.sum() for r in torch.split(loss_target, batch_sizes)]
        batched_source = list(
            zip(batched_rmse_source, batched_loss_source, batched_rul_score, batch_sizes)
        )
        batched_target = list(
            zip(batched_rmse_target, batched_loss_target, batched_rul_score, batch_sizes)
        )
        batched_scores = list(zip(batched_loss_target, batch_sizes))

        self.net.validation_epoch_end([batched_source, batched_target, batched_scores])
        expected_logs = {
            "val/regression_loss": expected_rmse_target,
            "val/source_regression_loss": expected_rmse_source,
            "val/domain_loss": expected_domain_loss,
            "val/rul_score": expected_rul_score,
            "val/score": expected_score,
        }

        for call in mock_log.mock_calls:
            metric = call[1][0]
            expected_value = expected_logs[metric].item()
            actual_value = call[1][1].item()
            with self.subTest(metric):
                self.assertAlmostEqual(expected_value, actual_value, places=5)


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.trade_off = 0.5
        self.net = baseline.Baseline(
            in_channels=14,
            seq_len=30,
            num_layers=4,
            kernel_size=3,
            base_filters=16,
            latent_dim=64,
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
        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    def test_batch_independence(self):
        inputs = torch.randn(16, 14, 30)
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        for n, output in enumerate(outputs):
            with self.subTest(n_output=n):
                # Mask loss for certain samples in batch
                batch_size = output.shape[0]
                mask_idx = torch.randint(0, batch_size, ())
                mask = torch.ones_like(output)
                mask[mask_idx] = 0
                output = output * mask

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

        loss = self.net.training_step(
            (torch.randn(16, 14, 30), torch.ones(16)), batch_idx=0
        )
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad ** 2))

    @torch.no_grad()
    def test_eval_metrics(self):
        criterion = torch.nn.MSELoss()
        target = torch.zeros(16, 14, 30)
        target_labels = torch.ones(16)

        expected_prediction = self.net.regressor(self.net.encoder(target))
        expected_loss = torch.sqrt(
            criterion(expected_prediction.squeeze(), target_labels)
        )
        actual_loss, _ = self.net.validation_step((target, target_labels), 0)

        self.assertEqual(expected_loss, actual_loss)


class TestUnsupervisedPretraining(unittest.TestCase):
    def setUp(self):
        self.net = pretraining.UnsupervisedPretraining(
            in_channels=14,
            seq_len=30,
            num_layers=4,
            kernel_size=3,
            base_filters=16,
            latent_dim=64,
            dropout=0.1,
            domain_tradeoff=0.001,
            domain_disc_dim=32,
            num_disc_layers=2,
            weight_decay=0,
            lr=0.01,
        )

    @torch.no_grad()
    def test_encoder(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net.encoder(inputs)
        self.assertEqual(torch.Size((16, 64)), outputs.shape)

    @torch.no_grad()
    def test_get_anchor_query_embeddings(self):
        anchors = torch.randn(16, 14, 30)
        queries = torch.randn(16, 14, 30)
        anchor_embeddings, query_embeddings = self.net._get_anchor_query_embeddings(
            anchors, queries
        )

        self.assertEqual(torch.Size((16, 64)), anchor_embeddings.shape)
        self.assertEqual(torch.Size((16, 64)), query_embeddings.shape)

        for norm in torch.norm(anchor_embeddings, dim=1).tolist():
            self.assertAlmostEqual(1.0, norm, places=6)
        for norm in torch.norm(query_embeddings, dim=1).tolist():
            self.assertAlmostEqual(1.0, norm, places=6)

    @torch.no_grad()
    def test_forward(self):
        anchors = torch.randn(16, 14, 30)
        anchors[5] = 1.0
        queries = torch.randn(16, 14, 30)
        queries[5] = 1.0
        self.net.eval()
        distances = self.net(anchors, queries)

        self.assertEqual(torch.Size((16,)), distances.shape)
        self.assertAlmostEqual(0.0, distances[5].item(), delta=1e-6)
        self.assertNotAlmostEqual(0.0, distances.sum().item())

    def test_batch_independence(self):
        torch.autograd.set_detect_anomaly(True)

        anchors = torch.randn(16, 14, 30)
        queries = torch.randn(16, 14, 30)
        anchors.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(anchors, queries)
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
        for i, grad in enumerate(anchors.grad[:batch_size]):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))
        anchors.grad = None

        torch.autograd.set_detect_anomaly(False)

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        inputs = (
            torch.randn(16, 14, 30),
            torch.randn(16, 14, 30),
            torch.randn(16),
            torch.randn(16),
        )
        loss = self.net.training_step(inputs, batch_idx=0)
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad ** 2))
