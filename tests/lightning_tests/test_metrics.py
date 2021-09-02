import unittest

import matplotlib.pyplot as plt
import torch

from lightning import metrics


class TestEmbeddingViz(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.EmbeddingViz(embedding_size=8)

    def test_updating(self):
        embeddings = torch.randn(5, 8)
        labels = torch.ones(5)
        ruls = torch.arange(0, 5)

        self.metric.update(embeddings, labels, ruls)
        self.metric.update(embeddings, labels, ruls)

        self.assertEqual(0, torch.sum(self.metric.embeddings[0] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[0] - labels))
        self.assertEqual(0, torch.sum(self.metric.embeddings[1] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[1] - labels))

    def test_compute(self):
        embeddings = torch.randn(16, 8)
        labels = torch.cat([torch.ones(10), torch.zeros(6)])
        ruls = torch.arange(0, 16)

        self.metric.update(embeddings, labels, ruls)
        fig = self.metric.compute()

        self.assertIsInstance(fig, plt.Figure)
        self.assertListEqual([20.0, 10.0], list(fig.get_size_inches()))


class TestRULScore(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.RULScore()

    def test_negative(self):
        inputs = torch.ones(2)
        targets = torch.ones(2) * 2
        actual_score = self.metric(inputs, targets)
        expected_score = torch.sum(
            torch.exp((inputs - targets) / self.metric.neg_factor) - 1
        )
        self.assertEqual(expected_score, actual_score)

    def test_positive(self):
        inputs = torch.ones(2) * 2
        targets = torch.ones(2)
        actual_score = self.metric(inputs, targets)
        expected_score = torch.sum(
            torch.exp((inputs - targets) / self.metric.pos_factor) - 1
        )
        self.assertEqual(expected_score, actual_score)


class TestRMSE(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.RMSELoss()

    def test_update(self):
        expected_sse, batch_size = self._add_one_batch()
        self.assertEqual(expected_sse, self.metric.losses)
        self.assertEqual(batch_size, self.metric.num_elements)

    def test_reset(self):
        self._add_one_batch()
        self.metric.reset()
        self.assertEqual(0.0, self.metric.losses)
        self.assertEqual(0, self.metric.num_elements)

    def _add_one_batch(self):
        batch_size = 16
        inputs = torch.ones(batch_size) * 2
        targets = torch.zeros_like(inputs)
        summed_squares = torch.sum((inputs - targets) ** 2)

        self.metric.update(inputs, targets)

        return summed_squares, batch_size

    def test_compute(self):
        batch_sizes = [16, 32, 32, 16]
        inputs = torch.randn(sum(batch_sizes)) + 8
        targets = torch.randn_like(inputs)
        expected_rmse = torch.sqrt(torch.mean((inputs - targets) ** 2))

        batched_inputs = torch.split(inputs, batch_sizes)
        batched_targets = torch.split(targets, batch_sizes)
        for inp, tgt in zip(batched_inputs, batched_targets):
            self.metric.update(inp, tgt)
        actual_rmse = self.metric.compute()

        self.assertAlmostEqual(expected_rmse.item(), actual_rmse.item(), delta=0.001)

    def test_forward(self):
        losses = 0.0
        num_elements = 0
        for batch_size in [16, 32, 32, 16]:
            inputs = torch.randn(batch_size) + 8
            targets = torch.randn_like(inputs)
            mse = torch.sum((inputs - targets) ** 2)
            expected_local_rmse = torch.sqrt(mse / batch_size)
            actual_rmse = self.metric(inputs, targets)
            losses += mse
            num_elements += batch_size
            self.assertEqual(expected_local_rmse, actual_rmse)
        expected_global_rmse = torch.sqrt(losses / num_elements)
        self.assertEqual(expected_global_rmse, self.metric.compute())


class TestMeanMetric(unittest.TestCase):
    def setUp(self):
        self.mean_metric = metrics.SimpleMetric()
        self.sum_metric = metrics.SimpleMetric(reduction="sum")

    def test_update(self):
        expected_loss, batch_size = self._add_one_batch()
        self.assertEqual(expected_loss, self.mean_metric.losses[0])
        self.assertEqual(batch_size, self.mean_metric.sizes[0])

    def test_reset(self):
        self._add_one_batch()
        self.mean_metric.reset()
        self.assertEqual([], self.mean_metric.losses)
        self.assertEqual([], self.mean_metric.sizes)

    def _add_one_batch(self):
        batch_size = 16
        loss = torch.tensor(500)

        self.mean_metric.update(loss, batch_size)

        return loss, batch_size

    def test_compute_mean(self):
        batch_sizes = [16] * 8 + [32]
        losses = torch.randn(sum(batch_sizes)) + 2
        expected_loss = losses.mean()

        batched_inputs = torch.split(losses, batch_sizes)
        for inp, sizes in zip(batched_inputs, batch_sizes):
            self.mean_metric.update(inp.mean(), sizes)
        actual_loss = self.mean_metric.compute()

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item(), places=5)

    def test_compute_sum(self):
        batch_sizes = [16] * 8 + [32]
        losses = torch.randn(sum(batch_sizes)) + 2
        expected_loss = losses.sum()

        batched_inputs = torch.split(losses, batch_sizes)
        for inp, sizes in zip(batched_inputs, batch_sizes):
            self.sum_metric.update(inp.sum(), sizes)
        actual_loss = self.sum_metric.compute()

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item(), delta=0.1)
