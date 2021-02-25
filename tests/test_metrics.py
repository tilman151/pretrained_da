import unittest

import matplotlib.pyplot as plt
import torch

from lightning import metrics


class TestEmbeddingViz(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.EmbeddingViz(32, 128)

    def test_updating(self):
        embeddings = torch.randn(16, 128)
        labels = torch.ones(16)
        ruls = torch.arange(0, 16)

        self.metric.update(embeddings, labels, ruls)
        self.metric.update(embeddings, labels, ruls)

        self.assertEqual(0, torch.sum(self.metric.embeddings[:16] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[:16] - labels))
        self.assertEqual(0, torch.sum(self.metric.embeddings[16:32] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[16:32] - labels))

    def test_compute(self):
        embeddings = torch.randn(32, 128)
        labels = torch.cat([torch.ones(16), torch.zeros(16)])
        ruls = torch.arange(0, 32)

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
        self.assertEqual(1, self.metric.sample_counter)
        self.assertEqual(expected_sse, self.metric.losses[0])
        self.assertEqual(batch_size, self.metric.sizes[0])

    def test_reset(self):
        self._add_one_batch()
        self.metric.reset()
        self.assertEqual(0, self.metric.sample_counter)
        self.assertEqual(0, self.metric.losses.sum())
        self.assertEqual(0, self.metric.sizes.sum())

    def _add_one_batch(self):
        batch_size = 100
        inputs = torch.ones(batch_size) * 2
        targets = torch.zeros_like(inputs)
        summed_squares = torch.sum((inputs - targets) ** 2)

        self.metric.update(inputs, targets)

        return summed_squares, batch_size

    def test_compute(self):
        batch_sizes = [3000, 3000, 3000, 1000]
        inputs = torch.randn(sum(batch_sizes)) + 100
        targets = torch.randn_like(inputs)
        expected_rmse = torch.sqrt(torch.mean((inputs - targets) ** 2))

        batched_inputs = torch.split(inputs, batch_sizes)
        batched_targets = torch.split(targets, batch_sizes)
        for inp, tgt in zip(batched_inputs, batched_targets):
            self.metric.update(inp, tgt)
        actual_rmse = self.metric.compute()

        self.assertEqual(expected_rmse, actual_rmse)
