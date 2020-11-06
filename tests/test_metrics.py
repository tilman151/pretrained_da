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
        self.assertListEqual([20., 10.], list(fig.get_size_inches()))
