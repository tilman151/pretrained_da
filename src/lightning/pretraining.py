import pytorch_lightning as pl
import torch
import torch.nn as nn

from lightning import metrics
from models import networks


class UnsupervisedPretraining(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 seq_len,
                 num_layers,
                 kernel_size,
                 base_filters,
                 latent_dim,
                 lr,
                 record_embeddings=False):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.lr = lr
        self.record_embeddings = record_embeddings

        self.encoder = networks.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                        self.num_layers, self.latent_dim, self.seq_len)
        self.criterion_regression = nn.MSELoss()
        self.embedding_metric = metrics.EmbeddingViz(40000, self.latent_dim)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, anchors, queries):
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(anchors, queries)
        pred_distances = torch.pairwise_distance(anchor_embeddings, query_embeddings, eps=1e-8)

        return pred_distances

    def _get_anchor_query_embeddings(self, anchors, queries):
        batch_size = anchors.shape[0]
        combined = torch.cat([anchors, queries])
        embeddings = self._get_embeddings(combined)
        anchor_embeddings, query_embeddings = torch.split(embeddings, batch_size)

        return anchor_embeddings, query_embeddings

    def _get_embeddings(self, inputs):
        embeddings = self.encoder(inputs)
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

        return embeddings

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val/loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test/loss', loss)

        return loss

    def _get_loss(self, batch):
        anchors, queries, true_distances = batch
        pred_distances = self.forward(anchors, queries)
        loss = self.criterion_regression(pred_distances, true_distances)

        return loss
