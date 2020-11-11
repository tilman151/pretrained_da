import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lightning import metrics
from lightning.mixins import DataHparamsMixin
from models import networks


class UnsupervisedPretraining(pl.LightningModule, DataHparamsMixin):
    def __init__(self,
                 in_channels,
                 seq_len,
                 num_layers,
                 kernel_size,
                 base_filters,
                 latent_dim,
                 dropout,
                 lr,
                 weight_decay,
                 record_embeddings=False):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.record_embeddings = record_embeddings

        self.encoder = networks.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                        self.num_layers, self.latent_dim, self.seq_len, self.dropout)
        self.criterion_regression = nn.MSELoss()
        self.embedding_metric = metrics.EmbeddingViz(40000, self.latent_dim)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            return self._get_loss(batch), batch[0].shape[0]
        else:
            self._record_embeddings(batch, dataloader_idx)

    def _record_embeddings(self, batch, dataloader_idx):
        features, labels = batch
        embedding = self.encoder(features)
        domain_labels = torch.full_like(labels, fill_value=(2 - dataloader_idx), dtype=torch.int)
        self.embedding_metric.update(embedding, domain_labels, labels)

    def validation_epoch_end(self, validation_step_outputs):
        self._get_tensorboard().add_figure('val/embeddings', self.embedding_metric.compute(), self.global_step)
        self.embedding_metric.reset()

        val_loss = validation_step_outputs[0]
        _, batch_sizes = zip(*val_loss)
        loss = sum(loss * batch_size for loss, batch_size in val_loss) / sum(batch_sizes)
        self.log('val/loss', loss)

    def _get_tensorboard(self):
        if isinstance(self.logger.experiment, SummaryWriter):
            return self.logger.experiment
        elif isinstance(self.logger.experiment, list):
            for logger in self.logger.experiment:
                if isinstance(logger, SummaryWriter):
                    return logger
        else:
            raise ValueError('No TensorBoard logger specified. Cannot log embeddings.')

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test/loss', loss)

        return loss

    def _get_loss(self, batch):
        anchors, queries, true_distances = batch
        pred_distances = self.forward(anchors, queries)
        loss = self.criterion_regression(pred_distances, true_distances)

        return loss
