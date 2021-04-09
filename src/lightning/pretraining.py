import pytorch_lightning as pl
import torch
import torch.nn as nn

from lightning import metrics
from lightning.mixins import DataHparamsMixin
from models import networks


class UnsupervisedPretraining(pl.LightningModule, DataHparamsMixin):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout,
        domain_tradeoff,
        domain_disc_dim,
        num_disc_layers,
        lr,
        weight_decay,
        record_embeddings=False,
        encoder="cnn",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.domain_tradeoff = domain_tradeoff
        self.domain_disc_dim = domain_disc_dim
        self.num_disc_layers = num_disc_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.record_embeddings = record_embeddings
        self.encoder_type = encoder

        self.encoder = self._get_encoder()
        if self.domain_tradeoff > 0:
            self.domain_disc = networks.DomainDiscriminator(
                self.latent_dim,
                num_layers=self.num_disc_layers,
                hidden_dim=self.domain_disc_dim,
            )
        else:
            self.domain_disc = None

        self.criterion_regression = nn.MSELoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

        self.embedding_metric = metrics.EmbeddingViz(40000, self.latent_dim)
        self.regression_metric = metrics.SimpleMetric(40000)
        self.domain_metric = metrics.SimpleMetric(40000)

        self.save_hyperparameters()
        self.hparams["mode"] = "metric"

    def _get_encoder(self):
        if self.encoder_type == "cnn":
            encoder = networks.Encoder(
                self.in_channels,
                self.base_filters,
                self.kernel_size,
                self.num_layers,
                self.latent_dim,
                self.seq_len,
                self.dropout,
                norm_outputs=True,
            )
        elif self.encoder_type == "lstm":
            encoder = networks.EncoderEllefsenEtAl(
                self.in_channels,
                self.base_filters,
                self.kernel_size,
                self.num_layers,
                self.latent_dim,
                self.seq_len,
                self.dropout,
                norm_outputs=True,
            )
        else:
            raise ValueError(
                f"Unknown encoder type {self.encoder}. Use either 'cnn' or 'lstm'."
            )
        return encoder

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, anchors, queries):
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(
            anchors, queries
        )
        pred_distances = self._pairwise_distance(anchor_embeddings, query_embeddings)

        return pred_distances

    def training_step(self, batch, batch_idx):
        regression_loss, domain_loss = self._get_losses(batch)
        loss = regression_loss + self.domain_tradeoff * domain_loss
        self.log("train/loss", loss)
        self.log("train/regression_loss", regression_loss)
        self.log("train/domain_loss", domain_loss)

        return loss

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.embedding_metric.reset()
        self.regression_metric.reset()
        self.domain_metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            regression_loss, domain_loss = self._get_losses(batch)
            batch_size = batch[0].shape[0]
            self.regression_metric.update(regression_loss, batch_size)
            self.domain_metric.update(domain_loss, batch_size)
        elif self.record_embeddings:
            self._record_embeddings(batch, dataloader_idx)

    def _record_embeddings(self, batch, dataloader_idx):
        features, labels = batch
        embedding = self.encoder(features)
        domain_labels = torch.full_like(
            labels, fill_value=(2 - dataloader_idx), dtype=torch.int
        )
        self.embedding_metric.update(embedding, domain_labels, labels)

    def validation_epoch_end(self, validation_step_outputs):
        if self.record_embeddings:
            embedding_fig = self.embedding_metric.compute()
            self.logger.log_figure("val/embeddings", embedding_fig, self.global_step)

        regression_loss = self.regression_metric.compute()
        domain_loss = self.domain_metric.compute()

        self.log("val/regression_loss", regression_loss)
        self.log("val/domain_loss", domain_loss)
        self.log("val/checkpoint_score", regression_loss - 0.1 * domain_loss)

    def _get_losses(self, batch):
        anchors, queries, true_distances, domain_labels = batch
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(
            anchors, queries
        )
        pred_distances = self._pairwise_distance(anchor_embeddings, query_embeddings)
        regression_loss = self.criterion_regression(pred_distances, true_distances)

        if self.domain_tradeoff > 0:
            domain_pred = self.domain_disc(anchor_embeddings)
            domain_loss = self.criterion_domain(domain_pred, domain_labels)
        else:
            domain_loss = 0

        return regression_loss, domain_loss

    def _pairwise_distance(self, anchor_embeddings, query_embeddings):
        return torch.pairwise_distance(anchor_embeddings, query_embeddings, eps=1e-8)

    def _get_anchor_query_embeddings(self, anchors, queries):
        batch_size = anchors.shape[0]
        combined = torch.cat([anchors, queries])
        embeddings = self.encoder(combined)
        anchor_embeddings, query_embeddings = torch.split(embeddings, batch_size)

        return anchor_embeddings, query_embeddings
