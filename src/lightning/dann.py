import pytorch_lightning as pl
import torch
import torch.nn as nn

from lightning import metrics
from lightning.mixins import LoadEncoderMixin
from models import networks


class DANN(pl.LightningModule, LoadEncoderMixin):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout,
        domain_trade_off,
        domain_disc_dim,
        num_disc_layers,
        optim_type,
        lr,
        record_embeddings=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.domain_trade_off = domain_trade_off
        self.domain_disc_dim = domain_disc_dim
        self.num_disc_layers = num_disc_layers
        self.optim_type = optim_type
        self.lr = lr
        self.record_embeddings = record_embeddings

        self._test_tag = ""

        self.encoder = networks.Encoder(
            self.in_channels,
            self.base_filters,
            self.kernel_size,
            self.num_layers,
            self.latent_dim,
            self.seq_len,
            dropout=self.dropout,
            norm_outputs=False,
        )
        self.domain_disc = networks.DomainDiscriminator(
            self.latent_dim, self.num_disc_layers, self.domain_disc_dim
        )
        self.regressor = networks.Regressor(latent_dim)

        self.criterion_recon = nn.MSELoss()
        self.criterion_regression = metrics.RMSELoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()
        self.rul_score = metrics.RULScore()

        self.embedding_metric = metrics.EmbeddingViz(self.latent_dim)
        self.source_regression_metric = metrics.RMSELoss()
        self.target_regression_metric = metrics.RMSELoss()
        self.target_rul_score_metric = metrics.SimpleMetric(reduction="sum")
        self.target_checkpoint_score_metric = metrics.RMSELoss()
        self.domain_loss_metric = metrics.SimpleMetric()

        self.save_hyperparameters()

    @property
    def test_tag(self):
        return self._test_tag

    @test_tag.setter
    def test_tag(self, value):
        self._test_tag = value

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters()},
            {"params": self.regressor.parameters()},
            {"params": self.domain_disc.parameters()},
        ]
        if self.optim_type == "adam":
            return torch.optim.Adam(param_groups, lr=self.lr)
        else:
            return torch.optim.SGD(
                param_groups, lr=self.lr, momentum=0.9, weight_decay=0.01
            )

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def training_step(self, batch, batch_idx):
        source, source_labels, target = batch
        domain_labels = torch.cat(
            [torch.ones_like(source_labels), torch.zeros_like(source_labels)]
        )
        # Predict on source and reconstruct/domain classify both
        loss, regression_loss, domain_loss = self._get_losses(
            source, source_labels, target, domain_labels
        )

        self.log("train/loss", loss)
        self.log("train/regression_loss", regression_loss)
        self.log("train/domain_loss", domain_loss)

        return loss

    def _get_losses(self, source, source_labels, target, domain_labels):
        domain_prediction, prediction = self._train_forward(source, target)
        regression_loss = self.criterion_regression(prediction, source_labels)
        domain_loss = self.criterion_domain(domain_prediction, domain_labels)
        loss = regression_loss + self.domain_trade_off * domain_loss

        return loss, regression_loss, domain_loss

    def _train_forward(self, source, target):
        batch_size = source.shape[0]
        common = torch.cat([source, target])

        latent_code = self.encoder(common)
        regression_code, _ = torch.split(latent_code, batch_size)
        prediction = self.regressor(regression_code)
        domain_prediction = self.domain_disc(latent_code)

        return domain_prediction, prediction

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def on_test_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.embedding_metric.reset()
        self.source_regression_metric.reset()
        self.target_regression_metric.reset()
        self.target_checkpoint_score_metric.reset()
        self.target_rul_score_metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self._choose_step_by_dataloader(
            batch, dataloader_idx, self.record_embeddings
        )

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self._choose_step_by_dataloader(
            batch, dataloader_idx, record_embeddings=False
        )

    def _choose_step_by_dataloader(self, batch, dataloader_idx, record_embeddings):
        if dataloader_idx == 0:
            features, labels = batch
            self._evaluate_source(
                features,
                labels,
                record_embeddings,
            )
        elif dataloader_idx == 1:
            features, labels = batch
            self._evaluate_target(
                features,
                labels,
                record_embeddings,
            )
        else:
            anchors, queries, true_distances, _ = batch
            self._evaluate_pairs(anchors, queries, true_distances)

    def _evaluate_source(self, source, labels, record_embeddings=False):
        batch_size = source.shape[0]
        domain_labels = torch.ones_like(labels)
        domain_prediction, prediction = self._eval_forward(source)

        self.source_regression_metric.update(prediction, labels)
        domain_loss = self.criterion_domain(domain_prediction, domain_labels)
        self.domain_loss_metric.update(domain_loss, batch_size)

        if record_embeddings:
            latent_code = self.encoder(source)
            self.embedding_metric.update(latent_code, domain_labels, labels)

    def _evaluate_target(self, target, labels, record_embeddings=False):
        batch_size = target.shape[0]
        domain_labels = torch.zeros_like(labels)
        domain_prediction, prediction = self._eval_forward(target)

        self.target_regression_metric.update(prediction, labels)
        rul_score = self.rul_score(prediction, labels)
        self.target_rul_score_metric.update(rul_score, batch_size)
        domain_loss = self.criterion_domain(domain_prediction, domain_labels)
        self.domain_loss_metric.update(domain_loss, batch_size)

        if record_embeddings:
            latent_code = self.encoder(target)
            self.embedding_metric.update(latent_code, domain_labels, labels)

    def _eval_forward(self, features):
        latent_code = self.encoder(features)
        prediction = self.regressor(latent_code)
        domain_prediction = self.domain_disc(latent_code)
        return domain_prediction, prediction

    def _evaluate_pairs(self, anchors, queries, true_distances):
        anchor_rul = self.forward(anchors)
        query_rul = self.forward(queries)
        distances = query_rul - anchor_rul
        self.target_checkpoint_score_metric.update(distances, true_distances * 125)

    def validation_epoch_end(self, outputs):
        if self.record_embeddings:
            embedding_fig = self.embedding_metric.compute()
            self.logger.log_figure("val/embeddings", embedding_fig, self.global_step)
        self.log("val/regression_loss", self.target_regression_metric.compute())
        self.log("val/source_regression_loss", self.source_regression_metric.compute())
        self.log("val/domain_loss", self.domain_loss_metric.compute())
        self.log("val/rul_score", self.target_rul_score_metric.compute())
        self.log("val/score", self.target_checkpoint_score_metric.compute())

    def test_epoch_end(self, outputs):
        tag = f"test/{self.test_tag}" if self.test_tag else "test"
        self.log(f"{tag}/regression_loss", self.target_regression_metric.compute())
        self.log(f"{tag}/domain_loss", self.domain_loss_metric.compute())
        self.log(f"{tag}/rul_score", self.target_rul_score_metric.compute())
