import pytorch_lightning as pl
import torch
import torch.nn as nn

from lightning import metrics
from lightning.mixins import DataHparamsMixin, LoadEncoderMixin
from models import networks


class DANN(pl.LightningModule, DataHparamsMixin, LoadEncoderMixin):
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

        self.embedding_metric = metrics.EmbeddingViz(40000, self.latent_dim)

        self.save_hyperparameters()

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        encoder_lr = self.lr if "pretrained_checkpoint" in self.hparams else self.lr
        param_groups = [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
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
        loss, regression_loss, domain_loss = self._train(
            source, source_labels, target, domain_labels
        )

        self.log("train/loss", loss)
        self.log("train/regression_loss", regression_loss)
        self.log("train/domain_loss", domain_loss)

        return loss

    def _train(
        self, regressor_features, regressor_labels, auxiliary_features, domain_labels
    ):
        common = torch.cat([regressor_features, auxiliary_features])
        domain_prediction, prediction = self._complete_forward(common)

        regression_loss = self.criterion_regression(
            prediction.squeeze(), regressor_labels
        )
        domain_loss = self.criterion_domain(domain_prediction.squeeze(), domain_labels)
        loss = regression_loss + self.domain_trade_off * domain_loss

        return loss, regression_loss, domain_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx < 2:
            features, labels = batch
            domain_labels = (
                torch.ones_like(labels)
                if dataloader_idx == 0
                else torch.zeros_like(labels)
            )
            regression_loss, domain_loss, rul_score, batch_size = self._evaluate(
                features, labels, domain_labels, record_embeddings=self.record_embeddings
            )

            return regression_loss, domain_loss, rul_score, batch_size
        else:
            anchors, queries, true_distances, _ = batch
            anchor_rul = self.forward(anchors)
            query_rul = self.forward(queries)
            distances = query_rul - anchor_rul
            score = nn.functional.mse_loss(distances / 125, true_distances)

            return score, anchors.shape[0]

    def test_step(self, batch, batch_idx, dataloader_idx):
        features, labels = batch
        domain_labels = (
            torch.ones_like(labels) if dataloader_idx == 0 else torch.zeros_like(labels)
        )
        regression_loss, domain_loss, rul_score, batch_size = self._evaluate(
            features, labels, domain_labels, record_embeddings=False
        )

        return regression_loss, domain_loss, rul_score, batch_size

    def _evaluate(self, features, labels, domain_labels, record_embeddings=False):
        batch_size = features.shape[0]
        latent_code = self.encoder(features)
        prediction = self.regressor(latent_code)
        domain_prediction = self.domain_disc(latent_code)

        regression_loss = self.criterion_regression(prediction.squeeze(), labels)
        domain_loss = self.criterion_domain(domain_prediction.squeeze(), domain_labels)
        rul_score = self.rul_score(prediction.squeeze(), labels)

        if record_embeddings:
            latent_code = self.encoder(features)
            self.embedding_metric.update(latent_code, domain_labels, labels)

        return regression_loss, domain_loss, rul_score, batch_size

    def validation_epoch_end(self, outputs):
        if self.record_embeddings:
            embedding_fig = self.embedding_metric.compute()
            self.logger.log_figure("val/embeddings", embedding_fig, self.global_step)
            self.embedding_metric.reset()

        (
            regression_loss,
            source_regression_loss,
            domain_loss,
            _,
            score,
        ) = self._reduce_metrics(outputs)
        self.log("val/regression_loss", regression_loss)
        self.log("val/source_regression_loss", source_regression_loss)
        self.log("val/domain_loss", domain_loss)
        self.log("val/score", score)

    def test_epoch_end(self, outputs):
        (
            regression_loss,
            source_regression_loss,
            domain_loss,
            rul_score,
            _,
        ) = self._reduce_metrics(outputs)
        self.log(f"test/regression_loss", regression_loss)
        self.log(f"test/domain_loss", domain_loss)
        self.log("test/rul_score", rul_score)

    def _reduce_metrics(self, outputs):
        source_outputs, target_outputs, *_ = outputs

        (
            source_domain_loss,
            source_regression_loss,
            _,
            source_num_samples,
        ) = self.__reduce_metrics(source_outputs)
        (
            target_domain_loss,
            regression_loss,
            rul_score,
            target_num_samples,
        ) = self.__reduce_metrics(target_outputs)
        domain_loss = (source_domain_loss + target_domain_loss) / (
            source_num_samples + target_num_samples
        )
        if len(outputs) == 3:
            score_outputs = outputs[-1]
            score = sum(s * b for s, b in score_outputs) / sum(
                b for _, b in score_outputs
            )
        else:
            score = 0.0

        return regression_loss, source_regression_loss, domain_loss, rul_score, score

    def __reduce_metrics(self, outputs):
        regression_loss, domain_loss, rul_score, batch_size = zip(
            *outputs
        )  # separate output parts
        num_samples = sum(batch_size)
        regression_loss = torch.sqrt(
            sum(b * (loss ** 2) for b, loss in zip(batch_size, regression_loss))
            / num_samples
        )
        summed_domain_loss = sum(b * loss for b, loss in zip(batch_size, domain_loss))
        rul_score = sum(rul_score)
        return summed_domain_loss, regression_loss, rul_score, num_samples

    def _complete_forward(self, common):
        batch_size = common.shape[0] // 2

        latent_code = self.encoder(common)
        regression_code, _ = torch.split(latent_code, batch_size)
        prediction = self.regressor(regression_code)
        domain_prediction = self.domain_disc(latent_code)

        return domain_prediction, prediction
