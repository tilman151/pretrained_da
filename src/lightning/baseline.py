import pytorch_lightning as pl
import torch

from lightning import metrics
from lightning.mixins import LoadEncoderMixin
from models import networks


class Baseline(pl.LightningModule, LoadEncoderMixin):
    def __init__(self,
                 in_channels,
                 seq_len,
                 num_layers,
                 kernel_size,
                 base_filters,
                 latent_dim,
                 optim_type,
                 lr,
                 record_embeddings=False):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.optim_type = optim_type
        self.lr = lr
        self.record_embeddings = record_embeddings

        self.encoder = networks.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                        self.num_layers, self.latent_dim, self.seq_len, dropout=0)
        self.regressor = networks.Regressor(latent_dim)

        self.criterion_regression = metrics.RMSELoss()

        self.embedding_metric = metrics.EmbeddingViz(20000, self.latent_dim)

        self.save_hyperparameters()

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        encoder_lr = 0 if 'pretrained_checkpoint' in self.hparams else self.lr
        param_groups = [{'params': self.encoder.parameters(), 'lr': encoder_lr},
                        {'params': self.regressor.parameters()}]
        if self.optim_type == 'adam':
            return torch.optim.Adam(param_groups, lr=self.lr)
        else:
            return torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9, weight_decay=0.01)

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def training_step(self, batch, batch_idx):
        source, source_labels = batch
        predictions = self(source)
        loss = self.criterion_regression(predictions.squeeze(), source_labels)

        self.log('train/regression_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        source, source_labels = batch
        regression_loss, batch_size = self._evaluate(source, source_labels)

        return regression_loss, batch_size

    def test_step(self, batch, batch_idx, dataloader_idx):
        source, source_labels, target, target_labels = batch
        regression_loss, batch_size = self._evaluate(target, target_labels)

        if self.record_embeddings:
            domain_labels = torch.cat([torch.zeros_like(source_labels),
                                       torch.ones_like(source_labels)])
            latent_code = self.encoder(torch.cat([source, target]))
            ruls = torch.cat([source_labels, target_labels])
            self.embedding_metric.update(latent_code, domain_labels, ruls)

        return regression_loss, batch_size

    def _evaluate(self, features, labels):
        batch_size = features.shape[0]
        predictions = self(features)
        regression_loss = self.criterion_regression(predictions.squeeze(), labels)

        return regression_loss, batch_size

    def validation_epoch_end(self, outputs):
        regression_loss = self._reduce_metrics(outputs)
        self.log('val/regression_loss', regression_loss)

    def test_epoch_end(self, outputs):
        if self.record_embeddings:
            self.logger.experiment.add_figure('test/embeddings', self.embedding_metric.compute(), self.global_step)
            self.embedding_metric.reset()

        for fd, out in enumerate(outputs, start=1):
            regression_loss = self._reduce_metrics(out)
            self.log(f'test/regression_loss_fd{fd}', regression_loss)

    def _reduce_metrics(self, outputs):
        regression_loss, batch_size = zip(*outputs)
        num_samples = sum(batch_size)
        regression_loss = torch.sqrt(sum(b * (loss ** 2) for b, loss in zip(batch_size, regression_loss)) / num_samples)

        return regression_loss
