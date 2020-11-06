import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lightning import metrics
from models import networks


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class DAAN(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 seq_len,
                 num_layers,
                 kernel_size,
                 base_filters,
                 latent_dim,
                 domain_trade_off,
                 domain_disc_dim,
                 num_disc_layers,
                 source_rul_cap,
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
        self.domain_trade_off = domain_trade_off
        self.domain_disc_dim = domain_disc_dim
        self.num_disc_layers = num_disc_layers
        self.source_rul_cap = source_rul_cap
        self.optim_type = optim_type
        self.lr = lr
        self.record_embeddings = record_embeddings

        self.encoder = networks.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                        self.num_layers, self.latent_dim, self.seq_len)
        self.domain_disc = networks.DomainDiscriminator(self.latent_dim, self.num_disc_layers, self.domain_disc_dim)
        self.regressor = networks.Regressor(latent_dim)

        self.criterion_recon = nn.MSELoss()
        self.criterion_regression = RMSELoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

        self.embedding_metric = metrics.EmbeddingViz(40000, self.latent_dim)

        self.save_hyperparameters()

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [{'params': self.encoder.parameters()},
                        {'params': self.regressor.parameters()},
                        {'params': self.domain_disc.parameters()}]
        if self.optim_type == 'adam':
            return torch.optim.Adam(param_groups, lr=self.lr)
        else:
            return torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9, weight_decay=0.01)

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def training_step(self, batch, batch_idx):
        source, source_labels, target = batch
        domain_labels = torch.cat([torch.ones_like(source_labels),
                                   torch.zeros_like(source_labels)])
        # Predict on source and reconstruct/domain classify both
        loss, regression_loss, domain_loss = self._train(source, source_labels, target, domain_labels, cap=True)

        self.log('train/loss', loss)
        self.log('train/regression_loss', regression_loss)
        self.log('train/domain_loss', domain_loss)

        return loss

    def _train(self, regressor_features, regressor_labels, auxiliary_features, domain_labels, cap=False):
        common = torch.cat([regressor_features, auxiliary_features])
        domain_prediction, prediction = self._complete_forward(common)

        rul_mask = self._get_rul_mask(regressor_labels, cap)
        regression_loss = self.criterion_regression(prediction.squeeze(), regressor_labels)
        domain_loss = self.criterion_domain(domain_prediction.squeeze()[rul_mask], domain_labels[rul_mask])
        loss = regression_loss + self.domain_trade_off * domain_loss

        return loss, regression_loss, domain_loss

    def _get_rul_mask(self, classifier_labels, cap):
        if cap and self.source_rul_cap is not None:
            rul_mask = (classifier_labels > self.source_rul_cap)
        else:
            rul_mask = torch.ones_like(classifier_labels, dtype=torch.bool)

        return rul_mask.repeat(2)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        features, labels = batch
        domain_labels = torch.ones_like(labels) if dataloader_idx == 0 else torch.zeros_like(labels)
        return self._evaluate(features, labels, domain_labels, record_embeddings=self.record_embeddings)

    def test_step(self, batch, batch_idx, dataloader_idx):
        features, labels = batch
        domain_labels = torch.ones_like(labels) if dataloader_idx == 0 else torch.zeros_like(labels)
        return self._evaluate(features, labels, domain_labels, record_embeddings=False)

    def _evaluate(self, features, labels, domain_labels, record_embeddings=False):
        batch_size = features.shape[0]
        # Predict on target and reconstruct/domain classify both
        common = torch.cat([features, torch.empty_like(features)])
        domain_prediction, prediction = self._complete_forward(common)

        regression_loss = self.criterion_regression(prediction.squeeze(), labels)
        domain_loss = self.criterion_domain(domain_prediction[:batch_size].squeeze(), domain_labels)

        if record_embeddings:
            latent_code = self.encoder(features)
            self.embedding_metric.update(latent_code, domain_labels, labels)

        return regression_loss, domain_loss, batch_size

    def validation_epoch_end(self, outputs):
        if self.record_embeddings:
            self._get_tensorboard().add_figure('val/embeddings', self.embedding_metric.compute(), self.global_step)
            self.embedding_metric.reset()

        regression_loss, domain_loss = self._reduce_metrics(outputs)
        self.log(f'val/regression_loss', regression_loss)
        self.log(f'val/domain_loss', domain_loss)

    def _get_tensorboard(self):
        if isinstance(self.logger.experiment, SummaryWriter):
            return self.logger.experiment
        elif isinstance(self.logger.experiment, list):
            for logger in self.logger.experiment:
                if isinstance(logger, SummaryWriter):
                    return logger
        else:
            raise ValueError('No TensorBoard logger specified. Cannot log embeddings.')

    def test_epoch_end(self, outputs):
        regression_loss, domain_loss = self._reduce_metrics(outputs)
        self.log(f'test/regression_loss', regression_loss)
        self.log(f'test/domain_loss', domain_loss)

    def _reduce_metrics(self, outputs):
        outputs = [item for sublist in outputs for item in sublist]  # concat outputs of both dataloaders
        regression_loss, domain_loss, batch_size = zip(*outputs)  # separate output parts
        num_samples = sum(batch_size)
        regression_loss = torch.sqrt(sum(b * (loss ** 2) for b, loss in zip(batch_size, regression_loss)) / num_samples)
        domain_loss = sum(b * loss for b, loss in zip(batch_size, domain_loss)) / num_samples

        return regression_loss, domain_loss

    def _complete_forward(self, common):
        batch_size = common.shape[0] // 2

        latent_code = self.encoder(common)
        regression_code, _ = torch.split(latent_code, batch_size)
        prediction = self.regressor(regression_code)
        domain_prediction = self.domain_disc(latent_code)

        return domain_prediction, prediction
