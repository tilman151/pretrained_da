import pytorch_lightning as pl
import torch
import torch.nn as nn

import metrics
import models


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class Baseline(pl.LightningModule):
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

        self.encoder = models.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                      self.num_layers, self.latent_dim, self.seq_len)
        self.regressor = models.Regressor(latent_dim)

        self.criterion_regression = RMSELoss()

        self.embedding_metric = metrics.EmbeddingViz(20000, self.latent_dim)

        self.save_hyperparameters()

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [{'params': self.encoder.parameters()},
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
        source, source_labels, _ = batch
        predictions = self(source)
        loss = self.criterion_regression(predictions.squeeze(), source_labels)

        self.log('train/regression_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, record_embeddings=self.record_embeddings)

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch)

    def validation_epoch_end(self, outputs):
        if self.record_embeddings:
            self.logger.experiment.add_figure('val/embeddings', self.embedding_metric.compute(), self.global_step)
            self.embedding_metric.reset()

        regression_loss = self._reduce_metrics(outputs)
        self.log(f'val/regression_loss', regression_loss)
        self.log('checkpoint_on', regression_loss, logger=False)

    def test_epoch_end(self, outputs):
        regression_loss = self._reduce_metrics(outputs)
        self.log(f'test/regression_loss', regression_loss)

    def _reduce_metrics(self, outputs):
        regression_loss, batch_size = zip(*outputs)
        num_samples = sum(batch_size)
        regression_loss = torch.sqrt(sum(b * (loss ** 2) for b, loss in zip(batch_size, regression_loss)) / num_samples)

        return regression_loss

    def _evaluate(self, batch, record_embeddings=False):
        source, source_labels, target, target_labels = batch
        batch_size = source.shape[0]
        predictions = self(target)
        regression_loss = self.criterion_regression(predictions.squeeze(), target_labels)

        if record_embeddings:
            domain_labels = torch.cat([torch.zeros_like(source_labels),
                                       torch.ones_like(source_labels)])
            latent_code = self.encoder(torch.cat([source, target]))
            ruls = torch.cat([source_labels, target_labels])
            self.embedding_metric.update(latent_code, domain_labels, ruls)

        return regression_loss, batch_size


class AdaptiveAE(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 seq_len,
                 num_layers,
                 kernel_size,
                 base_filters,
                 latent_dim,
                 recon_trade_off,
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
        self.recon_trade_off = recon_trade_off
        self.domain_trade_off = domain_trade_off
        self.domain_disc_dim = domain_disc_dim
        self.num_disc_layers = num_disc_layers
        self.source_rul_cap = source_rul_cap
        self.optim_type = optim_type
        self.lr = lr
        self.record_embeddings = record_embeddings

        self.encoder = models.Encoder(self.in_channels, self.base_filters, self.kernel_size,
                                      self.num_layers, self.latent_dim, self.seq_len)
        self.decoder = models.Decoder(self.in_channels, self.base_filters, self.kernel_size,
                                      self.num_layers, self.latent_dim, self.seq_len)
        self.domain_disc = models.DomainDiscriminator(self.latent_dim, self.num_disc_layers, self.domain_disc_dim)
        self.regressor = models.Regressor(latent_dim)

        self.criterion_recon = nn.MSELoss()
        self.criterion_regression = RMSELoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

        self.embedding_metric = metrics.EmbeddingViz(20000, self.latent_dim)

        self.save_hyperparameters()

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [{'params': self.encoder.parameters()},
                        {'params': self.decoder.parameters()},
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
        loss, recon_loss, regression_loss, domain_loss = self._calc_loss(source, source_labels, target, domain_labels,
                                                                         cap=True)

        self.log('train/loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/regression_loss', regression_loss)
        self.log('train/domain_loss', domain_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, record_embeddings=self.record_embeddings)

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch)

    def validation_epoch_end(self, outputs):
        if self.record_embeddings:
            self.logger.experiment.add_figure('val/embeddings', self.embedding_metric.compute(), self.global_step)
            self.embedding_metric.reset()

        recon_loss, regression_loss, domain_loss = self._reduce_metrics(outputs)
        self.log(f'val/recon_loss', recon_loss)
        self.log(f'val/regression_loss', regression_loss)
        self.log(f'val/domain_loss', domain_loss)
        self.log('checkpoint_on', regression_loss, logger=False)

    def test_epoch_end(self, outputs):
        recon_loss, regression_loss, domain_loss = self._reduce_metrics(outputs)
        self.log(f'test/recon_loss', recon_loss)
        self.log(f'test/regression_loss', regression_loss)
        self.log(f'test/domain_loss', domain_loss)

    def _reduce_metrics(self, outputs):
        recon_loss, regression_loss, domain_loss, batch_size = zip(*outputs)
        num_samples = sum(batch_size)
        recon_loss = sum(b * loss for b, loss in zip(batch_size, recon_loss)) / num_samples
        regression_loss = torch.sqrt(sum(b * (loss ** 2) for b, loss in zip(batch_size, regression_loss)) / num_samples)
        domain_loss = sum(b * loss for b, loss in zip(batch_size, domain_loss)) / num_samples

        return recon_loss, regression_loss, domain_loss

    def _evaluate(self, batch, record_embeddings=False):
        source, source_labels, target, target_labels = batch
        batch_size = source.shape[0]
        domain_labels = torch.cat([torch.zeros_like(source_labels),
                                   torch.ones_like(source_labels)])
        # Predict on target and reconstruct/domain classify both
        _, recon_loss, regression_loss, domain_loss = self._calc_loss(target, target_labels, source, domain_labels)

        if record_embeddings:
            latent_code = self.encoder(torch.cat([source, target]))
            ruls = torch.cat([source_labels, target_labels])
            self.embedding_metric.update(latent_code, domain_labels, ruls)

        return recon_loss, regression_loss, domain_loss, batch_size

    def _calc_loss(self, regressor_features, regressor_labels, auxiliary_features, domain_labels, cap=False):
        common = torch.cat([regressor_features, auxiliary_features])
        domain_prediction, prediction, reconstruction = self._complete_forward(common)
        rul_mask = self._get_rul_mask(regressor_labels, cap)

        recon_loss = self.criterion_recon(common, reconstruction)
        regression_loss = self.criterion_regression(prediction.squeeze(), regressor_labels)
        domain_loss = self.criterion_domain(domain_prediction.squeeze()[rul_mask], domain_labels[rul_mask])
        loss = regression_loss + self.recon_trade_off * recon_loss + self.domain_trade_off * domain_loss

        return loss, recon_loss, regression_loss, domain_loss

    def _complete_forward(self, common):
        batch_size = common.shape[0] // 2

        latent_code = self.encoder(common)
        reconstruction = self.decoder(latent_code)
        regression_code, _ = torch.split(latent_code, batch_size)
        prediction = self.regressor(regression_code)
        domain_prediction = self.domain_disc(latent_code)

        return domain_prediction, prediction, reconstruction

    def _get_rul_mask(self, classifier_labels, cap):
        if cap:
            rul_mask = (classifier_labels > self.source_rul_cap)
        else:
            rul_mask = torch.ones_like(classifier_labels, dtype=torch.bool)

        return rul_mask.repeat(2)


class AdverserialAdaptiveAE(AdaptiveAE):
    def _build_domain_disc(self):
        sequence = [nn.Linear(self.latent_dim, self.domain_disc_dim),
                    nn.BatchNorm1d(self.domain_disc_dim),
                    nn.ReLU(True)]
        for i in range(self.num_disc_layers - 1):
            sequence.extend([nn.Linear(self.domain_disc_dim, self.domain_disc_dim),
                             nn.BatchNorm1d(self.domain_disc_dim),
                             nn.ReLU()])

        sequence.append(nn.Linear(self.domain_disc_dim, 1))

        return nn.Sequential(*sequence)

    def configure_optimizers(self):
        gen_parameters = list(self.encoder.parameters()) + \
                         list(self.decoder.parameters()) + \
                         list(self.regressor.parameters())
        gen_optim = torch.optim.SGD(gen_parameters, lr=self.lr, momentum=0.9, weight_decay=0.001)
        disc_optim = torch.optim.SGD(self.domain_disc.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.001)

        return [gen_optim, disc_optim], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        source, source_labels, target = batch

        if optimizer_idx == 0:
            result = self._generator_step(source, source_labels, target)
        else:
            result = self._discriminator_step(source, source_labels, target)

        return result

    def _generator_step(self, source, source_labels, target):
        loss, recon_loss, regression_loss, domain_loss = self._generator_loss(source, source_labels, target)
        result = pl.TrainResult(minimize=loss)
        result.log('train/recon_loss', recon_loss)
        result.log('train/regression_loss', regression_loss)
        result.log('train/domain_loss', domain_loss)

        return result

    def _generator_loss(self, source, source_labels, target):
        batch_size = source.shape[0]

        common = torch.cat([source, target])
        latent_code = self.encoder(common)
        reconstruction = self.decoder(latent_code)
        source_code, target_code = torch.split(latent_code, batch_size)
        prediction = self.regressor(source_code)
        domain_prediction_src = self.domain_disc(source_code)
        domain_prediction_trg = self.domain_disc(target_code)
        domain_labels_src = torch.ones_like(source_labels)
        domain_labels_trg = torch.zeros_like(source_labels)

        recon_loss = self.criterion_recon(common, reconstruction)
        regression_loss = self.criterion_regression(prediction.squeeze(), source_labels)

        domain_loss = 0.25 * self.criterion_domain(domain_prediction_src.squeeze(), domain_labels_src) + \
                        self.criterion_domain(domain_prediction_src.squeeze(), domain_labels_trg) + \
                        self.criterion_domain(domain_prediction_trg.squeeze(), domain_labels_trg) + \
                        self.criterion_domain(domain_prediction_trg.squeeze(), domain_labels_src)

        loss = regression_loss + self.recon_trade_off * recon_loss + self.domain_trade_off * domain_loss

        return loss, recon_loss, regression_loss, domain_loss

    def _discriminator_step(self, source, source_labels, target):
        loss = self._discriminator_loss(source, source_labels, target)
        result = pl.TrainResult(minimize=loss)
        result.log('train/disc_loss', loss)

        return result

    def _discriminator_loss(self, source, source_labels, target):
        batch_size = source.shape[0]
        common = torch.cat([source, target])
        latent_code = self.encoder(common)
        pred = self.domain_disc(latent_code.detach())
        source_pred, target_pred = torch.split(pred, batch_size)

        source_loss = self.criterion_domain(source_pred.squeeze(), torch.ones_like(source_labels))
        target_loss = self.criterion_domain(target_pred.squeeze(), torch.zeros_like(source_labels))

        loss = 0.5 * (source_loss + target_loss)

        return loss