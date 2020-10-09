import pytorch_lightning as pl
import torch
import torch.nn as nn

import layers


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


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
                 lr):
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
        self.lr = lr

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.domain_disc = self._build_domain_disc()
        self.classifier = self._build_classifier()

        self.criterion_recon = nn.MSELoss()
        self.criterion_regression = RMSELoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(32, self.in_channels, self.seq_len)

        return common

    def _build_encoder(self):
        sequence = [nn.Conv1d(self.in_channels, self.base_filters, self.kernel_size),
                    nn.BatchNorm1d(self.base_filters),
                    nn.ReLU(True)]
        for i in range(1, self.num_layers):
            sequence.extend([nn.Conv1d(i * self.base_filters, (i + 1) * self.base_filters, self.kernel_size),
                             nn.BatchNorm1d((i + 1) * self.base_filters),
                             nn.ReLU(True)])

        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        flat_dim = (self.seq_len - cut_off) * self.num_layers * self.base_filters
        sequence.extend([nn.Flatten(),
                         nn.Linear(flat_dim, self.latent_dim)])

        return nn.Sequential(*sequence)

    def _build_decoder(self):
        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        max_filters = self.num_layers * self.base_filters
        reduced_seq_len = self.seq_len - cut_off
        flat_dim = reduced_seq_len * max_filters

        sequence = [nn.Linear(self.latent_dim, flat_dim),
                    nn.BatchNorm1d(flat_dim),
                    nn.ReLU(True),
                    layers.DeFlatten(reduced_seq_len, max_filters)]
        for i in range(self.num_layers - 1, 0, -1):
            sequence.extend([nn.ConvTranspose1d((i + 1) * self.base_filters, i * self.base_filters, self.kernel_size),
                             nn.BatchNorm1d(i * self.base_filters),
                             nn.ReLU(True)])

        sequence.extend([nn.ConvTranspose1d(self.base_filters, self.in_channels, self.kernel_size),
                         nn.Tanh()])

        return nn.Sequential(*sequence)

    def _build_classifier(self):
        classifier = nn.Sequential(nn.BatchNorm1d(self.latent_dim),
                                   nn.ReLU(True),
                                   nn.Linear(self.latent_dim, 1))

        return classifier

    def _build_domain_disc(self):
        sequence = [layers.GradientReversalLayer(),
                    nn.Linear(self.latent_dim, self.domain_disc_dim),
                    nn.BatchNorm1d(self.domain_disc_dim),
                    nn.ReLU(True)]
        for i in range(self.num_disc_layers - 1):
            sequence.extend([nn.Linear(self.domain_disc_dim, self.domain_disc_dim),
                             nn.BatchNorm1d(self.domain_disc_dim),
                             nn.ReLU()])

        sequence.append(nn.Linear(self.domain_disc_dim, 1))

        return nn.Sequential(*sequence)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

    def forward(self, common):
        batch_size = common.shape[0] // 2

        latent_code = self.encoder(common)
        reconstruction = self.decoder(latent_code)
        classification_code, _ = torch.split(latent_code, batch_size)
        prediction = self.classifier(classification_code)
        domain_prediction = self.domain_disc(latent_code)

        return reconstruction, prediction, domain_prediction

    def training_step(self, batch, batch_idx):
        source, source_labels, target = batch
        domain_labels = torch.cat([torch.ones_like(source_labels),
                                   torch.zeros_like(source_labels)])
        loss, recon_loss, regression_loss, domain_loss = self._calc_loss(source, source_labels, target, domain_labels)

        result = pl.TrainResult(minimize=loss)
        result.log('train/loss', loss)
        result.log('train/recon_loss', recon_loss)
        result.log('train/regression_loss', regression_loss)
        result.log('train/domain_loss', domain_loss)

        return result

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, 'test')

    def _evaluate(self, batch, prefix):
        source, source_labels, target, target_labels = batch
        domain_labels = torch.cat([torch.zeros_like(source_labels),
                                   torch.ones_like(source_labels)])
        loss, recon_loss, regression_loss, domain_loss = self._calc_loss(target, target_labels, source, domain_labels)
        result = pl.EvalResult(checkpoint_on=regression_loss)
        result.log(f'{prefix}/loss', loss)
        result.log(f'{prefix}/recon_loss', recon_loss)
        result.log(f'{prefix}/regression_loss', regression_loss)
        result.log(f'{prefix}/domain_loss', domain_loss)

        return result

    def _calc_loss(self, classifier_features, classifier_labels, auxiliary_features, domain_labels):
        common = torch.cat([classifier_features, auxiliary_features])
        reconstruction, prediction, domain_prediction = self(common)

        recon_loss = self.criterion_recon(common, reconstruction)
        regression_loss = self.criterion_regression(prediction.squeeze(), classifier_labels)
        domain_loss = self.criterion_domain(domain_prediction.squeeze(), domain_labels)
        loss = regression_loss + self.recon_trade_off * recon_loss + self.domain_trade_off * domain_loss

        return loss, recon_loss, regression_loss, domain_loss


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
                         list(self.classifier.parameters())
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
        prediction = self.classifier(source_code)
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