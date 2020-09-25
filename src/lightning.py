import torch
import torch.nn as nn
import pytorch_lightning as pl


class DeFlatten(nn.Module):
    def __init__(self, seq_len, num_channels):
        super().__init__()

        self.seq_len = seq_len
        self.num_channels = num_channels

    def forward(self, inputs):
        return inputs.view(-1, self.num_channels, self.seq_len)


class AdaptiveAE(pl.LightningModule):
    def __init__(self, in_channels, seq_len, num_layers, kernel_size, base_filters, latent_dim):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        # self.domain_disc = self._build_domain_disc()
        # self.classifier = self._build_classifier()

        self.criterion_recon = nn.MSELoss()

        self.save_hyperparameters()

    @property
    def example_input_array(self):
        return torch.randn(16, self.in_channels, self.seq_len), torch.zeros(16)

    def _build_encoder(self):
        layers = [nn.Conv1d(self.in_channels, self.base_filters, self.kernel_size),
                  nn.BatchNorm1d(self.base_filters),
                  nn.ReLU(True)]
        for i in range(1, self.num_layers):
            layers.extend([nn.Conv1d(i * self.base_filters, (i + 1) * self.base_filters, self.kernel_size),
                           nn.BatchNorm1d((i + 1) * self.base_filters),
                           nn.ReLU(True)])

        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        flat_dim = (self.seq_len - cut_off) * self.num_layers * self.base_filters
        layers.extend([nn.Flatten(),
                       nn.Linear(flat_dim, self.latent_dim),
                       nn.BatchNorm1d(self.latent_dim),
                       nn.ReLU(True)])

        return nn.Sequential(*layers)

    def _build_decoder(self):
        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        max_filters = self.num_layers * self.base_filters
        reduced_seq_len = self.seq_len - cut_off
        flat_dim = reduced_seq_len * max_filters

        layers = [nn.Linear(self.latent_dim, flat_dim),
                  nn.BatchNorm1d(flat_dim),
                  nn.ReLU(True),
                  DeFlatten(reduced_seq_len, max_filters)]
        for i in range(self.num_layers - 1, 0, -1):
            layers.extend([nn.ConvTranspose1d((i + 1) * self.base_filters, i * self.base_filters, self.kernel_size),
                           nn.BatchNorm1d(i * self.base_filters),
                           nn.ReLU(True)])

        layers.extend([nn.ConvTranspose1d(self.base_filters, self.in_channels, self.kernel_size),
                       nn.Tanh()])

        return nn.Sequential(*layers)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs, targets=None):
        latent_code = self.encoder(inputs)
        outputs = self.decoder(latent_code)

        return outputs, latent_code

    def training_step(self, batch, batch_idx):
        features, targets = batch
        reconstruction, latent_code = self(features)
        loss = self.criterion_recon(features, reconstruction)

        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        reconstruction, latent_code = self(features)
        loss = self.criterion_recon(features, reconstruction)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val/recon_loss', loss)

        return result

    def test_step(self, batch, batch_idx):
        features, targets = batch
        reconstruction, latent_code = self(features)
        loss = self.criterion_recon(features, reconstruction)
        result = pl.EvalResult(loss)
        result.log('test/recon_loss', loss)

        return result
