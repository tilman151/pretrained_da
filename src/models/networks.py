import torch
import torch.nn as nn

from models import layers


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        num_layers,
        latent_dim,
        seq_len,
        dropout,
        norm_outputs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.norm_outputs = norm_outputs

        self.layers = self._build_encoder()

    def _build_encoder(self):
        sequence = [
            nn.Conv1d(self.in_channels, self.base_filters, self.kernel_size),
            nn.BatchNorm1d(self.base_filters),
            nn.ReLU(True),
        ]
        for i in range(1, self.num_layers):
            in_filters = min(i * self.base_filters, 64)
            out_filters = min((i + 1) * self.base_filters, 64)
            sequence.extend(
                [
                    nn.Conv1d(in_filters, out_filters, self.kernel_size),
                    nn.BatchNorm1d(out_filters),
                    nn.ReLU(True),
                    nn.Dropout2d(p=self.dropout),
                ]
            )

        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        flat_dim = (self.seq_len - cut_off) * min(
            self.num_layers * self.base_filters, 64
        )
        sequence.extend([nn.Flatten(), nn.Linear(flat_dim, self.latent_dim)])

        return nn.Sequential(*sequence)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        if self.norm_outputs:
            outputs = outputs / torch.norm(outputs, dim=1, keepdim=True)

        return outputs


class Decoder(nn.Module):
    def __init__(
        self, in_channels, base_filters, kernel_size, num_layers, latent_dim, seq_len
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.layers = self._build_decoder()

    def _build_decoder(self):
        cut_off = self.num_layers * (self.kernel_size - (self.kernel_size % 2))
        max_filters = self.num_layers * self.base_filters
        reduced_seq_len = self.seq_len - cut_off
        flat_dim = reduced_seq_len * max_filters

        sequence = [
            nn.Linear(self.latent_dim, flat_dim),
            nn.BatchNorm1d(flat_dim),
            nn.ReLU(True),
            layers.DeFlatten(reduced_seq_len, max_filters),
        ]
        for i in range(self.num_layers - 1, 0, -1):
            sequence.extend(
                [
                    nn.ConvTranspose1d(
                        (i + 1) * self.base_filters,
                        i * self.base_filters,
                        self.kernel_size,
                    ),
                    nn.BatchNorm1d(i * self.base_filters),
                    nn.ReLU(True),
                ]
            )

        sequence.extend(
            [
                nn.ConvTranspose1d(
                    self.base_filters, self.in_channels, self.kernel_size
                ),
                nn.Tanh(),
            ]
        )

        return nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.layers(inputs)


class Regressor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.layers = self._build_regressor()

    def _build_regressor(self):
        classifier = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_dim, 1),
        )

        return classifier

    def forward(self, inputs):
        return self.layers(inputs)


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = self._build_domain_disc()

    def _build_domain_disc(self):
        sequence = [
            layers.GradientReversalLayer(),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
        ]

        for i in range(self.num_layers - 1):
            sequence.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                ]
            )

        sequence.append(nn.Linear(self.hidden_dim, 1))

        return nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.layers(inputs)
