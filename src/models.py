import torch.nn as nn

import layers


class Encoder(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, num_layers, latent_dim, seq_len):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.layers = self._build_encoder()

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

    def forward(self, inputs):
        return self.layers(inputs)


class Decoder(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, num_layers, latent_dim, seq_len):
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

    def forward(self, inputs):
        return self.layers(inputs)
