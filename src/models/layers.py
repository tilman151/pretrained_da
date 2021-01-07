import torch
from torch import nn as nn


class DeFlatten(nn.Module):
    def __init__(self, seq_len, num_channels):
        super().__init__()

        self.seq_len = seq_len
        self.num_channels = num_channels

    def forward(self, inputs):
        return inputs.view(-1, self.num_channels, self.seq_len)


class _GradientReverse(torch.autograd.Function):
    """Gradient reversal forward and backward definitions."""

    @staticmethod
    def forward(ctx, inputs, **kwargs):
        """Forward pass as identity mapping."""
        return inputs

    @staticmethod
    def backward(ctx, grad):
        """Backward pass as negative of gradient."""
        return -grad


def gradient_reversal(x):
    """Perform gradient reversal on input."""
    return _GradientReverse.apply(x)


class GradientReversalLayer(nn.Module):
    """Module for gradient reversal."""

    def forward(self, inputs):
        """Perform forward pass of gradient reversal."""
        return gradient_reversal(inputs)
