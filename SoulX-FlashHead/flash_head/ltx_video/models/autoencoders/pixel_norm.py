import torch
from torch import nn


# Normalize pixel values by their root-mean-square across a dimension.
class PixelNorm(nn.Module):
    # Initialize pixel normalization with the target dimension and epsilon.
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    # Normalize the input tensor by its RMS across the specified dimension.
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)
