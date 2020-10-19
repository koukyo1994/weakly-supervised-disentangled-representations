import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class BetaVAE(nn.Module):
    def __init__(self, input_shape, beta=16.0, n_latents=10):
        super().__init__()

        self.input_shape = input_shape
        self.beta = beta
        self.n_latents = n_latents

        self.encoder = Encoder(input_shape=input_shape, n_latents=n_latents)
        self.decoder = Decoder(input_shape=input_shape, n_latents=n_latents)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.rand_like(std)
        else:
            return mu

    def forward(self, x, sigmoid=True):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def reconstruct(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed

    @staticmethod
    def bernoulli_recons_error(recons: torch.Tensor, original: torch.Tensor):
        return F.binary_cross_entropy(recons, original, reduction="none").mean(dim=0).sum()

    @staticmethod
    def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor):
        return (0.5 * mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0).sum()

    def loss_fn(self, x: torch.Tensor, reconstructed: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recons_error = self.bernoulli_recons_error(reconstructed, x)
        kld = self.gaussian_kl(mu, logvar)

        loss = recons_error + self.beta * kld
        return loss, recons_error, kld
