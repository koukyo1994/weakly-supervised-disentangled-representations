import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .utils import anneal


class BetaCVAE(nn.Module):
    def __init__(self, input_shape, n_latents: int, gamma: float, C_max: float, anneal_steps: int):
        super().__init__()

        self.input_shape = input_shape
        self.n_latents = n_latents
        self.gamma = gamma
        self.C_max = C_max
        self.anneal_steps = anneal_steps

        self.encoder = Encoder(input_shape=input_shape, n_latents=n_latents)
        self.decoder = Decoder(output_shape=input_shape, n_latents=n_latents)
        self.num_train_steps = 0

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        else:
            return mu

    def forward(self, x, *args, **kwargs):
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
        if self.training:
            self.num_train_steps += 1

        recons_error = self.bernoulli_recons_error(reconstructed, x)
        kld = self.gaussian_kl(mu, logvar)

        c = anneal(0.0, self.C_max, self.num_train_steps, self.anneal_steps) if self.training else 1
        loss = recons_error + self.gamma * torch.abs(kld - c)
        return loss, recons_error, kld
