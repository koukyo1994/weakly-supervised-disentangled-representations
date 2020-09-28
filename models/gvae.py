import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class GroupVAE(nn.Module):
    def __init__(self, input_shape, beta=16.0, n_latents=10, aggregation="label"):
        super().__init__()
        assert aggregation in {"label", "argmax"}, \
            "`aggregation` must be either 'label' or 'argmax'"

        self.beta = beta
        self.aggregation = aggregation

        self.encoder = Encoder(input_shape=input_shape, n_latents=n_latents)
        self.decoder = Decoder(output_shape=input_shape, n_latents=n_latents)

    def encode(self, x):
        assert x.ndim == 5, \
            "Input of GroupVAE model should be 5 dimension (batch, pairs, channels, height, width)"
        assert x.size(1) == 2, \
            "Second dimension of the input should be a pair (2 elements)"
        assert x.size(2) in {1, 3}, \
            "Third dimension of the input should be either 1 or 3"

        pair_0 = x[:, 0, :, :, :]  # (batch, channels, height, width)
        pair_1 = x[:, 1, :, :, :]  # (batch, channels, height, width)

        mu_0, logvar_0 = self.encoder(pair_0)
        mu_1, logvar_1 = self.encoder(pair_1)
        return (mu_0, logvar_0), (mu_1, logvar_1)

    @staticmethod
    def kl_divergence_between_z(mu_0: torch.Tensor,
                                mu_1: torch.Tensor,
                                logvar_0: torch.Tensor,
                                logvar_1: torch.Tensor) -> torch.Tensor:
        var_0 = logvar_0.exp()
        var_1 = logvar_1.exp()
        return var_0 / var_1 + (mu_0 - mu_1).square() / var_1 - 1 + var_1 - var_0

    @staticmethod
    def aggregate_labels(mu: torch.Tensor,
                         logvar: torch.Tensor,
                         new_mu: torch.Tensor,
                         new_logvar: torch.Tensor,
                         labels: torch.Tensor,
                         **kwargs):
        """
        Use labels to aggregate.

        Labels contains a one-hot encoding with 1 at the share factor positions.
        We enforce which dimension of the latent code learn which factor (1 dimension learns 1 factor)
        and we enforce that each factor of variation is encoded in a single dimension.

        Parameters
        ----------
        mu: torch.Tensor
            Mean of the encoder distribution for the original image
            shape: (batch_size, n_latents)
        logvar: torch.Tensor
            Logvar of the encoder distribution for the original image
            shape: (batch_size, n_latents)
        new_mu: torch.Tensor
            Average mean of the encoder distribution of the pair of images
            shape: (batch_size, n_latents)
        new_logvar: torch.Tensor
            Average logvar of the encoder distribution of the pair of images
            shape: (batch_size, n_latents)
        labels: torch.Tensor
            One-hot-encoding with the position(s) of dimension that should not be shared
            shape: (batch_size, n_latents)

        Returns: Tuple[torch.Tensor, torch.Tensor]
            Mean and logvar for the new observation
        """
        mu_averaged = torch.where(labels.byte(), mu, new_mu)
        logvar_averaged = torch.where(labels.byte(), logvar, new_logvar)
        return mu_averaged, logvar_averaged
