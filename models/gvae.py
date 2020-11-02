import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class GroupVAE(nn.Module):
    def __init__(self, input_shape, beta=16.0, n_latents=10, aggregation="label", label_mode=None):
        super().__init__()
        assert aggregation in {"label", "argmax"}, \
            "`aggregation` must be either 'label' or 'argmax'"
        if aggregation == "label":
            assert label_mode is not None, \
                "`label_mode` must be given if `aggregation` is 'label'"
            assert label_mode in {"single", "multi"}, \
                "`label_mode` must be either 'single' or 'multi'"

        self.input_shape = input_shape
        self.beta = beta
        self.n_latents = n_latents
        self.aggregation = aggregation
        self.label_mode = label_mode

        self.encoder = Encoder(input_shape=input_shape, n_latents=n_latents)
        self.decoder = Decoder(output_shape=input_shape, n_latents=n_latents)

    def forward(self, x, label, sigmoid=True):
        mu_0, logvar_0, mu_1, logvar_1 = self.encode(x)
        new_mu_0, new_logvar_0, new_mu_1, new_logvar_1 = self.aggregate(
            mu_0, logvar_0, mu_1, logvar_1, label)
        z_0 = self.reparameterize(new_mu_0, new_logvar_0)
        z_1 = self.reparameterize(new_mu_1, new_logvar_1)
        reconstructed_0 = self.decoder(z_0)
        reconstructed_1 = self.decoder(z_1)
        return reconstructed_0, reconstructed_1, new_mu_0, new_logvar_0, new_mu_1, new_logvar_1

    def reconstruct(self, x: torch.Tensor, label: torch.Tensor):
        mu_0, logvar_0, mu_1, logvar_1 = self.encode(x)
        new_mu_0, new_logvar_0, new_mu_1, new_logvar_1 = self.aggregate(
            mu_0, logvar_0, mu_1, logvar_1, label)
        z_0 = self.reparameterize(new_mu_0, new_logvar_0)
        z_1 = self.reparameterize(new_mu_1, new_logvar_1)
        reconstructed_0 = self.decoder(z_0)
        reconstructed_1 = self.decoder(z_1)
        return reconstructed_0, reconstructed_1

    def loss_fn(self,
                x: torch.Tensor,
                reconstructed_0: torch.Tensor,
                reconstructed_1: torch.Tensor,
                mu_0: torch.Tensor,
                logvar_0: torch.Tensor,
                mu_1: torch.Tensor,
                logvar_1: torch.Tensor):
        assert x.ndim == 5, \
            "Input of GroupVAE model should be 5 dimension (batch, pairs, channels, height, width)"
        assert x.size(1) == 2, \
            "Second dimension of the input should be a pair (2 elements)"
        assert x.size(2) in {1, 3}, \
            "Third dimension of the input should be either 1 or 3"

        recons_error_0 = self.bernoulli_recons_error(reconstructed_0, x[:, 0, :, :, :])
        recons_error_1 = self.bernoulli_recons_error(reconstructed_1, x[:, 1, :, :, :])
        recons_error = 0.5 * recons_error_0 + 0.5 * recons_error_1

        kld_0 = self.gaussian_kl(mu_0, logvar_0)
        kld_1 = self.gaussian_kl(mu_1, logvar_1)
        kld = 0.5 * kld_0 + 0.5 * kld_1

        loss = recons_error + self.beta * kld
        return loss, recons_error, kld

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
        return mu_0, logvar_0, mu_1, logvar_1

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        else:
            return mu

    def aggregate(self,
                  mu_0: torch.Tensor,
                  logvar_0: torch.Tensor,
                  mu_1: torch.Tensor,
                  logvar_1: torch.Tensor,
                  label: torch.Tensor):
        var_0 = logvar_0.exp()
        var_1 = logvar_1.exp()

        new_mu = 0.5 * mu_0 + 0.5 * mu_1
        new_logvar = (0.5 * var_0 + 0.5 * var_1).log()

        if self.aggregation == "label":
            one_hot_label = self.make_one_hot_label(label)
            mu_sample_0, logvar_sample_0 = self.aggregate_labels(
                mu_0,
                logvar_0,
                new_mu=new_mu,
                new_logvar=new_logvar,
                labels=one_hot_label)
            mu_sample_1, logvar_sample_1 = self.aggregate_labels(
                mu_1,
                logvar_1,
                new_mu=new_mu,
                new_logvar=new_logvar,
                labels=one_hot_label)
        else:
            kl_per_point = self.kl_divergence_between_z(mu_0, logvar_0, mu_1, logvar_1)
            mu_sample_0, logvar_sample_0 = self.aggregate_argmax(
                mu_0,
                logvar_0,
                new_mu=new_mu,
                new_logvar=new_logvar,
                kl_per_point=kl_per_point)
            mu_sample_1, logvar_sample_1 = self.aggregate_argmax(
                mu_1,
                logvar_1,
                new_mu=new_mu,
                new_logvar=new_logvar,
                kl_per_point=kl_per_point)
        return mu_sample_0, logvar_sample_0, mu_sample_1, logvar_sample_1

    def make_one_hot_label(self, label: torch.Tensor):
        assert self.label_mode is not None, "`label_mode` must be given"
        if self.label_mode == "single":
            assert label.ndim == 1, \
                "`label` must be one dimensional if `label_mode` is 'single'"
            return F.one_hot(label, num_classes=self.n_latents)
        elif self.label_mode == "multi":
            assert label.ndim == 3, \
                "`label` must be three dimensional if `label_mode` is 'multi'"
            new_label0 = torch.zeros(size=(label.size(0), self.n_latents),
                                     device=label.device,
                                     dtype=label.dtype)
            new_label1 = torch.zeros(size=(label.size(0), self.n_latents),
                                     device=label.device,
                                     dtype=label.dtype)
            new_label0[:, :label.size(2)] = label[:, 0, :]
            new_label1[:, :label.size(2)] = label[:, 1, :]
            return new_label0 != new_label1
        else:
            raise NotImplementedError

    @staticmethod
    def kl_divergence_between_z(mu_0: torch.Tensor,
                                mu_1: torch.Tensor,
                                logvar_0: torch.Tensor,
                                logvar_1: torch.Tensor) -> torch.Tensor:
        var_0 = logvar_0.exp()
        var_1 = logvar_1.exp()
        return var_0 / var_1 + (mu_0 - mu_1).pow(2) / var_1 - 1 + logvar_1 - logvar_0

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

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and logvar for the new observation
        """
        mu_averaged = torch.where(labels.byte(), mu, new_mu)
        logvar_averaged = torch.where(labels.byte(), logvar, new_logvar)
        return mu_averaged, logvar_averaged

    @staticmethod
    def aggregate_argmax(mu: torch.Tensor,
                         logvar: torch.Tensor,
                         new_mu: torch.Tensor,
                         new_logvar: torch.Tensor,
                         kl_per_point: torch.Tensor,
                         **kwargs):
        """
        Argmax aggregation with adaptive k.

        The bottom k dimensions in terms of distance are not averaged.
        K is estimated adaptively by binning the distance into two bins of equal width.

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
        kl_per_point: torch.Tensor
            Distance between two encoder distributions

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and logvar for the new observation
        """
        mask_tensor = torch.zeros_like(kl_per_point, dtype=torch.int)
        for i, kl in enumerate(kl_per_point):
            histogram = torch.histc(kl, bins=2, min=kl.min().detach(), max=kl.max().detach()).long()  # type: ignore
            top_indices = torch.topk(kl, k=histogram[1]).indices  # type: ignore
            mask_tensor[i, top_indices] = 1
        mu_averaged = torch.where(mask_tensor == 1, mu, new_mu)
        logvar_averaged = torch.where(mask_tensor == 1, logvar, new_logvar)
        return mu_averaged, logvar_averaged

    @staticmethod
    def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor):
        return (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)).sum()

    @staticmethod
    def bernoulli_recons_error(reconstructed: torch.Tensor, original: torch.Tensor):
        return F.binary_cross_entropy(reconstructed, original, reduction="none").mean(dim=0).sum()
