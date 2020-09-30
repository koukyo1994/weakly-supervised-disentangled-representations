import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from typing import Union


def save_reconstructed_images(save_path: Path,
                              original_pairs: torch.Tensor,
                              reconstructed_0: torch.Tensor,
                              reconstructed_1: torch.Tensor):
    assert reconstructed_0.size() == reconstructed_1.size(), \
        "Both images should be exactly the same size"
    original_0 = original_pairs[:, 0, :, :, :]
    original_1 = original_pairs[:, 1, :, :, :]

    B, C, H, W = reconstructed_0.size()
    original_image_0 = np.moveaxis(original_0.numpy(), 1, 3)
    original_image_1 = np.moveaxis(original_1.numpy(), 1, 3)
    reconstructed_image_0 = np.moveaxis(reconstructed_0.numpy(), 1, 3)
    reconstructed_image_1 = np.moveaxis(reconstructed_1.numpy(), 1, 3)

    fig, axes = plt.subplots(ncols=4, nrows=B, figsize=(8, 2 * B))

    col_titles = [
        "Original pair 0", "Original pair 1",
        "Reconstructed pair 0", "Reconstructed pair 1"
    ]
    row_titles = [f"Pair {i}" for i in range(B)]

    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row, rotation=90, size="large")

    for i in range(B):
        orig_0 = original_image_0[i]
        orig_1 = original_image_1[i]
        recon_0 = reconstructed_image_0[i]
        recon_1 = reconstructed_image_1[i]

        if orig_0.shape[2] == 1:
            orig_0 = orig_0[:, :, 0]
            orig_1 = orig_1[:, :, 0]
            recon_0 = recon_0[:, :, 0]
            recon_1 = recon_1[:, :, 0]

        for j, img in zip(range(4), [orig_0, orig_1, recon_0, recon_1]):
            axes[i, j].imshow(img)
            axes[i, j].tick_params(labelbottom=False, bottom=False)
            axes[i, j].tick_params(labelleft=False, left=False)

    fig.tight_layout()
    plt.savefig(save_path)


def latent_traversal(model,
                     x: Union[np.ndarray, torch.Tensor],
                     device: torch.device,
                     save_path: Union[str, Path],
                     n_imgs: int,
                     z_min=-1.5,
                     z_max=1.5):
    """
    Make latent traversal gif image and save it at the specified path.
    Parameters
    ----------
    model: torch.nn.Module
        trained VAE model
    x: Union[numpy.ndarray, torch.Tensor]
        source image for latent traversal
    device: torch.device
        torch.device('cpu') or torch.device('cuda')
    save_path: Union[str, pathlib.Path]
        path to save the gif
    n_imgs: int
        number of images for making gif
    z_min: float
        lower bound for latent dim's moving range
    z_max: float
        upper bound for latent dim's moving range
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    if not save_path.resolve().parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Make array either [1, 1, w, h] or [1, 3, w, h] tensor
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = torch.from_numpy(x.reshape(1, 1, x.shape[0], x.shape[1])).to(device)
        elif x.ndim == 3:
            if x.shape[2] == 1 or x.shape[2] == 3:
                x = torch.from_numpy(np.moveaxis(x, 2, 0)).unsqueeze(0).to(device)
            elif x.shape[0] == 1 or x.shape[0] == 3:
                x = torch.from_numpy(x).unsqueeze(0).to(device)
            else:
                raise NotImplementedError
        elif x.ndim == 4:
            if x.shape[0] != 1:
                raise NotImplementedError
            if x.shape[1] not in {1, 3}:
                raise NotImplementedError
            x = torch.from_numpy(x).to(device)
    else:
        if x.ndim == 2:
            x = x.view(1, 1, x.size(0), x.size(1)).to(device)
        elif x.ndim == 3:
            if x.size(2) == 1 or x.size(2) == 3:
                x = x.permute(2, 0, 1).unsqueeze(0).to(device)
            elif x.size(0) == 1 or x.size(0) == 3:
                x = x.unsqueeze(0).to(device)
            else:
                raise NotImplementedError
        elif x.ndim == 4:
            if x.size(0) != 1 or x.size(1) not in {1, 3}:
                raise NotImplementedError
            x = x.to(device)

    model.eval()
    model.to(device)
    with torch.no_grad():
        mu, logvar = model.encoder(x)
        latents = model.reparameterize(mu, logvar)

    latent_dims = latents.size(1)
    z = latents.cpu().numpy()[0]
    gif = []
    for z_ij in np.linspace(z_min, z_max, n_imgs):
        latent_traversals = []
        latent_traversals.append(z)
        for i in range(latent_dims):
            z_ = z.copy()
            z_[i] = z_ij
            latent_traversals.append(z_)
        z_traversed = torch.from_numpy(
            np.array(latent_traversals).astype(np.float32)).to(device)
        with torch.no_grad():
            reconstructed = model.decoder(z_traversed).detach().cpu().numpy()

        gif.append(np.concatenate([x.cpu().numpy(), reconstructed]))

    for i, imgs in enumerate(gif):
        fig, ax = plt.subplots(1, latent_dims + 2,
                               figsize=(latent_dims + 2, 1), dpi=100)
        for ai, xi in zip(ax.flatten(), imgs):
            assert xi.shape[0] in {1, 3}, "channel num should be 1 or 3"
            if xi.shape[0] == 1:
                xi = xi.reshape(xi.shape[1], xi.shape[2])
            elif xi.shape[0] == 3:
                xi = np.moveaxis(xi, 0, 2)
            ai.tick_params(labelbottom=False, bottom=False)  # remove x-axis ticks
            ai.tick_params(labelleft=False, left=False)  # remove y-axis ticks
            ai.imshow(xi)
        fig.savefig(save_path.resolve().parent / str(i), bbox_inches="tight", pad_inches=0)
        plt.close()

    images = []
    for i in range(n_imgs):
        images.append(Image.open(save_path.resolve().parent / (str(i) + ".png")))

    images[0].save(
        save_path, save_all=True, append_images=images[1:], optimize=False, duration=50, loop=20)

    for i in range(n_imgs):
        os.remove(save_path.resolve().parent / (str(i) + ".png"))
