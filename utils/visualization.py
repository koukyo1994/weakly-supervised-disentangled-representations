import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path


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
