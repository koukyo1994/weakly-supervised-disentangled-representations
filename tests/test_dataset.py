import os
import sys

import matplotlib.pyplot as plt
import torch.utils.data as torchdata

from pathlib import Path

sys.path.append(os.getcwd())
os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"

from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data  # noqa

from dataset.pytorch import WeaklySupervisedDataset  # noqa


def test_pytorch_dataset():
    save_path = Path("./assets/weak_dataset/dsprites_full/")

    dataset = get_named_ground_truth_data("dsprites_full")
    torch_dataset = WeaklySupervisedDataset(dataset, seed=0, k=2)
    loader = torchdata.DataLoader(torch_dataset, batch_size=10)

    observations, factors = next(iter(loader))

    observations = observations.numpy()
    factors = factors.numpy()

    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(5, 20))

    col_titles = ["Observation0", "Observation1"]
    row_titles = [f"Pair{i}" for i in range(10)]

    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row, rotation=90, size="large")

    for i, pair in enumerate(observations):
        observation0 = pair[0]
        observation1 = pair[1]

        if observation0.shape[0] == 1:
            observation0 = observation0[0, :, :]
            observation1 = observation1[0, :, :]

        axes[i, 0].imshow(observation0)
        axes[i, 0].set_xlabel(f"{factors[i, 0]}")
        axes[i, 0].tick_params(labelbottom=False, bottom=False)
        axes[i, 0].tick_params(labelleft=False, left=False)
        axes[i, 1].imshow(observation1)
        axes[i, 1].set_xlabel(f"{factors[i, 1]}")
        axes[i, 1].tick_params(labelbottom=False, bottom=False)
        axes[i, 1].tick_params(labelleft=False, left=False)

    fig.tight_layout()
    plt.savefig(save_path / "paired_image_from_torch_dataset.png")
