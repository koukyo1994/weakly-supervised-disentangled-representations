import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List, Union

from disentanglement_lib.data.ground_truth.ground_truth_data import GroundTruthData
from disentanglement_lib.visualize import visualize_util


def create_paired_factors(z: np.ndarray,
                          ground_truth_data: GroundTruthData,
                          random_state: np.random.RandomState,
                          return_index=False,
                          k=1):
    """
    Create the pairs of factors for given z

    Parameters
    ----------
    z: numpy.ndarray
        factors of 2d array, (n_samples, n_factors)
    ground_truth_data: GroundTruthData
        data_generator class of disentanglement_lib
    random_state: numpy.random.RandomState
        random state
    return_index: bool
        whether to return the final index of factors that has been changed
    k: int
        number of factors that are changed between pairs
    """
    z_ = z.copy()
    if k == -1:
        # The number of factors that are changed between pairs is randomized
        k_observed = random_state.randint(1, ground_truth_data.num_factors)
    else:
        # The number of factors that are changed between pairs is fixed to k
        k_observed = k

    # Which factor is changed is randomized
    index_list = random_state.choice(
        z_.shape[1], random_state.choice([1, k_observed]), replace=False)
    idx = -1
    for index in index_list:
        z_[:, index] = np.random.choice(
            range(ground_truth_data.factors_num_values[index]))
        idx = index
    if return_index:
        return z_, idx
    return z_, k_observed


def visualize_weakly_supervised_dataset(dataset: GroundTruthData, save_path: Union[str, Path], num_images=5):
    """
    Visualize the dataset by creating pairs of images varying k

    Parameters
    ----------
    dataset_name: str
        String with name of dataset as defined in named_data.py
    save_path: Union[str, Path]
        String or Path object in which to create the animation
    num_images: int
        Integer with the number of images for each k
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    random_state = np.random.RandomState(0)

    num_factors = dataset.num_factors
    fig, axes = plt.subplots(nrows=num_factors, ncols=num_images + 1, figsize=(2 * (num_images + 2), 2 * num_factors))

    col_titles = ["Original"] + [f"Pair {i+1}" for i in range(num_images)]
    row_titles = [f"k={k}" for k in range(1, num_images + 1)]

    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row, rotation=90, size="large")

    for k in range(1, num_factors + 1):
        factor = dataset.sample_factors(1, random_state)
        observation = dataset.sample_observations_from_factors(factor, random_state)[0]
        if observation.shape[2] == 1:
            observation = observation[:, :, 0]

        axes[k - 1, 0].imshow(observation)
        axes[k - 1, 0].tick_params(labelbottom=False, bottom=False)
        axes[k - 1, 0].tick_params(labelleft=False, left=False)
        for i in range(num_images):
            new_factor, _ = create_paired_factors(factor.copy(), dataset, random_state, k=k)
            observation = dataset.sample_observations_from_factors(new_factor, random_state)[0]
            if observation.shape[2] == 1:
                observation = observation[:, :, 0]

            axes[k - 1, i + 1].imshow(observation)
            axes[k - 1, i + 1].tick_params(labelbottom=False, bottom=False)
            axes[k - 1, i + 1].tick_params(labelleft=False, left=False)

    fig.tight_layout()
    plt.savefig(save_path / "paired_image.png")


def visualize_weakly_supervised_dataset_with_animation(dataset: GroundTruthData,
                                                       save_path: Union[str, Path],
                                                       num_animations=10,
                                                       num_frames=20,
                                                       fps=10):
    """
    Visualize the dataset by creating an animation

    For each latent factor, outputs 16 images where only that latent factor is
    varied while all others are kept constant

    Parameters
    ----------
    dataset_name: str
        String with name of dataset as defined in named_data.py
    save_path: Union[str, Path]
        String or Path object in which to create the animation
    num_animations: int
        Integer with the number of distinct animations to create
    num_frames: int
        Integer with the number of frames in each animation
    fps: int
        Integer with frame rate for the animation
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    random_state = np.random.RandomState(0)

    images: List[List[np.ndarray]] = []
    for i in range(num_animations):
        images.append([])
        factor = dataset.sample_factors(1, random_state)

        images[i].append(
            np.squeeze(
                dataset.sample_observations_from_factors(factor, random_state),
                axis=0))

        for _ in range(num_frames):
            factor, _ = create_paired_factors(factor, dataset, random_state)
            images[i].append(np.squeeze(
                dataset.sample_observations_from_factors(factor, random_state),
                axis=0))

    visualize_util.save_animation(
        np.array(images), save_path / "animation.gif", fps)


if __name__ == "__main__":
    import os

    os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"

    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    dataset = get_named_ground_truth_data("dsprites_full")

    save_path = Path("./assets/weak_dataset/dsprites_full/")

    visualize_weakly_supervised_dataset_with_animation(dataset, save_path)
    visualize_weakly_supervised_dataset(dataset, save_path, num_images=10)
