import numpy as np
import torch.utils.data as torchdata

from disentanglement_lib.data.ground_truth.ground_truth_data import GroundTruthData

from dataset.utils import create_paired_factors


class WeaklySupervisedDataset(torchdata.Dataset):
    def __init__(self, dataset: GroundTruthData, seed=0, iterator_len=20000, k=1):
        self.iterator_len = iterator_len
        self.dataset = dataset
        self.random_state = np.random.RandomState(seed)
        self.k = k

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, index: int):
        factor = self.dataset.sample_factors(1, self.random_state)
        observation = self.dataset.sample_observations_from_factors(
            factor, self.random_state)

        paired_factor, _ = create_paired_factors(factor,
                                                 ground_truth_data=self.dataset,
                                                 random_state=self.random_state,
                                                 return_index=False,
                                                 k=self.k)
        paired_observation = self.dataset.sample_observations_from_factors(
            paired_factor, self.random_state)

        observation_pair = np.concatenate([
            observation, paired_observation
        ], axis=0)
        factor_pair = np.concatenate([
            factor, paired_factor
        ], axis=0)

        return observation_pair, factor_pair
