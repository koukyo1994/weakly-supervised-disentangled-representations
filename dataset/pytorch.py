import numpy as np
import torch.utils.data as torchdata

from disentanglement_lib.data.ground_truth.ground_truth_data import GroundTruthData

from dataset.utils import create_paired_factors


class WeaklySupervisedDataset(torchdata.Dataset):
    def __init__(self, dataset: GroundTruthData, seed=0, iterator_len=20000, k=1, return_index=False):
        self.iterator_len = iterator_len
        self.dataset = dataset
        self.random_state = np.random.RandomState(seed)
        self.k = k
        self.return_index = return_index

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, index: int):
        factor = self.dataset.sample_factors(1, self.random_state)
        observation = self.dataset.sample_observations_from_factors(
            factor, self.random_state)
        observation = np.moveaxis(observation, 3, 1)

        paired_factor, idx = create_paired_factors(factor,
                                                   ground_truth_data=self.dataset,
                                                   random_state=self.random_state,
                                                   return_index=self.return_index,
                                                   k=self.k)
        paired_observation = self.dataset.sample_observations_from_factors(
            paired_factor, self.random_state)
        paired_observation = np.moveaxis(paired_observation, 3, 1)

        observation_pair = np.concatenate([
            observation, paired_observation
        ], axis=0)
        factor_pair = np.concatenate([
            factor, paired_factor
        ], axis=0)
        if not self.return_index:
            return observation_pair, factor_pair
        else:
            return observation_pair, idx
