import os
import sys
import pytest

import torch.utils.data as torchdata

sys.path.append(os.getcwd())
os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"

from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data  # noqa

from dataset.pytorch import WeaklySupervisedDataset  # noqa
from models import GroupVAE  # noqa


@pytest.fixture
def data_generator():
    generator = get_named_ground_truth_data("dsprites_full")
    return generator


def test_gvae_aggregate_label(data_generator):
    dataset = WeaklySupervisedDataset(data_generator, seed=0, k=1, return_index=False)
    loader = torchdata.DataLoader(dataset, batch_size=10)
    model = GroupVAE(input_shape=[1, 64, 64], beta=16.0, aggregation="label", label_mode="multi")

    x, label = next(iter(loader))
    reconstructed_0, reconstructed_1, mu_0, logvar_0, mu_1, logvar_1 = model(x, label)

    assert reconstructed_0.size() == reconstructed_1.size()
    assert reconstructed_0.size(0) == x.size(0)
    assert reconstructed_0.size(1) == x.size(2)
    assert reconstructed_0.size(2) == x.size(3)
    assert reconstructed_0.size(3) == x.size(4)

    assert mu_0.size() == mu_1.size()
    assert mu_0.size() == logvar_0.size()
    assert mu_0.size() == logvar_1.size()

    assert mu_0.size(0) == reconstructed_0.size(0)

    loss, recons_error, kld = model.loss_fn(
        x, reconstructed_0, reconstructed_1,
        mu_0, logvar_0, mu_1, logvar_1)

    assert loss == recons_error + 16.0 * kld
