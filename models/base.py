import torch

from typing import Callable


def make_weak_loss(z1: torch.Tensor,
                   z2: torch.Tensor,
                   labels: torch.Tensor,
                   loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
    return loss_fn(z1, z2, labels)
