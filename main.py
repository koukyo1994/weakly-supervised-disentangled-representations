import os

import torch
import torch.optim as optim

from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from utils import (get_parser, load_config, set_seed, AverageMeter, save_reconstructed_images)

os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"


def train_one_epoch(loader, model, optimizer, device):
    loss_meter = AverageMeter()
    recons_error_meter = AverageMeter()
    kld_meter = AverageMeter()

    model.train()
    tqdm_bar = tqdm(loader)
    for step, (observation, label) in enumerate(tqdm_bar):
        observation = observation.to(device)
        label = label.to(device)
        output = model(observation, label)
        loss, recons_error, kld = model.loss_fn(
            observation, *output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(
            loss.item(),
            n=observation.size(0))
        recons_error_meter.update(
            recons_error.item(),
            n=observation.size(0))
        kld_meter.update(kld.item(), n=observation.size(0))
        tqdm_bar.set_description(
            f"Step: [{step}/{len(loader)}]"
            f"loss: {loss_meter.val:.4f} loss (avg) {loss_meter.avg:.4f} " +
            f"recons_error: {recons_error_meter.val:.4f} recons_error (avg) {recons_error_meter.avg:.4f} " +
            f"kld: {kld_meter.val:.4f} kld (avg) {kld_meter.avg:.4f}")


def validate(loader, model, device, save_dir: Path, epoch: int):
    model.eval()
    original_pairs, labels = next(iter(valid_loader))
    original_pairs = original_pairs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        reconstructed_0, reconstructed_1 = model.reconstruct(original_pairs, labels)

    save_path = save_dir / f"reconstruction_epoch_{epoch}.png"
    save_reconstructed_images(
        save_path,
        original_pairs.cpu(),
        reconstructed_0.detach().cpu(),
        reconstructed_1.detach().cpu())


if __name__ == "__main__":
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    from dataset.pytorch import WeaklySupervisedDataset

    args = get_parser().parse_args()
    config = load_config(args.config)

    SAVE_DIR = Path(f"assets/run/{args.config.split('/')[-1].replace('.yml', '')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(config["globals"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_named_ground_truth_data(config["dataset"]["name"])
    torch_dataset = WeaklySupervisedDataset(
        dataset,
        **config["dataset"]["params"])

    loader = DataLoader(torch_dataset, batch_size=config["loader"]["batch_size"])
    valid_loader = DataLoader(torch_dataset, batch_size=8)
    model = get_model(config).to(device)

    optimizer = getattr(optim, config["optimizer"]["name"], optim.Adam)(
        model.parameters(),
        **config["optimizer"]["params"])

    for epoch in range(config["training"]["epochs"]):
        print("#" * 100)
        print(f"Epoch: {epoch}")
        train_one_epoch(loader, model, optimizer, device)
        validate(valid_loader, model, device, save_dir=SAVE_DIR, epoch=epoch)
