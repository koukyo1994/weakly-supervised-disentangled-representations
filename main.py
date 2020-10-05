import os

import torch
import torch.optim as optim

from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_metrics
from models import get_model
from utils import (get_parser, load_config, set_seed, AverageMeter,
                   save_reconstructed_images, latent_traversal,
                   export_model, RepresentationExtractor)

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


def validate(loader, model, device, save_dir: Path, exp_path: Path, epoch: int, config: dict):
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
    save_path = save_dir / f"latent_traversal_epoch_{epoch}.gif"
    latent_traversal(model,
                     x=original_pairs[0, 0, :, :, :],
                     device=device,
                     save_path=save_path,
                     n_imgs=30,
                     z_min=-2.5,
                     z_max=2.5)
    export_model(RepresentationExtractor(model.encoder, "mean"),
                 exp_path,
                 input_shape=model.input_shape,
                 use_script_module=True)
    compute_metrics(exp_path.parent.parent,
                    dataset_name=config["dataset"]["name"],
                    random_seed=config["dataset"]["params"]["seed"])


if __name__ == "__main__":
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    from dataset.pytorch import WeaklySupervisedDataset

    args = get_parser().parse_args()
    config = load_config(args.config)

    SAVE_DIR = Path(f"assets/run/{args.config.split('/')[-1].replace('.yml', '')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_OUTPUT_DIR = Path(
        f"experiment/{args.config.split('/')[-1].replace('.yml', '')}/checkpoints")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EXP_DIR = MODEL_OUTPUT_DIR / "representation" / "results"
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    EXP_PATH = EXP_DIR.parent / "pytorch_model.pt"

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
        if (epoch + 1) % config["logging"]["validate_interval"] == 0:
            validate(valid_loader,
                     model,
                     device,
                     save_dir=SAVE_DIR,
                     exp_path=EXP_PATH,
                     epoch=epoch,
                     config=config)
