import json
import os

import torch
import torch.optim as optim

import utils

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import compute_metrics
from models import get_model

os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"


def train_one_epoch(loader, model, optimizer, device, writer: SummaryWriter, epoch: int):
    loss_meter = utils.AverageMeter()
    recons_error_meter = utils.AverageMeter()
    kld_meter = utils.AverageMeter()

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
            f"Step: [{step + 1}/{len(loader)}]"
            f"loss: {loss_meter.val:.4f} loss (avg) {loss_meter.avg:.4f} " +
            f"recons_error: {recons_error_meter.val:.4f} recons_error (avg) {recons_error_meter.avg:.4f} " +
            f"kld: {kld_meter.val:.4f} kld (avg) {kld_meter.avg:.4f}")

        global_step = epoch * len(loader) + step
        writer.add_scalar(tag="loss/batch", scalar_value=loss_meter.val, global_step=global_step)
        writer.add_scalar(tag="recons/batch", scalar_value=recons_error_meter.val, global_step=global_step)
        writer.add_scalar(tag="kld/batch", scalar_value=kld_meter.val, global_step=global_step)

    writer.add_scalar(tag="loss/epoch", scalar_value=loss_meter.avg, global_step=epoch)
    writer.add_scalar(tag="recons/epoch", scalar_value=recons_error_meter.avg, global_step=epoch)
    writer.add_scalar(tag="kld/epoch", scalar_value=kld_meter.avg, global_step=epoch)


def validate(loader,
             dataset,
             model,
             device,
             save_dir: Path,
             exp_path: Path,
             epoch: int,
             config: dict,
             task_type: str,
             writer: SummaryWriter):
    model.eval()
    original_pairs, labels = next(iter(valid_loader))
    original_pairs = original_pairs.to(device)
    labels = labels.to(device)

    save_path = save_dir / f"reconstruction_epoch_{epoch}.png"

    with torch.no_grad():
        if task_type == "weak":
            reconstructed_0, reconstructed_1 = model.reconstruct(original_pairs, labels)

            utils.save_paired_reconstructed_images(
                save_path,
                original_pairs.cpu(),
                reconstructed_0.detach().cpu(),
                reconstructed_1.detach().cpu(),
                writer=writer,
                epoch=epoch)

        else:
            reconstructed = model.reconstruct(original_pairs)
            utils.save_reconstructed_images(
                save_path,
                original_pairs.cpu(),
                reconstructed.cpu(),
                writer=writer,
                epoch=epoch)
    save_path = save_dir / f"latent_traversal_epoch_{epoch}.gif"
    utils.latent_traversal(model,
                           x=original_pairs[0, 0, :, :, :] if task_type == "weak" else original_pairs[0, :, :, :],
                           device=device,
                           save_path=save_path,
                           n_imgs=30,
                           z_min=-2.5,
                           z_max=2.5)
    save_path = save_dir / f"latent_traversal_epoch_{epoch}.png"
    utils.latent_traversal_static(model,
                                  x=original_pairs[0, 0, :, :, :] if task_type == "weak" else original_pairs[0, :, :, :],
                                  device=device,
                                  save_path=save_path,
                                  n_imgs=10,
                                  z_min=-2.5,
                                  z_max=2.5,
                                  writer=writer,
                                  epoch=epoch)

    save_path = save_dir / f"latent_histogram_epoch_{epoch}.png"
    utils.latent_histogram(model, loader, device, save_path, task_type, writer=writer, epoch=epoch)
    utils.export_model(utils.RepresentationExtractor(model.encoder, "mean"),
                       exp_path,
                       input_shape=model.input_shape,
                       use_script_module=True)
    compute_metrics(exp_path.parent.parent,
                    dataset=dataset,
                    random_seed=config["dataset"]["params"]["seed"],
                    epoch=epoch)
    with open(exp_path.parent.parent.parent / "metric_results.json", "r") as f:
        metric_results = json.load(f)
    epoch_result = metric_results[f"epoch{epoch}"]
    for key in epoch_result:
        writer.add_scalar(tag=key, scalar_value=epoch_result[key], global_step=epoch)


if __name__ == "__main__":
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    from dataset.pytorch import WeaklySupervisedDataset, UnsupervisedDataset

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]
    if isinstance(global_params["seed"], list):
        multirun = True
        seeds = global_params["seed"]
    else:
        multirun = False
        seeds = [global_params["seed"]]

    SAVE_DIR = Path(f"assets/run/{args.config.split('/')[-1].replace('.yml', '')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print("*" * 100)
        print(f"SEED: {seed}")
        if multirun:
            MODEL_OUTPUT_DIR = Path(
                f"experiment/{args.config.split('/')[-1].replace('.yml', '')}/seed{seed}/checkpoints")
        else:
            MODEL_OUTPUT_DIR = Path(
                f"experiment/{args.config.split('/')[-1].replace('.yml', '')}/checkpoints")
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        EXP_DIR = MODEL_OUTPUT_DIR / "representation" / "results"
        EXP_DIR.mkdir(parents=True, exist_ok=True)

        RESULT_DIR = MODEL_OUTPUT_DIR.parent
        writer = SummaryWriter(log_dir=RESULT_DIR)

        EXP_PATH = EXP_DIR.parent / "pytorch_model.pt"

        utils.set_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = get_named_ground_truth_data(config["dataset"]["name"])
        task_type = config["dataset"]["type"]

        if task_type == "weak":
            torch_dataset = WeaklySupervisedDataset(
                dataset,
                seed=seed,
                **config["dataset"]["params"])
        else:
            torch_dataset = UnsupervisedDataset(  # type: ignore
                dataset,
                seed=seed,
                **config["dataset"]["params"])

        loader = DataLoader(torch_dataset, batch_size=config["loader"]["batch_size"])
        valid_loader = DataLoader(torch_dataset, batch_size=8)
        model = get_model(config).to(device)

        optimizer = getattr(optim, config["optimizer"]["name"], optim.Adam)(
            model.parameters(),
            **config["optimizer"]["params"])

        for epoch in range(config["training"]["epochs"]):
            print("#" * 100)
            print(f"Epoch: {epoch + 1}")
            train_one_epoch(loader, model, optimizer, device, writer, epoch)
            if (epoch + 1) % config["logging"]["validate_interval"] == 0:
                validate(valid_loader,
                         dataset,
                         model,
                         device,
                         save_dir=SAVE_DIR,
                         exp_path=EXP_PATH,
                         epoch=epoch + 1,
                         config=config,
                         task_type=task_type,
                         writer=writer)

        writer.close()
