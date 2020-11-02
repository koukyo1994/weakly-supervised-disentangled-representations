import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def train_and_validate(loader,
                       valid_loader,
                       dataset,
                       model,
                       optimizer,
                       device,
                       writer: SummaryWriter,
                       save_dir: Path,
                       exp_path,
                       config: dict,
                       task_type="weak"):
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

        writer.add_scalar(tag="loss/batch", scalar_value=loss_meter.val, global_step=step + 1)
        writer.add_scalar(tag="recons/batch", scalar_value=recons_error_meter.val, global_step=step + 1)
        writer.add_scalar(tag="kld/batch", scalar_value=kld_meter.val, global_step=step + 1)

        if (step + 1) % config["logging"]["validate_interval"] == 0:
            model.eval()
            original_pairs, labels = next(iter(valid_loader))
            original_pairs = original_pairs.to(device)
            labels = labels.to(device)

            save_path = save_dir / f"reconstruction_step_{step + 1}.png"

            with torch.no_grad():
                if task_type == "weak":
                    reconstructed_0, reconstructed_1 = model.reconstruct(original_pairs, labels)

                    utils.save_paired_reconstructed_images(
                        save_path,
                        original_pairs.cpu(),
                        reconstructed_0.detach().cpu(),
                        reconstructed_1.detach().cpu(),
                        writer=writer,
                        epoch=step + 1)
                else:
                    reconstructed = model.reconstruct(original_pairs)
                    utils.save_reconstructed_images(
                        save_path,
                        original_pairs.cpu(),
                        reconstructed.detach().cpu(),
                        writer=writer,
                        epoch=step + 1)
            save_path = save_dir / f"latent_traversal_step_{step + 1}.gif"
            utils.latent_traversal(model,
                                   x=original_pairs[0, 0, :, :, :] if task_type == "weak" else original_pairs[0, :, :, :],
                                   device=device,
                                   save_path=save_path,
                                   n_imgs=30,
                                   z_min=-2.5,
                                   z_max=2.5)
            save_path = save_dir / f"latent_traversal_step_{step + 1}.png"
            utils.latent_traversal_static(model,
                                          x=original_pairs[0, 0, :, :, :] if task_type == "weak" else original_pairs[0, :, :, :],
                                          device=device,
                                          save_path=save_path,
                                          n_imgs=10,
                                          z_min=-2.5,
                                          z_max=2.5,
                                          writer=writer,
                                          epoch=step + 1)

            save_path = save_dir / f"latent_histogram_step_{step + 1}.png"
            utils.latent_histogram(
                model, valid_loader, device, save_path, task_type, writer=writer, epoch=step + 1)
            utils.export_model(utils.RepresentationExtractor(model.encoder, "mean"),
                               exp_path,
                               input_shape=model.input_shape,
                               use_script_module=True)
            compute_metrics(exp_path.parent.parent,
                            dataset=dataset,
                            random_seed=config["globals"]["seed"],
                            epoch=step + 1,
                            prefix="step")
            with open(exp_path.parent.parent.parent / "metric_results.json", "r") as f:
                metric_results = json.load(f)
            step_result = metric_results[f"step{step + 1}"]
            for key in step_result:
                writer.add_scalar(tag=key, scalar_value=step_result[key], global_step=step + 1)

        model.train()


def aggregate_multirun_results(base_dir: Path, seeds: list, config: dict):
    metric_results_dict: dict = {}
    for seed in seeds:
        metric_results_path = base_dir / f"seed{seed}/metric_results.json"
        with open(metric_results_path, "r") as f:
            metric_results = json.load(f)

        # metric_results = {"step100": {"dci": 0.222, "sap_score": ...}, ...}
        for step_key in metric_results:
            step = int(step_key.replace("step", ""))
            if metric_results_dict.get(step) is None:
                metric_results_dict[step] = {}
            step_result = metric_results[step_key]
            for metric_key in step_result:
                if metric_results_dict[step].get(metric_key) is None:
                    metric_results_dict[step][metric_key] = []
                metric_results_dict[step][metric_key].append(step_result[metric_key])

    for step in metric_results_dict:
        metric_types = []
        metric_values = []
        for metric_key in metric_results_dict[step]:
            metrics = metric_results_dict[step]
            metric_types.extend([metric_key] * len(metrics[metric_key]))
            metric_values.extend(metrics[metric_key])

        metric_df = pd.DataFrame({
            "metric_types": metric_types,
            "metric_values": metric_values
        })
        ax = sns.violinplot(
            data=metric_df,
            x="metric_types",
            y="metric_values",
            color="skyblue",
            alpha=0.5)
        sns.stripplot(
            data=metric_df,
            x="metric_types",
            y="metric_values",
            color="k",
            alpha=0.5,
            ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("disentanglement metric")
        ax.set_title(f"{config['models']['name']}: {config['dataset']['name']}")
        ax.grid(True)
        ax.set_ylim(0, 1.0)
        plt.savefig(base_dir / f"metrics_step_{step}.png")
        plt.close()


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

    for seed in seeds:
        print("*" * 100)
        print(f"SEED: {seed}")
        if multirun:
            SAVE_DIR = Path(f"assets/run/v2/{args.config.split('/')[-1].replace('.yml', '')}/seed{seed}")
            MODEL_OUTPUT_DIR = Path(
                f"experiment/v2/{args.config.split('/')[-1].replace('.yml', '')}/seed{seed}/checkpoints")
        else:
            SAVE_DIR = Path(f"assets/run/v2/{args.config.split('/')[-1].replace('.yml', '')}")
            MODEL_OUTPUT_DIR = Path(
                f"experiment/v2/{args.config.split('/')[-1].replace('.yml', '')}/checkpoints")

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
            valid_dataset = WeaklySupervisedDataset(
                dataset,
                seed=seed,
                iterator_len=300)
        else:
            torch_dataset = UnsupervisedDataset(  # type: ignore
                dataset,
                seed=seed,
                **config["dataset"]["params"])
            valid_dataset = UnsupervisedDataset(  # type: ignore
                dataset,
                seed=seed,
                iterator_len=300)

        loader = DataLoader(torch_dataset, batch_size=config["loader"]["batch_size"])
        valid_loader = DataLoader(valid_dataset, batch_size=8)
        model = get_model(config).to(device)

        optimizer = getattr(optim, config["optimizer"]["name"], optim.Adam)(
            model.parameters(),
            **config["optimizer"]["params"])

        if not args.skip_train:
            train_and_validate(
                loader,
                valid_loader,
                dataset,
                model,
                optimizer,
                device,
                writer=writer,
                save_dir=SAVE_DIR,
                exp_path=EXP_PATH,
                config=config,
                task_type=task_type)

        writer.close()

    # aggregate results
    if multirun:
        aggregate_multirun_results(
            base_dir=Path(f"experiment/{args.config.split('/')[-1].replace('.yml', '')}"),
            seeds=seeds,
            config=config)
