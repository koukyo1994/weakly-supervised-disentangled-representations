import os

from torch.utils.data import DataLoader

from models import get_model
from utils import get_parser, load_config, set_seed

os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"


if __name__ == "__main__":
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    from dataset.pytorch import WeaklySupervisedDataset

    args = get_parser().parse_args()
    config = load_config(args.config)

    set_seed(config["globals"]["seed"])

    dataset = get_named_ground_truth_data(config["dataset"]["name"])
    torch_dataset = WeaklySupervisedDataset(
        dataset,
        **config["dataset"]["params"])

    loader = DataLoader(torch_dataset, batch_size=config["loader"]["batch_size"])
    model = get_model(config)
