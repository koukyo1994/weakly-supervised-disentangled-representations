import os

os.environ["DISENTANGLEMENT_LIB_DATA"] = "./data"


if __name__ == "__main__":
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

    dataset = get_named_ground_truth_data("dsprites_full")
