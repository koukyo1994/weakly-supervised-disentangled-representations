import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument("--config", "-c",
                        dest="config",
                        metavar="FILE",
                        help="path/to/config",
                        default="configs/gvae_single_label_d.yml")
    parser.add_argument("--skip-train", action="store_true", dest="skip_train")
    return parser
