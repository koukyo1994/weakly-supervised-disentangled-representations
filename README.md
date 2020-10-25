# Weakly-supervised Disentangled Representation Learning

An implementation of Variational Autoencoders(VAEs) based weakly-supervised disentangled representation learning methods in PyTorch.

## Supported algorithms

* [x] [GroupVAE](https://arxiv.org/abs/1809.02383)
* [ ] [Multi-LevelVAE](https://arxiv.org/abs/1705.08841)
* [x] [Adaptive-GVAE](https://arxiv.org/abs/2002.02886)
* [ ] [Adaptive-MLVAE](https://arxiv.org/abs/2002.02886)
* [ ] [SlowVAE](https://arxiv.org/abs/2007.10930)

For comparison, I also implement Unsupervised Disentangled Representation Learning methods.

* [x] [BetaVAE](https://openreview.net/forum?id=Sy2fzU9gl)
* [ ] [FactorVAE](https://arxiv.org/abs/1802.05983)
* [x] [BetaVAE with Capacity](https://arxiv.org/abs/1804.03599)
* [ ] [BetaTCVAE](https://arxiv.org/abs/1802.04942)
* [ ] [DIP VAE](https://arxiv.org/abs/1711.00848)

## Requirements

* Python >= 3.6

## Installation

```shell
git clone https://github.com/koukyo1994/weakly-supervised-disentangled-representations
cd weakly-supervised-disentangled-representations
pip install -r requirements.txt
```

## Data preparation

```shell
make prepare
```

## To Run the code

My implementation is a config-based pipeline, and it is easy to use. Everything you need to do is to write a new config file and load it explicitly when you run `main.py` in the format below:

```shell
python main.py --config configs/<your config>.yml
```

The structure of the config file is below:

```yaml
globals:
  seed: 1213  # Whatever integer you want

models:
  name: BetaVAE  # Model name. Make sure that model is implemented. You can check `models/__init__.py` to see which model is implemented right now.
  params:  # Model specific parameters. Please see the implementation of each models and check what kind of arguments are required
    input_shape: [1, 64, 64]
    n_latents: 10
    beta: 16.0

dataset:
  name: dsprites_full  # Valid name for `disentanglement_lib/data/ground_truth/named_data/get_named_ground_truth_data`
  type: unsupervised  # Either `unsupervised` or `weak`
  params:  # Arguments for the pytorch dataset in `dataset/pytorch.py`
    iterator_len: 20000

loader:  # Arguments for `torch.utils.data.DataLoader`
  batch_size: 64

optimizer:
  name: Adam  # Name of optimizer. All the optimizers implemented in `torch.optim` can be used.
  params:  # Argument for the optimizer
    lr: 0.0001

training:
  epochs: 1000

logging:
  validate_interval: 100  # Interval between validation. In validation phase, my pipeline output reconstruction image, latent_traversal gif and png, also histogram of latent vectors, and calculate disentanglement metrics. This will take some time so if you set this interval small, the whole calculation takes a lot of time.
```
