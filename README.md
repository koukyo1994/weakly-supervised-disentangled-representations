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
